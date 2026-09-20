"""Reuse CPU upload buffers and ForwardBatch objects for LoopSpec decode graphs.

CPU staging slots are selected by call order within each cycle. GPU payloads
and ForwardBatch objects are reused by role and exact batch size. Returned
batches reference mutable storage; they are not snapshots of earlier calls.
"""

from collections.abc import Callable, Mapping, Sequence
from copy import copy
from typing import Any, TypeVar, cast

from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
)
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

from .metadata import DecodeStagingSlot, DirectMetadataStagingSlot, MetadataRow


_StagingSlot = TypeVar("_StagingSlot", DecodeStagingSlot, DirectMetadataStagingSlot)


class DecodeBatchPool:
    """Reuse pinned staging slots and exact-width SGLang forward batches."""

    def __init__(self, runner: Any) -> None:
        """Create empty caches for one RecurrentRequest's model runner.

        runner is SGLang's ModelRunner with a RecurrentCudaGraphRunner installed.
        Use Any to avoid importing the model-loading runtime for annotations.
        Buffers and batches are created only when first requested.
        """
        self.runner = runner
        # Reusable pinned CPU buffers used to upload metadata, not model states
        # or KV data. Keys: ("decode", B) or ("direct", role, B).
        # Each list holds separate buffers for repeated uses of its key within
        # one executor.forward() call. Later calls reuse the allocated buffers.
        self.staging: dict[
            tuple[str, int] | tuple[str, str, int],
            list[DecodeStagingSlot | DirectMetadataStagingSlot],
        ] = {}
        # One ForwardBatch object per (role, B) for roles WITH attention.
        # Its GPU fields reference reusable graph buffers; its CPU lengths are
        # linked to the staging slot selected for the current decode() call.
        self.decode_batches: dict[tuple[str, int], ForwardBatch] = {}
        # One ForwardBatch object per (role, B) for pre/post WITHOUT attention.
        # Initially a shallow copy of the supplied template, then reused with
        # graph-buffer views. No separate CPU sequence-length array is attached.
        self.direct_batches: dict[tuple[str, int], ForwardBatch] = {}
        # Next list index to use in staging[key] during this executor.forward().
        # begin_cycle() clears these counters, but keeps all three caches above.
        self.cycle_uses: dict[tuple[str, int] | tuple[str, str, int], int] = {}

    def begin_cycle(self) -> None:
        """Restart CPU slot selection at index zero for every staging key.

        Keep allocated buffers and batches. This does not synchronize CUDA;
        each staging slot waits for its previous upload before CPU writes.
        """
        self.cycle_uses.clear()

    def _acquire(
        self,
        key: tuple[str, int] | tuple[str, str, int],
        factory: Callable[[], _StagingSlot],
    ) -> _StagingSlot:
        """Return the next CPU staging slot for key, creating it with factory if needed.

        Repeated calls for the same key in one cycle use different slots, so
        their CPU length arrays remain separate. The next cycle reuses those
        slots in the same order. Each key must consistently use one slot class.
        """
        slots = self.staging.setdefault(key, [])
        index = self.cycle_uses.get(key, 0)
        self.cycle_uses[key] = index + 1
        if index == len(slots):
            slots.append(factory())
        # The key's factory fixes its slot class; cast only informs type checkers.
        return cast(_StagingSlot, slots[index])

    def decode(
        self,
        role: str,
        cursors: Sequence[Any],
        token_ids: Sequence[int],
        state_input_slots: Sequence[int] | None = None,
        state_output_slots: Sequence[int] | None = None,
    ) -> tuple[ForwardBatch, list[int]]:
        """Prepare an attention-bearing role's batch and reserve its new KV slots.

        role is pre, recurrent, or post with attention layers. cursors contains
        one TimelineCursor per lane; Any avoids a branch-manager import solely
        for typing. token_ids and any supplied state-slot sequences follow the
        same lane order and length. State slots index hidden/injected banks,
        not request-pool rows or KV slots; None means no state copy in that direction.

        Return a reusable ForwardBatch and a Python list of newly reserved KV
        slot IDs. Metadata views are CUDA tensors; seq_lens_cpu is pinned CPU
        memory. The graph writes out_cache_loc when replayed. This method does
        not run the model or advance cursors; RecurrentRequest._decode does that.
        """
        batch_size = len(cursors)
        graph_runner = self.runner.decode_cuda_graph_runner
        staging = self._acquire(("decode", batch_size), lambda: DecodeStagingSlot(batch_size))

        metadata, orig_seq_lens, staged_tokens, seq_lens_cpu = staging.stage(
            cursors,
            token_ids,
            graph_runner.direct_graph_payload(role, batch_size),
            state_input_slots,
            state_output_slots,
        )
        out_cache_loc, graph_slots = graph_runner.graph_decode_locs(batch_size)

        seq_lens_sum = sum(cursor.seq_len + 1 for cursor in cursors)
        key = (role, batch_size)
        forward_batch = self.decode_batches.get(key)
        if forward_batch is None:
            forward_batch = ForwardBatch(
                forward_mode=ForwardMode.DECODE,
                batch_size=batch_size,
                input_ids=staged_tokens,
                req_pool_indices=metadata[MetadataRow.REQUEST],
                seq_lens=metadata[MetadataRow.SEQUENCE_LENGTH],
                out_cache_loc=out_cache_loc,
                seq_lens_sum=seq_lens_sum,
                orig_seq_lens=orig_seq_lens,
                seq_lens_cpu=seq_lens_cpu,
                positions=metadata[MetadataRow.POSITION],
                spec_algorithm=SpeculativeAlgorithm.NONE,
                capture_hidden_mode=CaptureHiddenMode.NULL,
            )
            self.decode_batches[key] = forward_batch
        else:
            # GPU fields already point into the same role/size payload updated
            # above. The CPU length array can come from a different staging slot.
            forward_batch.seq_lens_sum = seq_lens_sum
            forward_batch.seq_lens_cpu = seq_lens_cpu
        return forward_batch, graph_slots

    def direct(
        self,
        role: str,
        template: ForwardBatch,
        batch_size: int,
        metadata_rows: Mapping[MetadataRow, Sequence[int]],
    ) -> ForwardBatch:
        """Prepare an attention-free pre/post batch without reserving KV slots.

        role must be pre or post; another role raises ValueError. template is
        shallow-copied only when first creating this role/size batch. batch_size
        is the number of selected lanes, not necessarily the template's width.
        metadata_rows supplies B values per selected row: pre uses TOKEN,
        POSITION, STATE_OUTPUT; post uses STATE_INPUT. Unspecified payload rows
        are not read by these graphs.

        Return the cached batch with CUDA views into the uploaded payload and
        no CPU sequence-length array. This prepares inputs but does not replay
        the graph. Shared tensor storage must not be treated as a saved snapshot.
        """
        graph_runner = self.runner.decode_cuda_graph_runner
        staging = self._acquire(("direct", role, batch_size), lambda: DirectMetadataStagingSlot(batch_size))
        staged = staging.stage(metadata_rows, graph_runner.direct_graph_payload(role, batch_size))

        key = (role, batch_size)
        forward_batch = self.direct_batches.get(key)
        if forward_batch is None:
            forward_batch = copy(template)
            forward_batch.forward_mode = forward_batch.global_forward_mode = graph_runner.capture_forward_mode
            forward_batch.batch_size = batch_size
            forward_batch.input_ids = staged[MetadataRow.TOKEN]
            forward_batch.positions = staged[MetadataRow.POSITION]
            # Keep these per-lane fields at the selected graph width instead of
            # retaining wider template views (e.g. a 4-lane core batch feeding
            # a 3-lane post graph). Without attention, request/length fields are
            # placeholders; the graph reads only the role's selected metadata rows.
            forward_batch.req_pool_indices = staged[MetadataRow.REQUEST]
            forward_batch.seq_lens = staged[MetadataRow.SEQUENCE_LENGTH]
            forward_batch.seq_lens_cpu = None
            forward_batch.out_cache_loc = graph_runner.out_cache_loc[:batch_size]
            self.direct_batches[key] = forward_batch

        if role == "pre":
            forward_batch.recurrent_state_output_slots = staged[MetadataRow.STATE_OUTPUT]
        elif role == "post":
            forward_batch.recurrent_state_input_slots = staged[MetadataRow.STATE_INPUT]
        else:
            raise ValueError(f"invalid attention-free role: {role}")
        return forward_batch
