"""Capture and replay exact-width CUDA graphs for LoopSpec's three model roles.

Capture records metadata copies, lane-state transfers, optional KV mapping and
attention setup, and model execution. Replay runs those GPU operations without
calling the Python capture helpers again. DecodeBatchPool uploads each role's
packed metadata before RecurrentRequest calls execute_exact().
"""

from collections.abc import Callable, Iterable
from typing import Any

import torch

from sglang.srt.model_executor.runner.decode_cuda_graph_runner import (
    DecodeCudaGraphRunner,
)
from sglang.srt.model_executor.runner_backend.full_cuda_graph_backend import (
    FullCudaGraphBackend,
)
# These modules are already imported by DecodeCudaGraphRunner above; using their
# concrete types does not introduce a new SGLang initialization path.
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors
from sglang.srt.model_executor.runner.shape_key import ShapeKey

from ..modeling.contracts import RecurrentState
from .branching.decode_slots import DecodeSlotPool
from .graph_ops import (
    allocate_decode_slots,
    copy_state_rows,
    update_request_mapping,
)
from .metadata import MetadataRow, payload_size, payload_views


class RecurrentCudaGraphRunner(DecodeCudaGraphRunner):
    """Extend SGLang decode graphs with role-specific graph sets.

    SGLang already owns graph inputs for request rows, positions, cache slots,
    and attention metadata. Recurrent and post additionally copy the hidden
    state inputs ordinary causal LMs do not have.
    """

    def capture(self) -> None:
        """Allocate shared banks and capture each configured (role, batch size).

        Called by SGLang's runner initialization/capture lifecycle, not once per
        request. Batch sizes count internal lanes. Role-specific uint8 payloads
        feed shared graph input buffers; persistent state banks let a lane move
        between different batch rows and widths without losing its hidden state.
        Only attention-bearing roles allocate KV slots and update request maps.
        """
        # SGLang exposes graph-owned decode tensors through
        # DecodeInputBuffers.  These aliases keep LoopSpec's graph operations on
        # the exact same physical storage as SGLang's replay registry.
        self.input_ids = self.buffers.input_ids
        self.req_pool_indices = self.buffers.req_pool_indices
        self.seq_lens = self.buffers.seq_lens
        self.out_cache_loc = self.buffers.out_cache_loc
        self.positions = self.buffers.positions

        original_bs = self.capture_bs
        original_compile_bs = self.compile_bs
        if not original_bs:
            raise ValueError("recurrent CUDA graphs require tree time widths")

        # One CUDA upload destination per exact graph shape, not per request.
        self.recurrent_metadata_banks: dict[tuple[str, int], torch.Tensor] = {}
        self.recurrent_metadata_initialized: set[tuple[str, int]] = set()
        self.mapping_parent_req_pool_indices = torch.zeros_like(self.req_pool_indices)
        self.mapping_copy_lens = torch.full_like(self.seq_lens, -1)
        allocator = self.model_runner.token_to_kv_pool_allocator
        self.decode_slots = DecodeSlotPool(allocator.size, self.req_pool_indices.device)

        # TODO: Move cuda_graph_inputs() and its per-width buffer cache from
        # RecurrentServingModel into this runner; they are graph-owned scratch.
        hidden, injected = self.model_runner.model.cuda_graph_inputs(original_bs[-1])
        # Bank indices are persistent lane-state slots, NOT request-pool rows
        # or KV slot IDs. Per-width model inputs are separate gather buffers.
        self.recurrent_hidden_bank = torch.zeros_like(hidden)
        self.recurrent_injected_bank = (None if injected is None else torch.zeros_like(injected))
        self.recurrent_token_bank = torch.zeros_like(self.input_ids)
        self.recurrent_position_bank = torch.zeros_like(self.positions)
        self.state_input_slots = torch.full_like(self.seq_lens, -1)
        self.state_output_slots = torch.full_like(self.seq_lens, -1)

        roles = ("recurrent", "pre", "post")
        self.recurrent_mapping_roles = frozenset(
            role
            for role in roles
            if self.model_runner.model.attention_layers[role]
        )

        if not isinstance(self.backend, FullCudaGraphBackend):
            raise ValueError("LoopSpec requires SGLang's full CUDA graph backend")

        configured = getattr(self.model_runner.server_args, "recurrent_role_batch_sizes", None)
        if configured is None:
            raise ValueError("LoopSpec requires role CUDA graph sizes")
        role_batch_sizes = {role: list(configured[role]) for role in roles}
        if any(
            not sizes or not set(sizes) <= set(original_bs)
            for sizes in role_batch_sizes.values()
        ):
            raise ValueError("role CUDA graph sizes exceed configured batch sizes")
        self.role_batch_sizes = role_batch_sizes

        for role in roles:
            self.capture_role = role
            self.capture_bs = role_batch_sizes[role]
            self.compile_bs = [size for size in original_compile_bs if size in self.capture_bs]
            super().capture()

        self.capture_role = None
        self.compile_bs = original_compile_bs
        self.capture_bs = original_bs
        self._active_role = roles[0]
        self.recurrent_graph_keys: dict[tuple[str, int, str | None], ShapeKey] = {}
        for role, sizes in role_batch_sizes.items():
            self._active_role = role
            for size in sizes:
                self.recurrent_graph_keys[(role, size, None)] = (self._make_graph_key(size))
        self._active_role = roles[0]

    def _make_graph_key(
        self,
        size: int,
        stream_idx: int | None = None,
        variant_label: str | None = None,
    ) -> ShapeKey:
        """Return SGLang's shape key with the current role in its variant label.

        size is the exact lane count; stream_idx and variant_label preserve the
        upstream key interface. Capture uses capture_role; replay uses
        _active_role. Thus pre/core/post graphs of the same width never collide.
        """
        role = getattr(self, "capture_role", None) or getattr(self, "_active_role", None)
        if role is not None:
            variant_label = (role if variant_label is None else f"{role}:{variant_label}")
        return super()._make_graph_key(size, stream_idx, variant_label)

    def capture_one_shape(
        self,
        size: int,
        forward: Callable[..., RecurrentState | LogitsProcessorOutput],
        stream_idx: int | None = None,
        variant_label: str | None = None,
    ) -> None:
        """Wrap one model callable before upstream warmup and graph capture.

        size is the lane count for capture_role. forward is the model callable
        supplied by SGLang, possibly compile-wrapped. stream_idx/variant_label
        are upstream capture-key options. The parent stores the captured graph
        and its output; this method returns nothing in the supported SGLang API.
        """
        bs = size
        role = self.capture_role
        self.recurrent_metadata_banks[(role, bs)] = torch.empty(
            payload_size(bs),
            dtype=torch.uint8,
            device=self.req_pool_indices.device,
        )

        def role_forward(
            input_ids: torch.Tensor,
            positions: torch.Tensor,
            forward_batch: ForwardBatch,
            **kwargs: Any,
        ) -> RecurrentState | LogitsProcessorOutput:
            """Record one role's data movement and forward in execution order.

            input_ids/positions are CUDA [B] graph inputs; forward_batch is the
            upstream capture batch, mutated to carry the role and state inputs.
            kwargs are forwarded unchanged. Pre/core return RecurrentState;
            post returns logits. Python runs during warmup/capture only.
            """
            forward_batch.recurrent_role = role
            self._capture_staged_metadata(bs, role, input_ids, positions, forward_batch)
            if role != "pre":
                self._capture_state_inputs(bs, role, forward_batch)
            self._capture_token_position_transfer(bs, role, input_ids, positions)
            if self.manages_recurrent_mapping(role):
                self._capture_mapping_and_attention_preamble(forward_batch)
            output = forward(input_ids, positions, forward_batch, **kwargs)
            self._capture_state_scatter(bs, role, output)
            return output

        return super().capture_one_shape(
            size,
            role_forward,
            stream_idx=stream_idx,
            variant_label=variant_label,
        )

    def _capture_token_position_transfer(
        self,
        bs: int,
        role: str,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> None:
        """Record recurrent token/position scatter or post gather; pre is a no-op.

        input_ids/positions are CUDA [B] buffers for bs lanes. Recurrent saves
        them at STATE_INPUT bank indices; post restores the selected rows into
        its own batch order, replacing any dummy IDs staged by the caller.
        Triton kernels move the values device-to-device, without a CPU readback.
        """
        if role not in ("recurrent", "post"):
            return
        gather = role == "post"
        for values, bank in (
            (input_ids, self.recurrent_token_bank),
            (positions, self.recurrent_position_bank),
        ):
            source, destination = (bank, values) if gather else (values, bank)
            copy_state_rows(source, destination, self.state_input_slots[:bs], gather=gather)

    def _staged_metadata_views(
        self, role: str, bs: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (CUDA uint8 payload, int64 metadata [8, B]) without copying.

        role is pre/recurrent/post and bs is an exact captured width. Missing
        pairs raise ValueError. Both returned tensors alias the graph's upload
        destination; the payload also contains an int32 orig_seq_lens tail.
        """
        try:
            payload = self.recurrent_metadata_banks[(role, bs)]
        except KeyError as error:
            raise ValueError(f"{role} requires an exact CUDA graph width {bs}") from error
        rows, _ = payload_views(payload, bs)
        return payload, rows

    def _capture_staged_metadata(
        self,
        bs: int,
        role: str,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> None:
        """Seed one payload once, then record copies into shared graph buffers.

        bs/role select the payload; input_ids/positions and forward_batch supply
        safe initial CUDA capture values. Negative mapping-copy lengths disable
        KV allocation/map writes during capture. Replay consumes new values
        uploaded by DecodeBatchPool, not these Python initialization branches.
        Attention-free pre needs position/token/output slots; attention-free
        post needs only input slots. Attention roles copy all eight metadata rows.
        """
        _, values = self._staged_metadata_views(role, bs)
        key = (role, bs)
        if key not in self.recurrent_metadata_initialized:
            values.zero_()
            values[MetadataRow.REQUEST].copy_(forward_batch.req_pool_indices[:bs])
            values[MetadataRow.SEQUENCE_LENGTH].copy_(forward_batch.seq_lens[:bs])
            values[MetadataRow.POSITION].copy_(positions[:bs])
            values[MetadataRow.TOKEN].copy_(input_ids[:bs])
            values[MetadataRow.PARENT_REQUEST].copy_(forward_batch.req_pool_indices[:bs])
            values[MetadataRow.MAPPING_COPY_LENGTH].fill_(-1)
            slots = torch.arange(bs, dtype=torch.int64, device=values.device)
            values[MetadataRow.STATE_INPUT].copy_(slots)
            values[MetadataRow.STATE_OUTPUT].copy_(slots)
            self.recurrent_metadata_initialized.add(key)

        manages_mapping = self.manages_recurrent_mapping(role)
        if role == "pre" and not manages_mapping:
            transfers = (
                (self.positions, values[MetadataRow.POSITION]),
                (self.input_ids, values[MetadataRow.TOKEN]),
                (self.state_output_slots, values[MetadataRow.STATE_OUTPUT]),
            )
        elif role == "post" and not manages_mapping:
            # Attention-free tails read token, position, and hidden state from
            # the persistent lane banks. Only the selected lane IDs vary.
            transfers = (
                (self.state_input_slots, values[MetadataRow.STATE_INPUT]),
            )
        else:
            transfers = (
                (self.req_pool_indices, values[MetadataRow.REQUEST]),
                (self.seq_lens, values[MetadataRow.SEQUENCE_LENGTH]),
                (self.positions, values[MetadataRow.POSITION]),
                (self.input_ids, values[MetadataRow.TOKEN]),
                # TODO: Remove these two unused copies and their destination
                # allocations in capture(). Allocation/mapping kernels already
                # read PARENT_REQUEST and MAPPING_COPY_LENGTH from values directly.
                (
                    self.mapping_parent_req_pool_indices,
                    values[MetadataRow.PARENT_REQUEST],
                ),
                (
                    self.mapping_copy_lens,
                    values[MetadataRow.MAPPING_COPY_LENGTH],
                ),
                (self.state_input_slots, values[MetadataRow.STATE_INPUT]),
                (self.state_output_slots, values[MetadataRow.STATE_OUTPUT]),
            )

        for destination, source in transfers:
            destination[:bs].copy_(source)

    def direct_graph_payload(self, role: str, bs: int) -> torch.Tensor:
        """Return the mutable CUDA uint8 [payload_size(bs)] upload destination.

        DecodeBatchPool uses this for both attention-bearing decode and
        attention-free direct batches. role/bs must identify a captured graph.
        The caller orders upload before replay and before reusing the buffer;
        this returns shared storage, not a snapshot or a new allocation.
        """
        return self._staged_metadata_views(role, bs)[0]

    def manages_recurrent_mapping(self, role: str) -> bool:
        """Return whether this role graph owns KV mapping and attention setup.

        capture() selects roles with attention layers. Attention-free pre/post
        and unknown role names return False; this is not graph-width validation.
        """
        return role in self.recurrent_mapping_roles

    def _capture_state_inputs(
        self, bs: int, role: str, forward_batch: ForwardBatch
    ) -> None:
        """Attach per-width inputs and record hidden-state gather before forward.

        Called for recurrent/post only. STATE_INPUT maps bs batch rows to the
        persistent CUDA [capacity, hidden_size] bank. Gather fills model scratch
        [bs, hidden_size]; Raven injected state is gathered only for recurrent.
        forward_batch is mutated in place; Ouro's injected input stays None.
        """
        hidden, injected = self.model_runner.model.cuda_graph_inputs(bs)
        forward_batch.hidden_states = hidden
        forward_batch.injected = injected
        copy_state_rows(self.recurrent_hidden_bank, hidden, self.state_input_slots[:bs], gather=True)
        # Raven's injected prompt state is consumed only by the recurrent
        # adapter.  The post executor reads hidden alone, so gathering the
        # immutable injected row for post only adds a graph kernel and a full
        # state-row read/write.
        if injected is not None and role == "recurrent":
            copy_state_rows(self.recurrent_injected_bank, injected, self.state_input_slots[:bs], gather=True)

    def _capture_state_scatter(
        self,
        bs: int,
        role: str,
        output: RecurrentState | LogitsProcessorOutput,
    ) -> None:
        """Record pre/core output writes to STATE_OUTPUT bank slots; skip post.

        output must be RecurrentState for pre/core, with hidden [bs, hidden_size].
        Pre also stores Raven's injected state; core preserves that existing
        bank. Slots are lane-state indices, not the current batch row numbers.
        An invalid pre/core output raises TypeError during warmup/capture.
        """
        if role not in ("pre", "recurrent"):
            return
        if not isinstance(output, RecurrentState):
            raise TypeError(f"{role} graph returned an invalid recurrent state")
        copy_state_rows(output.hidden, self.recurrent_hidden_bank, self.state_output_slots[:bs], gather=False)
        # Pre initializes Raven's injected state.  Recurrent never mutates it,
        # so scattering it again after every recurrent graph is redundant.
        if output.injected is not None and role == "pre":
            copy_state_rows(output.injected, self.recurrent_injected_bank, self.state_output_slots[:bs], gather=False)

    def _capture_mapping_and_attention_preamble(
        self, forward_batch: ForwardBatch
    ) -> None:
        """Record slot allocation -> request mapping -> attention metadata setup.

        forward_batch is the attention-bearing role's fixed capture batch.
        Allocate one KV slot per active metadata row, copy a fork's prefix map
        if needed, and append the new slot. CPU DecodeSlotPool reservations must
        match those allocations at replay. Attention setup runs after the map
        update on the same stream, despite its upstream 'out_graph' method name.
        """
        bs = forward_batch.batch_size
        _, values = self._staged_metadata_views(forward_batch.recurrent_role, bs)
        allocate_decode_slots(
            self.decode_slots.free_slots,
            self.decode_slots.allocation_cursor,
            values[MetadataRow.MAPPING_COPY_LENGTH],
            self.out_cache_loc[:bs],
        )
        update_request_mapping(
            self.model_runner.req_to_token_pool.req_to_token,
            values[MetadataRow.REQUEST],
            values[MetadataRow.PARENT_REQUEST],
            values[MetadataRow.MAPPING_COPY_LENGTH],
            values[MetadataRow.POSITION],
            self.out_cache_loc[:bs],
        )
        # SGLang normally runs this from DecodeCudaGraphRunner.load_batch,
        # outside the graph. LoopSpec owns fixed-shape graph buffers and stages all
        # varying inputs before replay, so the Triton buffer-update kernels are
        # graph-safe here. This restores the original mapping -> metadata ->
        # attention ordering without exposing per-role setup kernels between
        # token decisions.
        self.attn_backend.init_forward_metadata_out_graph(forward_batch)

    def begin_recurrent_request(self, allocated_slots: int) -> None:
        """Reset decode slot ownership/ring after this request's eager prefill.

        allocated_slots counts the initial allocator range already occupied by
        prefill, not requests or lanes. RecurrentRequest.prefill calls this once;
        earlier request GPU work must have completed before resetting the ring.
        """
        self.decode_slots.begin(allocated_slots)

    def release_decode_slots(self, slots: Iterable[int]) -> None:
        """Recycle CPU-tracked KV IDs released by the branch manager.

        slots contains IDs owned exclusively by pruned graph cursors, not bank
        indices or K/V tensors. This changes ownership without clearing K/V;
        later allocation may refill the GPU ring. Stream ordering must protect
        any still-pending GPU use before the same slot is written again.
        """
        self.decode_slots.release(slots)

    def graph_decode_locs(self, count: int) -> tuple[torch.Tensor, list[int]]:
        """Reserve count KV slots; return (GPU output view, CPU slot ID list).

        DecodeBatchPool calls this before an attention-role replay, once per
        allocation batch. The CUDA out_cache_loc[:count] view is filled later
        by the captured allocator kernel; it does not yet contain this call's
        reserved IDs. CPU exhaustion raises before replay instead of wrapping
        into live slots. Recycled ring entries are uploaded only when needed.
        """
        slots = self.decode_slots.allocate(count)
        return self.out_cache_loc[:count], slots

    def can_run_graph(self, forward_batch: ForwardBatch) -> bool:
        """Check exact role/width eligibility, then ask the upstream backend.

        Replacement embeddings, unknown roles, and uncaptured widths return
        False. Used by the generic runner dispatcher; LoopSpec execute_exact()
        bypasses this check and validates role/width in load_batch instead.
        No model execution occurs here; graph-key lookup may populate its cache.
        """
        role = getattr(forward_batch, "recurrent_role", None)
        if forward_batch.replace_embeds is not None:
            return False
        if role not in self.role_batch_sizes:
            return False
        raw_bs = forward_batch.batch_size
        if raw_bs not in self.role_batch_sizes[role]:
            return False
        graph_key = self._graph_key(forward_batch, role, raw_bs)
        return self.backend.can_run(forward_batch, graph_key)

    def _graph_key(
        self, forward_batch: ForwardBatch, role: str, raw_bs: int
    ) -> ShapeKey:
        """Return/cache a ShapeKey for (role, exact width, upstream LoRA variant).

        forward_batch supplies LoRA IDs when that upstream option is enabled.
        raw_bs is the unpadded lane count. Cache misses set _active_role before
        key construction. This only builds a key; it does not capture a graph
        or certify that the requested model feature is supported by LoopSpec.
        """
        variant = (
            self._resolve_lora_variant(forward_batch)
            if self.model_runner.server_args.enable_lora
            else None
        )
        cache_key = (role, raw_bs, variant)
        graph_key = self.recurrent_graph_keys.get(cache_key)
        if graph_key is None:
            self._active_role = role
            graph_key = self._make_graph_key(raw_bs, variant_label=variant)
            self.recurrent_graph_keys[cache_key] = graph_key
        return graph_key

    def load_batch(
        self,
        forward_batch: ForwardBatch,
        pp_proxy_tensors: PPProxyTensors | None = None,
    ) -> None:
        """Validate a staged batch and select its exact role/width replay key.

        forward_batch supplies role, width, and any variant metadata; its packed
        GPU inputs were already uploaded by DecodeBatchPool. Reject PP proxies,
        PDMux streams, unknown roles, and uncaptured widths. No padding, generic
        buffer fill, or attention setup happens here; the graph owns those
        required metadata operations. This mutates runner replay-selection state.
        """
        if pp_proxy_tensors is not None:
            raise ValueError("LoopSpec does not support pipeline-parallel proxies")
        if self.enable_pdmux:
            raise ValueError("LoopSpec does not support PDMux graph streams")
        role = getattr(forward_batch, "recurrent_role", None)
        if role not in self.role_batch_sizes:
            raise ValueError(f"invalid LoopSpec CUDA graph role: {role!r}")
        raw_bs = forward_batch.batch_size
        if raw_bs not in self.role_batch_sizes[role]:
            raise ValueError(f"{role} requires an exact CUDA graph width {raw_bs}")

        self._active_role = role
        self.raw_bs = raw_bs
        self.raw_num_token = raw_bs * self.num_tokens_per_bs
        self.bs = raw_bs
        self._replay_graph_key = self._graph_key(forward_batch, role, raw_bs)

    def execute(
        self,
        forward_batch: ForwardBatch,
        pp_proxy_tensors: PPProxyTensors | None = None,
    ) -> RecurrentState | LogitsProcessorOutput:
        """Select/replay one staged role graph and expose its reusable output.

        load_batch validates forward_batch and rejects non-None PP proxies.
        Post returns the backend's logits object; pre/core wrap captured hidden
        and injected views with this batch's positions. No output tensor is
        cloned and no completion synchronization is added. Consumers must use
        or copy the data before later graph execution reuses its storage.
        """
        role = getattr(forward_batch, "recurrent_role", None)
        self.load_batch(forward_batch, pp_proxy_tensors)
        output = self.backend.replay(self._replay_graph_key, forward_batch)

        if role == "post":
            if not hasattr(output, "next_token_logits"):
                raise TypeError("post CUDA graph returned invalid logits")
            return output

        # Pre/core graphs intentionally return recurrent state rather than a
        # causal-LM output. Exact LoopSpec graph widths need no generic padding
        # or output adapter.
        if not isinstance(output, RecurrentState):
            raise TypeError(f"{role} CUDA graph returned an invalid state")

        return RecurrentState(
            hidden=output.hidden[: self.raw_num_token],
            positions=forward_batch.positions,
            injected=(None if output.injected is None else output.injected[: self.raw_num_token]),
        )

    def execute_exact(
        self, forward_batch: ForwardBatch
    ) -> RecurrentState | LogitsProcessorOutput:
        """Replay a LoopSpec-owned exact-width role graph.

        DecodeBatchPool only emits role/width pairs captured by this runner, so
        the generic ModelRunner feature dispatcher and duplicate eligibility
        pass are unnecessary. load_batch remains the validation boundary.
        RecurrentRequest._forward calls this for graph-mode batches; eager
        prefill uses ModelRunner.forward instead. Reject non-graph modes and
        return execute()'s shared state/logits output without copying or waiting.
        """
        if not forward_batch.forward_mode.is_cuda_graph():
            raise ValueError("execute_exact requires a CUDA-graph batch")
        return self.execute(forward_batch)
