"""Pack CPU timeline metadata into the buffers consumed by LoopSpec CUDA graphs.

For B pipeline lanes, one upload contains int64 rows [8, B] followed by
int32 original sequence lengths [B]. These are metadata, not hidden/KV tensors.
"""

from collections.abc import Mapping, Sequence
from enum import IntEnum
from typing import Any

import torch


class MetadataRow(IntEnum):
    """Row indices in an int64 [8, batch_size] view; columns are pipeline lanes."""

    REQUEST = 0  # Destination row in SGLang's request-to-token mapping.
    SEQUENCE_LENGTH = 1  # Cached prefix length plus the token being decoded.
    POSITION = 2  # Zero-based position of that token (the old prefix length).
    TOKEN = 3  # Input token ID.
    PARENT_REQUEST = 4  # Source mapping row for a newly forked timeline.
    MAPPING_COPY_LENGTH = 5  # Prefix entries to copy; 0 means only append KV.
    STATE_INPUT = 6  # Hidden/injected bank row to gather; -1 skips the copy.
    STATE_OUTPUT = 7  # State bank row to scatter into; -1 skips the copy.


DECODE_METADATA_ROWS = len(MetadataRow)


def payload_size(batch_size: int) -> int:
    """Return the upload size in bytes for batch_size pipeline lanes.

    Includes eight int64 fields and one int32 length per lane (68 bytes).
    Does not include DecodeStagingSlot's separate CPU-only sequence lengths.
    """
    return (
        DECODE_METADATA_ROWS * batch_size * torch.int64.itemsize
        + batch_size * torch.int32.itemsize
    )


def payload_views(
    payload: torch.Tensor,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (metadata rows, original lengths) as views without copying.

    payload must be a contiguous uint8 tensor of payload_size(batch_size)
    bytes, on CPU or CUDA. Returns int64 [8, batch_size] and int32 [batch_size]
    tensors on the same device, sharing the supplied tensor's storage.
    """
    rows_nbytes = DECODE_METADATA_ROWS * batch_size * torch.int64.itemsize
    rows = payload[:rows_nbytes].view(torch.int64).view(DECODE_METADATA_ROWS, batch_size)
    original_lengths = payload[rows_nbytes:].view(torch.int32)
    return rows, original_lengths


class _PinnedUploadSlot:
    """Keep a pinned H2D source alive until its asynchronous copy completes."""

    def __init__(self) -> None:
        """Create the completion event used to protect this slot's CPU buffer."""
        self._upload_finished = torch.cuda.Event()
        self._upload_pending = False

    def _wait_for_upload(self) -> None:
        """Wait for the last recorded upload before overwriting its CPU source.

        A pinned non_blocking copy can still be reading host memory after
        copy_ returns. This waits for that copy, not subsequent graph execution.
        Does nothing when no upload has been recorded since the previous wait.
        """
        if not self._upload_pending:
            return
        self._upload_finished.synchronize()
        self._upload_pending = False

    def _record_upload(self, device: torch.device | str | int) -> None:
        """Record completion after the copy queued on device's current stream.

        device identifies the upload's CUDA destination. This records an event
        without waiting; the next reuse of this CPU slot performs the wait.
        """
        self._upload_finished.record(torch.cuda.current_stream(device))
        self._upload_pending = True


class DecodeStagingSlot(_PinnedUploadSlot):
    """Pinned request metadata uploaded into an exact CUDA graph payload."""

    def __init__(self, batch_size: int) -> None:
        """Allocate reusable pinned CPU storage for exactly batch_size lanes."""
        super().__init__()
        self.batch_size = batch_size
        self.payload_nbytes = payload_size(batch_size)

        # Host layout: [uploaded metadata + orig_seq_lens | CPU seq_lens].
        # The last int32[B] segment stays on CPU for ForwardBatch.seq_lens_cpu.
        self.host = torch.empty(
            self.payload_nbytes + batch_size * torch.int32.itemsize,
            dtype=torch.uint8,
            pin_memory=True,
        )

        self.host_rows, self.host_original_lengths = payload_views(self.host[: self.payload_nbytes], batch_size)
        # NumPy shares the pinned tensor's storage; assigning here fills it.
        self.host_rows_numpy = self.host_rows.numpy()
        self.host_sequence_lengths = self.host[self.payload_nbytes :].view(torch.int32)

    def stage(
        self,
        cursors: Sequence[Any],
        token_ids: Sequence[int],
        direct_graph_payload: torch.Tensor,
        state_input_slots: Sequence[int] | None = None,
        state_output_slots: Sequence[int] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Upload one attention-bearing decode batch and return its tensor views.

        cursors contains one TimelineCursor per lane, in token_ids order.
        Use Any to avoid importing the branch manager and its SGLang runtime
        dependencies just for annotations. Cursor fields are read, not updated.
        Each supplied sequence must contain batch_size entries.

        direct_graph_payload is the selected role/width graph's CUDA uint8
        buffer of payload_size(batch_size) bytes. Optional state slots select
        persistent hidden/injected bank rows, not request rows or KV slots;
        None fills the corresponding field with the -1 skip sentinel.

        Returns (CUDA int64 metadata [8, B], CUDA int32 orig_seq_lens [B],
        CUDA int64 token IDs [B], pinned CPU int32 seq_lens [B]). These are
        reusable views, not snapshots. Upload is asynchronous: the consuming
        graph must follow it on the same stream or explicitly wait for it.
        """
        batch_size = self.batch_size

        request_ids = [cursor.req_pool_idx for cursor in cursors]
        # Cursor lengths still describe the prefix before this decode token.
        sequence_lengths = [cursor.seq_len + 1 for cursor in cursors]
        positions = [cursor.seq_len for cursor in cursors]
        # A fork copies its parent's prefix mapping once; an existing timeline
        # uses its own row with copy length 0. The graph appends the new KV slot.
        parent_request_ids = [
            cursor.req_pool_idx
            if cursor.pending_parent_req_pool_idx is None
            else cursor.pending_parent_req_pool_idx
            for cursor in cursors
        ]
        mapping_copy_lengths = [
            0
            if cursor.pending_parent_req_pool_idx is None
            else cursor.seq_len
            for cursor in cursors
        ]
        input_slots = (
            [-1] * batch_size
            if state_input_slots is None
            else state_input_slots
        )
        output_slots = (
            [-1] * batch_size
            if state_output_slots is None
            else state_output_slots
        )
        rows = [
            request_ids,
            sequence_lengths,
            positions,
            token_ids,
            parent_request_ids,
            mapping_copy_lengths,
            input_slots,
            output_slots,
        ]

        # One NumPy assignment is faster than converting each tiny row.
        self._wait_for_upload()
        self.host_rows_numpy[:] = rows
        # orig_seq_lens and seq_lens_cpu are distinct SGLang fields, but both
        # contain the new full lengths for this non-padded decode batch.
        self.host_original_lengths.numpy()[:] = sequence_lengths
        self.host_sequence_lengths.numpy()[:] = sequence_lengths

        payload = direct_graph_payload
        payload.copy_(self.host[: self.payload_nbytes], non_blocking=True)
        self._record_upload(payload.device)
        device_rows, device_original_lengths = payload_views(payload, batch_size)
        return (
            device_rows,
            device_original_lengths,
            device_rows[MetadataRow.TOKEN],
            self.host_sequence_lengths,
        )


class DirectMetadataStagingSlot(_PinnedUploadSlot):
    """Pinned upload for one attention-free role graph."""

    def __init__(self, batch_size: int) -> None:
        """Allocate a zero-filled pinned payload for batch_size pre/post lanes.

        Attention-free roles use the same payload layout as decode, but their
        graphs read only the relevant token/position/state-slot fields.
        """
        super().__init__()
        self.batch_size = batch_size
        self.host = torch.zeros(payload_size(batch_size), dtype=torch.uint8, pin_memory=True)
        self.host_rows, _ = payload_views(self.host, batch_size)
        self.host_rows_numpy = self.host_rows.numpy()

    def stage(
        self,
        rows: Mapping[MetadataRow, Sequence[int]],
        device_payload: torch.Tensor,
    ) -> torch.Tensor:
        """Update selected rows, upload the payload, and return CUDA metadata.

        rows maps metadata row indices to batch_size integers. Pre supplies
        POSITION, TOKEN, and STATE_OUTPUT; post supplies STATE_INPUT. Omitted
        rows keep their previous values (initially zero), so callers must fill
        every field their graph consumes.

        device_payload is the role/width graph's CUDA uint8 buffer with exactly
        payload_size(batch_size) bytes. Returns an int64 [8, batch_size] view
        sharing that buffer. Copy completion and view reuse follow the same
        asynchronous contract as DecodeStagingSlot.stage.
        """
        self._wait_for_upload()
        for row, values in rows.items():
            self.host_rows_numpy[row] = values

        device_payload.copy_(self.host, non_blocking=True)
        self._record_upload(device_payload.device)
        device_rows, _ = payload_views(device_payload, self.batch_size)
        return device_rows
