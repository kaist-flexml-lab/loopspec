"""Copy lane state and update KV indices inside fixed-shape CUDA graphs.

State-bank rows, request-pool rows, and KV slot IDs are separate index spaces.
These operations copy state or slot IDs, not the attention K/V tensors.
Python tensor arguments use Any to avoid adding a torch import just for types;
their runtime type is torch.Tensor. Kernel arguments are Triton device pointers.

RecurrentCudaGraphRunner.capture_one_shape installs these operations around
each role's model forward. Warmup/capture calls the Python wrappers below;
subsequent graph replays run the captured kernels without calling Python again.
Full-prompt prefill uses a separate eager path, not these decode graph kernels.
"""

from typing import Any

import triton
import triton.language as tl


@triton.jit
def _copy_state_rows_kernel(
    source: tl.tensor,
    destination: tl.tensor,
    row_indices: tl.tensor,
    width: tl.constexpr,
    block_size: tl.constexpr,
    gather: tl.constexpr,
) -> None:
    """Copy one column block for one batch row; write destination in place.

    source/destination point to packed rows of width elements. row_indices
    points to one bank-row index per batch row. gather selects bank-to-batch
    rather than batch-to-bank copying; negative indices leave output unchanged.
    width, block_size, and gather are compile-time constants. Return nothing.

    Launched by copy_state_rows with grid (B, ceil(W / block_size)). The graph
    runner uses it before/after model execution as detailed in that wrapper.
    """
    row = tl.program_id(0)
    block = tl.program_id(1)
    columns = block * block_size + tl.arange(0, block_size)
    state_row = tl.load(row_indices + row)
    # Mask unused bank indices and the padded columns of the final block.
    valid = (state_row >= 0) & (columns < width)
    if gather:
        values = tl.load(source + state_row * width + columns, mask=valid)
        tl.store(destination + row * width + columns, values, mask=valid)
    else:
        values = tl.load(source + row * width + columns, mask=valid)
        tl.store(destination + state_row * width + columns, values, mask=valid)


def copy_state_rows(
    source: Any,
    destination: Any,
    row_indices: Any,
    *,
    gather: bool,
) -> None:
    """Enqueue a state-bank copy on the current CUDA stream; return nothing.

    source/destination are contiguous CUDA tensors with matching dtype and row
    width W. For gather=True their shapes are [N, W] and [B, W]; for False,
    [B, W] and [N, W]. Contiguous 1-D token/position arrays use W=1 instead.
    row_indices is a contiguous CUDA integer tensor [B] selecting bank rows,
    NOT request-pool rows or KV slots. Negative indices skip writes.

    Gather writes destination[b] = source[row_indices[b]]; scatter writes
    destination[row_indices[b]] = source[b]. Nonnegative indices must be in
    range; scatter destinations must be unique and copies must not race through
    overlapping source/destination storage. The caller ensures these contracts.
    This copies existing values; it neither allocates nor initializes bank slots.

    RecurrentCudaGraphRunner calls this in three places inside each role graph:
    - _capture_state_inputs, before recurrent/post: gather hidden by STATE_INPUT;
      recurrent also gathers Raven's injected state. Pre has no state gather.
    - _capture_token_position_transfer, before model execution: recurrent
      scatters input token IDs/positions to banks; post gathers them back.
      Both use STATE_INPUT indices, and these 1-D copies have W=1.
    - _capture_state_scatter, after pre/recurrent: scatter output hidden by
      STATE_OUTPUT; pre also stores Raven's injected state. Post has no scatter.
    """
    width = source.shape[1] if source.ndim > 1 else 1
    block_size = min(256, triton.next_power_of_2(width))
    _copy_state_rows_kernel[
        (row_indices.shape[0], triton.cdiv(width, block_size))
    ](
        source,
        destination,
        row_indices,
        width,
        block_size,
        gather,
    )


@triton.jit
def _allocate_decode_slots_kernel(
    free_slots: tl.tensor,
    allocation_cursor: tl.tensor,
    valid_markers: tl.tensor,
    out_cache_locs: tl.tensor,
    num_rows: tl.constexpr,
    block_size: tl.constexpr,
    capacity: tl.constexpr,
) -> None:
    """Use one program to assign ring entries to valid rows, in batch order.

    Pointers address the free-slot ring, scalar cursor, row-validity markers,
    and output slot IDs. num_rows is B, block_size covers B lanes, and capacity
    is the ring length; all three are compile-time constants. Update the cursor
    and valid output entries in place. Negative markers leave output untouched.

    Launched by allocate_decode_slots with grid (1,). One program assigns all
    B rows so a single cursor update covers the batch; no per-row atomics run.
    """
    offsets = tl.arange(0, block_size)
    in_bounds = offsets < num_rows
    valid = (
        tl.load(valid_markers + offsets, mask=in_bounds, other=-1) >= 0
    ) & in_bounds
    valid_i64 = valid.to(tl.int64)
    # Exclusive valid-row ranks: [True, False, True] consumes ring entries 0, 1,
    # not 0, 2. Skipped rows do not consume slots or advance the cursor.
    ranks = tl.cumsum(valid_i64, axis=0) - valid_i64
    count = tl.sum(valid_i64, axis=0)
    base = tl.load(allocation_cursor)
    tl.store(allocation_cursor, base + count)
    # Keep the device cursor monotonic and wrap only the table lookup.  A CPU
    # refill can therefore append recycled IDs at the current ring position
    # without resetting the cursor or copying a temporary CUDA tensor.
    slot_offsets = (base + ranks) % capacity
    slots = tl.load(free_slots + slot_offsets, mask=valid, other=0)
    tl.store(out_cache_locs + offsets, slots, mask=valid)


def allocate_decode_slots(
    free_slots: Any,
    allocation_cursor: Any,
    valid_markers: Any,
    out_cache_locs: Any,
) -> None:
    """Enqueue KV slot selection from the CPU-managed ring; return nothing.

    All arguments are contiguous CUDA integer tensors. free_slots is int64 [C]
    containing reusable KV slot IDs; allocation_cursor is a scalar int64 count.
    valid_markers and out_cache_locs have shape [B]. LoopSpec supplies mapping copy
    lengths as markers: >=0 allocates one slot, <0 skips the row. Marker values
    are NOT allocation sizes. out_cache_locs receives IDs, not K/V tensor data.

    DecodeSlotPool's CPU bookkeeping reserves capacity and refills the ring
    before replay. This kernel does not check exhaustion: it advances the cursor
    and wraps table indices modulo C. Calls sharing a cursor must be ordered,
    as its update is not atomic. B and C must be positive.

    Called first in RecurrentCudaGraphRunner._capture_mapping_and_attention_preamble,
    after staged metadata/state transfers and before the role's model forward.
    Recurrent always takes this path for supported models; pre/post take it only
    when they contain attention. MAPPING_COPY_LENGTH supplies valid_markers, and
    the output is the graph's out_cache_loc buffer. update_request_mapping runs
    next, then attention metadata setup, then the model writes the new K/V.
    """
    block_size = triton.next_power_of_2(valid_markers.shape[0])
    _allocate_decode_slots_kernel[(1,)](
        free_slots,
        allocation_cursor,
        valid_markers,
        out_cache_locs,
        valid_markers.shape[0],
        block_size,
        free_slots.shape[0],
    )


@triton.jit
def _update_request_mapping_kernel(
    mapping: tl.tensor,
    destination_rows: tl.tensor,
    source_rows: tl.tensor,
    copy_lens: tl.tensor,
    positions: tl.tensor,
    out_cache_locs: tl.tensor,
    mapping_stride: tl.constexpr,
    block_size: tl.constexpr,
) -> None:
    """Copy one prefix block and let block zero append the current slot ID.

    mapping points to request-row/position entries containing KV slot IDs.
    The other pointers each supply one value per batch row: destination/source
    request rows, prefix length, current token position, and new KV slot ID.
    mapping_stride is the row stride in elements; block_size is the number of
    prefix entries per program. Both are compile-time constants. Return nothing.

    Launched by update_request_mapping with grid (B, ceil(context_length / 256)).
    Each program copies up to 256 prefix entries for one request; only block 0
    appends the new entry. This covers any staged prefix length without changing
    the captured launch shape from token to token.
    """
    row = tl.program_id(0)
    block = tl.program_id(1)
    offsets = block * block_size + tl.arange(0, block_size)
    copy_len = tl.load(copy_lens + row)
    destination = tl.load(destination_rows + row)
    source = tl.load(source_rows + row)
    # Fork only the mapping entries: parent and child still share prefix K/V.
    # A zero copy length only appends; a negative length skips all writes.
    mask = offsets < copy_len
    values = tl.load(mapping + source * mapping_stride + offsets, mask=mask)
    tl.store(mapping + destination * mapping_stride + offsets, values, mask=mask)
    position = tl.load(positions + row)
    out_cache_loc = tl.load(out_cache_locs + row).to(tl.int32)
    # Exactly one program/lane appends. The position must lie outside the copied
    # prefix, so concurrent prefix blocks never overwrite this new entry.
    tl.store(
        mapping + destination * mapping_stride + position + offsets,
        out_cache_loc,
        mask=(copy_len >= 0) & (block == 0) & (offsets == 0),
    )


def update_request_mapping(
    mapping: Any,
    request_rows: Any,
    source_rows: Any,
    copy_lens: Any,
    positions: Any,
    out_cache_locs: Any,
) -> None:
    """Enqueue prefix-map copying and a new KV mapping entry; return nothing.

    mapping is SGLang's CUDA int32 [request_capacity, max_context_len] table,
    with contiguous columns. Other arguments are contiguous CUDA integer [B]
    arrays. request_rows/source_rows select destination/parent request-pool rows,
    NOT hidden-state bank rows. positions gives each new token's absolute index;
    out_cache_locs gives its allocated KV slot ID.

    copy_lens > 0 copies mapping[parent, :length] to the destination; zero only
    appends; negative skips all writes for that row. For active rows, append
    mapping[request_rows[b], positions[b]] = out_cache_locs[b]. Actual K/V
    values are not copied here; attention later writes them into those slots.

    The caller ensures valid request rows, 0 <= copy_len <= position < context
    length for active rows, and no conflicting writes or cross-row copy hazards.
    All [B] arrays must include skipped rows too: their metadata is still loaded
    even though no mapping entries are written. Attention metadata setup must
    follow this launch on the same stream or with an explicit dependency; this
    function does not synchronize.

    Called in RecurrentCudaGraphRunner._capture_mapping_and_attention_preamble
    immediately after allocate_decode_slots, for attention-bearing roles only.
    Arguments come from MetadataRow.REQUEST, PARENT_REQUEST,
    MAPPING_COPY_LENGTH, POSITION, and the just-filled graph out_cache_loc.
    A child's first use of a timeline copies its parent's prefix mapping;
    later uses set copy_lens=0 and only append. The following attention-metadata
    setup consumes this updated mapping before the model's attention executes.
    """
    batch_size = request_rows.shape[0]
    max_context_len = mapping.shape[1]
    block_size = 256
    _update_request_mapping_kernel[
        (batch_size, triton.cdiv(max_context_len, block_size))
    ](
        mapping,
        request_rows,
        source_rows,
        copy_lens,
        positions,
        out_cache_locs,
        mapping.stride(0),
        block_size,
    )
