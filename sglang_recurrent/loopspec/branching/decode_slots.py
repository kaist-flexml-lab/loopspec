"""CPU-managed KV slot ring shared by captured LoopSpec decode graphs.

Slot IDs address physical token storage in SGLang's KV pool, not logical
branches, request-mapping rows, or hidden-state bank rows. This module moves
IDs and tracks their ownership; it does not allocate or copy K/V tensors.
"""

from collections.abc import Iterable, Sequence

import torch


class DecodeSlotPool:
    """Reserve IDs on CPU and make the same allocation order visible to graphs.

    The CPU allocate() call precedes each attention-role replay. The captured
    allocate_decode_slots kernel consumes matching IDs from free_slots and
    advances allocation_cursor on GPU. There is no per-allocation GPU-to-CPU
    readback. Calls must be serialized, with refills and graph uses ordered on
    the same CUDA stream or through explicit dependencies; this is not a
    thread-safe pool.
    """

    def __init__(self, capacity: int, device: torch.device | str | int) -> None:
        """Create stable ring/upload buffers; call begin() before allocating.

        capacity is the positive number of usable KV token slots, whose IDs
        are 1..capacity; slot zero is excluded. device identifies the CUDA
        device holding the ring. A non-CUDA device raises ValueError.
        """
        self.capacity = capacity
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise ValueError("decode slot pools require a CUDA device")

        # CUDA int64 [capacity]: reusable-ID ring, not a compact set of all
        # currently free IDs. Entries already consumed can contain live IDs.
        self.free_slots = torch.arange(1, capacity + 1, dtype=torch.int64, device=self.device)
        # Pinned CPU int64 [capacity]: source for asynchronous ring refills.
        self.refill_host = torch.empty(capacity, dtype=torch.int64, pin_memory=True)
        self._refill_finished = torch.cuda.Event()
        self._refill_pending = False

        # CUDA int64 scalar: total consumed offset since begin(), including
        # the prefill offset. The kernel indexes free_slots modulo capacity.
        self.allocation_cursor = torch.zeros((), dtype=torch.int64, device=self.device)
        self.compactions = 0  # CPU queue rebuilds/refills, not K/V compactions.
        self.live: set[int] = set()

    def begin(self, allocated_slots: int) -> None:
        """Reset this request's allocation order after full-prompt prefill.

        allocated_slots is the occupied initial range 1..allocated_slots,
        not a count of arbitrary scattered IDs. It must be in [0, capacity].
        RecurrentCudaGraphRunner.begin_recurrent_request calls this once after
        eager prefill. Previous request GPU work must be finished before reset.
        Prefill IDs stay excluded from decode ownership/recycling. Return None.
        """
        if not 0 <= allocated_slots <= self.capacity:
            raise ValueError("invalid graph decode KV allocation offset")
        if self.compactions:
            # A previous request rewrote the ring. Restore its original ID
            # order in place so captured graphs keep the same buffer address.
            torch.arange(1, self.capacity + 1, dtype=torch.int64, device=self.device, out=self.free_slots)

        self.next: int = allocated_slots  # Next index in the CPU sequence.
        self.ring_head: int = allocated_slots % self.capacity  # Next GPU ring index.
        self.allocation_cursor.fill_(allocated_slots)

        # Only sequence[next:] is unconsumed. Earlier entries can still be live.
        self.sequence: list[int] = list(range(1, self.capacity + 1))
        self.recycled: list[int] = []  # Released IDs not yet appended to the ring.
        self.live = set()
        self.compactions = 0

    def _wait_for_refill(self) -> None:
        """Wait for the previous upload before overwriting its pinned source.

        This CPU wait protects refill_host, not every graph's access to the
        device ring or K/V. GPU consumers require the caller's stream ordering.
        Return immediately if no upload has been recorded; return None.
        """
        if not self._refill_pending:
            return
        self._refill_finished.synchronize()
        self._refill_pending = False

    def _append_to_ring(self, slots: Sequence[int], start: int) -> None:
        """Upload IDs into the GPU ring starting at a circular buffer index.

        slots contains at most capacity reusable KV IDs; start is in
        [0, capacity). allocate() chooses a region after the unconsumed IDs.
        Use one copy, or two if the region wraps, then record upload completion
        for safe pinned-buffer reuse. No allocation cursor is advanced here.
        Empty slots do nothing; nonempty uploads are asynchronous. Return None.
        """
        count = len(slots)
        if count == 0:
            return

        self._wait_for_refill()

        first_count = min(count, self.capacity - start)
        source = self.refill_host[:count]
        source.copy_(torch.as_tensor(slots, dtype=torch.int64))
        self.free_slots[start : start + first_count].copy_(source[:first_count], non_blocking=source.is_pinned())
        if first_count < count:
            self.free_slots[: count - first_count].copy_(source[first_count:], non_blocking=source.is_pinned())

        self._refill_finished.record(torch.cuda.current_stream(self.device))
        self._refill_pending = True

    def allocate(self, count: int) -> list[int]:
        """Reserve count KV IDs on CPU and return them in graph-consumption order.

        count is a nonnegative slot count (one per active attention row), not
        bytes or layers. graph_decode_locs calls this before replay; the GPU
        kernel must consume the same count in the same allocation order.
        Refill the ring from released IDs only when the current queue is short.
        Raise RuntimeError on exhaustion or an attempt to allocate a live ID.
        This does not run the GPU allocator or write actual K/V values.
        """
        end = self.next + count
        if end > len(self.sequence):
            # Preserve unread IDs first, then append recycled IDs. Compact only
            # the CPU queue; keep the GPU cursor moving through its fixed ring.
            remaining = self.sequence[self.next :]
            sequence = remaining + self.recycled
            if count > len(sequence):
                raise RuntimeError(
                    "recurrent decode ran out of KV cache slots "
                    f"(capacity={self.capacity}, live={len(self.live)}, "
                    f"available={len(sequence)}, requested={count})"
                )

            refill_start = (self.ring_head + len(remaining)) % self.capacity
            self._append_to_ring(self.recycled, refill_start)
            self.sequence = sequence
            self.recycled = []
            self.next = 0
            self.compactions += 1
            end = count

        slots = self.sequence[self.next : end]
        if any(slot in self.live for slot in slots):
            raise RuntimeError("recurrent decode allocated a live KV slot")
        self.live.update(slots)
        self.next = end
        # CPU prediction of the ring position after the matching GPU allocation.
        self.ring_head = (self.ring_head + count) % self.capacity
        return slots

    def release(self, slots: Iterable[int]) -> None:
        """Give owned decode KV IDs back to the CPU recycle queue; return None.

        Timeline release calls this through the graph runner. Each ID must be
        live and no longer needed by surviving timelines; duplicate/unowned IDs
        raise RuntimeError. Process IDs sequentially, without rollback on error.
        No K/V is cleared and no GPU upload occurs until a later refill. The
        caller must order pending GPU uses before any reuse overwrites the slot.
        """
        for slot in slots:
            if slot not in self.live:
                raise RuntimeError("recurrent decode released an unowned KV slot")
            self.live.remove(slot)
            self.recycled.append(slot)
