"""Track logical branches and their request-row/KV ownership for one generation.

A logical branch can have many timelines: pre, each core iteration, and each
post readout depth with attention. Each timeline has a SGLang request-pool row
whose positions map to KV slot IDs. Forks share prefix KV; they copy mapping
entries lazily in a decode graph, not K/V tensors here. Hidden-state bank slots
are managed separately by LosslessLoopSpecExecutor, not by this module.
"""

from __future__ import annotations

from array import array
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Literal

from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.radix_cache import RadixCache
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

from ..pipeline import PipelineLane


# "core" names branch timelines; the model/graph role for them is "recurrent".
TimelineRole = Literal["pre", "core", "post"]


def _prepare_prefill_request(request: Req, input_len: int) -> None:
    """Mutate an internal Req so ScheduleBatch can prepare its prompt extension.

    request already holds the timeline's prompt IDs. input_len is the number of
    tokens to extend (the full prompt in this module). Set SGLang bookkeeping
    only; do not allocate KV, run the model, or return a new request.
    """
    request.full_untruncated_fill_ids = request.origin_input_ids
    request.fill_len = len(request.origin_input_ids)
    request.logprob_start_len = 0
    request.set_extend_input_len(input_len)


@dataclass
class TimelineCursor:
    """Track one branch/role/depth prefix and the resources owned by its row.

    seq_len is the prefix length before the next decode: its next token has
    position seq_len and new full length seq_len + 1. A pending fork initially
    has only this logical prefix length; its mapping entries are copied on first
    replay. RecurrentRequest._decode records the new KV ID, clears the pending
    parent, and advances seq_len after submitting the graph (not a GPU barrier).
    """

    req_pool_idx: int  # SGLang mapping row; not a logical branch or hidden slot.
    seq_len: int  # Includes the prompt and all tokens appended on this timeline.
    request: Req | None = None  # Bound internal Req; real managed cursors have one.
    reserved_fork_row: bool = False  # Return to our lease list, not SGLang, on release.
    pending_parent_req_pool_idx: int | None = None  # Copy this row's prefix once.
    # Decode KV IDs this cursor must eventually release, including inherited
    # ownership. Excludes borrowed ancestor slots and eager-prefill allocations.
    owned_kv_slots: list[int] = field(default_factory=list)


class _TimelineManager:
    """Manage SGLang request rows and KV slots for recurrent timelines."""

    def __init__(
        self,
        runner: ModelRunner,
        sampling_params: SamplingParams,
    ) -> None:
        """Bind shared runner pools and valid sampling settings for internal Reqs.

        runner owns request/KV pools and the recurrent graph runner.
        sampling_params configures SGLang's internal bookkeeping, not LoopSpec's
        actual token decisions. Construction does not allocate rows or run CUDA.
        """
        self.runner = runner
        self.sampling_params = sampling_params
        # All reserved Req objects keep their public row bindings for the whole
        # generation. The second list is the stack of currently unleased objects.
        self._fork_requests: list[Req] = []
        self._available_fork_requests: list[Req] = []
        self._tree_cache: RadixCache | None = None

    def _disabled_tree_cache(self) -> RadixCache:
        """Lazily return the disabled prefix cache needed by ScheduleBatch setup.

        It references runner pools but does not provide cross-request prefix
        reuse. LoopSpec handles branch prefix sharing itself; reuse this cache
        object for the timeline prefill batches within the manager.
        """
        if self._tree_cache is None:
            self._tree_cache = RadixCache(
                CacheInitParams(
                    disable=True,
                    req_to_token_pool=self.runner.req_to_token_pool,
                    token_to_kv_pool_allocator=(self.runner.token_to_kv_pool_allocator),
                    page_size=self.runner.server_args.page_size,
                ),
            )
        return self._tree_cache

    def reset_for_request(self) -> None:
        """Reset shared allocator bookkeeping before a new prompt is prepared.

        Reject leftover reserved fork rows. The caller must have finished prior
        request GPU work; these pools are shared and this is not concurrent-
        request isolation. Clearing allocators does not zero the K/V tensors.
        """
        if self._fork_requests:
            raise RuntimeError("LoopSpec fork rows leaked across requests")
        self.runner.req_to_token_pool.clear()
        self.runner.token_to_kv_pool_allocator.clear()

    def reserve_fork_rows(self) -> None:
        """Bind the remaining public request-pool rows for this LoopSpec request.

        LoopSpec is an external batch-one scheduler. Its many branch timelines are
        short-lived views within that one request, not independently scheduled
        SGLang requests. Keep stable Req objects bound to the internal rows for
        the request lifetime so the decode hot path only leases a row.

        Called after root prefill rows exist. This reserves request mapping rows,
        not KV token slots. Raise if already reserved, none remain, or allocation
        fails; no new model forward is performed.
        """
        if self._fork_requests:
            raise RuntimeError("LoopSpec fork rows are already reserved")
        count = self.runner.req_to_token_pool.available_size()
        if count <= 0:
            raise RuntimeError("LoopSpec has no request-pool rows for branches")
        requests = [
            Req(f"loopspec-fork-row-{index}", "", array("q"), self.sampling_params)
            for index in range(count)
        ]

        rows = self.runner.req_to_token_pool.alloc(requests)
        if rows is None or len(rows) != count:
            raise RuntimeError("failed to reserve LoopSpec request-pool rows")
        self._fork_requests = requests
        self._available_fork_requests = list(reversed(requests))

    def finish_request(self) -> None:
        """Return all reserved fork rows after every timeline lease was released.

        Reject active leases, free each bound Req through SGLang, and clear the
        lease lists. Root rows are released separately by release(). This does
        not free GPU pool storage or individually reclaim eager-prefill KV.
        """
        if len(self._available_fork_requests) != len(self._fork_requests):
            raise RuntimeError("cannot finish LoopSpec with active branch rows")
        for request in self._fork_requests:
            self.runner.req_to_token_pool.free(request)
        self._fork_requests.clear()
        self._available_fork_requests.clear()

    def create_prefilled(
        self,
        timeline_name: str,
        input_ids: Sequence[int],
    ) -> tuple[TimelineCursor, ForwardBatch]:
        """Allocate one prompt timeline and return its cursor and eager batch.

        timeline_name labels the internal Req (for example core-0 or post-7).
        input_ids is the full prompt, copied into that Req. ScheduleBatch sets
        up the request row, KV allocations/mapping, and sampling metadata with
        prefix reuse disabled. Despite the name, this method does not execute
        prefill: RecurrentRequest must forward the returned batch to populate KV.
        Cursor seq_len already reflects the prepared prompt length.
        """
        request = Req(timeline_name, "", array("q", input_ids), self.sampling_params)
        _prepare_prefill_request(request, len(input_ids))
        schedule_batch = ScheduleBatch.init_new(
            reqs=[request],
            req_to_token_pool=self.runner.req_to_token_pool,
            token_to_kv_pool_allocator=self.runner.token_to_kv_pool_allocator,
            tree_cache=self._disabled_tree_cache(),
            model_config=self.runner.model_config,
            enable_overlap=False,
            spec_algorithm=SpeculativeAlgorithm.NONE,
        )
        schedule_batch.prepare_for_extend()

        # Some SGLang paths leave token IDs staged on CPU; ForwardBatch needs
        # the device tensor for the eager model call that follows this method.
        if (
            schedule_batch.input_ids is None
            and schedule_batch.prefill_input_ids_cpu is not None
        ):
            schedule_batch.input_ids = (
                schedule_batch.prefill_input_ids_cpu.to(schedule_batch.device, non_blocking=True)
            )
            schedule_batch.prefill_input_ids_cpu = None

        forward_batch = ForwardBatch.init_new(schedule_batch, self.runner)
        cursor = TimelineCursor(
            req_pool_idx=request.req_pool_idx,
            seq_len=int(schedule_batch.seq_lens.item()),
            request=request,
        )

        return cursor, forward_batch

    def fork(self, parent_cursor: TimelineCursor) -> TimelineCursor:
        """Lease a fresh row and return a cursor borrowing the parent's prefix.

        parent_cursor is the same role/depth on the parent branch. The child
        starts at its seq_len with no owned decode slots. Record the source row
        for the first graph to copy its mapping; no GPU copy or KV allocation
        occurs here. The parent row must remain valid until that copy is ordered.
        Raise on lease exhaustion or an unexpectedly unbound reserved Req.
        """
        if not self._available_fork_requests:
            raise RuntimeError("recurrent branch ran out of request-pool rows")

        request = self._available_fork_requests.pop()
        request_row = request.req_pool_idx
        if request_row is None:
            raise RuntimeError("reserved LoopSpec request row was released early")

        cursor = TimelineCursor(
            req_pool_idx=request_row,
            seq_len=parent_cursor.seq_len,
            request=request,
            reserved_fork_row=True,
            pending_parent_req_pool_idx=parent_cursor.req_pool_idx,
        )

        return cursor

    def release(self, cursor: TimelineCursor) -> None:
        """Release one unused timeline's owned decode KV IDs and request row.

        cursor must no longer be needed by a live branch or its descendants.
        Release only its owned slots, not borrowed prefix KV. Fork rows return
        to the local lease stack; original prefill rows return to SGLang's pool.
        The caller must order outstanding graph use before row/slot reuse.
        This is bookkeeping, not a GPU synchronization or K/V tensor clear.
        """
        self.runner.decode_cuda_graph_runner.release_decode_slots(cursor.owned_kv_slots)
        if cursor.request is None:
            raise RuntimeError("timeline cursor has no SGLang request")

        if cursor.reserved_fork_row:
            self._available_fork_requests.append(cursor.request)
        else:
            self.runner.req_to_token_pool.free(cursor.request)


@dataclass
class _BranchNode:
    """Own one logical branch's timelines and retain shared ancestor resources.

    Cursor dictionaries are indexed by role-specific depth, not batch row:
    pre uses 0; core uses zero-based recurrent iterations; post uses the q1/q2/
    final endpoint iteration. Attention-free roles have no cursors. An inactive
    node can remain while children still depend on its prefix ownership.
    """

    branch_id: int  # Assigned by pipeline.py, independently of physical row IDs.
    parent: _BranchNode | None = None
    pre_cursors: dict[int, TimelineCursor] = field(default_factory=dict)
    core_cursors: dict[int, TimelineCursor] = field(default_factory=dict)
    post_cursors: dict[int, TimelineCursor] = field(default_factory=dict)
    child_count: int = 0  # Direct child nodes still attached, including inactive ones.
    is_live: bool = True  # Selected by the scheduler, not necessarily owning a row yet.


class BranchManager:
    """Own branch ancestry and reclaim timeline resources after pruning."""

    def __init__(
        self,
        runner: ModelRunner,
        sampling_params: SamplingParams,
    ) -> None:
        """Create per-generation branch bookkeeping over the runner's shared pools.

        runner and sampling_params are passed to the timeline allocator. This
        does not construct the root or allocate resources; prefill does that.
        """
        self._timeline_manager = _TimelineManager(runner, sampling_params)
        self._branch_nodes: dict[int, _BranchNode] = {}

    def reset_for_request(self) -> None:
        """Clear allocator bookkeeping and logical nodes before prompt prefill.

        Called by RecurrentRequest.prefill under sequential request execution.
        The timeline manager rejects reserved rows left over from a prior request.
        """
        self._timeline_manager.reset_for_request()
        self._branch_nodes.clear()

    def create_prefilled_timeline(
        self,
        timeline_name: str,
        input_ids: Sequence[int],
    ) -> tuple[TimelineCursor, ForwardBatch]:
        """Prepare a named full-prompt timeline without executing the model.

        Forward timeline_name and prompt input_ids to the timeline manager;
        return its cursor and eager ForwardBatch. The caller runs prefill and
        later supplies these cursors to initialize_root().
        """
        return self._timeline_manager.create_prefilled(timeline_name, input_ids)

    def initialize_root(
        self,
        *,
        pre_cursors: dict[int, TimelineCursor],
        core_cursors: dict[int, TimelineCursor],
        post_cursors: dict[int, TimelineCursor],
    ) -> None:
        """Attach completed prompt timelines to branch 0 and reserve fork rows.

        Dictionaries are retained by reference: pre_cursors uses key 0 when pre
        has attention; core_cursors uses each recurrence index; post_cursors uses
        attention-bearing readout depths. Empty dictionaries represent absent
        role timelines. Called once after prompt forwards, before decode begins.
        """
        if self._branch_nodes:
            raise RuntimeError("root branch is already initialized")

        root = _BranchNode(branch_id=0, pre_cursors=pre_cursors, core_cursors=core_cursors, post_cursors=post_cursors)
        self._branch_nodes = {root.branch_id: root}
        self._timeline_manager.reserve_fork_rows()

    def _get_or_create_branch(
        self,
        branch_id: int,
        parent_branch_id: int | None,
    ) -> _BranchNode:
        """Return an existing logical node or attach a new one to its parent.

        branch_id/parent_branch_id are pipeline ancestry IDs, not request rows.
        Existing nodes are returned unchanged, without rechecking their parent.
        A new node requires an existing parent; otherwise raise RuntimeError.
        Increment child_count but allocate no timeline rows until they are used.
        """
        node = self._branch_nodes.get(branch_id)
        if node is not None:
            return node

        parent_node = self._branch_nodes.get(parent_branch_id)
        if parent_node is None:
            raise RuntimeError(f"branch {branch_id} has unavailable parent {parent_branch_id}")

        node = _BranchNode(branch_id=branch_id, parent=parent_node)
        parent_node.child_count += 1
        self._branch_nodes[branch_id] = node

        return node

    def get_or_create_cursors(
        self,
        role: TimelineRole,
        stage_ids: Sequence[int],
        branch_ids: Sequence[int],
        parent_branch_ids: Sequence[int | None],
    ) -> list[TimelineCursor]:
        """Return ordered cursors for one role, lazily leasing missing timelines.

        stage_ids, branch_ids, and parent_branch_ids must have matching lengths
        and batch order. role is pre/core/post; stage IDs use that role's cursor
        dictionary keys, not hidden slots. Existing cursors are reused. A missing
        cursor forks the same role/depth from the immediate parent's cursor,
        which must already exist. Do not advance lengths or copy mapping here.
        """
        cursors: list[TimelineCursor] = []
        cursor_field = f"{role}_cursors"
        for stage_id, branch_id, parent_branch_id in zip(
            stage_ids, branch_ids, parent_branch_ids, strict=True
        ):
            node = self._get_or_create_branch(branch_id, parent_branch_id)
            branch_cursors: dict[int, TimelineCursor] = getattr(node, cursor_field)
            cursor = branch_cursors.get(stage_id)

            if cursor is None:
                parent_cursors: dict[int, TimelineCursor] = getattr(node.parent, cursor_field)
                cursor = self._timeline_manager.fork(parent_cursors[stage_id])
                branch_cursors[stage_id] = cursor

            cursors.append(cursor)

        return cursors

    def _release_unused_branch(self, node: _BranchNode) -> None:
        """Release an inactive leaf, then recursively release eligible ancestors.

        node is retained if live or if any child still depends on it. Otherwise
        release every owned timeline, remove the logical node, and decrement its
        parent's child_count before checking the parent. No lane/state-bank slots
        are managed here; those belong to the executor's separate slot pool.
        """
        if node.is_live or node.child_count:
            return

        for cursors in (node.pre_cursors, node.core_cursors, node.post_cursors):
            for cursor in cursors.values():
                self._timeline_manager.release(cursor)

        self._branch_nodes.pop(node.branch_id, None)
        if node.parent is not None:
            node.parent.child_count -= 1
            self._release_unused_branch(node.parent)

    @staticmethod
    def _transfer_timeline_ownership(
        child_cursors: dict[int, TimelineCursor],
        parent_cursors: dict[int, TimelineCursor],
    ) -> None:
        """Move one role's ownership from an unused parent to its sole child.

        Mutate both depth-to-cursor dictionaries. If the child has not forked a
        depth, move the entire parent cursor, preserving its request row. If it
        has, prepend the parent's owned decode KV IDs to the child's ownership
        list and clear the parent's list so releasing it cannot free shared KV.
        This transfers CPU ownership only, not GPU mapping entries or K/V data.
        """
        for stage_id, parent_cursor in tuple(parent_cursors.items()):
            child_cursor = child_cursors.get(stage_id)
            if child_cursor is None:
                child_cursors[stage_id] = parent_cursors.pop(stage_id)
            else:
                child_cursor.owned_kv_slots[:0] = (parent_cursor.owned_kv_slots)
                parent_cursor.owned_kv_slots = []

    def _detach_from_unused_parent(self, node: _BranchNode) -> None:
        """Let a sole child own its inactive parent's prefix and detach from it.

        Skip a missing/live parent or one with multiple children. For each role,
        inherit cursors or KV ownership, then clear node.parent and release the
        unused parent if eligible. The caller must have ordered any pending
        prefix-map reads before the parent's request rows can be reused.
        """
        parent_node = node.parent
        if (parent_node is None or parent_node.is_live or parent_node.child_count != 1):
            return
        for child_cursors, parent_cursors in (
            (node.pre_cursors, parent_node.pre_cursors),
            (node.core_cursors, parent_node.core_cursors),
            (node.post_cursors, parent_node.post_cursors),
        ):
            self._transfer_timeline_ownership(child_cursors, parent_cursors)

        node.parent = None
        parent_node.child_count -= 1
        self._release_unused_branch(parent_node)

    def retain_branches(self, lanes: Iterable[PipelineLane]) -> None:
        """Keep lanes' logical branches and reclaim unneeded timeline resources.

        lanes is the scheduler's surviving frontier; only branch/parent_branch
        are read, not hidden-state slots. Create nodes for new lanes even if no
        forward has allocated their timelines yet. Pipeline IDs increase from
        parent to child, so ascending creation and descending pruning preserve
        ancestry. Inactive ancestors stay until children release/inherit their
        resources. An empty iterable eventually releases all branches and ends
        the reserved fork-row lifetime. This does not synchronize GPU execution.
        """
        lanes = tuple(lanes)
        for lane in sorted(lanes, key=lambda item: item.branch):
            if lane.branch and lane.branch not in self._branch_nodes:
                self._get_or_create_branch(lane.branch, lane.parent_branch)

        retained_branch_ids = {lane.branch for lane in lanes}
        unused_nodes = [
            node for node in self._branch_nodes.values()
            if node.is_live and node.branch_id not in retained_branch_ids
        ]
        for node in sorted(unused_nodes, key=lambda item: item.branch_id, reverse=True):
            node.is_live = False
            self._release_unused_branch(node)

        # Snapshot because ownership transfer can remove parent nodes in place.
        for node in tuple(self._branch_nodes.values()):
            self._detach_from_unused_parent(node)
        if not self._branch_nodes:
            self._timeline_manager.finish_request()
