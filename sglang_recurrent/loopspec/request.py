"""SGLang-backed batch-1 serving runtime for lossless recurrent LoopSpec.

The external workload contains one request at a time. Internal lanes process
token replicas at different recurrent depths. This class connects those lanes
to SGLang batches, branch-local KV timelines, and graph-owned state banks.
"""

from collections.abc import Collection, Iterable, Mapping, Sequence
from copy import copy

from sglang.srt.sampling.sampling_params import SamplingParams

from ..modeling.contracts import RecurrentState
from .batch_pool import DecodeBatchPool
from .branching.manager import BranchManager, TimelineCursor
from .metadata import MetadataRow

# These modules are already loaded by the runtime imports above; importing
# their types here adds no new model/runtime module initialization.
from torch import Tensor

from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.model_runner import ModelRunner

from .pipeline import PipelineLane


class RecurrentRequest:
    """One request with native rows for its role and recurrence timelines."""

    def __init__(
        self,
        runner: ModelRunner,
        prompt_ids: Sequence[int],
        max_new_tokens: int,
    ) -> None:
        """Create per-request bookkeeping without running the model.

        runner must have a LoopSpec schedule and recurrent graph runner installed.
        prompt_ids is a nonempty token sequence, copied into this request.
        max_new_tokens is the completion budget including the prefill prediction.
        This object owns batch reuse and KV branch tracking, not token selection.
        """
        if not prompt_ids:
            raise ValueError("the prompt must contain at least one token")
        self.runner = runner
        model = runner.model
        self.prompt_ids = list(prompt_ids)
        self.stages = runner.recurrent_stages
        schedule = runner.pipeline_schedule
        if schedule is None:
            raise ValueError("LoopSpec requires a pipeline schedule")
        self.endpoint_stages = schedule.endpoint_stages
        self.has_pre_attention = bool(model.attention_layers["pre"])
        self.has_post_attention = bool(model.attention_layers["post"])
        # Last prompt position, NOT the number of tokens processed by prefill.
        # For a lane predicting completion[index], its input is completion[index-1]
        # at position len(prompt_ids) + index - 1.
        self.prefill_len = len(self.prompt_ids) - 1
        # SGLang reads these settings when constructing internal Req objects and
        # building SamplingBatchInfo in prepare_for_extend(). LoopSpec calls model
        # forward, not runner.sample(), so this sampling_info does not select tokens.
        # The executor's decision policy uses the real request's sampling settings,
        # including for the first token after prefill. These internal settings must
        # still be valid; max_new_tokens also affects SGLang's prefill-only checks.
        sampling_params = SamplingParams(temperature=0, max_new_tokens=max_new_tokens + 1)
        self.batches = DecodeBatchPool(runner)
        self.branch_manager = BranchManager(runner, sampling_params)

    def begin_decode_cycle(self) -> None:
        """Reset CPU staging-slot counters at the start of executor.forward().

        Keep allocated buffers and cached batches. This call does not itself
        wait for GPU completion; staging slots protect their own pending uploads.
        """
        self.batches.begin_cycle()

    def retain_branches(self, lanes: Iterable[PipelineLane]) -> None:
        """Keep KV timelines needed by lanes and release discarded branches.

        Lane branch IDs identify logical ancestry, not physical request rows.
        Passing no lanes closes the branch manager's request. Hidden-state bank
        slot ownership is tracked separately by the executor.
        """
        self.branch_manager.retain_branches(lanes)

    def _forward(
        self,
        role: str,
        forward_batch: ForwardBatch,
        state: RecurrentState | None = None,
    ) -> RecurrentState | LogitsProcessorOutput:
        """Run pre, recurrent, or post using eager prefill or an exact decode graph.

        Attach role to the batch. Eager prefill passes state explicitly between
        roles, so its hidden/injected tensors are attached to the batch too.
        Decode omits state: captured kernels gather it from GPU state banks.
        Return RecurrentState for pre/core or LogitsProcessorOutput for post.
        Returning from a graph launch does not imply GPU work has completed.
        """
        forward_batch.recurrent_role = role
        if state is not None:
            forward_batch.hidden_states = state.hidden
            forward_batch.injected = state.injected
        if forward_batch.forward_mode.is_cuda_graph():
            return self.runner.decode_cuda_graph_runner.execute_exact(forward_batch)
        result = self.runner.forward(forward_batch)
        return result.logits_output

    def prefill(self) -> Tensor:
        """Populate prompt KV timelines and return final-depth logits [1, vocab_size].

        Process every prompt token eagerly: pre once, core at every recurrence,
        and post at every readout depth that needs its own prefix KV. An
        attention-free post runs only at final depth. Only the final prompt
        position's final-depth logits are returned; the caller selects the first
        completion token. The returned tensor stays on the model's device.

        Reset shared request/KV pools first: this requires the server's single
        external-request execution model, not concurrent HTTP request execution.
        """
        processed_ids = self.prompt_ids
        self.branch_manager.reset_for_request()

        if self.has_pre_attention:
            pre_cursor, pre_batch = (self.branch_manager.create_prefilled_timeline("pre", processed_ids))
            state = self._forward("pre", pre_batch)
        else:
            pre_cursor = None
            state = None

        endpoint_states: dict[int, RecurrentState] = {}
        core_cursors: list[TimelineCursor] = []
        for stage in range(self.stages):
            cursor, core_batch = (self.branch_manager.create_prefilled_timeline(f"core-{stage}", processed_ids))
            if state is None:
                # Without pre attention, no separate pre KV timeline is needed.
                # Borrow the first core batch's prompt metadata for the pre pass.
                pre_batch = copy(core_batch)
                pre_batch.mark_forward_metadata_ready()
                state = self._forward("pre", pre_batch)
                self.pre_decode_template = pre_batch
            state.positions = core_batch.positions
            state = self._forward("recurrent", core_batch, state)
            core_cursors.append(cursor)

            if self.has_post_attention and stage in self.endpoint_stages:
                # Later core iterations may reuse tensor storage. Preserve this
                # depth's state for its post-prefill pass after the core loop.
                endpoint_states[stage] = RecurrentState(
                    hidden=state.hidden.clone(),
                    positions=state.positions,
                    injected=(None if state.injected is None else state.injected.clone()),
                )

        final_output: LogitsProcessorOutput | None = None
        post_cursors: dict[int, TimelineCursor] = {}
        if self.has_post_attention:
            # q1/q2/final post attention each needs prompt KV from its own input
            # depth. These passes fill caches; only final-depth logits are returned.
            for stage in self.endpoint_stages:
                cursor, post_batch = (self.branch_manager.create_prefilled_timeline(f"post-{stage}", processed_ids))
                endpoint_state = endpoint_states[stage]
                endpoint_state.positions = post_batch.positions
                endpoint_output = self._forward("post", post_batch, endpoint_state)
                post_cursors[stage] = cursor
                if stage == self.stages - 1:
                    final_output = endpoint_output
        else:
            # No post KV cache to populate at proposal depths. Reuse the final
            # core batch for the norm/LM-head pass that produces the first token.
            state.positions = core_batch.positions
            core_batch.mark_forward_metadata_ready()
            final_output = self._forward("post", core_batch, state)

        if final_output is None:
            raise RuntimeError("prefill did not reach the final endpoint")
        self.branch_manager.initialize_root(
            pre_cursors={} if pre_cursor is None else {0: pre_cursor},
            core_cursors=dict(enumerate(core_cursors)),
            post_cursors=dict(post_cursors),
        )

        graph_runner = self.runner.decode_cuda_graph_runner
        allocator = self.runner.token_to_kv_pool_allocator
        # Eager prefill has already consumed the initial KV allocation range.
        # Start graph-side allocation after that range to avoid overwriting it.
        allocated_slots = allocator.size - allocator.available_size()
        graph_runner.begin_recurrent_request(allocated_slots)

        return final_output.next_token_logits[-1:]

    def finish_without_pipeline(self) -> None:
        """Release the full-prompt root when no decode lane will start."""
        self.branch_manager.retain_branches(())

    def _decode(
        self,
        cursors: Sequence[TimelineCursor],
        token_ids: Sequence[int],
        role: str,
        state_input_slots: Sequence[int] | None = None,
        state_output_slots: Sequence[int] | None = None,
    ) -> tuple[RecurrentState | LogitsProcessorOutput, ForwardBatch]:
        """Run one token per attention-bearing timeline and advance its cursor.

        cursors and token_ids have equal lengths and matching lane order. role
        selects pre/recurrent/post. Optional state-slot lists index hidden/injected
        banks; they are not KV slots. Return the role output and reusable batch.
        Record each reserved KV slot and increment prefix lengths only after
        _forward returns successfully; this is not a CUDA completion barrier.
        """
        forward_batch, graph_slots = self.batches.decode(role, cursors, token_ids, state_input_slots, state_output_slots)
        output = self._forward(role, forward_batch)
        # The graph has been submitted: record the matching CPU KV ownership.
        # Its mapping kernel handles any pending parent-prefix copy on this pass.
        for cursor, slot in zip(cursors, graph_slots, strict=True):
            cursor.owned_kv_slots.append(slot)
            cursor.pending_parent_req_pool_idx = None
        for cursor in cursors:
            cursor.seq_len += 1
        return output, forward_batch

    def _direct_state_role(
        self,
        role: str,
        template: ForwardBatch,
        batch_size: int,
        metadata_rows: Mapping[MetadataRow, Sequence[int]],
    ) -> RecurrentState | LogitsProcessorOutput:
        """Prepare and replay attention-free pre/post without creating KV timelines.

        template seeds the cached batch on its first use; metadata_rows supplies
        current token/position/state-slot values for batch_size selected lanes.
        Return pre state or post logits; no cursor or KV ownership is advanced.
        """
        forward_batch = self.batches.direct(role, template, batch_size, metadata_rows)
        return self._forward(role, forward_batch)

    def pre_batch(
        self,
        tokens: Sequence[int],
        indexes: Sequence[int],
        branches: Sequence[int],
        parent_branches: Sequence[int | None],
        state_output_slots: Sequence[int],
    ) -> None:
        """Initialize the selected new lanes' hidden states with one pre pass.

        All sequences have matching order and length. tokens are the input IDs;
        indexes are their lanes' next-token completion indices (index=1 means
        the input is completion[0]). branches/parent_branches are logical IDs
        used only when pre has attention. state_output_slots selects GPU bank
        rows receiving pre's hidden and, for Raven, injected state. Return nothing;
        later recurrent graphs read the results from those bank slots.
        """
        if not self.has_pre_attention:
            batch_size = len(tokens)
            self._direct_state_role("pre", self.pre_decode_template, batch_size,
                {
                    MetadataRow.POSITION: [self.prefill_len + index for index in indexes],
                    MetadataRow.TOKEN: tokens,
                    MetadataRow.STATE_OUTPUT: state_output_slots,
                },
            )
            return

        cursors = self.branch_manager.get_or_create_cursors("pre", [0] * len(branches), branches, parent_branches)
        self._decode(cursors, tokens, "pre", state_output_slots=state_output_slots)

    def recurrent(
        self,
        stage_ids: Sequence[int],
        token_ids: Sequence[int],
        branch_ids: Sequence[int],
        parent_branch_ids: Sequence[int | None],
        state_slots: Sequence[int],
    ) -> ForwardBatch:
        """Execute ONE core iteration per lane, using that iteration's KV timeline.

        All sequences share lane order and length. stage_ids are zero-based
        core repetition numbers, not transformer layer IDs. branch IDs select
        logical ancestry; state_slots selects bank rows for both input and output.
        The executor calls this K times per cycle. Return the reusable batch as
        a possible post template; updated hidden states remain in the GPU bank.
        """
        cursors = self.branch_manager.get_or_create_cursors("core", stage_ids, branch_ids, parent_branch_ids)
        _, forward_batch = self._decode(cursors, token_ids, "recurrent", state_input_slots=state_slots, state_output_slots=state_slots)
        return forward_batch

    def post(
        self,
        forward_batch: ForwardBatch,
        rows: Collection[int],
        endpoint_stages: Sequence[int],
        branch_ids: Sequence[int],
        parent_branch_ids: Sequence[int | None],
        state_slots: Sequence[int],
    ) -> LogitsProcessorOutput:
        """Compute next-token logits for selected q1/q2/final lanes, in rows order.

        forward_batch is the last core batch, used as an attention-free template.
        rows contains indices into the full core batch (the caller passes a dict,
        whose keys preserve order). state_slots covers that full batch; select
        only rows' entries. endpoint_stages and both branch-ID sequences already
        cover only the selected lanes in the same order. Endpoint stages are the
        last core iteration before each readout, not a post-layer index.

        Return LogitsProcessorOutput with next_token_logits [len(rows), vocab_size].
        Sampling and final proposal verification happen later in the executor.
        """
        selected_state_slots = [state_slots[row] for row in rows]
        if not self.has_post_attention:
            return self._direct_state_role("post", forward_batch, len(rows),
                {MetadataRow.STATE_INPUT: selected_state_slots},
            )

        cursors = self.branch_manager.get_or_create_cursors("post", endpoint_stages, branch_ids, parent_branch_ids)
        # In this decode path logits come from hidden states, not token IDs.
        # Attention uses the cursor positions and KV mapping prepared above.
        return self._decode(cursors, [0] * len(rows), "post", state_input_slots=selected_state_slots)[0]
