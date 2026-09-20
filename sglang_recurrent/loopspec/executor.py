"""Execute the logical LoopSpec pipeline using role graphs and decision policies.

pipeline.py chooses which token replicas remain live. This executor advances
them through pre/core/post, manages their GPU state-bank slot numbers, and
returns token proposals or final verification results. RecurrentRequest handles
KV timelines; the decision policy handles greedy or stochastic token selection.
"""

from __future__ import annotations

import heapq
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Literal, TypedDict

import torch

from sglang.srt.model_executor.model_runner import ModelRunner

from .decisions import GreedyDecisionPolicy
from .pipeline import (
    PipelineLane,
    PipelineResult,
    PipelineSchedule,
    continue_pipelined,
)
from .request import RecurrentRequest
from .sampling.policy import SamplingDecisionPolicy


# Integer counters for executed lane rows and final proposal verification.
# These count internal work, not concurrent HTTP requests or committed tokens.
LOOPSPEC_METRICS = (
    "pipeline_cycles",
    "pre_rows",
    "recurrent_rows",
    "post_rows",
    "verified_proposals",
    "first_rejections",
    "second_verifications",
    "second_acceptances",
    "second_rejections",
)

_EndpointKind = Literal["q1", "q2", "final"]


class _SamplingOptions(TypedDict):
    """Real request settings used by the stochastic decision policy."""

    temperature: float
    top_k: int
    top_p: float


@dataclass(slots=True)
class _CyclePlan:
    """Inputs for one forward(lanes) call; lists follow the supplied lane order.

    Batch row indices, logical branch IDs, and GPU bank slot numbers are
    different index spaces. No hidden tensors or KV data are stored here.
    """

    start_stages: list[int]  # First core iteration to execute, counted from zero.
    end_stages: list[int]  # Last core iteration this cycle: start + K - 1.
    token_ids: list[int]  # Input token of each lane, not its next-token prediction.
    branch_ids: list[int]  # Logical IDs used to select branch-local KV timelines.
    parent_branch_ids: list[int | None]  # Parent IDs for branches not registered yet.
    state_slots: list[int]  # Hidden/injected bank rows; kept stable across recurrence.
    pre_rows: list[int]  # Batch indices of new lanes (stage == 0).
    endpoints: dict[int, _EndpointKind]  # Batch index -> q1/q2/final, NOT stage -> kind.


class _StateSlotPool:
    """Assign persistent graph state rows and recycle pruned lanes."""

    def __init__(self, capacity: int) -> None:
        """Make bank indices 0 through capacity-1 available; allocate no tensors."""
        self.capacity = capacity
        self._free: list[int] = list(range(capacity))
        self._live: set[int] = set()

    def acquire(self) -> int:
        """Reserve the lowest free bank index, or raise RuntimeError if exhausted."""
        try:
            slot = heapq.heappop(self._free)
        except IndexError as error:
            raise RuntimeError(
                "pipeline requires more recurrent state slots than "
                f"configured (max_rows={self.capacity})"
            ) from error
        self._live.add(slot)
        return slot

    def retain(self, slots: Iterable[int]) -> None:
        """Keep the listed live indices and make all other live indices reusable.

        This updates CPU ownership only; it does not clear GPU bank contents.
        A newly assigned lane must initialize its slot before consuming it.
        """
        retained = set(slots)
        released = self._live - retained
        self._live.intersection_update(retained)
        for slot in released:
            heapq.heappush(self._free, slot)


class LosslessLoopSpecExecutor:
    """Execute one external request and implement pipeline.py's executor interface."""

    def __init__(
        self,
        runner: ModelRunner,
        prompt_ids: Sequence[int],
        max_new_tokens: int,
        *,
        sampling: _SamplingOptions | None = None,
        generator: torch.Generator | None = None,
        should_stop: Callable[[Sequence[int]], bool],
    ) -> None:
        """Set up lane state, eagerly prefill the prompt, and select the first token.

        runner must have LoopSpec role graphs and a PipelineSchedule. prompt_ids
        supplies the nonempty prompt; max_new_tokens includes the first completion.
        sampling=None selects greedy decisions; otherwise use the supplied request
        temperature/top-k/top-p. generator controls stochastic token selection,
        not Raven's hidden initializer. should_stop receives committed completion
        IDs only and may also update the server request or emit streaming output.
        """
        schedule: PipelineSchedule | None = runner.pipeline_schedule
        if schedule is None:
            raise ValueError("LoopSpec requires a pipeline schedule")
        graph_runner = runner.decode_cuda_graph_runner
        if not graph_runner.manages_recurrent_mapping("recurrent"):
            raise RuntimeError("LoopSpec requires graph-managed decode KV")

        self.schedule: PipelineSchedule = schedule
        self.endpoint_kind: dict[int, _EndpointKind] = schedule.endpoint_kinds
        self._state_slots = _StateSlotPool(schedule.max_rows)
        self.request = RecurrentRequest(runner, prompt_ids, max_new_tokens)

        if sampling is None:
            self.decisions = GreedyDecisionPolicy(schedule.max_rows, runner.device)
        else:
            self.decisions = SamplingDecisionPolicy(
                schedule.max_rows,
                runner.device,
                sampling,
                runner.loopspec_categorical_sampler,
                generator=generator,
            )

        self.should_stop = should_stop
        self.metrics: dict[str, int] = dict.fromkeys(LOOPSPEC_METRICS, 0)
        self._deferred_lanes: tuple[PipelineLane, ...] | None = None
        final_logits = self.request.prefill()
        self._first_token = self._sample_first_token(prompt_ids[-1], final_logits)

    def _sample_first_token(
        self, last_prompt_token: int, final_logits: torch.Tensor
    ) -> int:
        """Select one token from final_logits [1, vocab_size] after eager prefill.

        last_prompt_token labels a temporary lane used by the decision policy.
        This lane never enters the decode pipeline and has no earlier proposal
        to verify. Reuse the same policy as later tokens, including GPU-to-CPU
        decision readback, and return a Python token ID.
        """
        lane = PipelineLane(stage=self.schedule.stages - 1, index=0, token=last_prompt_token)
        endpoints = {0: "final"}
        post_output = SimpleNamespace(next_token_logits=final_logits)
        decision_batch = self.decisions.queue(post_output, endpoints, [lane])
        decision = self.decisions.read([lane], endpoints, decision_batch)[0]
        return self.decisions.resolve(
            lane,
            endpoints[0],
            decision,
            decision_batch.proposal_states.get(0),
            self.schedule,
        ).prediction

    def first_token(self) -> int:
        """Return the completion sampled after full-prompt prefill."""
        return self._first_token

    def _flush_deferred_retention(self) -> None:
        """Apply previously deferred KV branch retention once and clear the record.

        Called inside forward() before pipeline.py mutates the retained lane
        objects. It does not decide which lanes survive; that list was supplied
        by the previous cycle's retain_branches() call.
        """
        lanes = self._deferred_lanes
        if lanes is None:
            return
        self._deferred_lanes = None
        self.request.retain_branches(lanes)

    def finish_without_pipeline(self) -> None:
        """Close prefill's root branch when generation stops after the first token."""
        self.request.finish_without_pipeline()

    def _prepare_cycle(self, lanes: Sequence[PipelineLane]) -> _CyclePlan:
        """Prepare one cycle without running the model or mutating lane fields.

        lanes is the ordered list of live token replicas. Keep each existing
        lane's state-bank slot and reserve a slot for each new lane. Reset CPU
        staging counters, then select pre rows and post readouts by core depth.
        Return all role inputs together so each call uses the same lane mapping.
        """
        self.request.begin_decode_cycle()

        schedule = self.schedule
        start_stages = [lane.stage for lane in lanes]
        end_stages = [stage + schedule.step_size - 1 for stage in start_stages]

        return _CyclePlan(
            start_stages=start_stages,
            end_stages=end_stages,
            token_ids=[lane.token for lane in lanes],
            branch_ids=[lane.branch for lane in lanes],
            parent_branch_ids=[lane.parent_branch for lane in lanes],
            state_slots=[
                (lane.state if lane.state is not None else self._state_slots.acquire())
                for lane in lanes
            ],
            pre_rows=[row for row, lane in enumerate(lanes) if lane.stage == 0],
            endpoints={
                # row identifies a lane in this batch; end_stage is its last
                # core iteration. Only the latter indexes the readout schedule.
                row: self.endpoint_kind[end_stage]
                for row, end_stage in enumerate(end_stages)
                if end_stage in self.endpoint_kind
            },
        )

    def _make_pipeline_result(
        self,
        lane: PipelineLane,
        state_slot: int,
        endpoint_kind: _EndpointKind | None,
        decision: list[int] | None,
        proposal_state: torch.Tensor | None,
    ) -> PipelineResult:
        """Combine a lane's bank slot with its optional readout decision.

        endpoint_kind is q1/q2/final or None between readouts. decision is the
        policy's CPU integer row: greedy returns [token]; sampling returns a
        five-field decision interpreted by that policy. proposal_state is a
        CUDA probability vector [vocab_size] for sampling, otherwise None.
        Return one PipelineResult and update final-verification counters here.
        """
        if endpoint_kind is None:
            # No post pass ran for this lane. The scheduler uses state only;
            # prediction=0 is a placeholder, not a prediction of token ID zero.
            return PipelineResult(state=state_slot, prediction=0)

        outcome = self.decisions.resolve(
            lane,
            endpoint_kind,
            decision,
            proposal_state,
            self.schedule,
        )

        if endpoint_kind == "final":
            self.metrics["verified_proposals"] += 1
            second_verified = (
                outcome.first_rejected and lane.second_token is not None
            )
            self.metrics["first_rejections"] += outcome.first_rejected
            self.metrics["second_verifications"] += second_verified
            self.metrics["second_acceptances"] += (
                second_verified
                and outcome.accepted_stage == self.schedule.second_stage
            )
            self.metrics["second_rejections"] += (
                second_verified and outcome.second_rejected
            )

        return PipelineResult(
            state=state_slot,
            prediction=outcome.prediction,
            proposed=outcome.proposed,
            proposal_state=outcome.proposal_state,
            accepted_stage=outcome.accepted_stage,
            proposal_action=outcome.proposal_action,
        )

    @torch.inference_mode()
    def forward(
        self, lanes: Sequence[PipelineLane]
    ) -> list[PipelineResult]:
        """Advance every supplied lane by K core iterations and return ordered results.

        One call is one pipeline cycle: pre runs for new lanes, core runs K times
        for all lanes, then post runs for lanes ending at q1/q2/final. Post reads
        the same state-bank slots before another cycle can overwrite their hidden
        states. Read endpoint decisions back to CPU before building results.
        This method does not commit tokens or prune the logical lane tree;
        continue_pipelined() processes the results and calls retain_branches().
        """
        cycle = self._prepare_cycle(lanes)

        if cycle.pre_rows:
            self.request.pre_batch(
                [cycle.token_ids[row] for row in cycle.pre_rows],
                [lanes[row].index for row in cycle.pre_rows],
                [cycle.branch_ids[row] for row in cycle.pre_rows],
                [cycle.parent_branch_ids[row] for row in cycle.pre_rows],
                state_output_slots=[cycle.state_slots[row] for row in cycle.pre_rows],
            )

        # Each call executes one core iteration per lane. Token IDs and bank
        # slots stay fixed; the hidden in each slot advances by one iteration.
        for offset in range(self.schedule.step_size):
            forward_batch = self.request.recurrent(
                [stage + offset for stage in cycle.start_stages],
                cycle.token_ids,
                cycle.branch_ids,
                cycle.parent_branch_ids,
                cycle.state_slots,
            )

        decision_batch = None
        if cycle.endpoints:
            # Post selects a subset of the just-computed lanes. Keep branch/depth
            # arrays in that subset's order; request.post selects state slots
            # from the full cycle.state_slots using the endpoint dictionary keys.
            post_output = self.request.post(
                forward_batch,
                cycle.endpoints,
                [cycle.end_stages[row] for row in cycle.endpoints],
                [cycle.branch_ids[row] for row in cycle.endpoints],
                [cycle.parent_branch_ids[row] for row in cycle.endpoints],
                state_slots=cycle.state_slots,
            )
            decision_batch = self.decisions.queue(post_output, cycle.endpoints, lanes)

        # All GPU work for this cycle is queued. Keep CPU bookkeeping here so
        # it overlaps execution before decision readback synchronizes.
        # Finish the previous cycle's deferred branch cleanup before returning:
        # pipeline.py will then update the same lane objects in place.
        self._flush_deferred_retention()
        self.metrics["pipeline_cycles"] += 1
        self.metrics["pre_rows"] += len(cycle.pre_rows)
        self.metrics["recurrent_rows"] += (len(lanes) * self.schedule.step_size)
        self.metrics["post_rows"] += len(cycle.endpoints)

        decisions_by_row: dict[int, list[int]] = {}
        proposal_states: dict[int, torch.Tensor] = {}
        if decision_batch is not None:
            decisions_by_row = self.decisions.read(lanes, cycle.endpoints, decision_batch)
            proposal_states = decision_batch.proposal_states

        return [
            self._make_pipeline_result(
                lane,
                cycle.state_slots[row],
                cycle.endpoints.get(row),
                decisions_by_row.get(row),
                proposal_states.get(row),
            )
            for row, lane in enumerate(lanes)
        ]

    def retain_branches(
        self,
        lanes: Sequence[PipelineLane],
        accepted_stage: int | None = None,
    ) -> None:
        """Release unused state slots and retain KV branches for the surviving lanes.

        lanes is the next cycle's list after pipeline.py has selected branches.
        accepted_stage is the accepted q1/q2 core iteration, or None. Its presence
        lets KV cleanup wait until the next forward(); its numerical value is
        not used here. State-slot ownership is updated immediately in either case.
        """
        self._state_slots.retain(lane.state for lane in lanes if lane.state is not None)

        if accepted_stage is not None:
            # Save references to these lane objects, not copies of their fields.
            # The next forward() processes this cleanup before returning control
            # to pipeline.py, where those objects are updated for another cycle.
            self._deferred_lanes = tuple(lanes)
            return
        self.request.retain_branches(lanes)

    def stats(self) -> dict[str, int | float]:
        """Return a copy of the work/acceptance counters, currently all integers.

        The wider value type allows generate_loopspec() to append a host timestamp
        to the returned dictionary without changing this executor's counters.
        """
        return dict(self.metrics)


@torch.inference_mode()
def generate_loopspec(
    runner: ModelRunner,
    prompt_ids: Sequence[int],
    max_new_tokens: int,
    *,
    sampling: _SamplingOptions | None = None,
    generator: torch.Generator | None = None,
    should_stop: Callable[[Sequence[int]], bool],
) -> tuple[list[int], dict[str, int | float]]:
    """Generate completion IDs for one prompt and return IDs plus timing/counters.

    runner is initialized for LoopSpec. prompt_ids excludes any completion tokens;
    max_new_tokens is a positive total completion budget including the first token.
    sampling=None means greedy; otherwise use the supplied filtering settings and
    optional CUDA generator. should_stop is called after the first token and each
    later commit with the full completion so far; True stops generation.

    Returned IDs exclude the prompt and uncommitted drafts. Statistics include
    integer work/acceptance counts and decode_tokens (excluding the first token).
    decode_started_at is a perf_counter timestamp taken after the first stopping
    callback; the server records the end separately. No end-to-end tok/s is
    computed here. Synchronize the model device before returning.
    """
    decode_started_at: float | None = None

    def timed_should_stop(output_ids: Sequence[int]) -> bool:
        """Run the caller's stopping/output callback and mark decode start once."""
        nonlocal decode_started_at
        stop = should_stop(output_ids)
        if decode_started_at is None:
            # The final endpoint decision has synchronized before this call.
            # Start after processing the first completion, matching baseline.
            decode_started_at = time.perf_counter()
        return stop

    executor = LosslessLoopSpecExecutor(
        runner,
        prompt_ids,
        max_new_tokens,
        sampling=sampling,
        generator=generator,
        should_stop=timed_should_stop,
    )
    torch.cuda.synchronize(runner.device)

    first_token = executor.first_token()
    output = [first_token]
    if timed_should_stop(output):
        executor.finish_without_pipeline()
    else:
        output = continue_pipelined(first_token, max_new_tokens, executor)
    torch.cuda.synchronize(runner.device)

    stats = executor.stats()
    stats["decode_tokens"] = max(len(output) - 1, 0)
    if decode_started_at is not None:
        stats["decode_started_at"] = decode_started_at
    return output, stats
