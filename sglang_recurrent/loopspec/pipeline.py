"""Schedule token replicas and select branches without handling CUDA or KV data.

One cycle advances every live lane by K recurrent iterations. Readouts at q1,
optional q2, and final depth propose or commit the next token. The executor owns
tensor computation, sampling, and physical state/KV storage; this module owns
the logical tree and the ordered list of committed completion token IDs.
"""

from dataclasses import dataclass
from typing import Any, Mapping, Protocol, Sequence


@dataclass
class PipelineLane:
    """One input token progressing through repeated executions of the core.

    stage counts only core iterations: pre runs before iteration 0, and post
    reads out after q1/q2/final iterations without incrementing stage itself.
    If completion tokens are c0, c1, ..., a lane with index=1 processes c0
    to predict c1. The prompt is not included in this index.

    Tensor-valued sampling fields use Any to avoid importing torch solely
    for annotations. Hidden-state bank slots are ordinary integer indices.
    """

    stage: int  # Next zero-based recurrent iteration to execute, not a layer ID.
    index: int  # Zero-based completion index of the prediction, not the input token.
    token: int  # Input token being processed; prediction is the following token.
    state: int | None = None  # State-bank slot; None until the lane first runs.
    branch: int = 0  # Logical ID: 0 initially, then increasing IDs; no in-call reuse.
    parent_branch: int | None = None  # Branch whose readout created this replica.
    # Parent's last core iteration before the readout that created this lane:
    # K-1 for q1, K*X-1 for q2, S-1 for a final fallback; initial lane uses None.
    origin_stage: int | None = None
    first_token: int | None = None  # q1's candidate for the next token after token.
    # LoopSpec sampling: CUDA torch.Tensor [vocab_size], q1 probabilities;
    # None before q1 or in greedy mode. Not a hidden-state tensor.
    first_state: Any = None
    second_token: int | None = None  # Optional alternative q2 prediction.
    # LoopSpec sampling: CUDA torch.Tensor [vocab_size], q2 residual proposal
    # probabilities; None without a q2 proposal or in greedy mode.
    second_state: Any = None


@dataclass(frozen=True)
class PipelineResult:
    """One lane result, including an optional next-token proposal."""

    state: int  # State-bank slot containing the updated hidden state.
    prediction: int  # Next-token ID at a readout; unused between readouts.
    proposed: bool = False  # Whether this readout offers a speculative child.
    # LoopSpec sampling: CUDA torch.Tensor [vocab_size] of proposal probabilities,
    # stored as first_state/second_state; None for greedy/non-proposal results.
    proposal_state: Any = None
    accepted_stage: int | None = None  # Accepted q1/q2 stage; None means neither.
    proposal_action: str | None = None  # q2 child kind: "residual" or "argmax".


class PipelineExecutor(Protocol):
    """Batched model execution and final speculative-sampling boundary."""

    schedule: "PipelineSchedule"

    def should_stop(self, output_ids: Sequence[int]) -> bool:
        """Return whether the committed completion IDs meet a stopping condition."""
        ...

    def forward(self, lanes: Sequence[PipelineLane]) -> Sequence[PipelineResult]:
        """Advance lanes by K iterations and return one result per lane, in order."""
        ...

    def retain_branches(
        self,
        lanes: Sequence[PipelineLane],
        accepted_stage: int | None = None,
    ) -> None:
        """Retain storage for live lanes, using the accepted q1/q2 stage if any.

        An empty sequence releases the request's remaining branches. Physical
        cleanup and any deferred retention are the executor's responsibility.
        """
        ...


def _descends_from(
    branch: int | None,
    ancestor: int,
    parents: Mapping[int, int | None],
) -> bool:
    """Return whether branch equals ancestor or reaches it along parent IDs.

    parents is the scheduler's acyclic logical ancestry map. Missing parents
    terminate the walk; no request-pool or KV indices are involved.
    """
    while branch is not None and branch != ancestor:
        branch = parents.get(branch)
    return branch == ancestor


def _prune_ancestry(
    lanes: Sequence[PipelineLane],
    root: int,
    parents: Mapping[int, int | None],
) -> dict[int, int | None]:
    """Return live ancestor paths with the selected branch as their new root.

    lanes must all descend from root in parents; an escaped path raises
    RuntimeError. The input map and lanes are not modified.
    """
    live = {root}
    for lane in lanes:
        branch = lane.branch
        while branch != root:
            if branch is None or branch not in parents:
                raise RuntimeError("pipeline branch escaped the selected ancestry")
            live.add(branch)
            branch = parents[branch]
    return {
        branch: None if branch == root else parents[branch]
        for branch in live
    }


@dataclass(frozen=True)
class PipelineSchedule:
    """Validated LoopSpec schedule and every shape derived from S, K, and X."""

    stages: int  # S: total recurrent iterations per token, not transformer layers.
    step_size: int  # K: iterations per cycle; q1 reads out after the first K.
    second_step: int | None = None  # X: q2 reads out after K*X iterations.

    def __post_init__(self) -> None:
        """Reject invalid S/K/X combinations with ValueError after construction."""
        if not isinstance(self.stages, int) or self.stages <= 0:
            raise ValueError(f"stages S must be a positive integer, got {self.stages}")
        if (not isinstance(self.step_size, int) or not 1 <= self.step_size < self.stages or self.stages % self.step_size):
            raise ValueError(f"step_size K must be a positive divisor of S with K < S; got K={self.step_size}, S={self.stages}")
        if self.second_step is not None and (not isinstance(self.second_step, int) or not 1 < self.second_step < self.depth):
            raise ValueError(f"second_step X must satisfy 1 < X < S/K so K*X < S; got X={self.second_step}, S/K={self.depth}")

    @property
    def depth(self) -> int:
        """Return S/K, the number of cycles needed for one lane to reach final."""
        return self.stages // self.step_size

    @property
    def proposal_stage(self) -> int:
        """Return q1's zero-based last recurrent iteration: K - 1."""
        return self.step_size - 1

    @property
    def second_stage(self) -> int | None:
        """Return q2's zero-based last iteration K*X - 1, or None if disabled."""
        if self.second_step is None:
            return None
        return self.step_size * self.second_step - 1

    @property
    def endpoint_kinds(self) -> dict[int, str]:
        """Map readout iteration indices to q1, optional q2, and final roles."""
        endpoints = {self.proposal_stage: "q1"}
        if self.second_stage is not None:
            endpoints[self.second_stage] = "q2"
        endpoints[self.stages - 1] = "final"
        return endpoints

    @property
    def endpoint_stages(self) -> tuple[int, ...]:
        """Return readout iteration indices in increasing depth order."""
        return tuple(sorted(self.endpoint_kinds))

    @property
    def row_birth_sizes(self) -> list[int]:
        """Return full-tree birth bounds for times 0 through depth - 1.

        Each entry counts new stage-zero lanes, not allocated request-pool
        rows. With q2 enabled, retain both q1 and q2 alternatives in the bound.
        """
        if self.second_step is None:
            return [1] * self.depth

        # Sampling may retain q1 and add q2, so capture the original full-tree
        # bound. Before q2 arrives there is one birth per time; afterwards
        # b_t = b_(t-1) + b_(t-X).
        births = [1] * self.second_step
        for step in range(self.second_step, self.depth):
            births.append(births[step - 1] + births[step - self.second_step])
        return births

    @property
    def max_rows(self) -> int:
        """Return the live-lane capacity bound, not the request-pool capacity."""
        return sum(self.row_birth_sizes)

    def request_pool_capacity(
        self,
        *,
        has_pre_attention: bool = True,
        has_post_attention: bool = True,
    ) -> int:
        """Return a sufficient physical request-row capacity, not an exact peak.

        Count timelines per role/depth, not physical attention layers. Defaults
        conservatively include both pre and post until the model is available.
        The bound includes allocation before deferred branch cleanup and the
        original prefill rows, which never join the reserved fork-row free list
        even after returning to SGLang's public pool.
        """
        births = self.row_birth_sizes
        # One extra cycle: forward allocates before flushing prior retention.
        births.append(
            1 if self.second_step is None
            else births[-1] + births[self.depth - self.second_step]
        )
        post_steps = [stage // self.step_size + 1 for stage in self.endpoint_stages]
        root_rows = (
            self.stages + int(has_pre_attention)
            + int(has_post_attention) * len(post_steps)
        )
        # At age a a branch has touched K*a core timelines, optional pre,
        # and the post endpoints reached so far. Include the delayed frontier
        # through b[D], plus one full logical root and the original prefill.
        return 2 * root_rows + sum(
            births[self.depth + 1 - age]
            * (
                self.step_size * age + int(has_pre_attention)
                + int(has_post_attention) * sum(step <= age for step in post_steps)
            )
            for age in range(1, self.depth + 1)
        )

    def role_max_batch_sizes(self) -> dict[str, int]:
        """Return direct upper bounds for pre, recurrent, and post graphs."""
        births = self.row_birth_sizes
        # At the maximum-width frontier, a lane at pipeline step j was born
        # at time depth-1-j. Post runs at the q1, q2, and final steps.
        post_steps = {0, self.depth - 1}
        if self.second_step is not None:
            post_steps.add(self.second_step - 1)
        return {
            "pre": births[-1],
            "recurrent": sum(births),
            "post": sum(births[self.depth - 1 - step] for step in post_steps),
        }

    def role_batch_sizes(self) -> dict[str, list[int]]:
        """Capture every batch from one through each role's direct maximum."""
        return {
            role: list(range(1, maximum + 1))
            for role, maximum in self.role_max_batch_sizes().items()
        }


def continue_pipelined(
    first_token: int,
    max_new_tokens: int,
    executor: PipelineExecutor,
) -> list[int]:
    """Return committed completion IDs, including the already chosen first token.

    first_token is the full-prompt prefill result; max_new_tokens is the total
    completion budget including it and must be positive. executor advances
    lanes, verifies proposals, checks stopping conditions, and manages storage.
    The returned list excludes the prompt and all uncommitted draft tokens.
    """
    if max_new_tokens <= 0:
        raise ValueError("a committed first token requires a positive budget")

    schedule = executor.schedule
    proposal_stage = schedule.proposal_stage
    second_stage = schedule.second_stage
    proposal_stages = {proposal_stage, second_stage} - {None}
    stride = schedule.step_size
    final_stage = schedule.stages - 1
    max_rows = schedule.max_rows
    committed = [first_token]
    if max_new_tokens == 1:
        # Prefill has already reserved the root request rows and KV slots.
        # There will be no decode cycle whose normal terminal retention would
        # release them, so close the request explicitly.
        executor.retain_branches(())
        return committed

    # Logical branch IDs are local to this generation call. Pruning removes
    # ancestry/storage but never recycles an ID; the next call starts at 0 again.
    next_branch = 1
    parents: dict[int, int | None] = {0: None}
    lanes = [
        PipelineLane(
            stage=0,
            index=1,
            token=first_token,
        )
    ]

    while len(committed) < max_new_tokens:
        results = list(executor.forward(lanes))
        if len(results) != len(lanes):
            raise ValueError("executor must return one result per lane")

        # Lanes to run next cycle: collect existing lanes that have not reached
        # final depth, append newly created lanes, then remove branches rejected
        # by final verification. All remaining lanes enter the next forward together.
        following: list[PipelineLane] = []
        # New lanes starting at stage=0 with tokens proposed by this cycle's q1/q2.
        # Append them to following after processing all existing lane results.
        # This is list order, not a requirement to wait for existing lanes to finish.
        children: list[PipelineLane] = []
        # The lane reaching final depth this cycle and its result, or None.
        # Commit its next token and select the branch to keep only after collecting
        # all lane results, so a child created this cycle can also be selected.
        final: tuple[PipelineLane, PipelineResult] | None = None
        # K-1 if final verification accepts q1, or K*X-1 if it accepts q2:
        # the zero-based last core iteration before the accepted proposal's readout.
        # Pass this to branch retention to decide whether cleanup can be deferred.
        # Leave None without final verification, without acceptance, or at completion.
        accepted_stage: int | None = None

        for lane, result in zip(lanes, results):
            end_stage = lane.stage + stride - 1
            if end_stage == final_stage:
                if final is not None:
                    raise RuntimeError("multiple branches reached the final stage")
                final = lane, result
                continue

            # Update the existing lane for the next cycle; do not create a copy.
            lane.stage = end_stage + 1
            lane.state = result.state
            continued = lane
            if end_stage in proposal_stages and result.proposed:
                # Save this proposal on the parent lane for final verification.
                # A q2 proposal does not replace the saved q1 proposal or remove
                # its child branch; both remain until final verification selects
                # which branch to keep. Greedy proposes q2 only if its token
                # differs from q1; sampling may propose a token from a residual
                # distribution computed using the q1 and q2 probabilities.
                if end_stage == proposal_stage:
                    continued.first_token = result.prediction
                    continued.first_state = result.proposal_state
                elif result.proposal_action in ("residual", "argmax"):
                    continued.second_token = result.prediction
                    continued.second_state = result.proposal_state
                else:
                    raise RuntimeError("q2 proposal has no branch action")

                if lane.index + 1 < max_new_tokens:
                    # The proposal becomes a new input token at stage zero;
                    # its hidden state is initialized by the executor's pre role.
                    children.append(
                        PipelineLane(
                            stage=0,
                            index=lane.index + 1,
                            token=result.prediction,
                            branch=next_branch,
                            parent_branch=lane.branch,
                            origin_stage=end_stage,
                        )
                    )
                    parents[next_branch] = lane.branch
                    next_branch += 1
            following.append(continued)
        following.extend(children)

        if final is not None:
            lane, result = final
            if lane.index != len(committed):
                raise RuntimeError("pipeline attempted to commit out of order")
            committed.append(result.prediction)
            if executor.should_stop(committed):
                executor.retain_branches(())
                return committed

            if lane.index + 1 < max_new_tokens:
                # Reuse the accepted proposal's progress only when both token
                # and originating readout match. Otherwise start a fresh lane.
                # Each parent creates at most one child at each readout, so
                # matching parent_branch and origin_stage identifies at most one.
                selected = next(
                    (
                        child
                        for child in following
                        if child.parent_branch == lane.branch
                        and child.token == result.prediction
                        and child.origin_stage == result.accepted_stage
                    ),
                    None,
                )

                if selected is None:
                    selected = PipelineLane(
                        stage=0,
                        index=lane.index + 1,
                        token=result.prediction,
                        branch=next_branch,
                        parent_branch=lane.branch,
                        origin_stage=final_stage,
                    )
                    parents[next_branch] = lane.branch
                    next_branch += 1
                    following.append(selected)

                accepted_stage = result.accepted_stage
                selected_branch = selected.branch
                # Keep the selected continuation and its speculative descendants;
                # the executor later releases storage for discarded branches.
                following = [
                    child
                    for child in following
                    if _descends_from(child.branch, selected_branch, parents)
                ]
                parents = _prune_ancestry(following, selected_branch, parents)
            else:
                following = []

        if len(following) > max_rows:
            raise RuntimeError("speculative tree exceeded branch row limit")
        lanes = following
        executor.retain_branches(tuple(lanes), accepted_stage)
        if not lanes and len(committed) < max_new_tokens:
            raise RuntimeError("pipeline drained before generation completed")

    return committed
