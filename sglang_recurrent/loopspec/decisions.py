"""Choose greedy tokens and transfer compact endpoint decisions to the CPU.

LosslessLoopSpecExecutor calls queue -> read -> resolve for prefill's first token
and decode readouts. SamplingDecisionPolicy reuses the transfer/result classes,
but packs five integers per row instead of greedy's single token ID. Neither
policy updates executor statistics here; the executor counts final results.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import torch

from .pipeline import PipelineLane, PipelineSchedule


@dataclass
class DecisionBatch:
    """Describe one pending readback and optional GPU proposal distributions.

    staged describes the shared readback buffer, not an independent snapshot.
    proposal_states is empty for greedy and first-token sampling; sampling q1/q2
    may populate it with original lane-row keys and CUDA [vocab_size] vectors.
    These vectors are probabilities, not recurrent hidden states or slot IDs.
    """

    staged: tuple[int, int]  # (Number of endpoint rows, integer columns per row).
    proposal_states: dict[int, torch.Tensor] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class DecisionOutcome:
    """Policy-independent result consumed by the pipeline executor."""

    prediction: int  # Proposed token at q1/q2, or the selected final token.
    proposed: bool = False  # Whether this readout offers a new speculative child.
    proposal_state: torch.Tensor | None = None  # CUDA [vocab_size], sampling only.
    accepted_stage: int | None = None  # Accepted proposal's zero-based core depth.
    proposal_action: str | None = None  # q2 child kind: "argmax" or "residual".
    first_rejected: bool = False  # Final verification rejected the q1 proposal.
    second_rejected: bool = False  # Final verification also rejected q2, if tried.


class DecisionReadback:
    """Copy compact GPU decisions to CPU with one synchronization.

    One buffer/event serves one outstanding batch: callers must read it before
    staging another batch. This is not a multi-thread/multi-stream queue.
    """

    def __init__(self, max_rows: int, device: torch.device | str) -> None:
        """Allocate pinned CPU storage for up to max_rows five-integer decisions.

        max_rows is the internal lane capacity, not external request count.
        device must describe CUDA. The event binds when recorded; the caller
        must use the current stream on the decisions tensor's CUDA device.
        """
        if torch.device(device).type != "cuda":
            raise RuntimeError("LoopSpec decision readback requires CUDA")
        self.host = torch.empty(max_rows * 5, dtype=torch.int64, pin_memory=True)
        self.ready = torch.cuda.Event()

    def stage(self, decisions: torch.Tensor) -> tuple[int, int]:
        """Enqueue GPU-to-CPU copying and return (rows, columns) without waiting.

        decisions is CUDA int64 [M, C], with M <= max_rows. C=1 for greedy;
        sampling uses C=5: token, proposed, accepted_code, q1_reject, q2_reject.
        Input rows follow the endpoint dictionary's iteration order. The caller
        ensures capacity and does not overwrite the source before the copy.
        """
        rows, columns = decisions.shape
        target = self.host[: rows * columns].view(rows, columns)
        target.copy_(decisions, non_blocking=True)
        # Record after the copy so read() waits for CPU-visible results, not
        # merely for the GPU token-selection kernels that produced them.
        self.ready.record()
        return rows, columns

    def read(self, staged: tuple[int, int]) -> list[list[int]]:
        """Wait for the staged copy and return independent Python integer rows.

        staged is the tuple returned by the most recent stage() call. tolist()
        detaches the result from the reusable pinned buffer, so later batches
        cannot change an already returned decision list.
        """
        rows, columns = staged
        self.ready.synchronize()
        return self.host[: rows * columns].view(rows, columns).tolist()


class GreedyDecisionPolicy:
    """Select greedy tokens and copy them back with one synchronization."""

    def __init__(self, max_rows: int, device: torch.device | str) -> None:
        """Create reusable readback storage for at most max_rows CUDA decisions."""
        self.readback = DecisionReadback(max_rows, device)

    def queue(
        self,
        post_output: Any,
        endpoints: Mapping[int, str],
        lanes: Sequence[PipelineLane],
    ) -> DecisionBatch:
        """Queue argmax selection for endpoint logits and their CPU transfer.

        post_output exposes CUDA next_token_logits [M, vocab_size]. It is a
        LogitsProcessorOutput for decode or a SimpleNamespace for first-token
        selection; Any avoids importing SGLang only to annotate this interface.
        Its rows follow endpoints order, where each key is an index into lanes
        and each value is "q1", "q2", or "final". Greedy argmax needs neither
        the labels nor lane history; retain these arguments for the policy API.
        Return a pending DecisionBatch with one integer column and no proposals.
        """
        del endpoints, lanes
        choices = torch.argmax(post_output.next_token_logits, dim=-1, keepdim=True)
        return DecisionBatch(self.readback.stage(choices))

    def read(
        self,
        lanes: Sequence[PipelineLane],
        endpoints: Mapping[int, str],
        batch: DecisionBatch,
    ) -> dict[int, list[int]]:
        """Wait for queue()'s result and map each [token_id] to its original row.

        endpoints must keep the same keys/order used for queue(). Sparse keys
        are preserved: logits rows 0,1 can belong to lane rows 2,5. zip(strict)
        rejects row-count mismatches. lanes is unused by greedy readback but is
        needed by sampling's rejection path, so both policies accept it.
        """
        del lanes
        decisions = self.readback.read(batch.staged)
        return dict(zip(endpoints, decisions, strict=True))

    def resolve(
        self,
        lane: PipelineLane,
        endpoint_kind: str,
        decision: list[int],
        proposal_state: torch.Tensor | None,
        schedule: PipelineSchedule,
    ) -> DecisionOutcome:
        """Interpret one CPU [token_id] as a proposal or final greedy decision.

        lane holds earlier q1/q2 token IDs; endpoint_kind is "q1", "q2", or
        "final". q1 always proposes, q2 proposes only if its argmax differs from
        q1, and final compares against q1 then q2 to identify reusable progress.
        With no q1 token (including the temporary prefill lane), final only
        selects a token and does not report a rejection or acceptance.

        schedule converts an accepted proposal into its core iteration index.
        proposal_state is passed through unchanged (normally None for greedy),
        keeping the shared outcome interface. Return a DecisionOutcome without
        mutating lane, allocating a branch, launching CUDA, or changing metrics.
        """
        prediction = decision[0]
        proposed = endpoint_kind == "q1" or (endpoint_kind == "q2" and prediction != lane.first_token)

        accepted_stage = None
        first_rejected = second_rejected = False
        # None means no prior proposal; token ID 0 is a valid proposal to compare.
        if endpoint_kind == "final" and lane.first_token is not None:
            first_rejected = prediction != lane.first_token
            if not first_rejected:
                accepted_stage = schedule.proposal_stage
            elif lane.second_token is not None:
                second_rejected = prediction != lane.second_token
                if not second_rejected:
                    accepted_stage = schedule.second_stage

        return DecisionOutcome(
            prediction=prediction,
            proposed=proposed,
            proposal_state=proposal_state,
            accepted_stage=accepted_stage,
            proposal_action=(
                "argmax"
                if endpoint_kind == "q2" and proposed
                else None
            ),
            first_rejected=first_rejected,
            second_rejected=second_rejected,
        )
