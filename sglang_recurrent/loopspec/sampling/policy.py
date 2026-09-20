"""Sampling endpoint decisions for the LoopSpec pipeline."""

from dataclasses import dataclass

import torch

from ..decisions import DecisionBatch, DecisionOutcome, DecisionReadback
from .distributions import (
    prepare_sampling_probabilities,
    sample_probabilities,
)


@dataclass
class SamplingDecisionBatch(DecisionBatch):
    verification_distributions: dict | None = None


class SamplingDecisionPolicy:
    """Filter, sample, verify, and resolve speculative endpoint logits."""

    def __init__(
        self,
        max_rows,
        device,
        options,
        sampler,
        *,
        generator=None,
    ):
        if torch.device(device).type != "cuda":
            raise RuntimeError("LoopSpec sampling requires CUDA")
        self.readback = DecisionReadback(max_rows, device)
        self.options = options
        self.sampler = sampler
        self.generator = generator
        self.constants = torch.tensor(
            [0, 1], dtype=torch.int64, device=device
        )
        self.proposal_token_host = torch.empty(
            max_rows, dtype=torch.int64, pin_memory=True
        )
        self.proposal_token_device = torch.empty(
            max_rows, dtype=torch.int64, device=device
        )
        self.verification_token_host = torch.empty(
            max_rows, dtype=torch.int64, pin_memory=True
        )
        self.verification_token_device = torch.empty(
            max_rows, dtype=torch.int64, device=device
        )
        self.sample_row_host = torch.empty(
            max_rows, dtype=torch.int64, pin_memory=True
        )
        self.sample_row_device = torch.empty(
            max_rows, dtype=torch.int64, device=device
        )
        self.rejection_host = torch.empty(
            (max_rows, 3), dtype=torch.int64, pin_memory=True
        )
        self.rejection_ready = torch.cuda.Event()
        self.top_p_thresholds = torch.full(
            (max_rows, 1),
            options["top_p"],
            dtype=torch.float32,
            device=device,
        )

    @staticmethod
    def _stage_values(values, host_buffer, device_buffer):
        count = len(values)
        host = host_buffer[:count]
        host.copy_(torch.as_tensor(values, dtype=torch.int64))
        staged = device_buffer[:count]
        staged.copy_(host, non_blocking=True)
        return staged

    def _sample_drafts(self, entries, q1_context, staged_q1_rows):
        proposal_states = {}
        q2_entries = []
        for row, kind, lane, distribution in entries:
            if kind == "q1":
                proposal_states[row] = distribution
            elif kind == "q2":
                q2_entries.append((row, lane, distribution))

        q2_actions = {}
        draft_samples = {}
        if q2_entries:
            q2_targets = torch.stack(
                [distribution for _, _, distribution in q2_entries]
            )
            selected_tokens = self._stage_values(
                [lane.first_token for _, lane, _ in q2_entries],
                self.proposal_token_host,
                self.proposal_token_device,
            )
            first_distributions = torch.stack(
                [lane.first_state for _, lane, _ in q2_entries]
            )
            proposals, actions, sampled = self.sampler.sample_conditional_residual(
                first_distributions,
                q2_targets,
                selected_tokens,
                self.generator,
                result_slot=0,
            )
            for index, (row, _, _) in enumerate(q2_entries):
                proposal_states[row] = proposals[index]
                q2_actions[row] = actions[index]
                draft_samples[row] = sampled[index]

        q1_entries = [row for row, kind, _, _ in entries if kind == "q1"]
        if q1_entries:
            if q1_context is None:
                distributions = torch.stack(
                    [
                        distribution
                        for _, kind, _, distribution in entries
                        if kind == "q1"
                    ]
                )
                samples = sample_probabilities(
                    distributions,
                    self.generator,
                    self.sampler,
                    result_slot=2,
                )
            else:
                samples = self.sampler.sample_top_p(
                    *q1_context,
                    self.generator,
                    result_slot=2,
                    sample_rows=staged_q1_rows,
                )
            draft_samples.update(
                {
                    row: samples[index]
                    for index, row in enumerate(q1_entries)
                }
            )
        return proposal_states, q2_actions, draft_samples

    def _sample_finals(self, entries):
        distributions = {
            row: distribution
            for row, kind, _, distribution in entries
            if kind == "final"
        }
        direct = [
            (row, distribution)
            for row, kind, lane, distribution in entries
            if kind == "final" and lane.first_token is None
        ]
        direct_samples = {}
        if direct:
            sampled = sample_probabilities(
                torch.stack([distribution for _, distribution in direct]),
                self.generator,
                self.sampler,
                result_slot=1,
            )
            direct_samples = {
                row: sampled[index] for index, (row, _) in enumerate(direct)
            }

        verification = [
            (row, lane, distribution)
            for row, kind, lane, distribution in entries
            if kind == "final" and lane.first_token is not None
        ]
        decisions = {}
        if verification:
            target_masses = torch.stack(
                [distribution[lane.first_token] for _, lane, distribution in verification]
            )
            proposal_masses = torch.stack(
                [lane.first_state[lane.first_token] for _, lane, _ in verification]
            )
            tokens = self._stage_values(
                [lane.first_token for _, lane, _ in verification],
                self.verification_token_host,
                self.verification_token_device,
            )
            packed = self.sampler.verify(
                target_masses,
                proposal_masses,
                tokens,
                self.generator,
            )
            decisions = {
                row: packed[index]
                for index, (row, _, _) in enumerate(verification)
            }
        return distributions, direct_samples, decisions

    def _pack(self, entries, q2_actions, drafts, direct, verification):
        zero, one = self.constants
        none = (zero, zero, zero)
        values = []
        for row, kind, lane, _ in entries:
            if kind == "q1":
                decision = drafts[row], one, *none
            elif kind == "q2":
                action = q2_actions[row]
                decision = (
                    torch.where(action > 0, drafts[row], lane.first_token),
                    (action > 0).to(torch.int64),
                    action,
                    zero,
                    zero,
                )
            elif lane.first_token is None:
                decision = direct[row], zero, *none
            else:
                decision = verification[row]
            values.extend(decision)
        return torch.stack(values).reshape(-1, 5)

    def queue(self, post_output, endpoints, lanes):
        logits = post_output.next_token_logits
        q1_rows = [
            row for row, kind in enumerate(endpoints.values()) if kind == "q1"
        ]
        staged_q1_rows = self._stage_values(
            q1_rows, self.sample_row_host, self.sample_row_device
        )
        distributions, q1_context = prepare_sampling_probabilities(
            logits,
            top_p_thresholds=self.top_p_thresholds,
            sampler=self.sampler,
            **self.options,
        )
        entries = [
            (row, kind, lanes[row], distribution)
            for (row, kind), distribution in zip(
                endpoints.items(), distributions, strict=True
            )
        ]
        proposal_states, q2_actions, drafts = self._sample_drafts(
            entries, q1_context, staged_q1_rows
        )
        verification_distributions, direct, verification = (
            self._sample_finals(entries)
        )
        staged = self.readback.stage(
            self._pack(entries, q2_actions, drafts, direct, verification)
        )
        return SamplingDecisionBatch(
            staged,
            proposal_states=proposal_states,
            verification_distributions=verification_distributions,
        )

    def read(self, lanes, endpoints, batch):
        values = self.readback.read(batch.staged)
        decisions = dict(zip(endpoints, values, strict=True))
        rejected = [
            (row, decision)
            for row, decision in decisions.items()
            if decision[0] < 0
        ]
        if not rejected:
            return decisions
        for index, (row, _) in enumerate(rejected):
            lane = lanes[row]
            self.sampler.queue_rejection(
                lane.first_state,
                batch.verification_distributions[row],
                lane.second_token,
                lane.second_state,
                self.generator,
                self.rejection_host[index],
            )
        self.rejection_ready.record()
        self.rejection_ready.synchronize()
        results = self.rejection_host[: len(rejected)].tolist()
        for (_, decision), result in zip(rejected, results, strict=True):
            token, second_accepted, second_rejected = result
            decision[0] = token
            decision[2] = 2 if second_accepted else 0
            decision[4] = second_rejected
        return decisions

    def resolve(self, lane, endpoint_kind, decision, proposal_state, schedule):
        prediction, proposed, accepted_code, first_rejected, second_rejected = (
            decision
        )
        proposed = bool(proposed)
        if accepted_code == 1:
            accepted_stage = schedule.proposal_stage
        elif accepted_code == 2:
            accepted_stage = schedule.second_stage
        else:
            accepted_stage = None
        return DecisionOutcome(
            prediction=prediction,
            proposed=proposed,
            proposal_state=proposal_state if proposed else None,
            accepted_stage=accepted_stage,
            proposal_action=(
                "residual"
                if endpoint_kind == "q2" and proposed
                else None
            ),
            first_rejected=bool(first_rejected),
            second_rejected=bool(second_rejected),
        )
