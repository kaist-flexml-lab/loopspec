"""Filter endpoint logits and sample token IDs without CPU readback.

Rows are post/readout batch rows, not recurrence indices or KV slots. The
sampling policy uses the resulting distributions for proposals/verification.
Sampler arguments use Any to avoid importing the Triton-backed sampler solely
for annotations; serving supplies a CategoricalSampler instance.
"""

from typing import Any

import torch


def filtered_probabilities(
    logits: torch.Tensor,
    temperature: float,
    top_k: int,
    top_p: float,
    top_p_thresholds: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return filtered, normalized CUDA float32 probabilities [B, V].

    logits is a CUDA floating-point tensor [B, vocabulary_size]. The caller
    supplies valid sampling settings: temperature > 0, top_k >= 0, and
    0 < top_p <= 1. This is not the temperature-zero greedy path. top_k=0 or
    top_k>=V disables top-k; top_p=1 disables top-p.

    Optional top_p_thresholds is CUDA float32 [N, 1], N>=B, providing per-row
    thresholds instead of allocating a tensor filled with top_p. The scalar
    top_p still selects the filtering path. This standalone filter does not
    use sampler scratch or draw tokens; serving prepares top-p scratch through
    prepare_sampling_probabilities() instead.

    Top-p keeps probabilities at least as large as the first sorted entry
    whose cumulative mass reaches the threshold, including boundary ties.
    Joint top-k/top-p uses both cutoffs from the original softmax distribution,
    then renormalizes once. Reject non-CUDA logits with RuntimeError.
    """
    if not logits.is_cuda:
        raise RuntimeError("LoopSpec sampling requires CUDA logits")
    vocabulary_size = logits.shape[-1]
    effective_top_k = 0 if top_k >= vocabulary_size else top_k
    # Accumulate softmax in float32 even when model logits are bf16/fp16.
    probs = (
        torch.softmax(logits, dim=-1, dtype=torch.float32)
        if temperature == 1
        else torch.softmax(logits.float() / temperature, dim=-1)
    )
    if effective_top_k == 0 and top_p == 1:
        return probs

    if effective_top_k == 0 or top_p == 1:
        if effective_top_k:
            # Top-k only. Keep the native-kernel import lazy for other paths.
            from sgl_kernel import top_k_renorm_prob

            return top_k_renorm_prob(probs, effective_top_k)

        sorted_probs = probs.sort(dim=-1, descending=True).values
        cumulative = sorted_probs.cumsum(dim=-1)
        thresholds = (
            torch.full((logits.shape[0], 1), top_p, dtype=torch.float32, device=logits.device)
            if top_p_thresholds is None else top_p_thresholds[: logits.shape[0]]
        )
        cutoff_indices = torch.searchsorted(cumulative, thresholds, right=False).clamp_max_(logits.shape[-1] - 1)
        # Compare probability values, not sorted ranks: ties at the cutoff
        # remain in the support. The output retains the original token order.
        cutoffs = sorted_probs.gather(-1, cutoff_indices)
        filtered = torch.where(probs >= cutoffs, probs, 0)
        return filtered / filtered.sum(dim=-1, keepdim=True)

    # Both filters are active. Compute top-p before any top-k renormalization.
    sorted_probs = probs.sort(dim=-1, descending=True).values
    cumulative = sorted_probs.cumsum(dim=-1)
    thresholds = (
        torch.full((logits.shape[0], 1), top_p, dtype=torch.float32, device=logits.device)
        if top_p_thresholds is None else top_p_thresholds[: logits.shape[0]]
    )
    cutoff_indices = torch.searchsorted(cumulative, thresholds, right=False).clamp_max_(vocabulary_size - 1)
    top_p_cutoffs = sorted_probs.gather(-1, cutoff_indices)
    top_k_cutoffs = sorted_probs[:, effective_top_k - 1 : effective_top_k]

    # The larger probability cutoff selects the intersection of both supports.
    cutoffs = torch.maximum(top_k_cutoffs, top_p_cutoffs)
    filtered = torch.where(probs >= cutoffs, probs, 0)
    return filtered / filtered.sum(dim=-1, keepdim=True)


def prepare_sampling_probabilities(
    logits: torch.Tensor,
    temperature: float,
    top_k: int,
    top_p: float,
    sampler: Any,
    *,
    top_p_thresholds: torch.Tensor | None = None,
) -> tuple[
    torch.Tensor,
    tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None,
]:
    """Return filtered probabilities and optional context for later top-p draws.

    logits/settings/thresholds follow filtered_probabilities(). sampler is the
    required CategoricalSampler workspace. No RNG is used and no token IDs
    are produced here. The first result is CUDA float32 probabilities [B, V].
    The second result is
    (unfiltered_probs [B,V], sorted_probs [B,V], cutoff_indices [B,1]) for the
    top-p-only path, otherwise None. The context also relies on the sampler's
    retained top-p chunk sums; consume it before another top-p preparation
    overwrites those sums. Sampling row selection and output-buffer selection
    belong to the later sampler call, not this preparation step.
    SamplingDecisionPolicy.queue uses this mode to filter all endpoint rows,
    process q2 proposals, then sample only q1 rows using the retained context.
    """
    vocabulary_size = logits.shape[-1]
    effective_top_k = 0 if top_k >= vocabulary_size else top_k
    if not logits.is_cuda:
        raise RuntimeError("LoopSpec sampling requires CUDA logits")

    if effective_top_k == 0 and top_p < 1:
        # Reuse the sorted cutoff and retained chunk sums for normalization
        # and sampling instead of recomputing reductions from filtered values.
        probs = (
            torch.softmax(logits, dim=-1, dtype=torch.float32)
            if temperature == 1
            else torch.softmax(logits.float() / temperature, dim=-1)
        )
        sorted_probs = probs.sort(dim=-1, descending=True).values
        cumulative = sorted_probs.cumsum(dim=-1)
        thresholds = (
            torch.full((logits.shape[0], 1), top_p, dtype=torch.float32, device=logits.device)
            if top_p_thresholds is None
            else top_p_thresholds[: logits.shape[0]]
        )
        cutoff_indices = torch.searchsorted(cumulative, thresholds, right=False).clamp_max_(vocabulary_size - 1)

        return sampler.normalize_top_p_for_sampling(
            probs, sorted_probs, cutoff_indices
        ), (probs, sorted_probs, cutoff_indices)

    # Unfiltered, top-k-only, and joint-filter paths use the generic sampler.
    filtered = filtered_probabilities(
        logits,
        temperature,
        effective_top_k,
        top_p,
        top_p_thresholds=top_p_thresholds,
    )
    return filtered, None


def sample_probabilities(
    probs: torch.Tensor,
    generator: torch.Generator | None,
    sampler: Any,
    *,
    result_slot: int = 0,
) -> torch.Tensor:
    """Return one sampled token ID per probability row, without CPU readback.

    probs is CUDA float32 [B, V] containing valid categorical probabilities.
    generator is a device-compatible RNG or None; sampler is the existing
    CategoricalSampler whose workspace supports this B and V. result_slot
    selects its reusable output buffer. Return a CUDA int64 [B] view, valid
    until that output region is overwritten; clone it if longer retention is
    needed. Reject non-CUDA input with RuntimeError. The policy uses this for
    q1 without a top-p context and for direct final-endpoint sampling.
    """
    if not probs.is_cuda:
        raise RuntimeError("LoopSpec sampling requires CUDA probabilities")
    return sampler.sample(probs, generator, result_slot=result_slot)
