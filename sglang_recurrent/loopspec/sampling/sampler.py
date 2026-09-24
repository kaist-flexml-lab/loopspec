"""Triton kernels and persistent sampling workspace for recurrent LoopSpec.

B denotes distribution rows, V vocabulary size, and C=ceil(V/2048) chunks.
Rows are endpoint batch rows, not recurrence indices, branches, or KV slots.
Serving supplies CUDA float32 probabilities with unit vocabulary-axis stride;
matrix row strides may vary. Integer token/row IDs use int64, and vector
metadata must be contiguous. These layout contracts are not fully validated.

Kernel tl.tensor arguments represent device pointers. Strides are in elements,
and tl.constexpr dimensions/flags specialize the launch. Kernels write their
output buffers and return None. Calls share mutable scratch space and must be
ordered on one stream or through explicit dependencies, not run concurrently.
"""

import torch
import triton
import triton.language as tl

from .rejection import RejectionSampler


@triton.jit
def _categorical_chunk_sums_kernel(
    probabilities: tl.tensor,
    chunk_sums: tl.tensor,
    row_stride: tl.constexpr,
    vocabulary_size: tl.constexpr,
    chunks: tl.constexpr,
    chunk_size: tl.constexpr,
) -> None:
    """Sum each vocabulary chunk into float32 chunk_sums [B, C].

    probabilities points to [B, V] nonnegative weights; columns are contiguous.
    row_stride locates successive rows, chunk_size bounds each program, and
    vocabulary_size masks the last chunk. Grid (B, C), called by sample() and
    _sample_with_uniforms() before _categorical_pick_kernel.
    """
    row = tl.program_id(0)
    chunk = tl.program_id(1)
    offsets = chunk * chunk_size + tl.arange(0, chunk_size)
    values = tl.load(
        probabilities + row * row_stride + offsets,
        mask=offsets < vocabulary_size,
        other=0.0,
    ).to(tl.float32)
    tl.store(chunk_sums + row * chunks + chunk, tl.sum(values, axis=0))


@triton.jit
def _categorical_pick_kernel(
    probabilities: tl.tensor,
    chunk_sums: tl.tensor,
    uniforms: tl.tensor,
    samples: tl.tensor,
    row_stride: tl.constexpr,
    vocabulary_size: tl.constexpr,
    chunks: tl.constexpr,
    chunk_block: tl.constexpr,
    chunk_size: tl.constexpr,
) -> None:
    """Select a chunk, then a token within it, using one uniform per row.

    probabilities is [B, V], chunk_sums is its precomputed [B, C] mass, uniforms
    is float32 [B] in [0,1), and samples receives int64 token IDs [B]. Weights
    need not sum to one but must have positive row mass. chunk_block is the
    power-of-two padded C. Grid (B,), launched after the chunk-sum kernel by
    sample() or the rejection callback _sample_with_uniforms().
    """
    row = tl.program_id(0)
    chunk_offsets = tl.arange(0, chunk_block)
    sums = tl.load(
        chunk_sums + row * chunks + chunk_offsets,
        mask=chunk_offsets < chunks,
        other=0.0,
    )
    total = tl.sum(sums, axis=0)
    target = tl.load(uniforms + row) * total
    chunk_cdf = tl.cumsum(sums, axis=0)
    selected_chunk = tl.argmax(
        (chunk_cdf > target).to(tl.int32),
        axis=0,
        tie_break_left=True,
    )
    # Convert the full-distribution target to a target within the chosen chunk.
    preceding_mass = tl.sum(
        tl.where(chunk_offsets < selected_chunk, sums, 0.0), axis=0
    )

    token_offsets = tl.arange(0, chunk_size)
    token_indices = selected_chunk * chunk_size + token_offsets
    values = tl.load(
        probabilities + row * row_stride + token_indices,
        mask=token_indices < vocabulary_size,
        other=0.0,
    ).to(tl.float32)
    token_cdf = tl.cumsum(values, axis=0)
    selected_token = tl.argmax(
        (token_cdf > target - preceding_mass).to(tl.int32),
        axis=0,
        tie_break_left=True,
    )
    tl.store(samples + row, selected_chunk * chunk_size + selected_token)


@triton.jit
def _categorical_pick_top_p_kernel(
    probabilities: tl.tensor,
    sorted_probabilities: tl.tensor,
    cutoff_indices: tl.tensor,
    chunk_sums: tl.tensor,
    uniforms: tl.tensor,
    samples: tl.tensor,
    sample_rows: tl.tensor,
    row_stride: tl.constexpr,
    sorted_row_stride: tl.constexpr,
    vocabulary_size: tl.constexpr,
    chunks: tl.constexpr,
    chunk_block: tl.constexpr,
    chunk_size: tl.constexpr,
    indirect_rows: tl.constexpr,
) -> None:
    """Sample selected rows directly from retained, unnormalized top-p weights.

    probabilities/sorted_probabilities are [B, V] in token/descending order;
    cutoff_indices supplies one sorted index per row, and chunk_sums [B, C]
    already excludes values below that cutoff. With indirect_rows, sample_rows
    is int64 [S] selecting source rows; otherwise S=B and this pointer is unused.
    uniforms [S] and output samples [S] follow selected-row order. Grid (S,),
    called by sample_top_p() after normalize_top_p_for_sampling() has prepared
    masses. Original token IDs, not sorted ranks, are written.
    """
    sample_row = tl.program_id(0)
    row = (
        tl.load(sample_rows + sample_row)
        if indirect_rows
        else sample_row
    )
    chunk_offsets = tl.arange(0, chunk_block)
    sums = tl.load(
        chunk_sums + row * chunks + chunk_offsets,
        mask=chunk_offsets < chunks,
        other=0.0,
    )
    total = tl.sum(sums, axis=0)
    target = tl.load(uniforms + sample_row) * total
    chunk_cdf = tl.cumsum(sums, axis=0)
    selected_chunk = tl.argmax(
        (chunk_cdf > target).to(tl.int32),
        axis=0,
        tie_break_left=True,
    )
    preceding_mass = tl.sum(
        tl.where(chunk_offsets < selected_chunk, sums, 0.0), axis=0
    )

    cutoff_index = tl.load(cutoff_indices + row)
    cutoff = tl.load(
        sorted_probabilities + row * sorted_row_stride + cutoff_index
    )
    token_offsets = tl.arange(0, chunk_size)
    token_indices = selected_chunk * chunk_size + token_offsets
    mask = token_indices < vocabulary_size
    values = tl.load(
        probabilities + row * row_stride + token_indices,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    values = tl.where(values >= cutoff, values, 0.0)
    token_cdf = tl.cumsum(values, axis=0)
    selected_token = tl.argmax(
        (token_cdf > target - preceding_mass).to(tl.int32),
        axis=0,
        tie_break_left=True,
    )
    tl.store(samples + sample_row, selected_chunk * chunk_size + selected_token)


@triton.jit
def _second_proposal_chunk_sums_kernel(
    first_probabilities: tl.tensor,
    second_probabilities: tl.tensor,
    selected_tokens: tl.tensor,
    chunk_sums: tl.tensor,
    first_row_stride: tl.constexpr,
    second_row_stride: tl.constexpr,
    vocabulary_size: tl.constexpr,
    chunks: tl.constexpr,
    chunk_size: tl.constexpr,
) -> None:
    """Sum q2 residual weights r[t]=max(q2[t]-q1[t],0), excluding the q1 token.

    first/second_probabilities are normalized [B, V] readout distributions;
    selected_tokens is the existing q1 token ID [B]. Write chunk_sums [B, C].
    Grid (B, C), first kernel in sample_conditional_residual(); zero total
    residual mass is allowed and later disables the extra q2 proposal.
    """
    row = tl.program_id(0)
    chunk = tl.program_id(1)
    offsets = chunk * chunk_size + tl.arange(0, chunk_size)
    mask = offsets < vocabulary_size
    first = tl.load(
        first_probabilities + row * first_row_stride + offsets,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    second = tl.load(
        second_probabilities + row * second_row_stride + offsets,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    selected = tl.load(selected_tokens + row)
    values = tl.maximum(second - first, 0.0)
    values = tl.where(offsets == selected, 0.0, values)
    tl.store(chunk_sums + row * chunks + chunk, tl.sum(values, axis=0))


@triton.jit
def _categorical_pick_second_proposal_kernel(
    first_probabilities: tl.tensor,
    second_probabilities: tl.tensor,
    selected_tokens: tl.tensor,
    chunk_sums: tl.tensor,
    uniforms: tl.tensor,
    samples: tl.tensor,
    first_row_stride: tl.constexpr,
    second_row_stride: tl.constexpr,
    vocabulary_size: tl.constexpr,
    chunks: tl.constexpr,
    chunk_block: tl.constexpr,
    chunk_size: tl.constexpr,
) -> None:
    """Pick a possible q2 token from the same residual used for chunk sums.

    first/second_probabilities are [B, V], selected_tokens [B] excludes q1's
    token, chunk_sums is [B, C], and uniforms/samples are [B]. Grid (B,), called
    between residual summation and normalization by sample_conditional_residual.
    Sampling runs even for inactive rows; its ID must be ignored when the later
    action flag is zero, including zero-mass rows.
    """
    row = tl.program_id(0)
    chunk_offsets = tl.arange(0, chunk_block)
    sums = tl.load(
        chunk_sums + row * chunks + chunk_offsets,
        mask=chunk_offsets < chunks,
        other=0.0,
    )
    total = tl.sum(sums, axis=0)
    target = tl.load(uniforms + row) * total
    chunk_cdf = tl.cumsum(sums, axis=0)
    selected_chunk = tl.argmax(
        (chunk_cdf > target).to(tl.int32),
        axis=0,
        tie_break_left=True,
    )
    preceding_mass = tl.sum(
        tl.where(chunk_offsets < selected_chunk, sums, 0.0), axis=0
    )

    token_offsets = tl.arange(0, chunk_size)
    token_indices = selected_chunk * chunk_size + token_offsets
    mask = token_indices < vocabulary_size
    first = tl.load(
        first_probabilities + row * first_row_stride + token_indices,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    second = tl.load(
        second_probabilities + row * second_row_stride + token_indices,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    selected = tl.load(selected_tokens + row)
    values = tl.maximum(second - first, 0.0)
    values = tl.where(token_indices == selected, 0.0, values)
    token_cdf = tl.cumsum(values, axis=0)
    selected_token = tl.argmax(
        (token_cdf > target - preceding_mass).to(tl.int32),
        axis=0,
        tie_break_left=True,
    )
    tl.store(samples + row, selected_chunk * chunk_size + selected_token)


@triton.jit
def _normalize_second_proposal_kernel(
    first_probabilities: tl.tensor,
    second_probabilities: tl.tensor,
    selected_tokens: tl.tensor,
    chunk_sums: tl.tensor,
    normalized: tl.tensor,
    actions: tl.tensor,
    first_row_stride: tl.constexpr,
    second_row_stride: tl.constexpr,
    output_row_stride: tl.constexpr,
    vocabulary_size: tl.constexpr,
    chunks: tl.constexpr,
    chunk_block: tl.constexpr,
    chunk_size: tl.constexpr,
) -> None:
    """Write residual probabilities [B, V] and q2 proposal-action flags [B].

    Inputs match _second_proposal_chunk_sums_kernel. Normalize the positive
    q2-q1 residual with the selected q1 token excluded; zero mass gives an
    all-zero row. actions is int64: 1 only when mass>0 AND q2[x]<q1[x] at the
    q1 token x. This is a proposal gate, not final acceptance. Grid (B, C),
    last kernel in sample_conditional_residual(); only chunk zero writes flags.
    """
    row = tl.program_id(0)
    chunk = tl.program_id(1)
    sum_offsets = tl.arange(0, chunk_block)
    sums = tl.load(
        chunk_sums + row * chunks + sum_offsets,
        mask=sum_offsets < chunks,
        other=0.0,
    )
    mass = tl.sum(sums, axis=0)
    has_mass = mass > 0

    offsets = chunk * chunk_size + tl.arange(0, chunk_size)
    mask = offsets < vocabulary_size
    first = tl.load(
        first_probabilities + row * first_row_stride + offsets,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    second = tl.load(
        second_probabilities + row * second_row_stride + offsets,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    selected = tl.load(selected_tokens + row)
    selected_first = tl.load(
        first_probabilities + row * first_row_stride + selected
    )
    selected_second = tl.load(
        second_probabilities + row * second_row_stride + selected
    )
    proposal = tl.maximum(second - first, 0.0)
    proposal = tl.where(offsets == selected, 0.0, proposal)
    output = tl.where(has_mass, proposal / mass, 0.0)
    tl.store(normalized + row * output_row_stride + offsets, output, mask=mask)
    gated = selected_second < selected_first
    action = (has_mass & gated).to(tl.int64)
    tl.store(actions + row, action, mask=chunk == 0)


@triton.jit
def _top_p_filtered_chunk_sums_kernel(
    probabilities: tl.tensor,
    sorted_probabilities: tl.tensor,
    cutoff_indices: tl.tensor,
    chunk_sums: tl.tensor,
    row_stride: tl.constexpr,
    sorted_row_stride: tl.constexpr,
    vocabulary_size: tl.constexpr,
    chunks: tl.constexpr,
    chunk_size: tl.constexpr,
) -> None:
    """Sum weights at or above each row's top-p probability cutoff.

    probabilities/sorted_probabilities are [B, V]; contiguous cutoff_indices
    contains B integer ranks (shape [B] or [B,1]). Keep boundary ties and write
    chunk_sums [B, C]. Grid (B, C), called by normalize_top_p_for_sampling()
    with retained top-p scratch.
    """
    row = tl.program_id(0)
    chunk = tl.program_id(1)
    cutoff_index = tl.load(cutoff_indices + row)
    cutoff = tl.load(
        sorted_probabilities + row * sorted_row_stride + cutoff_index
    )
    offsets = chunk * chunk_size + tl.arange(0, chunk_size)
    values = tl.load(
        probabilities + row * row_stride + offsets,
        mask=offsets < vocabulary_size,
        other=0.0,
    ).to(tl.float32)
    filtered = tl.where(values >= cutoff, values, 0.0)
    tl.store(chunk_sums + row * chunks + chunk, tl.sum(filtered, axis=0))


@triton.jit
def _normalize_top_p_kernel(
    probabilities: tl.tensor,
    sorted_probabilities: tl.tensor,
    cutoff_indices: tl.tensor,
    chunk_sums: tl.tensor,
    normalized: tl.tensor,
    row_stride: tl.constexpr,
    sorted_row_stride: tl.constexpr,
    output_row_stride: tl.constexpr,
    vocabulary_size: tl.constexpr,
    chunks: tl.constexpr,
    chunk_block: tl.constexpr,
    chunk_size: tl.constexpr,
) -> None:
    """Write normalized [B, V] top-p probabilities in original token order.

    Inputs/cutoffs match _top_p_filtered_chunk_sums_kernel, which must run first
    to fill chunk_sums [B, C]. Divide retained values by positive total mass,
    zero excluded values, and write normalized using output_row_stride.
    Grid (B, C), second kernel in normalize_top_p_for_sampling(); no RNG.
    """
    row = tl.program_id(0)
    chunk = tl.program_id(1)
    sum_offsets = tl.arange(0, chunk_block)
    sums = tl.load(
        chunk_sums + row * chunks + sum_offsets,
        mask=sum_offsets < chunks,
        other=0.0,
    )
    mass = tl.sum(sums, axis=0)
    cutoff_index = tl.load(cutoff_indices + row)
    cutoff = tl.load(
        sorted_probabilities + row * sorted_row_stride + cutoff_index
    )
    offsets = chunk * chunk_size + tl.arange(0, chunk_size)
    mask = offsets < vocabulary_size
    values = tl.load(
        probabilities + row * row_stride + offsets,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    output = tl.where(values >= cutoff, values / mass, 0.0)
    tl.store(normalized + row * output_row_stride + offsets, output, mask=mask)


@triton.jit
def _verification_decisions_kernel(
    target_masses: tl.tensor,
    proposal_masses: tl.tensor,
    uniforms: tl.tensor,
    tokens: tl.tensor,
    decisions: tl.tensor,
    rows: int,
    block_size: tl.constexpr,
) -> None:
    """Test q1 acceptance and write int64 decision rows [B, 5].

    target_masses/proposal_masses are float32 [B] probabilities p[x]/q1[x]
    at tokens [B], not full distributions; q1[x] must be positive. uniforms is
    float32 [B]. Accept when u<min(1,p[x]/q1[x]). Output fields are token-or--1,
    proposed=0, accepted_code=1-or-0, first_rejected, second_rejected=0.
    A -1 token asks policy.read() to queue residual rejection handling later.
    verify() launches grid (1,) with block_size covering all runtime rows.
    """
    row = tl.arange(0, block_size)
    mask = row < rows
    target = tl.load(target_masses + row, mask=mask)
    proposal = tl.load(proposal_masses + row, mask=mask)
    acceptance = tl.minimum(1.0, target / proposal)
    accepted = tl.load(uniforms + row, mask=mask) < acceptance
    token = tl.load(tokens + row, mask=mask)
    prediction = tl.where(accepted, token, -1)
    base = row * 5
    tl.store(decisions + base, prediction, mask=mask)
    tl.store(decisions + base + 1, 0, mask=mask)
    tl.store(decisions + base + 2, accepted.to(tl.int64), mask=mask)
    tl.store(decisions + base + 3, (~accepted).to(tl.int64), mask=mask)
    tl.store(decisions + base + 4, 0, mask=mask)


class CategoricalSampler:
    """Own CUDA scratch/output buffers shared by one serialized sampler.

    Probability inputs follow the module's dtype/layout contract. Methods
    enqueue GPU work without readback unless documented otherwise. Token IDs,
    action flags, and verification decisions are reusable workspace views;
    callers must consume/copy them before the same region is overwritten.
    Normalized distributions are newly allocated tensors, not scratch views.
    """

    def __init__(
        self,
        max_rows: int,
        vocabulary_size: int,
        device: torch.device | str | int,
        *,
        result_slots: int = 3,
    ) -> None:
        """Allocate workspace for up to max_rows distributions of vocabulary_size.

        device must identify CUDA; dimensions must be positive. result_slots
        is the number of independent token-output buffers, not KV slots. LoopSpec
        uses three: q2 drafts (0), direct final samples (1), and q1 drafts (2).
        Construct the rejection helper but do not capture its graph yet.
        The server runner later calls warmup(), warmup_filtered(), and capture.
        """
        device = torch.device(device)
        if device.type != "cuda":
            raise RuntimeError("LoopSpec categorical sampling requires CUDA")
        self.max_rows = max_rows
        self.vocabulary_size = vocabulary_size
        self.chunk_size = 2048
        self.chunks = triton.cdiv(vocabulary_size, self.chunk_size)
        # Generic [max_rows, C] mass scratch for categorical/q2/rejection work.
        self.chunk_sums = torch.empty((max_rows, self.chunks), dtype=torch.float32, device=device)
        # Separate scratch preserves q1 top-p masses while q2 uses chunk_sums.
        self.top_p_chunk_sums = torch.empty_like(self.chunk_sums)
        self.uniforms = torch.empty(max_rows, dtype=torch.float32, device=device)
        # Separate result slots prevent q2/q1/final draws in one policy.queue()
        # from overwriting each other's IDs before the decisions are packed.
        self.samples = torch.empty((result_slots, max_rows), dtype=torch.int64, device=device)
        self.verification_decisions = torch.empty((max_rows, 5), dtype=torch.int64, device=device)
        self.proposal_actions = torch.empty(max_rows, dtype=torch.int64, device=device)
        self.rejections = RejectionSampler(vocabulary_size, device, self._sample_with_uniforms)

    def sample(
        self,
        probabilities: torch.Tensor,
        generator: torch.Generator | None = None,
        *,
        result_slot: int = 0,
    ) -> torch.Tensor:
        """Draw one token per [B, V] row; return a CUDA int64 [B] output view.

        probabilities is float32 with nonnegative values and positive row mass;
        rows need not sum to one. B<=max_rows, V matches the workspace, and
        columns must be contiguous. generator is a CUDA-compatible RNG or None.
        result_slot selects samples[result_slot, :B]. Enqueue RNG, chunk sums,
        then CDF selection; no synchronization. distributions.sample_probabilities
        uses this for q1 without a retained top-p context and direct finals.
        """
        rows, vocabulary_size = probabilities.shape
        if rows > self.max_rows or vocabulary_size != self.vocabulary_size:
            raise ValueError("categorical sampler workspace shape mismatch")
        torch.rand(rows, out=self.uniforms[:rows], generator=generator)
        samples = self.samples[result_slot, :rows]
        _categorical_chunk_sums_kernel[(rows, self.chunks)](
            probabilities,
            self.chunk_sums,
            probabilities.stride(0),
            vocabulary_size,
            self.chunks,
            self.chunk_size,
            num_warps=8,
        )
        _categorical_pick_kernel[(rows,)](
            probabilities,
            self.chunk_sums,
            self.uniforms,
            samples,
            probabilities.stride(0),
            vocabulary_size,
            self.chunks,
            triton.next_power_of_2(self.chunks),
            self.chunk_size,
            num_warps=8,
        )
        return samples

    def _sample_with_uniforms(
        self,
        probabilities: torch.Tensor,
        uniforms: torch.Tensor,
        samples: torch.Tensor,
    ) -> torch.Tensor:
        """Fill and return caller-owned sample IDs using supplied random values.

        probabilities is CUDA float32 [B, V] with the same contract as sample();
        uniforms is contiguous float32 [B] in [0,1), samples is int64 [B].
        The caller ensures workspace bounds. No RNG or CPU readback occurs.
        RejectionSampler._run invokes this callback while warming/capturing its
        residual fallback; graph replay later runs the recorded kernels.
        """
        rows, vocabulary_size = probabilities.shape
        _categorical_chunk_sums_kernel[(rows, self.chunks)](
            probabilities,
            self.chunk_sums,
            probabilities.stride(0),
            vocabulary_size,
            self.chunks,
            self.chunk_size,
            num_warps=8,
        )
        _categorical_pick_kernel[(rows,)](
            probabilities,
            self.chunk_sums,
            uniforms,
            samples,
            probabilities.stride(0),
            vocabulary_size,
            self.chunks,
            triton.next_power_of_2(self.chunks),
            self.chunk_size,
            num_warps=8,
        )
        return samples

    def capture_rejection_graph(self) -> None:
        """Prepare the residual fallback graph before serving; return None.

        Delegate to RejectionSampler.capture(), which allocates fixed buffers,
        warms kernels, synchronizes capture work, and records the graph once.
        The server runner calls this after sampler warmup. Later calls reuse
        the captured graph rather than replacing its buffers.
        """
        self.rejections.capture()

    def queue_rejection(
        self,
        first_proposal: torch.Tensor,
        target: torch.Tensor,
        second_token: int | None,
        second_proposal: torch.Tensor | None,
        generator: torch.Generator | None,
        output_host: torch.Tensor,
    ) -> None:
        """Queue one rejected q1's fallback and an asynchronous host result copy.

        first_proposal/target are CUDA float32 [V] q1/final distributions.
        second_proposal is the saved q2 residual distribution [V], or None;
        when present, second_token is its valid proposed token ID. generator
        supplies device RNG. output_host is caller-owned pinned CPU int64 [3]
        receiving (token, second_accepted, second_rejected). Return None.
        SamplingDecisionPolicy.read calls this after detecting q1 rejection;
        it waits on its event before reading the host result. Requires prior
        graph capture; this wrapper itself adds no completion barrier.
        """
        self.rejections.queue(
            first_proposal,
            target,
            second_token,
            second_proposal,
            generator,
            output_host,
        )

    def sample_conditional_residual(
        self,
        first_probabilities: torch.Tensor,
        second_probabilities: torch.Tensor,
        selected_tokens: torch.Tensor,
        generator: torch.Generator | None = None,
        *,
        result_slot: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (residual probabilities, action flags, candidate token IDs).

        first/second_probabilities are matching CUDA float32 [B, V] normalized
        q1/q2 distributions; selected_tokens is contiguous int64 [B] containing
        already-drawn q1 IDs x. generator and result_slot work as in sample().
        Form r[t]=max(q2[t]-q1[t],0) with r[x]=0, then sum, sample, and normalize.

        Return a new [B, V] tensor (zero if no residual mass), plus int64 [B]
        workspace views for actions and IDs. action=1 requires both positive
        residual mass and q2[x]<q1[x]; otherwise the candidate must be ignored.
        The q1 child is not replaced. policy._sample_drafts calls this at q2,
        before q1 sampling, using generic scratch rather than top_p_chunk_sums.
        """
        rows, vocabulary_size = second_probabilities.shape
        if first_probabilities.shape != second_probabilities.shape:
            raise ValueError("directional probability shapes must match")
        if rows > self.max_rows or vocabulary_size != self.vocabulary_size:
            raise ValueError("categorical sampler workspace shape mismatch")
        torch.rand(rows, out=self.uniforms[:rows], generator=generator)
        samples = self.samples[result_slot, :rows]
        normalized = torch.empty_like(second_probabilities)
        actions = self.proposal_actions[:rows]
        _second_proposal_chunk_sums_kernel[(rows, self.chunks)](
            first_probabilities,
            second_probabilities,
            selected_tokens,
            self.chunk_sums,
            first_probabilities.stride(0),
            second_probabilities.stride(0),
            vocabulary_size,
            self.chunks,
            self.chunk_size,
            num_warps=8,
        )
        _categorical_pick_second_proposal_kernel[(rows,)](
            first_probabilities,
            second_probabilities,
            selected_tokens,
            self.chunk_sums,
            self.uniforms,
            samples,
            first_probabilities.stride(0),
            second_probabilities.stride(0),
            vocabulary_size,
            self.chunks,
            triton.next_power_of_2(self.chunks),
            self.chunk_size,
            num_warps=8,
        )
        _normalize_second_proposal_kernel[(rows, self.chunks)](
            first_probabilities,
            second_probabilities,
            selected_tokens,
            self.chunk_sums,
            normalized,
            actions,
            first_probabilities.stride(0),
            second_probabilities.stride(0),
            normalized.stride(0),
            vocabulary_size,
            self.chunks,
            triton.next_power_of_2(self.chunks),
            self.chunk_size,
            num_warps=8,
        )
        return normalized, actions, samples

    def verify(
        self,
        target_masses: torch.Tensor,
        proposal_masses: torch.Tensor,
        tokens: torch.Tensor,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Return a CUDA int64 [B, 5] view with q1 verification decisions.

        target_masses/proposal_masses are contiguous float32 [B] final/q1
        probabilities at the candidate tokens (int64 [B]), not full vectors.
        Require B<=max_rows and positive proposal masses; generator supplies
        acceptance uniforms. Fields are (token-or--1, proposed=0, accepted_code,
        first_rejected, second_rejected=0). accepted_code is 1 for accepted q1,
        otherwise 0; -1 defers token selection to the rejection graph.
        Called by policy._sample_finals; does not sample a residual itself.
        """
        rows = target_masses.shape[0]
        torch.rand(rows, out=self.uniforms[:rows], generator=generator)
        decisions = self.verification_decisions[:rows]
        _verification_decisions_kernel[(1,)](
            target_masses,
            proposal_masses,
            self.uniforms,
            tokens,
            decisions,
            rows,
            triton.next_power_of_2(self.max_rows),
            num_warps=1,
        )
        return decisions

    def normalize_top_p_for_sampling(
        self,
        probabilities: torch.Tensor,
        sorted_probabilities: torch.Tensor,
        cutoff_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Normalize all rows and retain chunk masses for a later top-p draw.

        probabilities is CUDA float32 [B, V] in token order; sorted_probabilities
        holds the same values descending per row. cutoff_indices is contiguous
        int64 [B] or [B,1], indexing each sorted row. Keep values >= the cutoff,
        including ties, and divide by positive retained mass into a new [B,V]
        tensor. No RNG is used. Write top_p_chunk_sums, which survives ordinary/
        q2 sampling on chunk_sums. prepare_sampling_probabilities() calls this;
        consume its context before another preparation overwrites the masses.
        """
        rows, vocabulary_size = probabilities.shape
        if rows > self.max_rows or vocabulary_size != self.vocabulary_size:
            raise ValueError("categorical sampler workspace shape mismatch")
        normalized = torch.empty_like(probabilities)
        _top_p_filtered_chunk_sums_kernel[(rows, self.chunks)](
            probabilities,
            sorted_probabilities,
            cutoff_indices,
            self.top_p_chunk_sums,
            probabilities.stride(0),
            sorted_probabilities.stride(0),
            vocabulary_size,
            self.chunks,
            self.chunk_size,
            num_warps=8,
        )
        _normalize_top_p_kernel[(rows, self.chunks)](
            probabilities,
            sorted_probabilities,
            cutoff_indices,
            self.top_p_chunk_sums,
            normalized,
            probabilities.stride(0),
            sorted_probabilities.stride(0),
            normalized.stride(0),
            vocabulary_size,
            self.chunks,
            triton.next_power_of_2(self.chunks),
            self.chunk_size,
            num_warps=8,
        )
        return normalized

    def sample_top_p(
        self,
        probabilities: torch.Tensor,
        sorted_probabilities: torch.Tensor,
        cutoff_indices: torch.Tensor,
        generator: torch.Generator | None = None,
        *,
        result_slot: int = 0,
        sample_rows: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return CUDA int64 [S] IDs using previously prepared top-p masses.

        probabilities/sorted_probabilities/cutoff_indices must match the last
        normalize_top_p_for_sampling call, including row order. The input is
        the original unfiltered [B,V] probabilities, not its normalized output.
        sample_rows is optional contiguous CUDA int64 [S] selecting valid
        source rows; None selects all B rows. Require S<=max_rows. generator
        supplies S uniforms; result_slot chooses the reusable output buffer.
        Empty selection returns an empty view without a pick-kernel launch.
        Called by the policy after processing q2 to draw only q1 rows, and by
        warmup_filtered() to exercise that path. No normalization/readback here.
        """
        rows, vocabulary_size = probabilities.shape
        if rows > self.max_rows or vocabulary_size != self.vocabulary_size:
            raise ValueError("categorical sampler workspace shape mismatch")
        sample_count = rows if sample_rows is None else sample_rows.shape[0]
        torch.rand(
            sample_count,
            out=self.uniforms[:sample_count],
            generator=generator,
        )
        samples = self.samples[result_slot, :sample_count]
        if sample_count:
            _categorical_pick_top_p_kernel[(sample_count,)](
                probabilities,
                sorted_probabilities,
                cutoff_indices,
                self.top_p_chunk_sums,
                self.uniforms,
                samples,
                # The kernel ignores this pointer when indirect_rows=False.
                samples if sample_rows is None else sample_rows,
                probabilities.stride(0),
                sorted_probabilities.stride(0),
                vocabulary_size,
                self.chunks,
                triton.next_power_of_2(self.chunks),
                self.chunk_size,
                num_warps=8,
                indirect_rows=sample_rows is not None,
            )
        return samples

    def warmup(self) -> None:
        """Exercise categorical/q2/verification kernels, then synchronize CUDA.

        Called by the server runner before serving, with synthetic one-row
        probabilities and a private seeded RNG. Requires max_rows>=1 and at
        least two result slots. Includes an inactive zero-residual q2 row.
        Mutates scratch/output buffers, returns None, and does not capture the
        rejection graph or guarantee compilation of every later specialization.
        Top-p preparation and row-selected sampling run in warmup_filtered().
        """
        device = self.chunk_sums.device
        probabilities = torch.zeros(
            (1, self.vocabulary_size), dtype=torch.float32, device=device
        )
        probabilities[:, 0] = 1
        selected = torch.zeros(1, dtype=torch.int64, device=device)
        generator = torch.Generator(device=device).manual_seed(0)
        self.sample(probabilities, generator, result_slot=0)
        self.sample_conditional_residual(
            probabilities,
            probabilities,
            selected,
            generator,
            result_slot=1,
        )
        mass = torch.ones(1, dtype=torch.float32, device=device)
        token = torch.zeros(1, dtype=torch.int64, device=device)
        self.verify(mass, mass, token, generator)
        torch.cuda.synchronize(device)

    def warmup_filtered(self, logits_dtype: torch.dtype = torch.bfloat16) -> None:
        """Exercise top-p filtering from model-dtype logits, then synchronize.

        The server runner passes its model dtype. Use synthetic [1,V] logits,
        top_p=0.7 and a private seeded RNG; no model forward or user tokens.
        Requires max_rows>=1 and at least three result slots for q1 slot 2.
        Prepare probabilities without drawing a token, then sample an explicit
        q1 row, matching the policy's deferred top-p path and indirect kernel.
        Mutate workspace, import distributions lazily, and return None after
        GPU completion.
        """
        from .distributions import prepare_sampling_probabilities

        logits = torch.zeros(
            (1, self.vocabulary_size),
            dtype=logits_dtype,
            device=self.chunk_sums.device,
        )
        thresholds = torch.full(
            (1, 1), 0.7, dtype=torch.float32, device=logits.device
        )
        _, context = prepare_sampling_probabilities(
            logits,
            temperature=1.0,
            top_k=1 << 30,
            top_p=0.7,
            top_p_thresholds=thresholds,
            sampler=self,
        )
        assert context is not None  # Fixed top-p-only settings above.
        self.sample_top_p(
            *context,
            torch.Generator(device=logits.device).manual_seed(0),
            result_slot=2,
            sample_rows=torch.zeros(1, dtype=torch.int64, device=logits.device),
        )
        torch.cuda.synchronize(logits.device)
