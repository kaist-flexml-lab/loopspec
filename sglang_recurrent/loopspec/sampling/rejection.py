"""Captured residual fallback used after a first LoopSpec proposal rejects.

Q1 and P below are the normalized q1/final readout distributions, after any
top-p filtering. R is the saved distribution used to draw the extra q2 token:
the normalized positive q2-q1 residual with the q1 token excluded, NOT the
q2 readout distribution itself. Residuals here are not top-p filtered again.

Each graph replay handles one rejected q1. Buffers are reused, so callers must
serialize GPU work on one stream (or establish equivalent dependencies) and
wait for the result copy before reading its CPU destination. Not thread-safe.
"""

from collections.abc import Callable

import torch


class RejectionSampler:
    """Own fixed buffers and the CUDA graph for rare q1 rejection handling."""

    def __init__(
        self,
        vocabulary_size: int,
        device: torch.device | str | int,
        sample_with_uniforms: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
    ) -> None:
        """Configure a one-row fallback; do not allocate or capture its graph.

        vocabulary_size is V, the positive number of token IDs. device must
        identify CUDA. sample_with_uniforms is CategoricalSampler's callback:
        it receives probabilities [1,V], supplied uniforms [1], and an int64
        output [1], fills/returns that output without drawing additional RNG.
        capture() must be called before queue(). Return nothing.
        """
        self.vocabulary_size = vocabulary_size
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise RuntimeError("LoopSpec rejection sampling requires CUDA")
        self.sample_with_uniforms = sample_with_uniforms
        self.graph: torch.cuda.CUDAGraph | None = None

    def _run(self) -> None:
        """Compute fallback tensors from fixed buffers; write self.result [1,3].

        self.first/target/second are CUDA float32 [1,V] holding Q1, P, and R.
        metadata is int64 [2]: (q2 token or dummy 0, has_q2). uniforms is
        float32 [2]: q2 acceptance and fallback sampling draws. First form
        F=normalize(max(P-Q1,0)); when q2 exists, test its token y with
        min(1,F[y]/R[y]). On rejection sample normalize(max(F-R,0)); without
        q2 sample F. Zero-residual fallbacks are described inline below.

        Always compute a fallback sample, even if q2 is accepted, then select
        the output. Store int64 (token, second_accepted, second_rejected) in
        self.result. No top-p filtering or new RNG is performed here. Python
        runs during warmup/capture; later graph.replay() executes the recorded
        GPU operations without invoking this Python method. Return nothing.
        """
        # The q1 rejection was already decided by verify(). Its conditional
        # target is the normalized positive difference of final P and q1 Q1.
        first_delta = (self.target - self.first).clamp_min(0)
        first_mass = first_delta.sum(dim=-1, keepdim=True)
        has_first_residual = first_mass > 0
        target_mass = self.target.sum(dim=-1, keepdim=True)
        normalized_target = self.target / torch.where(target_mass > 0, target_mass, torch.ones_like(target_mass))
        first_residual = first_delta / torch.where(has_first_residual, first_mass, torch.ones_like(first_mass))
        # A first rejection with no positive residual is impossible for exact
        # normalized distributions, but can occur after floating-point sum
        # error. Sampling the normalized target keeps this rare path finite.
        first_residual = torch.where(has_first_residual, first_residual, normalized_target)

        # Both slices have shape [1]. The dummy token is still gathered when
        # q2 is absent, but has_second prevents that token from being accepted.
        token = self.metadata[:1]
        has_second = self.metadata[1:].to(torch.bool)
        # Verify the extra token against F, not directly against final P.
        # self.second is its actual residual proposal R, not the q2 readout.
        target_mass = first_residual.gather(1, token.unsqueeze(1)).squeeze(1)
        proposal_mass = self.second.gather(1, token.unsqueeze(1)).squeeze(1)
        proposal_mass = torch.where(has_second, proposal_mass, torch.ones_like(proposal_mass))
        acceptance = torch.minimum(torch.ones_like(target_mass), target_mass / proposal_mass)
        accepted = has_second & (self.uniforms[:1] < acceptance)

        # If q2 rejects, remove its proposal R from the first residual F.
        # Compute this unconditionally to keep one fixed graph for every case.
        second_delta = (first_residual - self.second).clamp_min(0)
        second_mass = second_delta.sum(dim=-1, keepdim=True)
        has_second_residual = second_mass > 0
        second_residual = second_delta / torch.where(has_second_residual, second_mass, torch.ones_like(second_mass))
        # No q2, or no positive second residual: use F instead. Otherwise use
        # normalize(max(F-R,0)). Neither path applies another top-p cutoff.
        fallback = torch.where(has_second.unsqueeze(1) & has_second_residual, second_residual, first_residual)
        # Consume the supplied sampling uniform even when q2 will be accepted;
        # prediction then selects the accepted token and discards this sample.
        sampled = self.sample_with_uniforms(fallback, self.uniforms[1:], self.sample)

        prediction = torch.where(accepted, token, sampled)
        self.result = torch.stack((prediction, accepted.to(torch.int64), (has_second & ~accepted).to(torch.int64)), dim=1)

    def capture(self) -> None:
        """Allocate stable buffers, warm the fallback, and capture it once.

        Called before serving through CategoricalSampler.capture_rejection_graph.
        Repeated calls after capture do nothing. Distribution buffers are CUDA
        float32 [1,V]; metadata is CUDA int64 [2] with a pinned CPU staging
        buffer; uniforms is CUDA float32 [2]; sample is CUDA int64 [1]. _run()
        creates the graph's int64 [1,3] result. Synthetic inputs exercise the
        no-q2 path; real queue() calls overwrite inputs/uniforms before replay.
        Return nothing. This captures only rejection fallback, not model work.
        """
        if self.graph is not None:
            return
        self.first = torch.empty((1, self.vocabulary_size), dtype=torch.float32, device=self.device)
        self.target = torch.empty_like(self.first)
        self.second = torch.empty_like(self.first)
        # metadata[0]: extra q2 token (dummy 0 if absent); [1]: presence flag.
        self.metadata = torch.zeros(2, dtype=torch.int64, device=self.device)
        self.metadata_host = torch.zeros(2, dtype=torch.int64, pin_memory=True)
        # queue() supplies both draws before each real replay; _run() has no RNG.
        self.uniforms = torch.empty(2, dtype=torch.float32, device=self.device)
        self.sample = torch.empty(1, dtype=torch.int64, device=self.device)
        self.first.zero_()
        self.target.zero_()
        self.target[:, 0] = 1
        self.second.zero_()

        # Finish input initialization before side-stream warmup, and complete
        # warmup before capture so callback kernels are ready to be recorded.
        capture_stream = torch.cuda.Stream(device=self.device)
        capture_stream.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(capture_stream):
            for _ in range(3):
                self._run()
        capture_stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=capture_stream):
            self._run()
        # Order later work on the caller's stream after the capture stream.
        torch.cuda.current_stream(self.device).wait_stream(capture_stream)
        self.graph = graph

    def queue(
        self,
        first_proposal: torch.Tensor,
        target: torch.Tensor,
        second_token: int | None,
        second_proposal: torch.Tensor | None,
        generator: torch.Generator | None,
        output_host: torch.Tensor,
    ) -> None:
        """Stage one rejection, replay its graph, and enqueue a CPU result copy.

        first_proposal/target are CUDA float32 [V] Q1/P distributions, already
        filtered and normalized. second_proposal is the saved CUDA float32 [V]
        residual R or None. When present, second_token must be a valid ID y
        drawn from R with R[y]>0; callers supply None for both when absent.
        generator is a CUDA-compatible RNG or None for the default generator.
        Always draw two uniforms, including when q2 is absent or accepted.

        output_host is caller-owned pinned CPU int64 [3], receiving (token,
        second_accepted, second_rejected). Keep it alive and do not read/reuse
        it until the queued copy completes. SamplingDecisionPolicy.read()
        waits on an event before reading its result rows. Raise if capture()
        has not run. Return None, not a completed result or a future; this
        method has no explicit completion wait for the final CPU copy.
        """
        if self.graph is None:
            raise RuntimeError("sampling rejection CUDA graph was not captured")
        self.first[0].copy_(first_proposal)
        self.target[0].copy_(target)
        has_second = second_proposal is not None
        if has_second:
            self.second[0].copy_(second_proposal)
        # When absent, the previous second distribution may remain in the
        # buffer; the presence flag makes _run() ignore its effect on results.
        self.metadata_host[0] = 0 if second_token is None else second_token
        self.metadata_host[1] = int(has_second)
        # Stage the CPU token/flag through reusable pinned memory. Distribution
        # copies, metadata upload, RNG, replay, and result copy use stream order.
        self.metadata.copy_(self.metadata_host, non_blocking=True)
        torch.rand(2, out=self.uniforms, generator=generator)
        self.graph.replay()
        output_host.copy_(self.result[0], non_blocking=True)
