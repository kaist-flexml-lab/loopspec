"""Add request-level decode metrics to SGLang's standard scheduler."""

import time
from collections.abc import Callable
from multiprocessing.connection import Connection
from types import ModuleType
from typing import Any

from sglang.srt.constants import HEALTH_CHECK_RID_PREFIX

from ..integrations.flashinfer import install_linear_backend
from ..integrations.sglang import (
    install_sglang_config_compatibility,
    register_recurrent_models,
    use_recurrent_tokenizer_special_tokens,
)
from .metrics import log_decode_metrics


# SGLang-owned objects use Any to preserve the deferred scheduler import below.
# Their concrete classes are documented at the corresponding entry points.
def _install_generation_metrics(
    scheduler_module: ModuleType,
    tokenizer_hook: Callable[[Any, Any], None] | None = None,
) -> None:
    """Replace the supplied scheduler module's Scheduler class in place.

    The optional hook receives (ModelConfig, tokenizer), where tokenizer may
    be None, and applies compatibility fixes after upstream initialization.
    Model execution remains owned by SGLang; this wrapper adds timing only.
    """

    class GenerationMetricsScheduler(scheduler_module.Scheduler):
        """Keep SGLang scheduling and attach metrics to each request."""

        def init_tokenizer(self) -> None:
            """Initialize the tokenizer, then align recurrent token settings."""
            super().init_tokenizer()
            if tokenizer_hook is not None:
                tokenizer_hook(self.model_config, self.tokenizer)

        @staticmethod
        def _finish_decode_metrics(
            req: Any,
            finished_at: float,
        ) -> None:
            """Log a completed SGLang Req once; timestamps are perf_counter seconds."""
            # Internal health generations must not count as experiment requests.
            if req.rid.startswith(HEALTH_CHECK_RID_PREFIX):
                return

            # Request-owned state prevents duplicate logs if seen again.
            if getattr(req, "_generation_metrics_logged", False):
                return

            started_at = getattr(req, "_generation_decode_started_at", None)
            output_token_count = len(req.output_ids_through_stop)

            # Without a measured decode interval, report zero tokens/time.
            if started_at is None or output_token_count <= 1:
                log_decode_metrics(scheduler_module.logger, 0, 0.0)
            else:
                # Count through the stop boundary, excluding the first token.
                log_decode_metrics(scheduler_module.logger, output_token_count - 1, finished_at - started_at)

            req._generation_metrics_logged = True

        def process_batch_result(self, batch: Any, result: Any) -> None:
            """Time SGLang's result handling without changing its output.

            batch is a ScheduleBatch; result is a GenerationBatchResult or
            EmbeddingBatchResult, matching the upstream dispatch interface.
            Recurrent serving uses generation results.
            """
            # Snapshot membership and lengths before SGLang appends tokens or
            # updates the batch. Keep references to requests that finish here.
            requests = tuple(batch.reqs)
            previous_lengths: dict[int, int] = {id(req): len(req.output_ids) for req in requests}

            super().process_batch_result(batch, result)

            # These are host-side result-processing boundaries, not CUDA
            # kernel timings; all requests in this batch share this timestamp.
            now = time.perf_counter()
            for req in requests:
                if (previous_lengths[id(req)] == 0 and len(req.output_ids) >= 1):
                    # Start decode timing after processing the first token.
                    req._generation_decode_started_at = now
                if req.finished():
                    self._finish_decode_metrics(req, now)

    # The upstream process entry point resolves this class when constructing
    # its scheduler. Install the wrapper before delegating to that entry point.
    scheduler_module.Scheduler = GenerationMetricsScheduler


def run_recurrent_scheduler_process(
    server_args: Any,
    port_args: Any,
    gpu_id: int,
    tp_rank: int,
    attn_cp_rank: int,
    moe_dp_rank: int,
    moe_ep_rank: int,
    pp_rank: int,
    dp_rank: int | None,
    pipe_writer: Connection,
) -> None:
    """Initialize recurrent support and run SGLang's baseline scheduler loop.

    server_args and port_args are SGLang ServerArgs and PortArgs. gpu_id
    selects the local device; the rank arguments identify tensor, attention
    context, MoE data/expert, pipeline, and data-parallel workers respectively.
    dp_rank may be None before SGLang configures the process. pipe_writer sends
    initialization information to the parent process.

    Keep argument order identical to SGLang's process target. The upstream
    entry point owns the blocking event loop and shutdown; no value is returned.
    """
    # Run setup inside the scheduler subprocess, before model construction.
    install_linear_backend(device=gpu_id)
    install_sglang_config_compatibility("LoopedRecurrentServingModel")
    register_recurrent_models()

    # Preserve lazy loading: importing this wrapper alone need not import the
    # upstream scheduler and all of its execution dependencies.
    from sglang.srt.managers import scheduler as scheduler_module

    _install_generation_metrics(
        scheduler_module,
        use_recurrent_tokenizer_special_tokens,
    )

    return scheduler_module.run_scheduler_process(
        server_args,
        port_args,
        gpu_id,
        tp_rank,
        attn_cp_rank,
        moe_dp_rank,
        moe_ep_rank,
        pp_rank,
        dp_rank,
        pipe_writer,
    )
