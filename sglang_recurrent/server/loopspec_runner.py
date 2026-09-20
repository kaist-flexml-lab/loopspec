"""Load the model/tokenizer and prepare role-specific LoopSpec CUDA graphs.

Called inside the scheduler subprocess after model registration and backend
setup. This module builds execution resources; loopspec_scheduler owns the IPC
loop, request handling, and tokenizer special-token alignment afterward.
"""

import copy
import json
from functools import partial
from types import MethodType
from typing import Any

from transformers import PreTrainedConfig, PreTrainedTokenizerBase

from sglang import bench_one_batch as one_batch
from sglang.srt.model_executor.cuda_graph_config import Backend
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.model_executor.runner import (
    decode_cuda_graph_runner as decode_graph_module,
)
from sglang.srt.model_executor.runner.eager_runner import EagerRunner
from sglang.srt.utils import set_random_seed
from sglang.srt.utils.hf_transformers import get_config

from ..modeling.contracts import recurrence_count
from ..loopspec.cuda_graph import RecurrentCudaGraphRunner
from ..loopspec.pipeline import PipelineSchedule
from ..loopspec.sampling.sampler import CategoricalSampler


def _graph_configuration(
    stages: int,
    step_size: int,
    second_step: int | None,
) -> tuple[PipelineSchedule, dict[str, list[int]], list[int], int]:
    """Return (schedule, role graph sizes, union graph sizes, request capacity).

    stages is the fixed recurrence count S, not the physical layer count.
    step_size is K; second_step is the optional q2 multiplier X (depth K*X).
    PipelineSchedule validates these values and bounds live pipeline rows.
    Graph sizes count internal rows, not independent user requests.
    """
    schedule = PipelineSchedule(stages, step_size, second_step)

    # Each pre/recurrent/post role can see a different number of active rows.
    role_batch_sizes = schedule.role_batch_sizes()
    # SGLang first allocates shared graph buffers for the union of all sizes;
    # RecurrentCudaGraphRunner then captures only the sizes each role needs.
    graph_batch_sizes = {size for sizes in role_batch_sizes.values() for size in sizes}
    graph_batch_sizes = sorted(graph_batch_sizes)

    # The model is not loaded yet; conservatively include pre/post attention.
    # This is request-row capacity, separate from graph lanes and KV tokens.
    max_requests = schedule.request_pool_capacity()
    return schedule, role_batch_sizes, graph_batch_sizes, max_requests


def _install_eager_recurrent_state_bridge(runner: ModelRunner) -> None:
    """Patch this runner's eager load_batch in place; return nothing.

    SGLang extracts a ForwardBatch from its registered eager buffers. The
    project-added recurrent_role and Raven injected state must survive that
    extraction too, particularly during the eager full-prompt prefill.
    """
    eager_runner = runner.eager_runner
    original_load_batch = eager_runner.load_batch

    def load_batch(
        self: EagerRunner,
        forward_batch: ForwardBatch,
        pp_proxy_tensors: PPProxyTensors | None = None,
        **kwargs: Any,
    ) -> ForwardBatch:
        """Load standard buffers, then restore the two project-owned fields.

        pp_proxy_tensors carries upstream pipeline-parallel intermediates;
        LoopSpec does not use PP, but preserves the upstream method signature.
        Additional keyword arguments are forwarded unchanged.
        """
        loaded = original_load_batch(forward_batch, pp_proxy_tensors=pp_proxy_tensors, **kwargs)
        # Preserve references, including the injected tensor; do not clone it.
        for name in ("recurrent_role", "injected"):
            if hasattr(forward_batch, name):
                setattr(loaded, name, getattr(forward_batch, name))
        return loaded

    # Bind only this instance. original_load_batch above is already bound.
    eager_runner.load_batch = MethodType(load_batch, eager_runner)


def _load_loopspec_config(server_args: Any) -> PreTrainedConfig:
    """Read the normalized HF config without loading weights or a tokenizer.

    server_args is SGLang ServerArgs with the project's recurrent options.
    The scheduler must register the recurrent config/parser in this process
    before calling this loader; this function does not install compatibility.
    Require the LoopSpec parser so recurrence/graph sizing uses the same config
    interpretation as the later model load; a different parser raises ValueError.
    """

    configured_parser = server_args.model_config_parser
    if configured_parser != "looped_transformer_loopspec":
        raise ValueError(
            "LoopSpec requires its recurrent SGLang config parser, got "
            f"{configured_parser!r}"
        )
    return get_config(
        server_args.model_path,
        trust_remote_code=server_args.trust_remote_code,
        revision=server_args.revision,
        model_override_args=json.loads(server_args.json_model_override_args),
        model_config_parser=configured_parser,
    )


def make_loopspec_runner(
    server_args: Any,
    port_args: Any,
    gpu_id: int,
    tp_rank: int,
) -> tuple[ModelRunner, PreTrainedTokenizerBase]:
    """Return (initialized PyTorch model runner, loaded tokenizer).

    server_args/port_args are SGLang ServerArgs/PortArgs. gpu_id selects the
    local device; tp_rank is the tensor-parallel rank (zero in supported LoopSpec).
    The runner owns model weights, KV/request pools, role CUDA graphs, and
    sampling buffers. The caller applies family-specific tokenizer fixes.
    """
    config = _load_loopspec_config(server_args)
    stages = recurrence_count(config)
    schedule, role_sizes, graph_sizes, max_requests = _graph_configuration(
        stages,
        server_args.recurrent_step_size,
        server_args.recurrent_second_step,
    )

    # Graph configuration is nested and mutable. Keep the scheduler's public
    # settings, especially its external one-request limit, unchanged.
    runner_args = copy.deepcopy(server_args)
    runner_args.recurrent_role_batch_sizes = role_sizes
    # Prefill executes eagerly; decode replays exact-width full CUDA graphs.
    runner_args.cuda_graph_config.prefill.backend = Backend.DISABLED
    runner_args.cuda_graph_config.decode.backend = Backend.FULL
    runner_args.cuda_graph_config.decode.bs = graph_sizes
    runner_args.cuda_graph_config.decode.max_bs = schedule.max_rows
    # This is internal branch/request-row capacity, not external concurrency.
    runner_args.max_running_requests = max_requests

    # Temporarily replace the classes/functions used by the upstream loader.
    # These are process-local module globals, not changes to SGLang on disk.
    original_graph_runner = decode_graph_module.DecodeCudaGraphRunner
    original_tokenizer_loader = one_batch.get_tokenizer
    # bench_one_batch.load_model does not forward revision/backend itself.
    patched_tokenizer_loader = partial(
        original_tokenizer_loader,
        tokenizer_revision=runner_args.revision,
        tokenizer_backend=runner_args.tokenizer_backend,
    )
    decode_graph_module.DecodeCudaGraphRunner = RecurrentCudaGraphRunner
    one_batch.get_tokenizer = patched_tokenizer_loader

    try:
        # Loads weights, allocates pools, captures model graphs, and creates
        # the tokenizer. The benchmark API returns a wrapper around ModelRunner.
        loaded_runner, tokenizer = one_batch.load_model(runner_args, port_args, gpu_id, tp_rank)
        runner = loaded_runner.torch_runner
        _install_eager_recurrent_state_bridge(runner)

        # Reset RNGs after model/graph initialization; this does not reload or
        # reinitialize model weights. Sampler warmups use their own generators.
        set_random_seed(runner_args.random_seed)
        runner.recurrent_stages = stages
        runner.pipeline_schedule = schedule

        # Prepare sampling too, even if the first request will be greedy.
        # Capacity follows internal live rows, not HTTP request concurrency.
        sampler = CategoricalSampler(
            schedule.max_rows,
            runner.model_config.hf_config.vocab_size,
            runner.device,
        )
        # Compile sampling/filter kernels and capture the rejection graph now
        # so this setup is completed before the scheduler reports ready.
        sampler.warmup()
        sampler.warmup_filtered(runner.model_config.dtype)
        sampler.capture_rejection_graph()
        runner.loopspec_categorical_sampler = sampler

        return runner, tokenizer
    finally:
        # Restore both globals after success or any loading/warmup failure.
        decode_graph_module.DecodeCudaGraphRunner = original_graph_runner
        one_batch.get_tokenizer = original_tokenizer_loader
