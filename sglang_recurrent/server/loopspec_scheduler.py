"""Serve LoopSpec requests using SGLang's tokenizer/detokenizer IPC protocol.

This module owns its request loop rather than subclassing SGLang's Scheduler.
External requests run one at a time; parallel LoopSpec lanes belong to one request.
"""

import logging
import signal
from array import array
from collections.abc import Sequence
from multiprocessing.connection import Connection
from time import perf_counter
from typing import Any

import psutil
import torch
import zmq
from transformers import PreTrainedTokenizerBase

from sglang.srt.constants import HEALTH_CHECK_RID_PREFIX
from sglang.srt.managers.io_struct import (
    BatchTokenIDOutput,
    BatchTokenizedGenerateReqInput,
    ShutdownReq,
    TokenizedGenerateReqInput,
)
from sglang.srt.managers.schedule_batch import (
    FINISH_LENGTH,
    Req,
)
from sglang.srt.managers.scheduler import configure_scheduler_process
from sglang.srt.plugins import load_plugins
from sglang.srt.utils.network import get_zmq_socket
from sglang.utils import get_exception_traceback

from ..integrations.flashinfer import install_linear_backend
from ..integrations.sglang import (
    install_sglang_config_compatibility,
    register_recurrent_models,
    use_recurrent_tokenizer_special_tokens,
)
from ..loopspec.executor import generate_loopspec
from .metrics import log_decode_metrics


logger = logging.getLogger(__name__)


def _unsupported_options(request: TokenizedGenerateReqInput) -> list[str]:
    """List unsupported request features; an empty list means none were found.

    SGLang has already tokenized the request and normalized sampling options.
    This check does not mutate it or validate prompt length/token budget.
    """
    params = request.sampling_params
    unsupported: list[str] = []

    # The local pipeline accepts token IDs, not multimodal/embedding inputs
    # or state owned by sessions, adapters, or distributed request routing.
    if request.mm_inputs:
        unsupported.append("multimodal input")
    if any(
        (
            request.need_wait_for_mm_inputs,
            request.num_items_assigned,
            request.mm_data_mooncake,
            request.encoder_urls,
        )
    ):
        unsupported.append("multimodal IPC state")

    if request.input_embeds is not None:
        unsupported.append("input_embeds")
    if request.token_type_ids is not None:
        unsupported.append("token_type_ids")
    if request.session_params is not None:
        unsupported.append("sessions")
    if request.lora_id is not None:
        unsupported.append("LoRA")
    if any(
        value is not None
        for value in (
            request.bootstrap_host,
            request.bootstrap_port,
            request.bootstrap_room,
            request.bootstrap_pair_key,
            request.decode_tp_size,
            request.routed_dp_rank,
            request.disagg_prefill_dp_rank,
        )
    ):
        unsupported.append("distributed request routing")

    # These optional outputs and logit transformations require execution paths
    # that generate_loopspec() does not implement.
    if request.return_logprob:
        unsupported.append("return_logprob")
    if request.return_hidden_states:
        unsupported.append("return_hidden_states")
    if request.return_routed_experts:
        unsupported.append("return_routed_experts")
    if request.return_indexer_topk:
        unsupported.append("return_indexer_topk")
    if request.return_entropy:
        unsupported.append("return_entropy")
    if request.return_bytes:
        unsupported.append("return_bytes")

    if request.custom_logit_processor is not None:
        unsupported.append("custom_logit_processor")
    if request.positional_embed_overrides is not None:
        unsupported.append("positional_embed_overrides")
    if request.require_reasoning:
        unsupported.append("hybrid reasoning")
    if request.multi_item_delimiter_indices is not None:
        unsupported.append("multi-item scoring")

    # Supported sampling is greedy or temperature/top-k/top-p sampling.
    # Reject other options rather than silently generating with different rules.
    if any(
        (
            params.json_schema,
            params.regex,
            params.ebnf,
            params.structural_tag,
        )
    ):
        unsupported.append("constrained decoding")
    if params.logit_bias:
        unsupported.append("logit_bias")
    if params.custom_params:
        unsupported.append("custom sampling parameters")
    if params.min_p != 0:
        unsupported.append("min_p")
    if params.min_new_tokens != 0:
        unsupported.append("min_new_tokens")
    if params.max_new_tokens is None:
        unsupported.append("unbounded max_new_tokens")
    if request.stream and params.stream_interval not in (None, 1):
        unsupported.append("stream_interval other than 1")
    if any(
        (
            params.frequency_penalty != 0,
            params.presence_penalty != 0,
            params.repetition_penalty != 1,
        )
    ):
        unsupported.append("sampling penalties")

    return unsupported


def _make_req(
    request: TokenizedGenerateReqInput,
    runner: Any,
    tokenizer: PreTrainedTokenizerBase,
) -> Req:
    """Adapt a tokenizer IPC request into SGLang's mutable generation Req.

    runner supplies the model's EOS IDs and vocabulary size. Req tracks
    committed output tokens, stop conditions, and incremental detokenization;
    it is not the LoopSpec RecurrentRequest that owns recurrent execution state.
    Preserve routing fields so normal and error responses reach the requester.
    """
    # SGLang can send input_embeds without input_ids. LoopSpec rejects that input,
    # but _run_request() builds Req before checking unsupported options, and
    # even its abort response uses len/slicing and incremental detokenization.
    # Replace None only to construct that error response, not to run the model
    # with an empty prompt. Use array("q"), matching Req.output_ids: SGLang joins
    # input and output arrays during detokenization, so [] would not suffice.
    input_ids = request.input_ids
    if input_ids is None:
        input_ids = array("q")
    req = Req(
        request.rid,
        request.input_text,
        input_ids,
        request.sampling_params,
        return_logprob=request.return_logprob,
        top_logprobs_num=request.top_logprobs_num,
        token_ids_logprob=request.token_ids_logprob,
        stream=request.stream,
        input_embeds=request.input_embeds,
        positional_embed_overrides=request.positional_embed_overrides,
        token_type_ids=request.token_type_ids,
        lora_id=request.lora_id,
        custom_logit_processor=request.custom_logit_processor,
        require_reasoning=request.require_reasoning,
        return_hidden_states=request.return_hidden_states,
        return_routed_experts=request.return_routed_experts,
        routed_experts_start_len=request.routed_experts_start_len,
        return_indexer_topk=request.return_indexer_topk,
        eos_token_ids=runner.model_config.hf_eos_token_id,
        bootstrap_host=request.bootstrap_host,
        bootstrap_port=request.bootstrap_port,
        bootstrap_room=request.bootstrap_room,
        routed_dp_rank=request.routed_dp_rank,
        disagg_prefill_dp_rank=request.disagg_prefill_dp_rank,
        vocab_size=runner.model_config.vocab_size,
        priority=request.priority,
        extra_key=request.extra_key,
        routing_key=request.routing_key,
        http_worker_ipc=request.http_worker_ipc,
        time_stats=request.time_stats,
        multi_item_delimiter_indices=request.multi_item_delimiter_indices,
    )

    # Req.__init__ has no tokenizer parameter and initializes this field to
    # None. Attach it afterward, as SGLang's own scheduler does, for stop-string
    # decoding and tokenizer-defined EOS/additional stop-token checks.
    req.tokenizer = tokenizer
    return req


def _send_output(
    socket: zmq.Socket,
    req: Req,
) -> None:
    """Send new output to the detokenizer and advance req's send offsets.

    socket is the PUSH socket connected to SGLang's detokenizer. An unfinished
    Req may be emitted only in streaming mode.
    """
    finish_reason = req.finished_reason or req.to_finish
    if finish_reason is None and not getattr(req, "stream", False):
        raise RuntimeError("LoopSpec attempted to emit an unfinished request")

    output_ids = req.output_ids_through_stop
    # Req keeps cumulative output. Offsets ensure each update sends only the
    # newly available token IDs and detokenizer context, without duplication.
    send_token_offset = req.send_token_offset
    decode_ids, read_offset = req.init_incremental_detokenize()
    incremental_decode_ids = decode_ids[req.send_decode_id_offset :]
    incremental_output_ids = list(output_ids[send_token_offset:])
    req.send_decode_id_offset = len(decode_ids)
    req.send_token_offset = len(output_ids)

    socket.send_pyobj(
        # SGLang expects a batch-shaped message even for our single request.
        BatchTokenIDOutput(
            rids=[req.rid],
            http_worker_ipcs=[req.http_worker_ipc],
            finished_reasons=[finish_reason.to_json() if finish_reason is not None else None],
            decoded_texts=[req.decoded_text],
            # Detokenization can depend on the prompt boundary. Keep SGLang's
            # surrounding-token context internal while preserving the public
            # completion-only output_ids contract below.
            decode_ids=[incremental_decode_ids],
            read_offsets=[read_offset],
            output_ids=[incremental_output_ids],
            skip_special_tokens=[req.sampling_params.skip_special_tokens],
            spaces_between_special_tokens=[req.sampling_params.spaces_between_special_tokens],
            no_stop_trim=[req.sampling_params.no_stop_trim],
            prompt_tokens=[len(req.origin_input_ids)],
            completion_tokens=[len(req.output_ids)],
            # LoopSpec does not report radix-cache hits or use SGLang's built-in
            # speculative decoder statistics; its counters go to our own log.
            cached_tokens=[0],
            spec_verify_ct=[],
            spec_num_correct_drafts=[],
            spec_correct_drafts_histogram=[],
            reasoning_tokens=[0],
            # Preserve the upstream message schema for unsupported outputs.
            input_token_logprobs_val=None,
            input_token_logprobs_idx=None,
            output_token_logprobs_val=None,
            output_token_logprobs_idx=None,
            input_top_logprobs_val=None,
            input_top_logprobs_idx=None,
            output_top_logprobs_val=None,
            output_top_logprobs_idx=None,
            input_token_ids_logprobs_val=None,
            input_token_ids_logprobs_idx=None,
            output_token_ids_logprobs_val=None,
            output_token_ids_logprobs_idx=None,
            output_token_entropy_val=None,
            output_hidden_states=None,
            routed_experts=[None],
            indexer_topk=[None],
            placeholder_tokens_idx=None,
            placeholder_tokens_val=None,
            retraction_counts=[0],
            time_stats=[req.time_stats],
        )
    )


def _run_request(
    request: TokenizedGenerateReqInput,
    runner: Any,
    tokenizer: PreTrainedTokenizerBase,
    send_to_detokenizer: zmq.Socket,
) -> None:
    """Handle one request to completion, including validation and output.

    runner is the model runner returned by make_loopspec_runner(). Invalid
    options/empty prompts produce an abort response without running the model.
    The stop callback below connects LoopSpec's committed tokens to SGLang's Req.
    """
    req = _make_req(request, runner, tokenizer)
    unsupported = _unsupported_options(request)
    if unsupported:
        req.set_finish_with_abort(f"LoopSpec does not support: {', '.join(unsupported)}")
        _send_output(send_to_detokenizer, req)
        return
    if not req.origin_input_ids:
        req.set_finish_with_abort("LoopSpec requires a nonempty prompt")
        _send_output(send_to_detokenizer, req)
        return
    if req.sampling_params.max_new_tokens == 0:
        # A valid request with no output budget completes without model work.
        req.finished_reason = FINISH_LENGTH(length=0)
        _send_output(send_to_detokenizer, req)
        return

    params = req.sampling_params
    sampling = None
    generator: torch.Generator | None = None
    # SGLang normalizes greedy sampling to top_k=1. None selects LoopSpec's
    # greedy policy; otherwise forward the supported sampling parameters.
    if params.top_k != 1:
        sampling = {
            "temperature": params.temperature,
            "top_k": params.top_k,
            "top_p": params.top_p,
        }
        if params.sampling_seed is not None:
            # Isolate this request's sampling RNG when the caller gives a seed.
            generator = torch.Generator(device=runner.device).manual_seed(params.sampling_seed)

    def should_stop(output_ids: Sequence[int]) -> bool:
        """Consume cumulative committed output and report whether Req finished."""
        # These are verified/committed tokens, not outstanding draft tokens.
        # Let SGLang handle EOS, stop tokens/strings, and the output limit.
        new_accepted_len = len(output_ids) - len(req.output_ids)
        req.output_ids.extend(output_ids[len(req.output_ids) :])
        req.update_finish_state(new_accepted_len)
        # Hold a possible partial stop string until it can be disambiguated.
        # Final output is sent once below, after generate_loopspec() returns.
        if (
            req.stream
            and not req.finished()
            and not req.check_match_stop_str_prefix()
        ):
            _send_output(send_to_detokenizer, req)
        return req.finished()

    output_ids, stats = generate_loopspec(
        runner,
        req.origin_input_ids,
        params.max_new_tokens,
        sampling=sampling,
        generator=generator,
        should_stop=should_stop,
    )
    # The callback and executor must agree on the entire committed sequence.
    if list(req.output_ids) != list(output_ids):
        raise RuntimeError("LoopSpec request state diverged from pipeline output")
    _send_output(send_to_detokenizer, req)

    # End at host-side output enqueue, not client receipt or a CUDA event.
    # generate_loopspec() records the start after processing the first token.
    decode_finished_at = perf_counter()
    # Health checks execute and respond normally, but are not experiment requests.
    if req.rid.startswith(HEALTH_CHECK_RID_PREFIX):
        return

    output_token_count = len(req.output_ids)
    decode_started_at = stats.pop("decode_started_at", None)
    # Retain LoopSpec statistics even without a decode interval.
    if decode_started_at is None or output_token_count <= 1:
        log_decode_metrics(logger, 0, 0.0, stats)
    else:
        # Decode timing excludes the first completion token.
        log_decode_metrics(logger, output_token_count - 1, decode_finished_at - decode_started_at, stats)


def _run_scheduler_loop(
    recv_from_tokenizer: zmq.Socket,
    runner: Any,
    tokenizer: PreTrainedTokenizerBase,
    send_to_detokenizer: zmq.Socket,
) -> None:
    """Block on tokenizer IPC and run requests sequentially until ShutdownReq.

    recv_from_tokenizer is a PULL socket; send_to_detokenizer is a PUSH socket.
    Batched input is unpacked into individual requests, not a GPU request batch.
    Other control messages are warned about and ignored. Incoming messages
    (including shutdown) are read only after the active request finishes.
    """

    while True:
        received = recv_from_tokenizer.recv_pyobj()
        if isinstance(received, ShutdownReq):
            return
        if isinstance(received, TokenizedGenerateReqInput):
            _run_request(received, runner, tokenizer, send_to_detokenizer)
        elif isinstance(received, BatchTokenizedGenerateReqInput):
            for request in received:
                _run_request(request, runner, tokenizer, send_to_detokenizer)
        else:
            logger.warning(
                "LoopSpec scheduler ignored unsupported control request %s",
                type(received).__name__,
            )


def run_loopspec_scheduler_process(
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
    """Load LoopSpec, report readiness, and serve requests in a child process.

    server_args/port_args are SGLang ServerArgs/PortArgs. gpu_id selects the
    local GPU; the rank arguments identify tensor, attention-context, MoE
    data/expert, pipeline, and data-parallel workers. LoopSpec requires one worker
    in each dimension, with rank zero (dp_rank may initially be None).
    pipe_writer is the multiprocessing connection used to notify the parent
    that initialization finished and report the model's request-size limits.
    Argument order matches SGLang's scheduler-process launch contract.
    """
    # Check here as well as in the CLI: this process target can be called alone.
    if any(
        size != 1
        for size in (
            server_args.tp_size,
            server_args.pp_size,
            server_args.dp_size,
            server_args.ep_size,
            server_args.attn_cp_size,
            server_args.moe_dp_size,
        )
    ):
        raise ValueError("LoopSpec currently requires every parallel size to be 1")
    if (
        tp_rank != 0
        or attn_cp_rank != 0
        or moe_dp_rank != 0
        or moe_ep_rank != 0
        or pp_rank != 0
        or dp_rank not in (None, 0)
    ):
        raise ValueError("LoopSpec must run on rank zero")

    # Use SGLang's process setup for logging, affinity, and parent-death handling.
    load_plugins()
    dp_rank = configure_scheduler_process(
        server_args,
        gpu_id,
        tp_rank,
        attn_cp_rank,
        moe_dp_rank,
        moe_ep_rank,
        pp_rank,
        dp_rank,
    )
    parent_process = psutil.Process().parent()

    try:
        # Compatibility and registry setup must precede model construction.
        install_linear_backend(device=gpu_id)
        install_sglang_config_compatibility("RecurrentServingModel")
        register_recurrent_models()
        # Load the builder only in the LoopSpec child process, after setup.
        from .loopspec_runner import make_loopspec_runner

        runner, tokenizer = make_loopspec_runner(server_args, port_args, gpu_id, tp_rank)
        use_recurrent_tokenizer_special_tokens(runner.model_config, tokenizer)

        # HTTP/tokenization stay in SGLang. Only this scheduler's execution loop
        # is project-owned; token output still goes through its detokenizer.
        context = zmq.Context(2)
        recv_from_tokenizer = get_zmq_socket(context, zmq.PULL, port_args.scheduler_input_ipc_name, False)
        send_to_detokenizer = get_zmq_socket(context, zmq.PUSH, port_args.detokenizer_ipc_name, False)
        pipe_writer.send(
            {
                "status": "ready",
                "max_total_num_tokens": runner.max_total_num_tokens,
                # Reserve room for at least one generated token in context/KV.
                "max_req_input_len": min(
                    runner.model_config.context_len - 1,
                    runner.max_total_num_tokens - 1,
                ),
            }
        )

        _run_scheduler_loop(
            recv_from_tokenizer,
            runner,
            tokenizer,
            send_to_detokenizer,
        )
    except Exception:
        # Notify SGLang's parent so it can tear down the failed server process
        # group instead of leaving the HTTP side waiting for a dead scheduler.
        traceback = get_exception_traceback()
        logger.error("LoopSpec scheduler hit an exception: %s", traceback)
        if parent_process is not None:
            parent_process.send_signal(signal.SIGQUIT)
