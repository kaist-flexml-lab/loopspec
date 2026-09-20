"""Launch recurrent models through SGLang's standard HTTP server."""

import argparse
import sys
from types import SimpleNamespace

from transformers import PreTrainedConfig

from sglang.srt.entrypoints.http_server import launch_server
from sglang.srt.server_args import ServerArgs

from ..integrations.sglang import (
    install_sglang_config_compatibility,
    use_ouro_tokenizer_special_tokens,
)
from .baseline_scheduler import run_recurrent_scheduler_process
from .loopspec_scheduler import run_loopspec_scheduler_process


def run_recurrent_detokenizer_process(server_args, port_args):
    """Run SGLang's detokenizer with recurrent tokenizer compatibility fixes."""

    from sglang.srt.managers.detokenizer_manager import (
        DetokenizerManager,
        run_detokenizer_process,
    )

    class RecurrentDetokenizerManager(DetokenizerManager):
        def init_tokenizer(self, args):
            super().init_tokenizer(args)
            config_dict, _ = PreTrainedConfig.get_config_dict(
                args.model_path,
                cache_dir=args.download_dir,
                revision=args.revision,
            )
            model_config = SimpleNamespace(model_type=config_dict.get("model_type"))
            use_ouro_tokenizer_special_tokens(model_config, self.tokenizer)

    return run_detokenizer_process(
        server_args,
        port_args,
        detokenizer_manager_class=RecurrentDetokenizerManager,
    )


def _positive_int(value):
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def main():
    parser = argparse.ArgumentParser()
    ServerArgs.add_cli_args(parser)
    parser.set_defaults(
        attention_backend="triton",
        dtype="bfloat16",
        host="0.0.0.0",
        max_running_requests=1,
        mem_fraction_static=0.55,
        port=30000,
        trust_remote_code=True,
    )
    parser.add_argument("--loopspec", action="store_true")
    parser.add_argument(
        "--step-size",
        type=_positive_int,
        default=1,
        help="K: recurrent layers per pipeline step; must divide S",
    )
    parser.add_argument(
        "--second-step",
        type=_positive_int,
        help=(
            "X: optional q2 step, placing q2 at K*X layers; "
            "must satisfy 1 < X < S/K"
        ),
    )

    argv = sys.argv[1:]
    if "--config" in argv:
        from sglang.srt.server_args_config_parser import ConfigArgumentMerger

        argv = ConfigArgumentMerger(parser).merge_config_with_args(argv)
    args = parser.parse_args(argv)

    if args.loopspec and args.max_running_requests != 1:
        parser.error("--max-running-requests greater than 1 is baseline-only")
    if args.loopspec and args.stream_interval != 1:
        parser.error("--loopspec currently requires --stream-interval=1")
    if args.loopspec and any(
        size != 1
        for size in (
            args.tp_size,
            args.pp_size,
            args.dp_size,
            args.ep_size,
            args.attn_cp_size,
            args.moe_dp_size,
        )
    ):
        parser.error("--loopspec requires all parallel sizes to equal 1")
    loopspec_attention_backends = {
        args.prefill_attention_backend or args.attention_backend,
        args.decode_attention_backend or args.attention_backend,
    }
    if args.loopspec and loopspec_attention_backends != {"triton"}:
        parser.error("--loopspec currently requires the Triton attention backend")
    if args.loopspec and args.disaggregation_mode != "null":
        parser.error("--loopspec does not support PD disaggregation")
    if args.loopspec and args.enable_pdmux:
        parser.error("--loopspec does not support PDMux")
    if args.loopspec and args.skip_tokenizer_init:
        parser.error("--loopspec requires SGLang's tokenizer manager")
    if (
        args.speculative_algorithm is not None
        or args.speculative_draft_model_path is not None
    ):
        parser.error("this server does not support SGLang speculative decoding")

    config_dict, _ = PreTrainedConfig.get_config_dict(
        args.model_path,
        cache_dir=args.download_dir,
        revision=args.revision,
    )
    model_type = config_dict.get("model_type")

    if model_type not in {"huginn_raven", "ouro"}:
        parser.error(
            f"this server serves recurrent models only, got {model_type}"
        )
    if model_type == "huginn_raven" and args.skip_tokenizer_init:
        parser.error("Raven requires SGLang's tokenizer manager")
    if args.pp_size != 1:
        parser.error("recurrent models currently require --pp-size=1")
    if args.model_impl not in ("auto", "sglang"):
        parser.error("recurrent models require SGLang's model implementation")

    if args.loopspec:
        architecture = "RecurrentServingModel"
        args.disable_overlap_schedule = True
        args.page_size = 1
        scheduler_target = run_loopspec_scheduler_process
    else:
        architecture = "LoopedRecurrentServingModel"
        if model_type == "huginn_raven":
            # Model-owned recurrent state is not represented by a radix-cache
            # key. CUDA-graph selection remains owned by SGLang.
            args.disable_radix_cache = True
            args.chunked_prefill_size = -1
        scheduler_target = run_recurrent_scheduler_process

    args.model_config_parser = install_sglang_config_compatibility(architecture)

    server_args = ServerArgs.from_cli_args(args)
    if args.loopspec:
        server_args.recurrent_step_size = args.step_size
        server_args.recurrent_second_step = args.second_step

    launch_server(
        server_args,
        run_scheduler_process_func=scheduler_target,
        run_detokenizer_process_func=run_recurrent_detokenizer_process,
    )
