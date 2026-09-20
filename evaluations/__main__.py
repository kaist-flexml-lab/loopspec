"""Run lm-eval with the compatibility and token-trace patches used here."""

from __future__ import annotations

import argparse
import contextvars
import json
import sys
import threading
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


_current_payload: contextvars.ContextVar[dict[str, Any]] = (
    contextvars.ContextVar("pipelined_lm_eval_payload")
)
_current_contexts: contextvars.ContextVar[list[Any] | None] = (
    contextvars.ContextVar("pipelined_lm_eval_contexts", default=None)
)
_write_lock = threading.Lock()


def install_config_independent_tokenizer_loading() -> None:
    """Load a tokenizer without importing the target's custom model config."""
    from transformers import AutoTokenizer, PreTrainedConfig

    current_loader = AutoTokenizer.from_pretrained

    def from_pretrained(*args, **kwargs):
        kwargs.setdefault("config", PreTrainedConfig())
        return current_loader(*args, **kwargs)

    AutoTokenizer.from_pretrained = staticmethod(from_pretrained)


def install_enable_thinking_support(enable_thinking: bool) -> None:
    """Pass enable_thinking from lm-eval to the tokenizer chat template."""
    from lm_eval.models.api_models import TemplateAPI

    def apply_chat_template(self, chat_history, add_generation_prompt=True):
        return self.tokenizer.apply_chat_template(
            chat_history,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            continue_final_message=not add_generation_prompt,
            enable_thinking=enable_thinking,
        )

    TemplateAPI.apply_chat_template = apply_chat_template


def _build_generation_record(
    context: Any,
    request: Any,
    response: Mapping[str, Any],
) -> dict[str, Any]:
    meta_info = response["meta_info"]
    record = {
        "schema_version": 2,
        "context": context,
        "input_kind": "input_ids",
        "input": request,
        "text": response["text"],
        "output_ids": response["output_ids"],
        "prompt_tokens": meta_info["prompt_tokens"],
        "completion_tokens": meta_info["completion_tokens"],
    }
    return record


def _append_records(
    path: Path,
    records: Sequence[Mapping[str, Any]],
) -> None:
    with _write_lock:
        encoded = "".join(
            json.dumps(
                record,
                ensure_ascii=False,
                separators=(",", ":"),
            )
            + "\n"
            for record in records
        )
        with path.open("a") as output:
            output.write(encoded)
            output.flush()


def install_token_trace(path: Path) -> None:
    """Patch lm-eval's SGLang adapter to preserve exact request/response IDs."""
    from lm_eval.models.sglang_generate_API import SGLANGGENERATEAPI

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("")

    original_create_payload = SGLANGGENERATEAPI._create_payload
    original_parse_generations = SGLANGGENERATEAPI.parse_generations
    original_amodel_call = SGLANGGENERATEAPI.amodel_call

    def create_payload(self, messages, *args, **kwargs):
        payload = original_create_payload(self, messages, *args, **kwargs)
        _current_payload.set(payload)
        return payload

    async def amodel_call(self, *args, cache_keys=None, **kwargs):
        contexts = None
        if cache_keys:
            contexts = [
                key[0] if isinstance(key, (list, tuple)) and key else key
                for key in cache_keys
            ]
        token = _current_contexts.set(contexts)
        try:
            return await original_amodel_call(self, *args, cache_keys=cache_keys, **kwargs)
        finally:
            _current_contexts.reset(token)

    def parse_generations(outputs, **kwargs):
        requests = _current_payload.get()["input_ids"]
        contexts = kwargs.get("contexts") or _current_contexts.get()
        records = [
            _build_generation_record(context, item, response)
            for context, item, response in zip(contexts, requests, outputs, strict=True)
        ]
        _append_records(path, records)
        return original_parse_generations(outputs, **kwargs)

    SGLANGGENERATEAPI._create_payload = create_payload
    SGLANGGENERATEAPI.amodel_call = amodel_call
    SGLANGGENERATEAPI.parse_generations = staticmethod(parse_generations)


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--trace-path", type=Path)
    parser.add_argument("--enable-thinking", action="store_true")
    arguments, lm_eval_arguments = parser.parse_known_args()

    sys.argv[1:] = lm_eval_arguments
    output_index = lm_eval_arguments.index("--output_path") + 1
    output_path = Path(lm_eval_arguments[output_index])
    previous_results = set(output_path.parent.glob(f"{output_path.stem}_*.json"))

    install_config_independent_tokenizer_loading()
    install_enable_thinking_support(arguments.enable_thinking)
    if arguments.trace_path is not None:
        install_token_trace(arguments.trace_path)

    from lm_eval.__main__ import cli_evaluate

    cli_evaluate()

    new_results = set(output_path.parent.glob(f"{output_path.stem}_*.json")) - previous_results
    if len(new_results) != 1:
        print(
            f"warning: lm-eval created {len(new_results)} result files; "
            "result hint unavailable",
            file=sys.stderr,
            flush=True,
        )
    else:
        result, = new_results
        print(f"pipelined_result={result.name}", flush=True)


if __name__ == "__main__":
    main()
