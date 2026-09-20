"""Interactive OpenAI-compatible streaming client for the demo server."""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Iterable

from .config import DEFAULT_CONFIG, DemoConfig, load_config


DEFAULT_URL = "http://server:30000"


@dataclass(frozen=True, slots=True)
class StreamResult:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    decode_tps: float | None

    @property
    def decode_time(self) -> float | None:
        if self.decode_tps is None:
            return None
        return max(self.completion_tokens - 1, 0) / self.decode_tps


def consume_stream(
    lines: Iterable[bytes],
    *,
    output: IO[str],
) -> StreamResult:
    """Print native SGLang SSE deltas and return final server metrics."""

    previous_text = ""
    final_meta: dict | None = None
    for raw_line in lines:
        line = raw_line.decode("utf-8").strip()
        if not line.startswith("data:"):
            continue
        data = line.removeprefix("data:").strip()
        if data == "[DONE]":
            break
        event = json.loads(data)
        if "error" in event:
            raise RuntimeError(str(event["error"]))
        meta = event.get("meta_info")
        if isinstance(meta, dict):
            final_meta = meta
        text = event.get("text", "")
        if not isinstance(text, str):
            raise RuntimeError("stream returned a non-string text field")
        delta = (
            text[len(previous_text) :]
            if text.startswith(previous_text)
            else text
        )
        if delta:
            output.write(delta)
            output.flush()
        previous_text = text

    if final_meta is None:
        raise RuntimeError("stream ended without meta_info")
    prompt_tokens = final_meta.get("prompt_tokens")
    completion_tokens = final_meta.get("completion_tokens")
    decode_tps = final_meta.get("decode_throughput")
    if type(prompt_tokens) is not int or type(completion_tokens) is not int:
        raise RuntimeError("stream ended without token counts")
    if decode_tps is not None:
        if (
            isinstance(decode_tps, bool)
            or not isinstance(decode_tps, (int, float))
            or decode_tps <= 0
        ):
            raise RuntimeError("stream returned an invalid decode_throughput")
    return StreamResult(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
        decode_tps=float(decode_tps) if decode_tps is not None else None,
    )


def _chat_tokenize_payload(config: DemoConfig, prompt: str) -> dict:
    return {
        "model": config.model.path,
        "messages": [{"role": "user", "content": prompt}],
        "chat_template_kwargs": {
            "enable_thinking": config.model.enable_thinking
        },
    }


def _generation_payload(
    config: DemoConfig,
    prompt: str | list[int],
) -> dict:
    generation = config.generation
    payload = {
        "stream": True,
        "sampling_params": {
            "max_new_tokens": generation.max_new_tokens,
            "temperature": generation.temperature,
            "top_p": generation.top_p,
            "sampling_seed": generation.seed,
            "ignore_eos": generation.ignore_eos,
            "stream_interval": 1,
        },
    }
    payload["text" if isinstance(prompt, str) else "input_ids"] = prompt
    return payload


def _post_json(url: str, payload: dict) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=3600) as response:
            result = json.load(response)
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"server returned HTTP {error.code}: {detail}") from error
    if not isinstance(result, dict):
        raise RuntimeError("server returned an invalid JSON response")
    return result


def _chat_tokens(config: DemoConfig, prompt: str, base_url: str) -> list[int]:
    result = _post_json(
        base_url.rstrip("/") + "/v1/tokenize",
        _chat_tokenize_payload(config, prompt),
    )
    tokens = result.get("tokens")
    if not isinstance(tokens, list) or any(
        type(token) is not int for token in tokens
    ):
        raise RuntimeError("chat template tokenization returned invalid tokens")
    return tokens


def stream_prompt(
    config: DemoConfig,
    prompt: str,
    *,
    base_url: str = DEFAULT_URL,
    output: IO[str] = sys.stdout,
) -> StreamResult:
    prepared_prompt = (
        _chat_tokens(config, prompt, base_url) if config.model.chat else prompt
    )
    payload = _generation_payload(config, prepared_prompt)
    request = urllib.request.Request(
        base_url.rstrip("/") + "/generate",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=3600) as response:
            return consume_stream(
                response,
                output=output,
            )
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"server returned HTTP {error.code}: {detail}") from error


def print_metrics(result: StreamResult, output: IO[str] = sys.stdout) -> None:
    tps = "N/A" if result.decode_tps is None else f"{result.decode_tps:.2f}"
    decode_time = (
        "N/A" if result.decode_time is None else f"{result.decode_time:.2f} s"
    )
    print(f"Prompt tokens:     {result.prompt_tokens}", file=output)
    print(f"Completion tokens: {result.completion_tokens}", file=output)
    print(f"Total tokens:      {result.total_tokens}", file=output)
    print(f"Decode Time:       {decode_time}", file=output)
    print(f"Decode TPS:        {tps}", file=output)


def clear_screen(output: IO[str] = sys.stdout) -> None:
    output.write("\033[2J\033[H")
    output.flush()


def _run_prompt(config: DemoConfig, prompt: str, base_url: str) -> None:
    result = stream_prompt(config, prompt, base_url=base_url)
    print("\n")
    print_metrics(result)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--prompt")
    args = parser.parse_args()
    config = load_config(Path(args.config))

    print(
        f"Model: {config.active_model} | Config: {config.active_configuration}"
    )
    if args.prompt is not None:
        _run_prompt(config, args.prompt, args.url)
        return 0

    print("Enter a prompt, /clear to clear the screen, or /quit to exit.")
    while True:
        try:
            prompt = input("\nPrompt> ")
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if prompt.strip() == "/quit":
            return 0
        if prompt.strip() == "/clear":
            clear_screen()
            continue
        if not prompt.strip():
            continue
        try:
            _run_prompt(config, prompt, args.url)
        except (OSError, RuntimeError, ValueError) as error:
            print(f"\nError: {error}", file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
