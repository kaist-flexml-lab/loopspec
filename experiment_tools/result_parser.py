#!/usr/bin/env -S uv run --locked
"""Parse lm-eval results and serving metrics into report rows."""

from __future__ import annotations

import json
import logging
import math
import re
from pathlib import Path
from typing import Any

METRIC_PRIORITY = (
    "math_verify",
    "exact_match,flexible-extract",
    "exact_match,strict-match",
    "exact_match",
    "pass@1",
    "pass_at_1",
    "acc_norm",
    "acc",
    "f1",
)
METRICS_RE = re.compile(r"generation_metrics\s+(\{.*\})$")
logger = logging.getLogger(__name__)

MODEL_FULL_DEPTHS = (
    ("Ouro-", 4),
    ("Recurrent-", 32),
)


class ParseError(ValueError):
    """A result set cannot be represented by the requested tables."""


def result_layout(
    directory: Path,
) -> tuple[str, str, str, str] | None:
    """Parse an experiment result directory name.

    LoopSpec names are either "k{K}-d{D}" (current: D is the second
    proposal depth) or "k{K}-x{X}" (legacy: X is the multiplier, so the
    second proposal depth is K*X). Both resolve to proposal depths.
    """
    match = re.fullmatch(
        r"(baseline|k\d+(?:-[xd]\d+)?)"
        r"(?:_gate-(?:always|decrease|stochastic)"
        r"_proposal-(?:residual|masked-q2))?_(.+)",
        directory.name,
    )
    if match:
        config, sampling = match.groups()
        decoding = "greedy" if sampling == "greedy" else "sampling"
        if config == "baseline":
            return "—", "—", decoding, sampling
        loopspec = re.fullmatch(r"k(\d+)(?:-([xd])(\d+))?", config)
        assert loopspec is not None
        first, kind, value = loopspec.groups()
        if kind is None:
            second = "—"
        elif kind == "d":
            second = value
        else:
            second = str(int(first) * int(value))
        return first, second, decoding, sampling
    return None


def sampling_label(sampling: str) -> str:
    """Format a result-directory sampling name for the report."""
    if sampling == "greedy":
        return "greedy"
    match = re.fullmatch(r"t([^_]+)_p(.+)", sampling)
    if match is not None:
        temperature, top_p = match.groups()
        return f"sampling ({temperature}, {top_p})"
    return f"sampling ({sampling})"


def number(value: Any, digits: int = 2) -> str:
    if (
        value is None
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        return "—"
    return f"{value:.{digits}f}"


def percent(value: Any) -> str:
    if (
        value is None
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        return "—"
    return f"{100 * value:.2f}%"


class GenerationMetrics(list[dict[str, Any]]):
    @staticmethod
    def _valid(metric: Any) -> bool:
        if not isinstance(metric, dict):
            return False
        decode_tokens = metric.get("decode_tokens")
        decode_seconds = metric.get("decode_seconds")
        if not (
            type(decode_tokens) is int
            and decode_tokens >= 0
            and isinstance(decode_seconds, (int, float))
            and not isinstance(decode_seconds, bool)
            and math.isfinite(decode_seconds)
            and decode_seconds >= 0
        ):
            return False

        rejection_fields = (
            "first_accept",
            "first_reject",
            "second_skip",
            "second_accept",
            "second_reject",
        )
        rejection_values = [metric.get(name) for name in rejection_fields]
        if any(value is not None for value in rejection_values):
            if not all(
                type(value) is int and value >= 0
                for value in rejection_values
            ):
                return False
            if (
                decode_tokens
                != metric["first_accept"] + metric["first_reject"]
                or metric["second_skip"]
                + metric["second_accept"]
                + metric["second_reject"]
                != metric["first_reject"]
            ):
                return False

        return True

    @classmethod
    def from_log(
        cls,
        log: Path,
        *,
        expected_generations: int,
    ) -> GenerationMetrics:
        lines = log.read_text(errors="replace").splitlines()
        ready = next(
            (
                index
                for index, line in enumerate(lines)
                if "The server is fired up and ready to roll!" in line
            ),
            None,
        )
        first_metric_line = ready + 1 if ready is not None else 0
        metrics = cls()
        for line_number, line in enumerate(
            lines[first_metric_line:],
            first_metric_line + 1,
        ):
            if "generation_metrics" not in line:
                continue
            match = METRICS_RE.search(line)
            if not match:
                raise ParseError(
                    f"{log}:{line_number}: malformed generation metrics"
                )
            try:
                value = json.loads(match.group(1))
            except json.JSONDecodeError as error:
                raise ParseError(
                    f"{log}:{line_number}: malformed generation metrics"
                ) from error
            if not cls._valid(value):
                raise ParseError(
                    f"{log}:{line_number}: invalid generation metrics"
                )
            metrics.append(value)
        if len(metrics) != expected_generations:
            raise ParseError(
                f"{log}: expected {expected_generations} generations, "
                f"found {len(metrics)}"
            )
        return metrics

    def total(self, name: str) -> float:
        return sum(
            value
            for metric in self
            if (
                isinstance((value := metric.get(name)), (int, float))
                and not isinstance(value, bool)
            )
        )

    def cycle_metric(
        self,
        first_depth: int,
        second_depth: int,
        full_depth: int,
    ) -> int:
        """Estimate recurrent work from endpoint rejection counts."""
        token_count = int(self.total("decode_tokens"))
        first_rejected = int(self.total("first_reject"))
        final_rejected = int(
            self.total("second_skip") + self.total("second_reject")
        )
        return (
            token_count * first_depth
            + first_rejected * (second_depth - first_depth)
            + final_rejected * (full_depth - second_depth)
        )

    def cycle_accept_length(
        self,
        first_depth: int,
        second_depth: int,
        full_depth: int,
    ) -> float | None:
        cycles = self.cycle_metric(first_depth, second_depth, full_depth)
        if cycles <= 0:
            return None
        return full_depth * self.total("decode_tokens") / cycles

    def summary(self) -> dict[str, str]:
        decode_seconds = self.total("decode_seconds")
        tok_s = (
            self.total("decode_tokens") / decode_seconds
            if decode_seconds
            else None
        )
        first_accepted = self.total("first_accept")
        first_rejected = self.total("first_reject")
        verified = first_accepted + first_rejected
        second_skipped = self.total("second_skip")
        second_accepted = self.total("second_accept")
        second_rejected = self.total("second_reject")
        return {
            "tok_s": number(tok_s),
            "first_rejection": percent(
                first_rejected / verified if verified else None
            ),
            "second_skip": percent(
                second_skipped / first_rejected if first_rejected else None
            ),
            "second_acceptance": percent(
                second_accepted / first_rejected if first_rejected else None
            ),
            "second_rejection": percent(
                second_rejected / first_rejected if first_rejected else None
            ),
            "final_rejection": percent(
                (second_skipped + second_rejected) / verified
                if verified
                else None
            ),
            "accept_len": "—",
        }


def primary_metric(values: dict[str, Any]) -> tuple[str, Any] | None:
    """Select the most useful non-stderr metric from one task summary."""
    numeric = {
        key: value
        for key, value in values.items()
        if isinstance(value, (int, float))
        and not isinstance(value, bool)
        and key not in ("sample_len", "sample_count")
        and "_stderr" not in key
    }
    for preferred in METRIC_PRIORITY:
        for key, value in numeric.items():
            if key == preferred or key.split(",", 1)[0] == preferred:
                return key, value
    return None


def task_summaries(payload: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    """Return aggregate groups and independent tasks, but not group subtasks."""
    results = payload.get("results", {})
    groups = payload.get("groups", {})
    group_subtasks = payload.get("group_subtasks", {})
    if not isinstance(results, dict):
        results = {}
    if not isinstance(groups, dict):
        groups = {}
    if not isinstance(group_subtasks, dict):
        group_subtasks = {}

    summaries = [
        (task, values)
        for task, values in groups.items()
        if isinstance(values, dict)
    ]
    group_names = set(groups)
    subtasks = {
        task
        for children in group_subtasks.values()
        if isinstance(children, list)
        for task in children
    }
    summaries.extend(
        (task, values)
        for task, values in results.items()
        if (
            task not in subtasks
            and task not in group_names
            and isinstance(values, dict)
        )
    )
    return summaries


def aggregate_metric(
    summaries: list[tuple[str, dict[str, Any]]],
    result: Path,
) -> tuple[float, int]:
    """Combine one task alias's independent lm-eval summaries by sample count."""
    selected: list[tuple[str, float, int]] = []
    for _, values in summaries:
        metric = primary_metric(values)
        if metric is None:
            continue
        key, score = metric
        if not isinstance(score, (int, float)) or isinstance(score, bool):
            continue
        sample_len = values.get("sample_len")
        weight = (
            int(sample_len)
            if (
                isinstance(sample_len, (int, float))
                and not isinstance(sample_len, bool)
                and sample_len > 0
            )
            else 1
        )
        selected.append((key, float(score), weight))

    if not selected:
        raise ParseError(f"{result}: no supported metric")
    metric_keys = {key for key, _, _ in selected}
    if len(metric_keys) != 1:
        details = ", ".join(sorted(metric_keys))
        raise ParseError(
            f"{result}: task summaries use incompatible primary metrics: "
            f"{details}"
        )
    total_samples = sum(weight for _, _, weight in selected)
    score = sum(value * weight for _, value, weight in selected) / total_samples
    return score, total_samples


def matching_server_log(result: Path) -> Path:
    logs = sorted(result.parent.glob("server_*.log"))
    marker = f"pipelined_result={result.name}"
    matching_runs = [
        log.stem.removeprefix("lm_eval_")
        for log in result.parent.glob("lm_eval_*.log")
        if marker in log.read_text(errors="replace")
    ]
    if len(matching_runs) == 1:
        server_log = result.with_name(f"server_{matching_runs[0]}.log")
        if server_log.is_file():
            return server_log
        raise ParseError(f"{result}: missing {server_log.name}")
    if len(matching_runs) > 1:
        raise ParseError(f"{result}: multiple lm-eval logs claim this result")
    if len(logs) == 1:
        logger.warning(
            "%s: no lm-eval result hint; using the only server log %s",
            result,
            logs[0].name,
        )
        return logs[0]
    raise ParseError(
        f"{result}: expected one server log without an lm-eval result hint, "
        f"found {len(logs)}"
    )


def result_concurrency(payload: dict[str, Any]) -> int | None:
    config = payload.get("config", {})
    config = config if isinstance(config, dict) else {}
    model_args = config.get("model_args", {})
    model_args = model_args if isinstance(model_args, dict) else {}
    value = model_args.get("num_concurrent")
    if (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and int(value) == value
    ):
        return int(value)
    return None


def model_full_depth(model: str) -> int | None:
    for prefix, depth in MODEL_FULL_DEPTHS:
        if model.startswith(prefix):
            return depth
    return None


def cycle_depths(
    model: str,
    first: str,
    second: str,
) -> tuple[int, int, int] | None:
    full_depth = model_full_depth(model)
    if full_depth is None:
        return None
    if first == "—":
        return None
    try:
        first_depth = int(first)
    except (TypeError, ValueError):
        return None
    if second == "—":
        second_depth = full_depth
    else:
        try:
            second_depth = int(second)
        except (TypeError, ValueError):
            return None
    if not 0 < first_depth < second_depth <= full_depth:
        return None
    return first_depth, second_depth, full_depth


def parse_results(result: Path) -> list[dict[str, Any]]:
    try:
        payload = json.loads(result.read_text())
    except (OSError, json.JSONDecodeError):
        return []
    summaries = task_summaries(payload)
    if not summaries:
        return []
    layout = result_layout(result.parent)
    if layout is None:
        return []
    first, second, decoding, sampling = layout
    score, samples = aggregate_metric(summaries, result)
    metrics = GenerationMetrics.from_log(
        matching_server_log(result),
        expected_generations=samples,
    )
    speed = metrics.summary()
    depths = cycle_depths(
        result.parent.parent.parent.name,
        first,
        second,
    )
    if depths is not None:
        speed["accept_len"] = number(
            metrics.cycle_accept_length(*depths)
        )
    return [
        {
            "source": result,
            "model": result.parent.parent.parent.name,
            "benchmark": result.parent.parent.name,
            "first": first,
            "second": second,
            "decoding": decoding,
            "sampling": sampling,
            "sampling_label": sampling_label(sampling),
            "samples": samples,
            "score": percent(score),
            "num_concurrent": result_concurrency(payload),
            **speed,
        }
    ]
