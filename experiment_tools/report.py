"""Render parsed experiment results as Markdown tables."""

from __future__ import annotations

from pathlib import Path
from typing import Any, NamedTuple

import yaml

from experiment_tools.result_parser import ParseError, parse_results


BENCHMARK_ORDER = tuple(
    yaml.safe_load(Path("experiments/tasks.yaml").read_text())
)


def speculative_sort_key(label: str) -> tuple[int, int, int, int]:
    if label == "baseline":
        return (0, 0, 0, 0)
    first, separator, second = label.partition(", ")
    return (
        1,
        -int(first),
        bool(separator),
        -int(second) if separator else 0,
    )


class AccuracyKey(NamedTuple):
    model: str
    benchmark: str
    sampling: str

    @property
    def column(self) -> AccuracyColumn:
        return AccuracyColumn(self.benchmark, self.sampling)

    def sort_key(self) -> tuple[int, str, str, bool, str]:
        try:
            benchmark_rank = BENCHMARK_ORDER.index(self.benchmark)
        except ValueError:
            benchmark_rank = len(BENCHMARK_ORDER)
        return (
            benchmark_rank,
            self.benchmark.casefold(),
            self.model.casefold(),
            self.sampling != "greedy",
            self.sampling,
        )


class AccuracyColumn(NamedTuple):
    benchmark: str
    sampling: str

    def sort_key(self) -> tuple[int, str, str, bool, str]:
        return AccuracyKey("", self.benchmark, self.sampling).sort_key()


class MainKey(NamedTuple):
    model: str
    benchmark: str
    sampling: str
    speculative: str

    def sort_key(
        self,
    ) -> tuple[
        tuple[int, str, str, bool, str],
        tuple[int, int, int, int],
    ]:
        accuracy = AccuracyKey(
            self.model,
            self.benchmark,
            self.sampling,
        )
        return (
            accuracy.sort_key(),
            speculative_sort_key(self.speculative),
        )


def escape(value: Any) -> str:
    return str(value).replace("|", "\\|")


def split_tables(
    rows: list[dict[str, Any]],
) -> tuple[
    dict[AccuracyKey, dict[str, Any]],
    dict[MainKey, dict[str, Any]],
]:
    """Classify runs and reject every ambiguous output cell."""
    accuracy: dict[AccuracyKey, dict[str, Any]] = {}
    main: dict[MainKey, dict[str, Any]] = {}
    for row in rows:
        concurrency = row.get("num_concurrent")
        if not isinstance(concurrency, int) or concurrency <= 0:
            raise ParseError(
                f"{row['source']}: config.model_args.num_concurrent must be "
                "a positive integer"
            )
        first = row["first"]
        if first == "—":
            label = "baseline"
        elif row["second"] == "—":
            label = first
        else:
            label = f"{first}, {row['second']}"

        if concurrency > 1:
            if label != "baseline":
                raise ParseError(
                    f"{row['source']}: Accuracy Eval cannot represent a "
                    "speculative configuration"
                )
            key = AccuracyKey(
                row["model"],
                row["benchmark"],
                row["sampling_label"],
            )
            destination = accuracy
        else:
            row = {**row, "speculative_label": label}
            key = MainKey(
                row["model"],
                row["benchmark"],
                row["sampling_label"],
                label,
            )
            destination = main

        if key in destination:
            previous = destination[key]
            coordinates = ", ".join(key)
            raise ParseError(
                f"report cell collision ({coordinates}):\n"
                f"  {previous['source']}\n"
                f"  {row['source']}"
            )
        destination[key] = row
    return accuracy, main


def accuracy_markdown(
    cells: dict[AccuracyKey, dict[str, Any]],
) -> list[str]:
    if not cells:
        return []
    columns = sorted(
        {key.column for key in cells},
        key=AccuracyColumn.sort_key,
    )
    models = sorted({key.model for key in cells}, key=str.casefold)

    benchmark_headers: list[str] = []
    previous = None
    for column in columns:
        benchmark = column.benchmark
        benchmark_headers.append(benchmark if benchmark != previous else "")
        previous = benchmark

    lines = [
        "## Accuracy Eval",
        "",
        "| Benchmark | "
        + " | ".join(escape(value) for value in benchmark_headers)
        + " |",
        "| " + " | ".join("---" for _ in range(len(columns) + 1)) + " |",
        "| Sampling | "
        + " | ".join(escape(column.sampling) for column in columns)
        + " |",
    ]
    for model in models:
        values = [
            cells.get(
                AccuracyKey(model, column.benchmark, column.sampling),
                {},
            ).get("score", "")
            for column in columns
        ]
        lines.append(
            "| " + " | ".join(escape(value) for value in (model, *values)) + " |"
        )
    return lines


def main_markdown(
    cells: dict[MainKey, dict[str, Any]],
) -> list[str]:
    if not cells:
        return []
    task_groups: dict[str, list[dict[str, Any]]] = {}
    for key in sorted(cells, key=MainKey.sort_key):
        task_groups.setdefault(key.benchmark, []).append(cells[key])

    lines = ["## Main Experiment"]
    for benchmark, task_rows in task_groups.items():
        baselines = {
            (row["model"], row["sampling_label"]): row
            for row in task_rows
            if row["speculative_label"] == "baseline"
        }

        lines.extend(
            [
                "",
                f"### {escape(benchmark)}",
                "",
                "| Model | Sampling | Speculative | tok/s | speedup | "
                "accept len |",
                "| --- | --- | --- | --- | --- | --- |",
            ]
        )
        previous_model = None
        previous_sampling = None
        for row in task_rows:
            model = row["model"]
            sampling = row["sampling_label"]
            shown_model = model if model != previous_model else ""
            shown_sampling = (
                sampling
                if model != previous_model or sampling != previous_sampling
                else ""
            )
            try:
                tok_s = float(row["tok_s"])
            except (TypeError, ValueError):
                tok_s = None
            baseline = baselines.get((model, sampling))
            try:
                baseline_tok_s = (
                    float(baseline["tok_s"]) if baseline is not None else None
                )
            except (TypeError, ValueError):
                baseline_tok_s = None
            speedup = (
                f"{tok_s / baseline_tok_s:.2f}x"
                if tok_s is not None and baseline_tok_s
                else ""
            )
            tok_s_text = f"{tok_s:.2f}" if tok_s is not None else ""
            values = (
                shown_model,
                shown_sampling,
                row["speculative_label"],
                tok_s_text,
                speedup,
                row.get("accept_len", ""),
            )
            lines.append(
                "| " + " | ".join(escape(value) for value in values) + " |"
            )
            previous_model = model
            previous_sampling = sampling
    return lines


def markdown(rows: list[dict[str, Any]]) -> str:
    accuracy, main = split_tables(rows)
    sections = [
        section
        for section in (accuracy_markdown(accuracy), main_markdown(main))
        if section
    ]
    return "\n\n".join("\n".join(section) for section in sections)


def report_for_root(results_root: Path) -> str:
    """Return the Markdown report for every completed result below a root."""
    rows: list[dict[str, Any]] = []
    for result in sorted(results_root.rglob("results_*.json")):
        rows.extend(parse_results(result))
    return markdown(rows) if rows else ""
