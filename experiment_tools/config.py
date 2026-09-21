"""Load experiment YAML into validated execution jobs."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

NAMED_STRATEGIES = {"baseline"}
# LoopSpec configuration keys name the proposal depths directly: "K" or "K:D".
LOOPSPEC_CONFIGURATION = re.compile(r"([1-9]\d*)(?::([1-9]\d*))?")
GpuId = int | str


# Configuration model

@dataclass(frozen=True, slots=True)
class Task:
    alias: str
    lm_eval_task: str
    max_gen_toks: int
    max_length: int
    num_fewshot: int | None = None
    apply_chat_template: bool = False
    enable_thinking: bool = False
    eval_seed: int | None = None
    confirm_run_unsafe_code: bool = False


@dataclass(frozen=True, slots=True)
class DecodingPolicy:
    """Resolved decoding policy used by generation jobs."""

    do_sample: bool
    temperature: float
    top_p: float


@dataclass(frozen=True, slots=True)
class Sampling:
    alias: str
    temperature: float | None = None
    top_p: float | None = None

    @property
    def is_greedy(self) -> bool:
        return self.temperature is None

    @property
    def decoding_policy(self) -> DecodingPolicy:
        if self.is_greedy:
            return DecodingPolicy(
                do_sample=False,
                temperature=0.0,
                top_p=1.0,
            )
        assert self.temperature is not None and self.top_p is not None
        return DecodingPolicy(
            do_sample=True,
            temperature=self.temperature,
            top_p=self.top_p,
        )


@dataclass(frozen=True, slots=True)
class Job:
    model: str
    strategy: str
    task: Task
    sampling: Sampling
    loopspec_first: int | None = None
    loopspec_second: int | None = None

    @property
    def model_path(self) -> str:
        return f"models/{self.model}"

    @property
    def uses_loopspec(self) -> bool:
        return self.strategy == "loopspec"

    @property
    def result_name(self) -> str:
        configuration = self.strategy
        if self.uses_loopspec:
            configuration = f"k{self.loopspec_first}"
            if self.loopspec_second is not None:
                configuration += f"-d{self.loopspec_second}"
        return f"{configuration}_{self.sampling.alias}"


@dataclass(frozen=True, slots=True)
class Experiment:
    gpus: tuple[GpuId, ...]
    memory_fraction: float
    server_seed: int
    max_running_requests: int
    num_concurrent: int
    timeout: int
    results_root: Path
    jobs: tuple[Job, ...]

    def result_directory(self, job: Job) -> Path:
        return self.results_root / job.model / job.task.alias / job.result_name


# YAML loading and validation

def _validated_int(
    name: str,
    value: Any,
    *,
    minimum: int,
) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _validated_path_alias(name: str, value: Any) -> str:
    if (
        not isinstance(value, str)
        or value in {"", ".", ".."}
        or Path(value).name != value
    ):
        raise ValueError(f"{name} must be a single safe path component")
    return value


def _validated_task(alias: Any, values: Any) -> Task:
    alias = _validated_path_alias("task aliases", alias)
    if not isinstance(values, Mapping):
        raise ValueError(f"task {alias} must be a mapping")
    task = Task(alias=alias, **values)
    if not isinstance(task.lm_eval_task, str) or not task.lm_eval_task:
        raise ValueError(f"task {alias}: lm_eval_task must be a nonempty string")
    for field, minimum in (
        ("max_gen_toks", 1),
        ("max_length", 2),
        ("num_fewshot", 0),
    ):
        value = getattr(task, field)
        if value is not None:
            _validated_int(
                f"task {alias}: {field}",
                value,
                minimum=minimum,
            )
    if task.max_gen_toks >= task.max_length - 1:
        raise ValueError(
            f"task {alias}: max_gen_toks must be less than max_length - 1"
        )
    for field in (
        "apply_chat_template",
        "enable_thinking",
        "confirm_run_unsafe_code",
    ):
        if type(getattr(task, field)) is not bool:
            raise ValueError(f"task {alias}: {field} must be a boolean")
    if task.enable_thinking and not task.apply_chat_template:
        raise ValueError(
            f"task {alias}: enable_thinking requires apply_chat_template"
        )
    if task.eval_seed is not None:
        _validated_int(
            f"task {alias}: eval_seed",
            task.eval_seed,
            minimum=0,
        )
    return task


def read_sampling(alias: str, values: Mapping[str, Any]) -> Sampling:
    alias = _validated_path_alias("sampling aliases", alias)
    options = {key: value for key, value in values.items() if key != "tasks"}
    if alias == "greedy":
        if options:
            raise ValueError("greedy sampling only accepts tasks")
        return Sampling(alias="greedy")

    unexpected = set(options) - {"temperature", "top_p"}
    if unexpected:
        raise ValueError(
            f"sampling {alias} has unknown options: "
            f"{', '.join(sorted(unexpected))}"
        )

    sampling = Sampling(alias=alias, **options)
    if sampling.temperature is None or sampling.top_p is None:
        raise ValueError(f"sampling {alias} requires temperature and top_p")
    temperature = sampling.temperature
    if (
        isinstance(temperature, bool)
        or not isinstance(temperature, (int, float))
        or not math.isfinite(temperature)
        or temperature <= 0
    ):
        raise ValueError(
            f"sampling {alias} temperature must be finite and positive"
        )
    top_p = sampling.top_p
    if (
        isinstance(top_p, bool)
        or not isinstance(top_p, (int, float))
        or not math.isfinite(top_p)
        or not 0 < top_p <= 1
    ):
        raise ValueError(f"sampling {alias} top_p must be in (0, 1]")
    return sampling


def _read_jobs(
    models: Mapping[str, Any],
    tasks: Mapping[str, Task],
) -> tuple[Job, ...]:
    jobs = []
    for model, model_values in models.items():
        model = _validated_path_alias("model names", model)
        for raw_configuration, raw_config in model_values["configs"].items():
            configuration = str(raw_configuration)
            loopspec = LOOPSPEC_CONFIGURATION.fullmatch(configuration)
            if configuration not in NAMED_STRATEGIES and loopspec is None:
                raise ValueError(
                    f"invalid configuration: {configuration}"
                )
            if not isinstance(raw_config, Mapping):
                raise ValueError(f"{configuration} config must be a mapping")
            options = dict(raw_config)
            sampling_configs = options.pop("sampling")
            job_options = {}

            if loopspec is not None:
                strategy = "loopspec"
                first = int(loopspec.group(1))
                second = loopspec.group(2)
                job_options["loopspec_first"] = first
                if second is not None:
                    second = int(second)
                    if second <= first or second % first:
                        raise ValueError(
                            f"invalid configuration: {configuration}: the "
                            "second depth must be a multiple of the first "
                            "depth and greater than it"
                        )
                    job_options["loopspec_second"] = second
            else:
                strategy = "baseline"

            if options:
                raise ValueError(f"unknown {configuration} options")

            for sampling_alias, sampling_values in sampling_configs.items():
                sampling = read_sampling(sampling_alias, sampling_values)
                for task_alias in sampling_values["tasks"]:
                    if task_alias not in tasks:
                        raise ValueError(f"unknown task: {task_alias}")
                    task = tasks[task_alias]
                    jobs.append(
                        Job(
                            model=model,
                            strategy=strategy,
                            task=task,
                            sampling=sampling,
                            **job_options,
                        )
                    )
    return tuple(jobs)


def _validated_gpus(values: Any) -> tuple[GpuId, ...]:
    if not isinstance(values, list) or not values:
        raise ValueError("gpus must contain at least one device")
    gpus = tuple(values)
    if any(
        not (
            (type(gpu) is int and gpu >= 0)
            or (
                isinstance(gpu, str)
                and re.fullmatch(r"\d+:\d+", gpu) is not None
            )
        )
        for gpu in gpus
    ):
        raise ValueError(
            "gpus must contain non-negative integers or quoted GPU:MIG IDs"
        )
    if len(set(gpus)) != len(gpus):
        raise ValueError("gpus must not contain duplicate device IDs")
    return gpus


def load_experiment(path: Path) -> Experiment:
    """Read and validate one experiment YAML file."""
    data = yaml.safe_load(path.read_text())
    expected = {
        "gpus",
        "memory_fraction",
        "server_seed",
        "max_running_requests",
        "num_concurrent",
        "timeout",
        "tasks",
        "results_root",
        "models",
    }
    if set(data) - expected:
        raise ValueError("unknown experiment options")
    task_values = yaml.safe_load((path.parent / data["tasks"]).read_text())
    tasks = {
        alias: _validated_task(alias, values)
        for alias, values in task_values.items()
    }
    jobs = _read_jobs(data["models"], tasks)
    if not jobs or len(set(jobs)) != len(jobs):
        raise ValueError("experiment must contain unique jobs")
    max_running_requests = _validated_int(
        "max_running_requests",
        data["max_running_requests"],
        minimum=1,
    )
    num_concurrent = _validated_int(
        "num_concurrent",
        data["num_concurrent"],
        minimum=1,
    )
    timeout = _validated_int("timeout", data["timeout"], minimum=1)
    loopspec = any(job.uses_loopspec for job in jobs)
    if loopspec and (
        max_running_requests != 1
        or num_concurrent != 1
    ):
        raise ValueError(
            "LoopSpec experiments require max_running_requests and "
            "num_concurrent to be 1"
        )

    server_seed = _validated_int(
        "server_seed",
        data["server_seed"],
        minimum=0,
    )
    memory_fraction = data["memory_fraction"]
    if (
        isinstance(memory_fraction, bool)
        or not isinstance(memory_fraction, (int, float))
        or not math.isfinite(memory_fraction)
        or not 0 < memory_fraction <= 1
    ):
        raise ValueError("memory_fraction must be finite and in (0, 1]")

    results_root = Path(data["results_root"])
    if results_root.is_absolute() or ".." in results_root.parts:
        raise ValueError("results_root must stay inside the project root")
    if not (
        results_root.name == path.stem
        or results_root.name.startswith(f"{path.stem}-")
    ):
        results_root /= path.stem

    return Experiment(
        gpus=_validated_gpus(data["gpus"]),
        memory_fraction=float(memory_fraction),
        server_seed=server_seed,
        max_running_requests=max_running_requests,
        num_concurrent=num_concurrent,
        timeout=timeout,
        results_root=results_root,
        jobs=jobs,
    )
