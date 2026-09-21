#!/usr/bin/env -S uv run --locked
"""Run the jobs in an experiment YAML file.

Each worker takes the next available job, starts a dedicated SGLang server,
runs lm-eval, saves the logs and artifacts, and stops that server.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from queue import Empty, Queue

import yaml

from experiment_tools.config import (
    Experiment,
    GpuId,
    Job,
)

# Service arguments and Compose configuration


def server_arguments(experiment: Experiment, job: Job) -> list[str]:
    arguments = [
        "--model",
        job.model_path,
        "--mem-fraction-static",
        str(experiment.memory_fraction),
        "--random-seed",
        str(experiment.server_seed),
        "--max-running-requests",
        str(experiment.max_running_requests),
    ]
    if not job.uses_loopspec:
        return arguments

    arguments += ["--loopspec", "--first", str(job.loopspec_first)]
    if job.loopspec_second is not None:
        arguments += ["--second", str(job.loopspec_second)]
    return arguments


def client_arguments(
    experiment: Experiment,
    job: Job,
    generation_tokens: Path,
) -> list[str]:
    model_arguments = [
        f"model={job.model_path}",
        "base_url=http://sglang:30000/generate",
        f"num_concurrent={experiment.num_concurrent}",
        "max_retries=1",
        f"timeout={experiment.timeout}",
        f"max_length={job.task.max_length}",
    ]
    policy = job.sampling.decoding_policy
    generation_arguments = [f"max_gen_toks={job.task.max_gen_toks}"]
    if not policy.do_sample:
        generation_arguments += ["do_sample=false", "temperature=0"]
    else:
        generation_arguments += [
            "do_sample=true",
            f"temperature={policy.temperature}",
            f"top_p={policy.top_p}",
        ]

    arguments = [
        "--trace-path",
        str(Path("/app") / generation_tokens),
    ]
    if job.task.enable_thinking:
        arguments.append("--enable-thinking")
    arguments += [
        "--model",
        "sglang-generate",
        "--include_path",
        "/app/evaluations/tasks",
        "--model_args",
        *model_arguments,
        "--gen_kwargs",
        *generation_arguments,
        "--tasks",
        job.task.lm_eval_task,
        "--batch_size",
        "1",
        "--output_path",
        str(experiment.result_directory(job) / "results.json"),
        "--log_samples",
    ]
    if job.task.num_fewshot is not None:
        arguments += ["--num_fewshot", str(job.task.num_fewshot)]
    if job.task.apply_chat_template:
        arguments.append("--apply_chat_template")
    if job.task.eval_seed is not None:
        arguments += ["--seed", str(job.task.eval_seed)]
    if job.task.confirm_run_unsafe_code:
        arguments.append("--confirm_run_unsafe_code")
    return arguments


def write_compose_override(
    experiment: Experiment,
    job: Job,
    gpu: GpuId,
    generation_tokens: Path,
) -> Path:
    """Write the Compose fragment for one complete job."""
    sglang = {
        "command": server_arguments(experiment, job),
        "gpus": [
            {
                "driver": "nvidia",
                "device_ids": [str(gpu)],
                "capabilities": ["gpu"],
            }
        ],
    }
    lm_eval: dict[str, object] = {
        "command": client_arguments(experiment, job, generation_tokens),
    }
    if job.task.confirm_run_unsafe_code:
        lm_eval["environment"] = {"HF_ALLOW_CODE_EVAL": "1"}
    override = {
        "services": {
            "sglang": sglang,
            "lm-eval": lm_eval,
        }
    }
    file = tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".yaml",
        prefix="pipelined-",
        delete=False,
    )
    with file:
        yaml.safe_dump(override, file, sort_keys=False)
    return Path(file.name)


# One lm-eval invocation

@dataclass(frozen=True, slots=True)
class RunArtifacts:
    """Artifact namespace owned by one lm-eval invocation."""

    output: Path
    timestamp: str

    @property
    def lm_eval_log(self) -> Path:
        return self.output / f"lm_eval_{self.timestamp}.log"

    @property
    def generation_tokens(self) -> Path:
        return self.output / f"generation_tokens_{self.timestamp}.jsonl"

    @property
    def server_log(self) -> Path:
        return self.output / f"server_{self.timestamp}.log"

    @classmethod
    def create(
        cls,
        experiment: Experiment,
        job: Job,
    ) -> RunArtifacts:
        output = experiment.result_directory(job)
        output.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S.%f")
        return cls(
            output=output,
            timestamp=timestamp,
        )


# Job and GPU scheduling

def run_job(
    experiment: Experiment,
    job_index: int,
    job: Job,
    gpu: GpuId,
) -> bool:
    """Run one job as a self-contained Compose project."""
    files = RunArtifacts.create(experiment, job)
    project = f"pipelined-{os.getpid()}-{job_index}"
    override = write_compose_override(
        experiment,
        job,
        gpu,
        files.generation_tokens,
    )
    compose = [
        "docker",
        "compose",
        "--project-name",
        project,
        "--file",
        "compose.yaml",
        "--file",
        str(override),
    ]
    label = (
        f"GPU={gpu} model={job.model} configuration={job.result_name} "
        f"task={job.task.alias} sampling={job.sampling.alias}"
    )
    print(f"[experiment] starting {label}", flush=True)

    returncode = 1
    execution_error = None
    cleanup_succeeded = False
    log_followers = []
    with ExitStack() as resources:
        resources.callback(override.unlink, missing_ok=True)
        try:
            started = subprocess.run(
                [*compose, "up", "--detach"],
                check=False,
                capture_output=True,
                text=True,
            )
            if started.returncode:
                detail = (started.stdout + started.stderr).strip()
                suffix = f": {detail}" if detail else ""
                execution_error = (
                    "docker compose up exited with "
                    f"{started.returncode}{suffix}"
                )
                print(f"[experiment] {execution_error}", flush=True)
            else:
                for service, log_path in (
                    ("lm-eval", files.lm_eval_log),
                    ("sglang", files.server_log),
                ):
                    log = resources.enter_context(log_path.open("w"))
                    follower = subprocess.Popen(
                        [
                            *compose,
                            "logs",
                            "--follow",
                            "--no-color",
                            "--no-log-prefix",
                            "--timestamps",
                            service,
                        ],
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        text=True,
                    )
                    log_followers.append(follower)

                waited = subprocess.run(
                    [*compose, "wait", "lm-eval"],
                    check=False,
                    capture_output=True,
                    text=True,
                )
                returncode = waited.returncode
        except (OSError, subprocess.SubprocessError) as error:
            execution_error = f"could not run compose job: {error}"
            print(f"[experiment] {execution_error}", flush=True)
        finally:
            print(f"[experiment] stopping {label}", flush=True)
            try:
                result = subprocess.run(
                    [*compose, "down", "--remove-orphans"],
                    check=False,
                    capture_output=True,
                    text=True,
                )
                cleanup_succeeded = result.returncode == 0
                if not cleanup_succeeded:
                    detail = (result.stdout + result.stderr).strip()
                    suffix = f": {detail}" if detail else ""
                    print(
                        "[experiment] docker compose down exited with "
                        f"{result.returncode}{suffix}",
                        flush=True,
                    )
            except (OSError, subprocess.SubprocessError) as error:
                print(
                    f"[experiment] could not clean up compose job: {error}",
                    flush=True,
                )

        for follower in log_followers:
            if not cleanup_succeeded:
                follower.terminate()
            follower.wait()
    completed = returncode == 0 and execution_error is None
    if completed:
        missing = [
            kind
            for kind, path in (
                ("generation_tokens", files.generation_tokens),
                ("server_log", files.server_log),
            )
            if not path.is_file()
        ]
        if missing:
            completed = False
            execution_error = "missing run artifacts: " + ", ".join(missing)
            print(f"[experiment] {execution_error}", flush=True)

    succeeded = completed and cleanup_succeeded
    state = "finished" if succeeded else "failed"
    print(f"[experiment] {state} {label}", flush=True)
    return succeeded


def run_gpu_jobs(
    experiment: Experiment,
    gpu: GpuId,
    jobs: Queue[tuple[int, Job]],
) -> bool:
    """Take the next job whenever this GPU becomes available."""
    while True:
        try:
            job_index, job = jobs.get_nowait()
        except Empty:
            return True
        if not run_job(experiment, job_index, job, gpu):
            return False


def run_loaded_experiment(
    experiment: Experiment,
) -> int:
    """Build prerequisites and execute one already-validated experiment."""

    print(
        f"[experiment] {len(experiment.jobs)} jobs "
        f"on GPUs {list(experiment.gpus)}"
    )
    build = subprocess.run(["docker", "compose", "build"], check=False)
    if build.returncode:
        return build.returncode

    pending: Queue[tuple[int, Job]] = Queue()
    for job in enumerate(experiment.jobs):
        pending.put(job)

    with ThreadPoolExecutor(max_workers=len(experiment.gpus)) as pool:
        futures = [
            pool.submit(run_gpu_jobs, experiment, gpu, pending)
            for gpu in experiment.gpus
        ]
    return 0 if all(future.result() for future in futures) else 1
