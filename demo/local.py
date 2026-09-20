"""Run the streaming server and terminal client as one local command."""

from __future__ import annotations

import argparse
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from .config import DEFAULT_CONFIG, DemoConfig, load_config


SERVER_URL = "http://127.0.0.1:30000"
STARTUP_TIMEOUT = 600
REPOSITORY = Path(__file__).resolve().parents[1]


def _is_ready() -> bool:
    try:
        with urllib.request.urlopen(SERVER_URL + "/model_info", timeout=1):
            return True
    except (OSError, urllib.error.URLError):
        return False


def _port_is_open() -> bool:
    try:
        with socket.create_connection(("127.0.0.1", 30000), timeout=0.25):
            return True
    except OSError:
        return False


def _tail(path: Path, lines: int = 40) -> str:
    try:
        return "".join(path.read_text(errors="replace").splitlines(True)[-lines:])
    except OSError:
        return ""


def _wait_until_ready(process: subprocess.Popen, log_path: Path) -> None:
    deadline = time.monotonic() + STARTUP_TIMEOUT
    while time.monotonic() < deadline:
        if process.poll() is not None:
            detail = _tail(log_path)
            raise RuntimeError(
                f"server exited with code {process.returncode}\n{detail}"
            )
        if _is_ready():
            return
        time.sleep(0.5)
    raise RuntimeError(
        f"server did not become ready within {STARTUP_TIMEOUT}s\n"
        f"{_tail(log_path)}"
    )


def _stop(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()


def _local_path(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else REPOSITORY / candidate


def _validate_checkpoints(config: DemoConfig) -> None:
    model_path = _local_path(config.model.path)
    if config.model.path.startswith("models/") and not model_path.is_dir():
        raise RuntimeError(f"model directory not found: {model_path}")


def _client_command(config_path: Path, prompt: str | None) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "demo.client",
        "--config",
        str(config_path),
        "--url",
        SERVER_URL,
    ]
    if prompt is not None:
        command += ["--prompt", prompt]
    return command


def run(config_path: Path, prompt: str | None) -> int:
    config = load_config(config_path)
    _validate_checkpoints(config)
    if _is_ready() or _port_is_open():
        raise RuntimeError("port 30000 is already in use")

    log_path = REPOSITORY / ".tmp" / "demo-server.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(
        f"Starting {config.active_model} ({config.active_configuration}) "
        f"on GPU {config.gpu}...",
        flush=True,
    )
    with log_path.open("w") as log:
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "demo.server",
                "--config",
                str(config_path),
            ],
            cwd=REPOSITORY,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            _wait_until_ready(process, log_path)
            return subprocess.run(
                _client_command(config_path, prompt),
                cwd=REPOSITORY,
                check=False,
            ).returncode
        finally:
            _stop(process)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--prompt")
    args = parser.parse_args()
    try:
        return run(Path(args.config).resolve(), args.prompt)
    except KeyboardInterrupt:
        print()
        return 130
    except (OSError, RuntimeError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
