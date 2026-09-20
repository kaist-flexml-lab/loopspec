"""Launch the configured demo server."""

from __future__ import annotations

import argparse
import os
import sys

from .config import DEFAULT_CONFIG, load_config, server_arguments


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    args = parser.parse_args()
    config = load_config(args.config)

    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu
    print(
        f"[demo] model={config.active_model} "
        f"config={config.active_configuration} gpu={config.gpu}",
        flush=True,
    )
    command = [
        sys.executable,
        "-m",
        "sglang_recurrent.server",
        *server_arguments(config),
    ]
    os.execv(sys.executable, command)


if __name__ == "__main__":
    main()
