"""Run experiments and report their results."""

import argparse
from pathlib import Path

import yaml


def _run_experiment(args: argparse.Namespace) -> int:
    from experiment_tools.config import load_experiment
    from experiment_tools.run import run_loaded_experiment

    args.experiment = load_experiment(args.config)
    return run_loaded_experiment(args.experiment)


def _report(args: argparse.Namespace) -> int:
    from experiment_tools.report import report_for_root

    rendered = report_for_root(args.results_root)
    if not rendered:
        raise ValueError(
            f"no completed lm-eval results found under {args.results_root}"
        )
    print(rendered)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="benchmark.py",
        description="Run recurrent-serving experiments and report results.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    run = commands.add_parser("run", help="run one experiment YAML")
    run.add_argument("config", type=Path)
    run.set_defaults(handler=_run_experiment, parser=run)

    report = commands.add_parser(
        "report",
        help="render completed lm-eval results as Markdown",
    )
    report.add_argument(
        "results_root",
        nargs="?",
        default=Path("results"),
        type=Path,
    )
    report.set_defaults(handler=_report, parser=report)
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args, extra_args = parser.parse_known_args(argv)
    if extra_args:
        parser.error("unrecognized arguments: " + " ".join(extra_args))

    try:
        return args.handler(args)
    except (
        OSError,
        KeyError,
        TypeError,
        yaml.YAMLError,
        ValueError,
    ) as error:
        args.parser.error(str(error))


if __name__ == "__main__":
    raise SystemExit(main())
