from __future__ import annotations

import argparse
from pathlib import Path

from .config import load_experiment_config
from .report import aggregate_runs, compare_aggregates, discover_latest_complete_runs
from .runner import initialize_staged_run, run_seed, run_stage


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="JOB-light baseline evaluation pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run", help="prepare, build, evaluate, and summarize")
    run.add_argument("--config", required=True, type=Path)
    run.add_argument("--seed", action="append", type=int)
    run.add_argument("--run-id")

    for name in ("prepare", "smoke", "build", "evaluate", "summarize"):
        stage = subparsers.add_parser(name, help=f"run only the {name} stage")
        stage.add_argument("--config", required=True, type=Path)
        stage.add_argument("--seed", required=True, type=int)
        stage.add_argument("--run-directory", type=Path)
        stage.add_argument("--run-id")

    aggregate = subparsers.add_parser("aggregate", help="aggregate complete seed runs")
    aggregate.add_argument("--config", required=True, type=Path)
    aggregate.add_argument("--run-directory", action="append", type=Path)
    aggregate.add_argument("--output", type=Path)
    compare = subparsers.add_parser("compare", help="compare method/variant aggregates")
    compare.add_argument("--aggregate-json", action="append", required=True, type=Path)
    compare.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "compare":
        compare_aggregates(args.aggregate_json, args.output)
        print(args.output)
        return 0
    config = load_experiment_config(args.config)
    if args.command == "run":
        seeds = tuple(args.seed or config.seeds)
        unknown = set(seeds) - set(config.seeds)
        if unknown:
            raise ValueError(f"seeds are not declared by config: {sorted(unknown)}")
        paths = [run_seed(config, seed, run_id=args.run_id) for seed in seeds]
        for path in paths:
            print(path)
        return 0

    if args.command in {"prepare", "smoke", "build", "evaluate", "summarize"}:
        if args.seed not in config.seeds:
            raise ValueError(f"seed {args.seed} is not declared by config")
        if args.command == "prepare" and args.run_directory is None:
            run_directory = initialize_staged_run(config, args.seed, args.run_id)
        elif args.run_directory is not None:
            run_directory = args.run_directory
        else:
            raise ValueError(f"{args.command} requires --run-directory")
        run_stage(config, args.seed, run_directory, args.command)
        print(run_directory)
        return 0

    base = (
        config.results_root
        / config.experiment_id
        / config.method_id
        / config.variant_id
    )
    paths = args.run_directory or discover_latest_complete_runs(base, config.seeds)
    output = args.output or base / "aggregate"
    aggregate_runs(paths, output)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
