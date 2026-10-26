"""Trajectory evaluation CLI: the shared stages plus ``query-index``.

``python -m evaluation.trajectory_pol_porto.traj_eval.cli run --config CONFIG``
works exactly like the JOB-light CLI (``prepare``, ``smoke``, ``build``,
``evaluate``, ``summarize``, ``time``, ``aggregate``, ``compare``).

``query-index --config CONFIG --output-directory DIR`` writes, per workload,
``<workload>_query_index.csv``: the pipeline's integer ``query_id`` next to the
v3 ``query_id``, ``slot_id`` and ``predicate_hash``, for joining
``predictions.csv`` back to the benchmark.
"""

from __future__ import annotations

from pathlib import Path

from evaluation.eval_core.cli import build_parser
from evaluation.eval_core.cli import main as core_main

DESCRIPTION = "Trajectory (POL / Porto, v3) cardinality-estimation evaluation pipeline"


def _parser():
    parser = build_parser(DESCRIPTION)
    subparsers = next(
        action for action in parser._actions if action.__class__.__name__ == "_SubParsersAction"
    )
    index = subparsers.add_parser("query-index", help="write query_id -> v3 identity per workload")
    index.add_argument("--config", required=True, type=Path)
    index.add_argument("--output-directory", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.command == "query-index":
        from evaluation.eval_core.config import load_experiment_config
        from evaluation.eval_core.workload_formats import load_queries

        from .workloads import FORMAT_NAME, write_query_index

        config = load_experiment_config(args.config)
        args.output_directory.mkdir(parents=True, exist_ok=True)
        for workload in config.workloads:
            if workload.format != FORMAT_NAME:
                continue
            path = args.output_directory / f"{workload.workload_id}_query_index.csv"
            write_query_index(load_queries(workload), path)
            print(path)
        return 0
    return core_main(argv, parser=parser)


if __name__ == "__main__":
    raise SystemExit(main())
