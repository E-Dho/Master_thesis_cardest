"""JOB-light evaluation CLI (the shared ``evaluation.eval_core.cli``).

Usage is unchanged: ``python -m evaluation.job_light_imdb_non_trajectory.joblight_eval.cli ...``.
"""

from __future__ import annotations

from evaluation.eval_core.cli import build_parser as _build_parser
from evaluation.eval_core.cli import main as _main

DESCRIPTION = "JOB-light baseline evaluation pipeline"


def build_parser():
    return _build_parser(DESCRIPTION)


def main(argv: list[str] | None = None) -> int:
    return _main(argv, description=DESCRIPTION)


if __name__ == "__main__":
    raise SystemExit(main())
