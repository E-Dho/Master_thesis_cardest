"""Write a timing plan for the shared launcher from trajectory configs.

The profile-controlled launcher (``evaluation/job_light_imdb_non_trajectory/
scripts/timing_launch.py``) times one accuracy run per plan line::

    CONFIG_PATH  SEED  RUN_DIRECTORY

For every config and every seed it declares, this picks the latest complete
accuracy run (the same rule as ``aggregate``), so the timing stage measures
exactly the runs the comparison reports. A config that does not declare the
profile in ``timing.profiles``, or a seed without a complete run, is an
error: a plan with gaps would leave the latency of a method incomplete.

    python -m evaluation.trajectory_pol_porto.traj_eval.timing_plan \\
        --profile cpu_1core --output plan.txt --config a.yaml --config b.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

from evaluation.eval_core.config import load_experiment_config
from evaluation.eval_core.report import discover_latest_complete_runs
from evaluation.eval_core.timing import declared_profiles


def plan_entries(configs: Iterable[Path], profile: str) -> List[Tuple[Path, int, Path]]:
    entries: List[Tuple[Path, int, Path]] = []
    for path in configs:
        config = load_experiment_config(path)
        if profile not in declared_profiles(config):
            raise ValueError(f"{path} does not declare timing profile {profile} (timing.profiles)")
        base = config.results_root / config.experiment_id / config.method_id / config.variant_id
        runs = discover_latest_complete_runs(base, config.seeds)
        entries.extend((config.source_path, seed, run) for seed, run in zip(config.seeds, runs))
    return entries


def write_plan(entries: List[Tuple[Path, int, Path]], output: Path, profile: str) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"# timing plan for {profile}: CONFIG SEED RUN_DIRECTORY"]
    lines += [f"{config} {seed} {run}" for config, seed, run in entries]
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--profile", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--config", required=True, action="append", type=Path)
    args = parser.parse_args(argv)
    entries = plan_entries(args.config, args.profile)
    print(write_plan(entries, args.output, args.profile))
    for config, seed, run in entries:
        print(f"  {config.name} seed {seed}: {run}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
