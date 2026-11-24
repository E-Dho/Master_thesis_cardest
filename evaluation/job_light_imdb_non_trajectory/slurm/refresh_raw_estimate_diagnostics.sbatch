#!/bin/bash
# Aggregate baseline raw-estimate diagnostics and refresh final JOB-light tables.
# Submit after the existing rule-table refresh and the nine baseline reruns.
#SBATCH --job-name=refresh_raw_estimate_diagnostics
#SBATCH --partition=base
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/work_beegfs/sunip956/master_thesis_trajectories/slurm_logs/%x_%j.out
#SBATCH --error=/work_beegfs/sunip956/master_thesis_trajectories/slurm_logs/%x_%j.err

set -euo pipefail

ROOT=/work_beegfs/sunip956/master_thesis_trajectories
REPO=$ROOT/Master_thesis_cardest_joblight_evaluation
EVAL=evaluation/job_light_imdb_non_trajectory
RESULTS=$REPO/$EVAL/results/joblight_phase1
PYTHON=$ROOT/joblight_evaluation/envs/learned-gpu/bin/python

cd "$REPO"
export PYTHONPATH="$REPO"
export PYTHONDONTWRITEBYTECODE=1

for config in neurocard_job_light.yaml neurocard_job_light_ranges.yaml distjoin_job_light.yaml; do
  "$PYTHON" -m evaluation.job_light_imdb_non_trajectory.joblight_eval.cli \
    aggregate --config "$EVAL/configs/$config"
done

RESULTS="$RESULTS" PYTHON="$PYTHON" "$PYTHON" - <<'PY'
import json
import os
import subprocess
from pathlib import Path

results = Path(os.environ["RESULTS"])
previous = json.loads((results / "final_comparison" / "comparison.json").read_text())
command = [
    os.environ["PYTHON"],
    "-m",
    "evaluation.job_light_imdb_non_trajectory.joblight_eval.cli",
    "compare",
]
for path in previous["inputs"]:
    command.extend(["--aggregate-json", str(path)])
command.extend(["--output", str(results / "final_comparison")])
subprocess.run(command, check=True)
PY

"$PYTHON" "$EVAL/scripts/check_comparison_provenance.py" \
  --comparison "$RESULTS/final_comparison/comparison.json" \
  --output "$RESULTS/final_comparison/provenance.json"
"$PYTHON" "$EVAL/scripts/make_latex_tables.py" \
  --results-root "$RESULTS" --refresh-own-model-build-records
