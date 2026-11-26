#!/bin/bash
# Final JOB-light tables with the rare-support-off own-model rows (decided 2026-11-25).
# Replaces both pending refreshes: refresh_corrected_own_model_tables.sbatch (23397707, whose
# rare-auxiliary inputs were cancelled) and refresh_raw_estimate_diagnostics.sbatch (23427924).
# Submit after the own-model accuracy and timing runs and the nine baseline reruns.
#SBATCH --job-name=refresh_rare_off_own_model_tables
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
PYTHON=$ROOT/joblight_evaluation/envs/learned-gpu/bin/python
EVAL=evaluation/job_light_imdb_non_trajectory
RESULTS=$REPO/$EVAL/results/joblight_phase1

cd "$REPO"
export PYTHONPATH="$REPO"
export PYTHONDONTWRITEBYTECODE=1

for CONFIG in \
  neurocard_job_light.yaml \
  neurocard_job_light_ranges.yaml \
  distjoin_job_light.yaml \
  own_model_job_light_rare_off_rule_step_11000.yaml \
  own_model_job_light_ranges_rare_off_rule_step_11000.yaml; do
  "$PYTHON" -m evaluation.job_light_imdb_non_trajectory.joblight_eval.cli aggregate --config "$EVAL/configs/$CONFIG"
done

RESULTS="$RESULTS" PYTHON="$PYTHON" "$PYTHON" - <<'PY'
import json
import os
import subprocess
from pathlib import Path

results = Path(os.environ["RESULTS"])
previous = json.loads((results / "final_comparison" / "comparison.json").read_text(encoding="utf-8"))
own_model = results / "own_model"
# Every earlier own-model row (rare-auxiliary rule rows and the benchmark-selected rows) is
# replaced by the two rare-support-off rule rows; the 7M FOJ sample stays excluded.
excluded = {results / "foj_sampling" / "n7168000" / "aggregate" / "comparison.json"}
kept = [Path(path) for path in previous["inputs"]
        if own_model not in Path(path).parents and Path(path) not in excluded]
replacement = [own_model / variant / "aggregate" / "comparison.json"
               for variant in ("rare_off_rule_step_11000", "rare_off_content_rule_step_11000")]
inputs = kept + replacement
if len(inputs) != len(set(inputs)):
    raise RuntimeError("comparison input list contains duplicates")
missing = [str(path) for path in inputs if not path.is_file()]
if missing:
    raise RuntimeError(f"missing aggregates: {missing}")
print("dropped:", *sorted(set(previous["inputs"]) - {str(p) for p in kept}), sep="\n  ")
command = [os.environ["PYTHON"], "-m", "evaluation.job_light_imdb_non_trajectory.joblight_eval.cli",
           "compare"]
for path in inputs:
    command.extend(["--aggregate-json", str(path)])
command.extend(["--output", str(results / "final_comparison")])
subprocess.run(command, check=True)
PY

"$PYTHON" "$EVAL/scripts/check_comparison_provenance.py" \
  --comparison "$RESULTS/final_comparison/comparison.json" \
  --output "$RESULTS/final_comparison/provenance.json"
"$PYTHON" "$EVAL/scripts/make_latex_tables.py" --results-root "$RESULTS" \
  --refresh-own-model-build-records
