#!/bin/bash
#SBATCH --job-name=refresh_own_model_tables
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
  $EVAL/configs/own_model_job_light_best_median_step_18000_corrected_model_code.yaml \
  $EVAL/configs/own_model_job_light_step_11000_corrected_model_code.yaml \
  $EVAL/configs/own_model_job_light_ranges_best_median_step_40000_corrected_model_code.yaml \
  $EVAL/configs/own_model_job_light_ranges_best_p99_step_19500_corrected_model_code.yaml; do
  "$PYTHON" -m evaluation.job_light_imdb_non_trajectory.joblight_eval.cli aggregate --config "$CONFIG"
done

RESULTS="$RESULTS" PYTHON="$PYTHON" "$PYTHON" - <<'PY'
import json
import os
import subprocess
from pathlib import Path

results = Path(os.environ["RESULTS"])
old_comparison = results / "final_comparison" / "comparison.json"
previous = json.loads(old_comparison.read_text(encoding="utf-8"))
obsolete = {
    str(results / "own_model" / variant / "aggregate" / "comparison.json")
    for variant in (
        "duet_neurocard_distjoin_rare_aux_best_median_step_18000",
        "duet_neurocard_distjoin_rare_aux_step_11000",
        "duet_neurocard_distjoin_rare_aux_best_median_step_40000",
        "duet_neurocard_distjoin_rare_aux_best_p99_step_19500",
    )
}
replacement = [
    results / "own_model" / variant / "aggregate" / "comparison.json"
    for variant in (
        "duet_neurocard_distjoin_rare_aux_best_median_step_18000_corrected_model_code",
        "duet_neurocard_distjoin_rare_aux_step_11000_corrected_model_code",
        "duet_neurocard_distjoin_rare_aux_best_median_step_40000_corrected_model_code",
        "duet_neurocard_distjoin_rare_aux_best_p99_step_19500_corrected_model_code",
    )
]
inputs = [Path(path) for path in previous["inputs"] if path not in obsolete] + replacement
if len(inputs) != len(set(inputs)):
    raise RuntimeError("comparison input list contains duplicates")
command = [
    os.environ["PYTHON"], "-m", "evaluation.job_light_imdb_non_trajectory.joblight_eval.cli",
    "compare",
]
for path in inputs:
    command.extend(["--aggregate-json", str(path)])
command.extend(["--output", str(results / "final_comparison")])
subprocess.run(command, check=True)
PY

"$PYTHON" "$EVAL/scripts/check_comparison_provenance.py" \
  --comparison "$RESULTS/final_comparison/comparison.json" \
  --output "$RESULTS/final_comparison/provenance.json"
"$PYTHON" $EVAL/scripts/make_latex_tables.py --results-root "$RESULTS"
