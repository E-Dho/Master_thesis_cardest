#!/bin/bash
#SBATCH --job-name=own_ranges_7c_sweep
#SBATCH --partition=base
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --hint=nomultithread
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --array=0-15%4
#SBATCH --output=/work_beegfs/sunip956/master_thesis_trajectories/slurm_logs/%x_%A_%a.out
#SBATCH --error=/work_beegfs/sunip956/master_thesis_trajectories/slurm_logs/%x_%A_%a.err

set -euo pipefail
ROOT=/work_beegfs/sunip956/master_thesis_trajectories
REPO=$ROOT/Master_thesis_cardest_joblight_evaluation
EVAL=evaluation/job_light_imdb_non_trajectory
SOURCE=$ROOT/external/master_thesis_cardest_7c156b247031d7f5dfbd96e812f111abe14a75f2
PYTHON=$ROOT/joblight_evaluation/envs/learned-gpu/bin/python
OUTPUT=$REPO/$EVAL/results/joblight_phase1/own_model/job_light_ranges_candidate_sweep_7c156b2

cd "$REPO"
test "$(git -C "$SOURCE" rev-parse HEAD)" = 7c156b247031d7f5dfbd96e812f111abe14a75f2
export PYTHONPATH="$SOURCE:$REPO:${PYTHONPATH:-}"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
"$PYTHON" "$EVAL/scripts/sweep_own_model_ranges_candidates.py" \
  --selection "$REPO/$EVAL/results/joblight_phase1/final_comparison/own_model_selection.md" \
  --queries "$ROOT/external/neurocard/neurocard/queries/job-light-ranges.csv" \
  --output-root "$OUTPUT" --candidate-index "$SLURM_ARRAY_TASK_ID" --device cpu
