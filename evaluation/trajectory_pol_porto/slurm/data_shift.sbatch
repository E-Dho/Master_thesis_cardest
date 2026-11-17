#!/bin/bash
#SBATCH --job-name=traj_data_shift
#SBATCH --partition=base
#SBATCH --cpus-per-task=2
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --output=/work_beegfs/sunip956/master_thesis_trajectories/slurm_logs/traj_data_shift_%A_%a.out
#SBATCH --error=/work_beegfs/sunip956/master_thesis_trajectories/slurm_logs/traj_data_shift_%A_%a.err
# One step of the data-shift ablation's data side (submit_data_shift.sh chains them):
#   STEP=select   the kept trips (split.json, kept/all trip ids)        -> $DATA/split
#   STEP=subset   the snapshot of the kept data                          -> $DATA/snapshot
#   STEP=gate     one shard (array task) of the benchmark counted on it  -> $DATA/gate
#   STEP=relabel  the benchmark with those counts (in-distribution control) -> $DATA/benchmark_subset.jsonl
#   STEP=prepare  the 80 % model preparation (MODEL_REPO = a checkout of model/data-shift) -> $DATA/prepared
# DATA = $THESIS_WORK_ROOT/data_shift/$TRAJ_DATASET/$SPLIT; every step refuses existing outputs.
set -euo pipefail
REPO_ROOT=${REPO_ROOT:-${SLURM_SUBMIT_DIR:-}}
source "$REPO_ROOT/evaluation/trajectory_pol_porto/slurm/traj_env.sh"
traj_resolve_repo_root
STEP=${STEP:?set STEP}
SPLIT=${SPLIT:?set SPLIT=random80 or temporal80}
case "$SPLIT" in random80) MODE=random ;; temporal80) MODE=temporal ;; *) echo "SPLIT must be random80 or temporal80" >&2; exit 2 ;; esac
DATA=$THESIS_WORK_ROOT/data_shift/$TRAJ_DATASET/$SPLIT
mkdir -p "$DATA"
echo "step=$STEP dataset=$TRAJ_DATASET split=$SPLIT commit=$(git -C "$REPO_ROOT" rev-parse HEAD) host=$(hostname)"
case "$STEP" in
  select|subset|gate|relabel)
    activate_mobilitydb_env
    cd "$REPO_ROOT"
    case "$STEP" in
      select) python -m evaluation.trajectory_pol_porto.traj_eval.data_shift select --snapshot "$SNAPSHOT_DIR" \
                --config "$V3_CONFIG" --mode "$MODE" --seed 0 --output "$DATA/split" ;;
      subset) python -m evaluation.trajectory_pol_porto.traj_eval.data_shift subset-snapshot --snapshot "$SNAPSHOT_DIR" \
                --config "$V3_CONFIG" --split "$DATA/split" --output "$DATA/snapshot" ;;
      gate) python -m evaluation.trajectory_pol_porto.traj_eval.snapshot_gate run --snapshot "$DATA/snapshot" \
              --config "$V3_CONFIG" --benchmark "$(traj_benchmark_for test)" --results-dir "$DATA/gate" \
              --shard "$SLURM_ARRAY_TASK_ID" --shards "${GATE_SHARDS:?}" --deadline-seconds "${DEADLINE_SECONDS:-20000}" ;;
      relabel) python -m evaluation.trajectory_pol_porto.traj_eval.data_shift relabel --benchmark "$(traj_benchmark_for test)" \
                 --results-dir "$DATA/gate" --split "$DATA/split" --snapshot "$DATA/snapshot" \
                 --output "$DATA/benchmark_subset.jsonl" ;;
    esac ;;
  prepare)
    MODEL_REPO=${MODEL_REPO:?set MODEL_REPO to a checkout of model/data-shift}
    FULL_PREPARED=${FULL_PREPARED:?set FULL_PREPARED to the full model preparation}
    PYTHON=${MODEL_PYTHON:-/work_beegfs/sunip956/micromamba/envs/geo-mlp/bin/python}
    cd "$MODEL_REPO"
    echo "model_commit=$(git rev-parse HEAD)"
    PYTHONPATH="$MODEL_REPO" "$PYTHON" -m model.scripts.subset_prepared_full_join --prepared "$FULL_PREPARED" \
      --split "$DATA/split" --output "$DATA/prepared" ;;
  *) echo "STEP must be select, subset, gate, relabel or prepare" >&2; exit 2 ;;
esac
