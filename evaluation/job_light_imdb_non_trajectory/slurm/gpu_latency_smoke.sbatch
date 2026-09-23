#!/bin/bash
#SBATCH --job-name=joblight_gpu_latency_smoke
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=/work_beegfs/sunip956/master_thesis_trajectories/slurm_logs/%x_%j.out
#SBATCH --error=/work_beegfs/sunip956/master_thesis_trajectories/slurm_logs/%x_%j.err

set -euo pipefail

ROOT=/work_beegfs/sunip956/master_thesis_trajectories
REPO=$ROOT/Master_thesis_cardest_joblight_evaluation
PYTHON=$ROOT/joblight_evaluation/envs/learned-gpu/bin/python
PACKAGES=$ROOT/joblight_evaluation/python_packages
NEUROCARD=$ROOT/external/neurocard
QUERIES=$NEUROCARD/neurocard/queries/job-light.csv
OWN_CHECKPOINT=${OWN_CHECKPOINT:?set OWN_CHECKPOINT to a project checkpoint}
OWN_CONFIG=${OWN_CONFIG:-$ROOT/Master_thesis_cardest/model/configs/job_light_duet_binary_native_anpm_40k_rare_auxiliary.yaml}
OUTPUT=$ROOT/joblight_evaluation/gpu_latency_smoke_${SLURM_JOB_ID}

mkdir -p "$OUTPUT"
cd "$REPO"
export PYTHONPATH="$PACKAGES:$REPO"
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}

"$PYTHON" -c 'import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))'

"$PYTHON" evaluation/job_light_imdb_non_trajectory/scripts/neurocard_bridge.py evaluate \
  --source-root "$NEUROCARD" \
  --dataset-root "$ROOT/datasets/job" \
  --model-kind job_light \
  --seed 0 \
  --device cuda \
  --cpu-threads 4 \
  --checkpoint "$NEUROCARD/neurocard/models/job-light-pretrained.pt" \
  --queries "$QUERIES" \
  --predictions "$OUTPUT/neurocard_predictions.csv" \
  --latency "$OUTPUT/neurocard_latency.csv" \
  --psamples 8000 \
  --query-limit 2 \
  --warmup-passes 1 \
  --repetitions 2 \
  --smoke

"$PYTHON" evaluation/job_light_imdb_non_trajectory/scripts/own_model_bridge.py smoke \
  --config "$OWN_CONFIG" \
  --checkpoint "$OWN_CHECKPOINT" \
  --output-directory "$OUTPUT/own_model" \
  --queries "$QUERIES" \
  --query-limit 2 \
  --warmup-passes 1 \
  --repetitions 2 \
  --device cuda

grep -q ',cuda,' "$OUTPUT/neurocard_latency.csv"
grep -q ',cuda,' "$OUTPUT/own_model/smoke_latency.csv"
echo "output=$OUTPUT"
echo "neurocard_latency_rows=$(($(wc -l < "$OUTPUT/neurocard_latency.csv") - 1))"
echo "own_model_latency_rows=$(($(wc -l < "$OUTPUT/own_model/smoke_latency.csv") - 1))"
