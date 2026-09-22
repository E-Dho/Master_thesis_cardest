#!/bin/bash
#SBATCH --job-name=distjoin_sampler_build
#SBATCH --partition=base
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=/work_beegfs/sunip956/master_thesis_trajectories/slurm_logs/%x_%j.out
#SBATCH --error=/work_beegfs/sunip956/master_thesis_trajectories/slurm_logs/%x_%j.err

set -euo pipefail

ROOT=/work_beegfs/sunip956/master_thesis_trajectories
SOURCE=$ROOT/external/distjoin
PYTHON=/work_beegfs/sunip956/micromamba/envs/geo-mlp/bin/python
PACKAGES=$ROOT/joblight_evaluation/envs/distjoin_packages

module load gcc12-env/12.1.0
module load gcc/12.1.0
export MAX_JOBS=${SLURM_CPUS_PER_TASK:-4}
export CC=gcc
export CXX=g++

cd "$SOURCE/MySampler"
"$PYTHON" setup.py build_ext --inplace --force

cd "$SOURCE"
export PYTHONPATH="$PACKAGES:$SOURCE:$SOURCE/MySampler"
"$PYTHON" -c 'import torch; import mysamplerAQP; print(torch.__version__, mysamplerAQP.__file__)'
