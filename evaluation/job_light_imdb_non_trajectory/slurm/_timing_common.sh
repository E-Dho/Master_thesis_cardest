# Shared settings for the profile-controlled timing launchers (sourced).
# Every run of a table must use the same node class: the CPU scripts request
# --nodelist=n237 and the GPU script --constraint=L40 (override on the command
# line with another node of the declared class).
# The class itself is declared in ../configs/timing_hardware.json and enforced
# by the launcher; a node of another class fails before anything is timed.
# and PLAN=<timing plan file>; see ../configs/timing_plan.example.txt.
ROOT=/work_beegfs/sunip956/master_thesis_trajectories
REPO=$ROOT/Master_thesis_cardest_joblight_evaluation
CLI_PYTHON=${CLI_PYTHON:-/work_beegfs/sunip956/micromamba/envs/geo-mlp/bin/python}
PACKAGES=$ROOT/joblight_evaluation/python_packages
PG_BIN=$ROOT/envs/mobilitydb/bin
PGDATA=$ROOT/joblight_evaluation/pgdata
SOCKET=$ROOT/joblight_evaluation/socket
PORT=55432
LAUNCHER=evaluation/job_light_imdb_non_trajectory/scripts/timing_launch.py
PLAN=${PLAN:?set PLAN to a timing plan file (CONFIG SEED RUN_DIRECTORY per line)}

cd "$REPO"
export PYTHONPATH="$PACKAGES:$REPO:${PYTHONPATH:-}"
export PYTHONDONTWRITEBYTECODE=1
# Slurm >= 22.05 does not pass the batch --cpus-per-task to srun implicitly.
export SRUN_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-1}
# Thread limits are exported per profile by the runner; clear inherited ones.
unset OMP_NUM_THREADS MKL_NUM_THREADS OPENBLAS_NUM_THREADS NUMEXPR_NUM_THREADS

echo "job_id=${SLURM_JOB_ID:-unknown} host=$(hostname)"
echo "start_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "repo_commit=$(git -C "$REPO" rev-parse HEAD 2>/dev/null || echo unknown)"
echo "plan=$PLAN"
lscpu | grep -E 'Model name|Thread\(s\) per core|Core\(s\) per socket|Socket\(s\)|NUMA node\(s\)' || true
