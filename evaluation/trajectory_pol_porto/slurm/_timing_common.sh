# Shared settings of the trajectory timing jobs (sourced by timing_*.sbatch).
# Needs REPO_ROOT, PROFILE and CONFIGS (colon-separated config paths). The plan
# (one line per config and seed: the latest complete accuracy run) is written
# at job start, so the job can be submitted before its accuracy runs finish.
# Every plan entry runs on the node of this job, so one plan = one hardware
# class; the shared launcher pins the client (and, for cpu_postgres_2core, the
# server) and the core's timing guard checks every measuring process.
REPO_ROOT=${REPO_ROOT:?set REPO_ROOT to a checkout of feature/trajectory-evaluation}
PROFILE=${PROFILE:?}
CONFIGS=${CONFIGS:?set CONFIGS to colon-separated config paths}
source "$REPO_ROOT/evaluation/trajectory_pol_porto/slurm/traj_env.sh"
traj_resolve_repo_root
# geo-mlp has numpy and threadpoolctl; the shared package directory of the IMDB
# timing runs adds psycopg for the planner adapter (as for PostgreSQL on IMDB).
CLI_PYTHON=${CLI_PYTHON:-/work_beegfs/sunip956/micromamba/envs/geo-mlp/bin/python}
PACKAGES=${PACKAGES:-$THESIS_WORK_ROOT/joblight_evaluation/python_packages}
LAUNCHER=$REPO_ROOT/evaluation/job_light_imdb_non_trajectory/scripts/timing_launch.py
CLI_MODULE=evaluation.trajectory_pol_porto.traj_eval.cli
PLAN_DIR=${PLAN_DIR:-$REPO_ROOT/evaluation/trajectory_pol_porto/results/timing_plans}
PLAN=$PLAN_DIR/${PROFILE}_${SLURM_JOB_ID:-local}.txt

cd "$REPO_ROOT"
export PYTHONPATH="$PACKAGES:$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
export JOBLIGHT_TIMING_HARDWARE_SPEC=$REPO_ROOT/evaluation/trajectory_pol_porto/configs/timing_hardware.json
export SRUN_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-1}
unset OMP_NUM_THREADS MKL_NUM_THREADS OPENBLAS_NUM_THREADS NUMEXPR_NUM_THREADS

echo "job_id=${SLURM_JOB_ID:-unknown} host=$(hostname) profile=$PROFILE"
echo "start_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "repo_commit=$(git -C "$REPO_ROOT" rev-parse HEAD 2>/dev/null || echo unknown)"
lscpu | grep -E 'Model name|Thread\(s\) per core|Core\(s\) per socket|Socket\(s\)|NUMA node\(s\)' || true

config_args=()
IFS=: read -r -a _config_list <<< "$CONFIGS"
for _config in "${_config_list[@]}"; do config_args+=(--config "$_config"); done
"$CLI_PYTHON" -m evaluation.trajectory_pol_porto.traj_eval.timing_plan \
  --profile "$PROFILE" --output "$PLAN" "${config_args[@]}"
echo "plan=$PLAN"
