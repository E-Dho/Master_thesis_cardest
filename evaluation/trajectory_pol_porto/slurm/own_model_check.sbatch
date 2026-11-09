#!/bin/bash
#SBATCH --job-name=traj_own_model_check
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/work_beegfs/sunip956/master_thesis_trajectories/slurm_logs/traj_own_model_check_%j.out
#SBATCH --error=/work_beegfs/sunip956/master_thesis_trajectories/slurm_logs/traj_own_model_check_%j.err
# P7 code check: the bridge with this repository's model code and with the
# pinned training state's (MODEL_ROOT, a worktree of
# model/pol-segments-first-training-state) must give byte-identical estimates
# and components for CHECK_QUERIES test queries of each relation, on CPU.
# Writes $CHECK_DIR/{repo,training_state}/<relation>_*.csv and check.json; exit 1 on any difference.
set -euo pipefail
REPO_ROOT=${REPO_ROOT:-${SLURM_SUBMIT_DIR:-}}
source "$REPO_ROOT/evaluation/trajectory_pol_porto/slurm/traj_env.sh"
traj_resolve_repo_root
MODEL_ROOT=${MODEL_ROOT:?set MODEL_ROOT to the training-state worktree}
MODELS=${MODELS:?set MODELS to the p7_models/<dataset>/seed_<s> directory}
CHECK_DIR=${CHECK_DIR:?}
CHECK_QUERIES=${CHECK_QUERIES:-500}
PY=${OWN_MODEL_PYTHON:-$THESIS_WORK_ROOT/joblight_evaluation/envs/learned-gpu/bin/python}
cd "$REPO_ROOT"
echo "repo=$(git rev-parse HEAD) model_root=$(git -C "$MODEL_ROOT" rev-parse HEAD) host=$(hostname)"
for relation in multi single; do
  for side in repo training_state; do
    extra=()
    if [ "$side" = training_state ]; then extra=(--model-root "$MODEL_ROOT"); fi
    out="$CHECK_DIR/$side"
    "$PY" evaluation/trajectory_pol_porto/scripts/own_model_traj_bridge.py evaluate "${extra[@]}" \
      --training-config "$MODELS/training_config.yaml" --checkpoint "$MODELS/checkpoint_best.pt" \
      --queries-jsonl "$TEST_BENCHMARK" --select-json "{\"relation\": \"$relation\"}" --query-limit "$CHECK_QUERIES" \
      --predictions "$out/${relation}_predictions.csv" --latency "$out/${relation}_latency.csv" --device cpu
  done
done
"$PY" - "$CHECK_DIR" <<'PYEOF'
import csv, json, sys
from pathlib import Path
root = Path(sys.argv[1])
def rows(side, name):
    with (root / side / name).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))
result = {"passed": True, "files": {}}
for relation in ("multi", "single"):
    for name in (f"{relation}_predictions.csv", f"{relation}_predictions_components.csv"):
        a, b = rows("repo", name), rows("training_state", name)
        differing = [x["query_id"] for x, y in zip(a, b) if x != y]
        ok = len(a) == len(b) and not differing
        result["files"][name] = {"rows": len(a), "identical": ok, "differing_query_ids": differing[:50]}
        result["passed"] &= ok
for side in ("repo", "training_state"):
    provenance = json.loads((root / side / "multi_predictions_components_provenance.json").read_text())
    result[f"{side}_model_package_file"] = provenance["model_package_file"]
    result[f"{side}_model_root_git"] = provenance["model_root_git"]
(root / "check.json").write_text(json.dumps(result, indent=2, sort_keys=True))
print(json.dumps(result, indent=2, sort_keys=True))
sys.exit(0 if result["passed"] and result["repo_model_package_file"] != result["training_state_model_package_file"] else 1)
PYEOF
