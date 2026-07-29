#!/bin/bash
#SBATCH --job-name=pol_50m_seed1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=05:00:00
#SBATCH --output=/work_beegfs/sunip956/master_thesis_trajectories/slurm_logs/pol_50m_seed1_%j.out
#SBATCH --error=/work_beegfs/sunip956/master_thesis_trajectories/slurm_logs/pol_50m_seed1_%j.err

set -euo pipefail

module load micromamba/1.4.2
export MAMBA_ROOT_PREFIX=/work_beegfs/sunip956/micromamba

SEED=1
THESIS_HOME=/zfshome/sunip956/master_thesis_trajectories
THESIS_WORK=/work_beegfs/sunip956/master_thesis_trajectories
REPO="$THESIS_HOME/generate-mobility-dataset"
RUN_ROOT="$THESIS_WORK/runs/pol_atl_5000a_365d_5min_trajbench_50m"
DATASET="$THESIS_WORK/datasets/pol_atl_5000a_365d_5min_seed${SEED}_trajbench_50m"
CONFIG="$RUN_ROOT/seed${SEED}.properties"
JAR="$REPO/target/pol-0.2-jar-with-dependencies.jar"

rm -rf "$DATASET"
mkdir -p "$RUN_ROOT" "$DATASET" "$THESIS_WORK/slurm_logs"
cd "$REPO"

grep -Ev '^(numOfAgents|seed|maps|oneStepTime|maxSimulationSteps)[[:space:]]*=' parameters.properties > "$CONFIG"
cat >> "$CONFIG" <<CFG
numOfAgents = 5000
seed = $SEED
maps = maps/atl/map
oneStepTime = 5
maxSimulationSteps = 105120
CFG

cat > "$DATASET/metadata.txt" <<META
run_id=pol_atl_5000a_365d_5min_seed${SEED}_trajbench_50m
job_name=pol_50m_seed1
repo_dir=$REPO
commit_sha=$(git rev-parse HEAD)
jar=$JAR
config=$CONFIG
dataset_dir=$DATASET
steps=105120
agents=5000
seed=$SEED
simulation_test=trajectory_benchmark
started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
META

micromamba run -p "$THESIS_WORK/envs/pol-java" java \
  -Xmx56g \
  -Djava.awt.headless=true \
  -Dlog4j2.configurationFactory=pol.log.CustomConfigurationFactory \
  -Dlog.rootDirectory="$DATASET" \
  -Dsimulation.test=trajectory_benchmark \
  -jar "$JAR" \
  -configuration "$CONFIG" \
  -until 105120 2>&1 | tee "$DATASET/java_run.log"

if grep -E "Exception in thread|Exception occurred|Error in ShapeFileImporter" "$DATASET/java_run.log" >/dev/null; then
  echo "Detected Java runtime exception in POL output" >&2
  exit 3
fi

for expected in "$DATASET/logs/Trajectory.tsv" "$DATASET/logs/AgentCharacteristicsTable.tsv"; do
  if [ ! -s "$expected" ]; then
    echo "Missing expected log: $expected" >&2
    exit 4
  fi
done

if find "$DATASET/logs" -type f \( -name 'AgentStateTable*' -o -name 'TravelJournal*' -o -name 'FinancialJournal*' -o -name 'FinancialAttributesJournal*' -o -name 'Checkin*' -o -name 'SocialNetwork*' -o -name 'FriendFamilyGraph*' -o -name 'WorkGraph*' \) | grep -q .; then
  echo "Unexpected heavy log was produced" >&2
  find "$DATASET/logs" -type f \( -name 'AgentStateTable*' -o -name 'TravelJournal*' -o -name 'FinancialJournal*' -o -name 'FinancialAttributesJournal*' -o -name 'Checkin*' -o -name 'SocialNetwork*' -o -name 'FriendFamilyGraph*' -o -name 'WorkGraph*' \)
  exit 5
fi

agent_rows=$(($(wc -l < "$DATASET/logs/AgentCharacteristicsTable.tsv") - 1))
if [ "$agent_rows" -ne 5000 ]; then
  echo "Expected 5000 AgentCharacteristics rows, found $agent_rows" >&2
  exit 6
fi

printf "finished_utc=%s\n" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$DATASET/metadata.txt"
