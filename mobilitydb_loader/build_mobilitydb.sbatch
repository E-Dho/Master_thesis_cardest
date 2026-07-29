#!/bin/bash
#SBATCH --job-name=build_mobilitydb
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=/work_beegfs/sunip956/master_thesis_trajectories/slurm_logs/build_mobilitydb_%j.out
#SBATCH --error=/work_beegfs/sunip956/master_thesis_trajectories/slurm_logs/build_mobilitydb_%j.err

set -euo pipefail
source /zfshome/sunip956/master_thesis_trajectories/mobilitydb_loader/common.sh

mkdir -p "$MOBILITYDB_ROOT" "$MOBILITYDB_LOG_DIR" "$MOBILITYDB_ROOT/src" "$MOBILITYDB_ROOT/build"
load_micromamba

if [ ! -x "$MOBILITYDB_ENV/bin/psql" ]; then
  micromamba create -y -p "$MOBILITYDB_ENV" -c conda-forge --override-channels \
    postgresql=16 postgis cmake make pkg-config git python=3.11 psycopg \
    gcc_linux-64 gxx_linux-64 gsl json-c geos proj protobuf-c \
    libxml2 libxslt openssl readline zlib
fi

activate_mobilitydb_env
export CC="$MOBILITYDB_ENV/bin/x86_64-conda-linux-gnu-cc"
export CXX="$MOBILITYDB_ENV/bin/x86_64-conda-linux-gnu-c++"
export CMAKE_PREFIX_PATH="$MOBILITYDB_ENV"
export PKG_CONFIG_PATH="$MOBILITYDB_ENV/lib/pkgconfig:$MOBILITYDB_ENV/share/pkgconfig:${PKG_CONFIG_PATH:-}"

if [ ! -d "$MOBILITYDB_SRC/.git" ]; then
  echo "Missing MobilityDB source at $MOBILITYDB_SRC; clone it from the login node before submitting this job." >&2
  exit 2
fi

rm -rf "$MOBILITYDB_BUILD"
mkdir -p "$MOBILITYDB_BUILD"

cmake -S "$MOBILITYDB_SRC" -B "$MOBILITYDB_BUILD" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$MOBILITYDB_ENV" \
  -DPOSTGRESQL_PG_CONFIG="$MOBILITYDB_ENV/bin/pg_config" \
  -DCMAKE_PREFIX_PATH="$MOBILITYDB_ENV" \
  2>&1 | tee "$MOBILITYDB_LOG_DIR/cmake_$(date -u +%Y%m%dT%H%M%SZ).log"

cmake --build "$MOBILITYDB_BUILD" --parallel "${SLURM_CPUS_PER_TASK:-4}" \
  2>&1 | tee "$MOBILITYDB_LOG_DIR/build_$(date -u +%Y%m%dT%H%M%SZ).log"

cmake --install "$MOBILITYDB_BUILD" \
  2>&1 | tee "$MOBILITYDB_LOG_DIR/install_$(date -u +%Y%m%dT%H%M%SZ).log"

{
  echo "built_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "mobilitydb_commit=$(git -C "$MOBILITYDB_SRC" rev-parse HEAD)"
  echo "mobilitydb_branch=$(git -C "$MOBILITYDB_SRC" rev-parse --abbrev-ref HEAD)"
  echo "pg_config_version=$("$MOBILITYDB_ENV/bin/pg_config" --version)"
  echo "pg_config_bindir=$("$MOBILITYDB_ENV/bin/pg_config" --bindir)"
  echo "pg_config_pkglibdir=$("$MOBILITYDB_ENV/bin/pg_config" --pkglibdir)"
  echo "pg_config_sharedir=$("$MOBILITYDB_ENV/bin/pg_config" --sharedir)"
  "$MOBILITYDB_ENV/bin/psql" --version
} > "$MOBILITYDB_ROOT/build_metadata.txt"

find "$MOBILITYDB_ENV" \( -path "*extension*" -name "*mobilitydb*" -o -name "*MobilityDB*" \) \
  > "$MOBILITYDB_ROOT/installed_mobilitydb_files.txt" || true
