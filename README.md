# Master Thesis CardEst

Trajectory cardinality-estimation benchmark tooling for POL-generated mobility data and MobilityDB.

## Contents

- `query_generation/`: generic, config-driven SQL query generator for `standard`, `temporal`, `spatial`, and `spatio_temporal` COUNT workloads.
- `mobilitydb_loader/`: CAU-cluster scripts for building/running MobilityDB and loading POL trajectory data into PostgreSQL/MobilityDB.
- `pol_runs/`: Slurm scripts and configs for reproducible POL benchmark simulations.

Generated datasets, database directories, query run outputs, and POL source checkouts are intentionally ignored. POL is kept as an external dependency rather than vendored in this repository.

## Query Generation

Generate SQL-only workloads:

```bash
python3 query_generation/query_generator.py \
  --config query_generation/pol_query_config.json \
  --output queries.jsonl \
  --queries-per-category 500 \
  --seed 1 \
  --no-execute
```

Generate and execute against a running MobilityDB/PostgreSQL instance:

```bash
python3 query_generation/query_generator.py \
  --config query_generation/pol_query_config.json \
  --output queries.jsonl \
  --queries-per-category 500 \
  --seed 1 \
  --execute \
  --host 127.0.0.1 \
  --port 55432 \
  --dbname pol_mobilitydb \
  --user sunip956
```

Run tests:

```bash
python3 -m unittest query_generation.test_query_generator
```
