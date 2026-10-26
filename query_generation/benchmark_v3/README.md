# POL benchmark v3 (`pol_exact_st_v3`)

Exact, segment-coupled spatio-temporal COUNT workload for POL. What a label
means is defined in [SEMANTICS.md](SEMANTICS.md); this file describes how to
produce and check the benchmark.

## What changes compared with v2

| | v2 (`segment_coupled_nonzero_v2`) | v3 |
|---|---|---|
| Spatio-temporal match | geometry and segment time span tested separately | exact: the object is inside the window *during* the interval (Liang-Barsky clipping on the raw columns) |
| Grid | 16 x 500 | 8 x 250 single (`COUNT(*)`) + 8 x 750 multi (`COUNT(DISTINCT trip_id)` only) |
| Temporal widths | 50 % uniform, 50 % exponential | log-uniform, 60 s to the full year |
| Spatial widths | 50 % uniform, 50 % exponential | log-uniform per axis, 10 m to the full extent |
| Domains | from an earlier, smaller dataset | measured on the 7-seed data (`domain_overrides`) |
| Spatio-temporal centers | space and time from different rows | jointly from one segment at one point in time |
| Attribute filters on spatial/temporal queries | none | non-standard multi queries: exactly 50 / 25 / 25 % with 0 / 1 / 2 add-ons |
| Spatial windows per query | possibly several | at most one |
| Zero labels | replaced in a second pass | rejected during generation, logged as known zeros |
| Duplicate queries | not checked | rejected (canonical predicate hash) |
| Center pools | unseeded `ORDER BY random()` per shard | one seeded `TABLESAMPLE` pool file per run |
| Interval bounds | `start < U AND end >= L` | closed: `start <= U AND end >= L` |

## Files

| File | Role |
|---|---|
| `pol_v3_config.json` | v3 settings; tables and domains come from `../pol_query_config.json` |
| `exact.py` | numpy reference of every predicate |
| `query_model.py` | SQL rendering from structured predicates, canonical hash |
| `sampler.py` | deterministic candidate per (category, slot, attempt) |
| `pools.py`, `build_pools.py` | seeded center pools |
| `generate.py` | one category: label, reject zero/duplicate, resume |
| `merge.py` | 16 categories -> `benchmark.jsonl`, all gates |
| `verify.py` | recount of a stratified sample: numpy (rows fetched), `EXISTS` (multi), independent PostGIS formulation (spatial and spatio-temporal) |
| `cluster_check.sql` | Phase 1 data checks (read-only) |
| `slurm/` | jobs and `submit_v3.sh` |

## On the cluster

Check out the branch `feature/exact-st-benchmark-v3` and use it as `REPO_ROOT`.

1. **Data check** (once, read-only):

   ```bash
   REPO_ROOT=$PWD STAGE=check query_generation/benchmark_v3/slurm/submit_v3.sh
   ```

   The report lands in `$MOBILITYDB_ROOT/query_runs/v3_cluster_check_*.txt`.
   Every check states its expected value in `cluster_check.sql`.

   Data domains: `pol_v3_config.json` carries the measured domains of all
   ordered attributes in `domain_overrides`. The pools job (first stage of
   every run) re-measures them and prints `domain_violations`; the list must
   be empty.

2. **Pilot** (10 queries per category, own run directory):

   ```bash
   REPO_ROOT=$PWD PILOT=1 query_generation/benchmark_v3/slurm/submit_v3.sh
   ```

   Look at `categories/*.summary.json` (label seconds, rejections),
   `benchmark_summary.json` and `verify_report.json` before the full run.

3. **Full run**:

   ```bash
   REPO_ROOT=$PWD query_generation/benchmark_v3/slurm/submit_v3.sh
   ```

   Each category task stops cleanly before its 24 h limit (exit 75). Then
   resubmit with the same `SEED` and `RUN_LABEL` and without rebuilding pools:

   ```bash
   REPO_ROOT=$PWD SKIP_POOLS=1 query_generation/benchmark_v3/slurm/submit_v3.sh
   ```

   Finished categories return immediately; the others resume after their
   last durable row without repeating any label query.

   Verification checks 25 queries per category (400 in total), writes each
   result to `verify_report.json.results.jsonl` immediately and stops cleanly
   before its 24 h limit (exit 75). To continue or repeat only the
   verification of a merged run:

   ```bash
   REPO_ROOT=$PWD STAGE=verify query_generation/benchmark_v3/slurm/submit_v3.sh
   ```

4. **Validation workload** (8 x 75 single + 8 x 225 multi = 2,400 queries,
   seed 20261026, none of them in the test benchmark):

   ```bash
   REPO_ROOT=$PWD VALIDATION=1 query_generation/benchmark_v3/slurm/submit_v3.sh
   ```

   Every candidate whose predicate hash occurs in the test benchmark is
   rejected (reason `excluded`), and the merge refuses any overlap. Its
   verification checks 10 queries per category.

All jobs that start PostgreSQL are submitted under the job name
`pol_mobilitydb` with `--dependency=singleton`, so Slurm runs them one at a
time (they share one data directory, and each stops PostgreSQL when it
ends). For a database job submitted under another name, add
`AFTER_JOB=<job id>`.

Outputs in `$MOBILITYDB_ROOT/query_runs/<RUN_LABEL>/`: `pools.json`,
`categories/<category>.{accepted,rejected}.jsonl` and `.summary.json`,
`benchmark.jsonl`, `benchmark_summary.json`, `verify_report.json` and
`verify_report.json.results.jsonl`.

## Locally

```bash
python -m pytest -q query_generation/benchmark_v3
```

`test_sql_duckdb.py` and `test_duckdb_end_to_end.py` run the rendered SQL on
DuckDB (with its spatial extension) against a synthetic POL-shaped database;
they are skipped if `duckdb` is not installed.
