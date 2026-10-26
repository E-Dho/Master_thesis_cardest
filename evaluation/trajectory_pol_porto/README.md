# Trajectory evaluation (POL and Porto, benchmark v3)

Evaluation of trajectory cardinality estimators on the v3 benchmarks
(`query_generation/benchmark_v3`, semantics `pol_exact_st_v3`). It uses the
same core as the JOB-light pipeline (`evaluation/eval_core`): configs, staged
runs, run directories, metrics, timing profiles, seed aggregation and
cross-method comparison work exactly as described in
`../job_light_imdb_non_trajectory/README.md`. This file only lists what is
different. The plan behind it is the project doc
`trajectory-evaluation-pipeline-plan.md`.

## Status

| Phase | Content | State |
|---|---|---|
| P1 | Shared core, v3 workload format, features, strata, coverage | done |
| P2 | Shared data snapshot + label-reproduction gate | implemented; cluster run pending |
| P3 | Exact-MBR and trajectory sampling, first end-to-end comparison | open |
| P4–P9 | MobilityDB, DeepDB-MBR, CASE+/TraHist (+ temporal), own model, RIS / MSCN, Porto | open |

`traj_eval/plugin.py` registers adapters as they are implemented; there are none yet.

## Workloads

A v3 `benchmark.jsonl` holds both label kinds. A workload is one selection of it:

```yaml
workloads:
  pol_v3_multi:
    format: traj_v3_jsonl
    queries_jsonl: /work_beegfs/.../query_runs/<RUN_LABEL>/benchmark.jsonl
    select:
      relation: multi            # COUNT(DISTINCT trip_id) labels
    expected_query_count: 6000   # counts the selected rows
    benchmark_config: ../../../query_generation/benchmark_v3/pol_v3_config.json
  pol_v3_single:
    format: traj_v3_jsonl
    queries_jsonl: /work_beegfs/.../query_runs/<RUN_LABEL>/benchmark.jsonl
    select:
      relation: single           # COUNT(*) labels
    expected_query_count: 2000
```

- `select` keys: `relation`, `dimension`, `interval`, `category`, `label_kind`.
  A value may be a list.
- Multi and single workloads are always separate workload IDs and are never
  pooled (like JOB-light and JOB-light-ranges).
- `benchmark_config` is optional. With it, every query gets the
  domain-relative features `window_area_fraction` and `interval_fraction`
  (unbounded windows and intervals clipped to the data domain), which the
  strata use.
- Loading enforces the v3 contract: a supported `semantics_version`, the label
  kind matching the relation, `cardinality` equal to the relation's label
  field, and unique predicate hashes.
- The rendered SQL is never parsed; everything comes from the structured
  predicates.

**Query IDs.** The core needs integer IDs. `query_id` is the 0-based position
within the selection, in file order. `traj-eval query-index --config C
--output-directory D` writes `<workload>_query_index.csv` with the v3
`query_id`, `slot_id` and `predicate_hash` for every ID.

## Subprocess bridges

External commands get the usual placeholders plus:

| Placeholder | Value |
|---|---|
| `{queries_jsonl}` | the benchmark file |
| `{workload_select_json}` | the selection as JSON, e.g. `{"relation": "multi"}` |

A bridge must select rows with
`evaluation.trajectory_pol_porto.traj_eval.workloads.select_rows(queries_jsonl,
json.loads(select_json))`, so its IDs are the pipeline's IDs. That module
imports only the standard library, so legacy environments can use it. The
prediction and latency CSV contract is unchanged.

**Approximate estimates.** An estimate that answers a different (approximating)
question, e.g. an MBR surrogate, is reported with `status = ok` and a
diagnostic starting with `approx:`. It is scored, but reported separately.
Queries a method cannot express are `unsupported`, e.g. attribute add-ons for
CASE+/TraHist and their temporal variants.

## Additional summary block

Every trajectory workload in `summary.json` gets `extension`
(`schema: trajectory_summary_v1`), next to the core's `accuracy` block:

- `coverage`: query count, `ok` exact / `ok` approximate / `unsupported` /
  `failed`, coverage fraction.
- `strata`: per stratum, the query and status counts, raw and smoothed
  Q-error p50/p90/p95/p99/max, 95 % percentile-bootstrap intervals for the raw
  percentiles (1,000 resamples, seeded, at least 10 queries), the signed error
  `log10(max(est,1)/max(true,1))` (p10/p50/p90/mean) and the fraction of
  overestimates. Strata:
  - `all` and `all:exact_only`;
  - `category:<dimension.interval.relation>`;
  - `truth_decade:[1eK,1eK+1)`;
  - `addons:<0|1|2>`;
  - `window_area_fraction:<decade>` and `interval_fraction:<decade>` (with
    `benchmark_config`).

`aggregate` averages every stratum metric across seeds (mean and sample std;
interval bounds are averaged too). It writes `trajectory_summary_v1.csv` and
`.md` next to `comparison.*`; `compare` writes the cross-method version. The
core's own tables are unchanged.

## Shared data snapshot (P2)

`traj_eval/snapshot.py` exports the benchmark database once into numpy arrays:
`tables/<table>/<column>.npy`, exactly the arrays the v3 `NumpyBackend` reads
(times as int64 microseconds of the stored naive timestamps; segments as
`sx, sy, ex, ey, ts, te`), ordered by primary key, plus `derived/trips/` with a
CSR index (`trip_id`, `offset`, `segment_count`) and per-trip bounding boxes
(`xmin..ymax`, `tmin_us`, `tmax_us`). The export runs in one read-only
repeatable-read transaction, streams fixed-width tables with binary `COPY`
(about 2 minutes for 50M segments), checks that segments are sorted per trip,
that every trip has segments and that `num_of_segments` matches, and writes a
manifest with database identity, SQL, NULL counts and a SHA-256 per array. It
writes into `<dir>.partial` and renames at the end, so a failed export never
leaves a half snapshot behind. `load_snapshot(dir)` memory-maps the arrays;
`snapshot verify --snapshot DIR` re-checks every checksum.

`traj_eval/snapshot_gate.py` recounts every benchmark label on the snapshot
with `NumpyBackend` (sharded, resumable, fsynced per query) and `merge` writes
`gate/<workload>_report.json`; it fails on any mismatch, gap or duplicate.
About 3.5 s per query at POL scale, i.e. 8 h serial for the test workload.

On the cluster (from a checkout of this branch):

```bash
REPO_ROOT=$PWD evaluation/trajectory_pol_porto/slurm/submit_snapshot.sh
```

The export starts PostgreSQL and is therefore submitted under the shared job
name `pol_mobilitydb` with `--dependency=singleton`, like every v3 database job
(`AFTER_JOB=<id>` additionally waits for a job of another name). The gate
arrays (test: 16 shards, validation: 8) need no database and start after the
export succeeded. Output: `$THESIS_WORK_ROOT/traj_snapshots/pol_v3/`.

## Memory budget

One level: 1× the own model's inference-only serialized size per dataset
(decided 2026-10-26). Adapters will receive it as `budget_bytes` and report
`serialized_bytes`.

## Timing

The standardized profiles of the core apply. Two things still point at the
IMDB pipeline and have to be switched for trajectory timing runs (P3/P4):

- `scripts/timing_launch.py` starts the JOB-light CLI;
- the hardware-class file defaults to the IMDB one (override with
  `JOBLIGHT_TIMING_HARDWARE_SPEC`).

## Tests

```bash
python -m pytest -q evaluation/trajectory_pol_porto/tests
```

The fixtures build a real v3 benchmark on the synthetic POL data with the v3
generator itself, so a change of the v3 record schema breaks these tests
rather than the cluster runs.
