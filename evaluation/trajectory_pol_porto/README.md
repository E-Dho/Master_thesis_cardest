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
| P2 | Shared data snapshot + label-reproduction gate | done (all 10,400 labels reproduced) |
| P3 | Exact-MBR and trajectory sampling, first end-to-end comparison | done, timed |
| P4 | MobilityDB / PostgreSQL planner (three variants) | done, timed |
| P5 | DeepDB-MBR | done, timed |
| P8b | MSCN | done, timed |
| P6 | TraHist, TraHist-Temporal, CASE+, CASE+-Temporal (shared temporal module, tuning on validation) | done on POL, timed |
| P7 | Own model (ResMADE + `traj_dedup_factor`), dedup-head ablation, error split | implemented; seeds 1-2 training, cluster runs pending |
| P9 | Porto | open (RIS: no-go for now) |

`traj_eval/plugin.py` registers the in-process adapters: `traj_exact_mbr` and
`traj_sampling` (P3), `traj_mobilitydb` (P4), `traj_mscn` (P8b),
`traj_trahist` and `traj_case_plus` (P6). DeepDB-MBR
(P5) and the own model (P7) use the core's `external` adapter with
`scripts/deepdb_mbr_bridge.py` and `scripts/own_model_traj_bridge.py`.

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
repeatable-read transaction, counts NULLs per column in SQL first, streams
tables without text columns or NULLs with binary `COPY` (about 2 minutes for
50M segments) and the others through a cursor, checks that segments are sorted per trip,
that every trip has segments and that `num_of_segments` matches, and writes a
manifest with database identity, SQL, NULL counts and a SHA-256 per array. It
writes into `<dir>.partial` and renames at the end, so a failed export never
leaves a half snapshot behind. `load_snapshot(dir)` memory-maps the arrays;
`snapshot verify --snapshot DIR` re-checks every checksum.

`traj_eval/snapshot_gate.py` recounts every benchmark label on the snapshot
with `NumpyBackend` (sharded, resumable, fsynced per query) and `merge` writes
`gate/<workload>_report.json`; it fails on any mismatch, gap or duplicate. A
results directory is bound to its inputs (`binding.json`: snapshot manifest,
benchmark and config hashes, shard count), and every result line carries its
query's `predicate_hash`; resumes and `merge` refuse anything else, so rerun
with new inputs into a new results directory.
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

### Segment-MBR overcount statistic

`traj_eval/mbr_overcount.py` measures how much a segment-level bounding box
overcounts the spatial part of the benchmark. For every query with a spatial
or spatio-temporal predicate it counts, with the same `NumpyBackend` path as
the gate and only that predicate's segment mask replaced:

- `exact_count`: the v3 predicate; must equal the stored label (checked);
- `decoupled_count`: line meets the window, time span meets the interval,
  tested separately (equals `exact_count` for spatial queries);
- `mbr_count`: segment MBR overlaps the window (and time span meets the
  interval).

`exact <= decoupled <= mbr`, so `mbr - exact` splits into the corner effect
(`mbr - decoupled`) and the space/time decoupling (`decoupled - exact`).
Per query it also counts segments by class among those satisfying every other
predicate: *corner* (both endpoints outside the window, line misses it, MBR
overlaps it: the false positives) and *crossing* (both endpoints outside, line
passes through: correct). Multi queries count distinct trips, single queries
segment rows; `standard.*` and `temporal.*` queries are excluded (no spatial
predicate). Sharded and resumable like the gate.

Provenance and gates: a `verify` step checks every snapshot array against its
manifest checksum once (`snapshot_verification.json`, with each array's size
and mtime); shards and merge refuse to start without it or if an array changed
since. The results directory is bound to the manifest, benchmark, config,
shard count, statistic version, git revision and a SHA-256 over the source
files of the computation; shards refuse uncommitted source changes
(`--allow-dirty` for local experiments only). The merge fails on any label
mismatch (recomputed against the benchmark, not taken from the stored flag)
and on any query that violates `exact <= decoupled <= mbr`, the
segment partition `mbr = exact + corner + decoupled_only`, or has a segment
with an endpoint inside the window whose line misses it.

```bash
REPO_ROOT=$PWD evaluation/trajectory_pol_porto/slurm/submit_mbr_overcount.sh   # WORKLOAD=test, 16 shards
```

Output in `$SNAPSHOT_DIR/analysis/mbr_overcount/v<version>/` (one directory per
statistic version, currently `v2`, so results of an older version never block a
run; `OVERCOUNT_ROOT` overrides it): `test_report.{json,md}`
(per category, per dimension × relation and per add-on count; provenance
hashes; the JSON also lists the ids of all overcounted queries),
`test_report_per_query.csv` (every evaluated query) and
`test_report_overcounted.csv` (only queries with `mbr_count > exact_count`,
most overcounted first, with the split into corner and decoupling overcount).

## P3 estimators

Both are in-process adapters (`traj_eval/adapters/`) built from the P2
snapshot: `build` writes the model to `<run>/model/model.npz` (its size is the
reported `serialized_bytes`), and `evaluate` answers queries one at a time.
`time` runs the same loop under the core's timing guard.

- **Exact-MBR** (`traj_exact_mbr`, one seed; deterministic). One box per
  trajectory (`derived/trips`). Trip and agent predicates are evaluated
  exactly with the v3 predicate code; the segment window and interval become
  box overlap, with space and time tested separately. It is never below the
  truth (tested), and exact when a query has no segment condition. It is the
  representation error of every one-box-per-trajectory method, without any
  model error, so it is reported next to DeepDB-MBR. Answers that rely on box
  overlap (any segment window or interval) are marked
  `approx:trajectory_box_overlap`, as the convention for surrogate answers
  requires; `all:exact_only` therefore covers its exact answers only.
  Unsupported: single-table queries, and attribute predicates on segment rows
  (`segment_idx`).
- **Trajectory sampling** (`traj_sampling`, three seeds). A seeded simple
  random sample of whole trips (with their segments and agents), evaluated
  exactly with `NumpyBackend` and scaled by N/n. This is unbiased (tested over
  300 seeds), and with n = N it reproduces every label (tested). Zero
  estimates stay 0. Size via exactly one of `sample_trips`, `sample_fraction`,
  `budget_bytes`. Unsupported: single-table agent queries.

Configs: `configs/pol_v3/` (`_base.yaml`, `exact_mbr.yaml`,
`traj_sampling_n{1000,10000,100000,1000000}.yaml`). On the cluster, after P2:

```bash
REPO_ROOT=$PWD evaluation/trajectory_pol_porto/slurm/submit_p3.sh
```

It submits one array job per config (one task per seed, `geo-mlp`
environment, no database) and an aggregate-and-compare job writing
`results/traj_phase1_pol_v3/p3_comparison/` (the core tables plus
`trajectory_summary_v1.{csv,md}`). Accuracy runs time a single pass; reported
latency comes from `cli time` under `cpu_1core`, which still needs the
trajectory timing launcher (see Timing).

Latency is recorded for answered queries only. The time to reject an
unsupported query is reported separately (`evaluation_detail.rejections` in
`summary.json`), and a workload a method cannot answer at all has null
inference metrics instead of rejection timings. Subprocess bridges must follow
the same rule: latency rows only for queries with status `ok`. The `time` stage
first checks that its predictions cover every query of each workload exactly
once; only a fully covered workload without answers may have no latency.

The stratified Markdown table shows smoothed Q-errors and the zero-estimate
count next to the raw ones, because one zero estimate makes the raw Q-error
huge (1e-12 guard), which hides everything else for sampling methods.

## P4 MobilityDB planner

`traj_mobilitydb` (`traj_eval/adapters/mobilitydb.py`) reads the planner's
row estimate with `EXPLAIN (FORMAT JSON)` on the loaded MobilityDB database;
nothing is executed. Single queries are planned as `SELECT 1 FROM ... WHERE
...`, multi queries as `SELECT t.trip_id ... GROUP BY t.trip_id`, so the top
node is the planner's distinct-trip estimate; `planner_details.csv` also keeps
the planner's row estimate of the distinct step's input (below Gather, Sort and
partial aggregates; per-process rows of parallel plans scaled to the total with
PostgreSQL's parallel divisor), which separates the join-size error from the
distinct-count error. Variants (`adapter.variant`, one config
each):

| Variant | SQL | Reported as |
|---|---|---|
| `v3_sql` (headline) | the label SQL of the benchmark itself (Liang-Barsky terms included) | exact question |
| `segment_stbox` | segment windows/intervals as `s.segment_tgeom && stbox(...)`, rest v3 SQL | `approx:segment_stbox` |
| `trip_stbox` | trips only, windows/intervals as `t.trip_tgeom && stbox(...)` | `approx:trip_stbox`; single queries and segment attribute predicates unsupported |

- Statistics are the database's own; `adapter.analyze` stays false.
  `prepare` writes `database_identity.json` (versions, planner settings,
  `last_analyze`, a SHA-256 fingerprint of `pg_stats` for the benchmark
  tables); `build` and `evaluate` stop if the fingerprint changed. The model
  size is the `pg_statistic` storage of those tables.
- `adapter.session_timezone: UTC`, and the stbox templates carry `+00`. The
  probe (2026-10-30, job 22578953) showed the `tgeompoint` times equal the
  naive `t_s`/`t_e` read as UTC (0 or −1 µs from float epochs) while the
  server default is `CET`; without the UTC session the boxes would be shifted
  by 1–2 h.
- The `stbox` literal syntax depends on the MobilityDB version, so it is a
  template (`adapter.stbox.{xt,x,t}`); the shipped one is the MobilityDB 1.4.0
  syntax (all three forms parse on the cluster). Spatial literals must carry
  the data SRID ("Operation on mixed SRID" otherwise), so `prepare` of an stbox
  variant executes each template once on one row and stops on any error.
  A template is the literal's text only (`SRID={srid};STBOX X(...)`); the
  adapter adds `stbox '...'`. Quoted templates are refused, because the config
  parser strips leading and trailing quote characters (this broke the first
  P4 run at c312134: the closing quote was lost).
- Cluster database (probe): PostgreSQL 16.10, MobilityDB 1.4.0, PostGIS 3.5.0
  (MobilityDB's own version string names PostGIS 3.6.0, i.e. it was built
  against newer headers; recorded, not shown to be a fault). Statistics exist
  for all 23 columns, but `last_analyze` is empty, so their age is unknown;
  `segments.trip_id` has `n_distinct` ≈ 40k against 4.3M trips, which drives
  the distinct-trip estimates. That is the planner as configured, and it is
  what the baseline reports.
- The three configs must stay identical apart from the variant keys (tested),
  because the config loader supports one level of `extends`.

On the cluster:

```bash
REPO_ROOT=$PWD evaluation/trajectory_pol_porto/slurm/submit_p4.sh
```

One job under the shared database job name `pol_mobilitydb`
(`--dependency=singleton`) starts PostgreSQL once and runs the three configs
in the `mobilitydb` environment; then an aggregate-and-compare job writes
`results/traj_phase1_pol_v3/p4_comparison/` with the P3 configs included
(`COMPARE_WITH=""` for P4 alone).

Tests: rendering, plan parsing and config checks run everywhere; with
`TRAJ_EVAL_TEST_PG` (PostgreSQL + PostGIS, no MobilityDB needed) the
`v3_sql` variant runs end to end, its `cpu_postgres_2core` time stage runs on
one pinned client core, and a changed `pg_stats` must stop a run.

## P5 DeepDB-MBR

One row per trajectory, learned with DeepDB's single-table SPN
(`create_naive_all_split_ensemble`, deepdb-public at `655ada1`, the IMDB
DeepDB environment, Python 3.8). `traj_eval/deepdb_mbr.py`:

- `export` (pipeline environment): `trips_mbr.csv` + `layout.json` from the P2
  snapshot, shared by all seeds: `trip_id`, trip attributes
  (`t_num_of_segments`, `t_trip_time_start/end`), agent attributes aligned via
  the v3 join (`a_age`, `a_educationlevel`, ...), and the box
  `b_xmin..b_ymax`, `b_tmin`, `b_tmax`; times as epoch seconds of the stored
  microseconds.
- `translate` (standard library only, imported by the bridge): exactly
  Exact-MBR's question (tested on every fixture query): trip and agent
  predicates on their columns, a segment window as closed box overlap, a
  segment interval as `b_tmin <= upper AND b_tmax >= lower`; diagnostic
  `approx:trajectory_box_overlap` when the box is used. Unsupported as for
  Exact-MBR: single-table queries and `segment_idx` (decided 2026-10-30).

`scripts/deepdb_mbr_bridge.py` (DeepDB environment): `prepare-shared` (CSV to
DeepDB HDF, once), `prepare`, `smoke` (throwaway SPN on the first 20,000
rows), `build` (per seed, all trips as samples: `--sample-size 10000000`,
RDC threshold 0.3), `evaluate` (translation, DeepDB `parse_query`,
`cardinality`; the expectation cache is cleared before each query, as on
IMDB). SQL literals are the shortest round-trip decimals without exponents.
Two compatibility shims (`install_compat`, recorded as `numpy_compat` in the
prepare manifest and the build metrics) keep the pinned DeepDB/SPFlow working
in the cluster's DeepDB environment (numpy 1.24, newer scikit-learn): the mixed
column domains of `rspn.build_ds_context` become an object array, as older
numpy produced, and an RDC pair gets NaN (SPFlow: independent) only if its CCA
raises *and* the pair is verified degenerate: finite features and every
canonical correlation at most 1e-8, computed scale-free from orthonormal bases
of the centred feature matrices (variation below 1e-12 per entry counts as
rounding noise), so rdc = 0 is exact. Examples: a constant variable, or
two-valued variables with orthogonal patterns on a tiny slice, where
scikit-learn 1.2 divides 0 by 0. Any other CCA failure, non-finite input
included, is re-raised; the number of fallbacks is recorded
(`cca_degenerate_fallbacks` in the smoke and build metrics). The first cluster
attempt (2801ae3) failed on the first of these in smoke.
On the cluster:

```bash
REPO_ROOT=$PWD evaluation/trajectory_pol_porto/slurm/submit_p5.sh
```

Shared export + HDF (`deepdb_mbr_prepare_shared.sbatch`), then one
`run_external.sbatch` task per seed (8 CPUs, 128 GB, 2 days, as the IMDB
DeepDB build), then aggregate and compare into `p5_comparison/`. Tests: with
`TRAJ_EVAL_DEEPDB_PYTHON` and `TRAJ_EVAL_DEEPDB_SOURCE` the real bridge runs
through every pipeline stage on the fixture.

## P8b MSCN

`traj_mscn` (`traj_eval/adapters/mscn.py`) re-implements the published model
and training (`mscn/model.py`, `train.py` at `5007328`): set convolution over
samples, predicates and joins; sigmoid on min-max normalised log labels;
q-error loss; Adam 1e-3, 100 epochs, batch 1024, 256 hidden units; 1,000
samples per table. The adapted part is the featurisation of v3 predicates
(attribute one-hot, kind one-hot, nominal one-hot, six domain-normalised
bounds plus presence bits) and the bitmaps, evaluated with the exact v3
semantics on rows sampled from the snapshot. One model serves both label
kinds (single queries have an empty join set).

- Trained on `adapter.training_jsonl` (the POL v3 training workload, 10,000
  queries); `build` refuses any predicate hash shared with an evaluated
  workload. `adapter.validation_jsonl` is reported in `model_meta.json`
  (overall and per relation), never used for selection.
- `mscn.yaml`: trained on single + multi (decided). `mscn_multi_only.yaml`:
  the decided fallback, trained on multi only; it reports single queries as
  unsupported.
- Weights, normalisation and samples are stored in `model.npz`
  (`serialized_bytes`); `parameter_count` is the weight count.

```bash
REPO_ROOT=$PWD evaluation/trajectory_pol_porto/slurm/submit_p8b.sh
```

One `run_inprocess.sbatch` task per seed with 8 CPUs, then aggregate and
compare into `p8b_comparison/`.

## P6 TraHist and the temporal extension

**Temporal module** (`traj_eval/temporal.py`, spec "Temporal extension for
CASE+ and TraHist"): equi-depth time buckets (segments by time midpoint,
boundaries computed on the full data and stored), sparse per-statistic
bucket counts `F_k` and links `L_ij` between consecutive occupied buckets of
one trajectory, and `D([a, b]) = sum F - sum L` with the link term as a
dominance count. A query interval selects every bucket it intersects. Exact
per statistic for every bucket-aligned interval (brute-force test).

**`traj_trahist`** (`traj_eval/adapters/trahist.py`), two configs:

- `trahist.yaml` (base): upstream TraHist (`YJ960/TraHist` at `870e8ce`,
  imported from its checkout `adapter.trahist_source`; the repository has no
  licence, so nothing is copied). Construction and estimator unchanged:
  CountGrid on the trajectory points (the polyline vertices: every segment
  start plus each trip's last segment end), LabelMap partition, per-bucket
  means `Sp`/`ST`, ratio estimator over the cell-resolution samples of the
  query rectangle. Spatial only: queries with a time predicate are
  unsupported.
- `trahist_temporal.yaml` (extension by us): same grid and LabelMap; per
  bucket and time bucket the point count and the `F`/`L` of its cells'
  distinct-trajectory counts. `ST` upstream is the *mean of per-cell*
  distinct counts, so a bucket's `F`/`L` are the sums of its cells' `F`/`L`
  (the spec's note on a union over the bucket would not reproduce
  upstream). The time-restricted `Sp`/`ST` go into TraHist's own estimator;
  with the full time range the result equals the base estimate exactly
  (gate, tested against upstream). Temporal-only queries use the full
  extent; `trips.trip_time` predicates use the point times and carry
  `approx:trip_time_as_point_time`.

Adaptations: the query rectangle is clipped to the grid extent before
upstream's sampling (upstream clips out-of-grid samples onto the border
cells and would count them repeatedly; v3's one-sided windows reach 1,000 km
past the data); points are float64 (the upstream runner rounds to float32).
Unsupported: single-table queries, attribute predicates (decided), segment
attribute predicates. Deterministic, one seed. `model_meta.json` reports the
upstream theoretical size `(cells + 2 buckets) x 8` bytes plus the temporal
entries, next to the serialized size.

**`traj_case_plus`** (`traj_eval/adapters/case_plus.py`), re-implemented from
the CASE paper (Fanaeepour et al., GeoInformatica 2015, Algorithms 1-3 and
Fig. 12; no public code):

- construction: exact grid traversal of every segment (cells intersected,
  edges crossed, once per trajectory; a vertex crossing goes x first, the
  paper's perturbation); Stage 2 flags a trajectory when two of its cells in
  one row (column) are not joined by a straight run of visited cells with
  crossed edges (our reading of "same row or column, not neighbours, no
  connection edge"), and every trajectory whose cells and crossed edges are
  not a tree (a cycle, a jump between segments) is flagged too, since Euler
  counting without planarization is only exact for a tree; a flagged
  trajectory is planarized to the MBR of its cells: unvisited MBR cells,
  uncrossed interior MBR edges and all interior MBR vertices become virtual
  counts (Fig. 12: real edges are the crossed ones);
- query (Algorithm 2) on the cells the query rectangle intersects:
  0 without real faces, else `(F+F') - (E+E') + V'` (no `V'` for one row or
  column); CASE+ (Algorithm 3) subtracts `min(min F', min E', min V')`;
- `case_plus_temporal.yaml` (extension by us): time-restricted CASE+ is
  CASE+ on every trajectory cut to the query's time buckets. A trajectory
  spanning buckets `i..j` gets the full CASE construction for each of its
  `s(s+1)/2` restrictions `lo..hi`, summed per key `(i, j, lo, hi)`; a query
  over `a..b` takes per span the key `lo = max(a, i)`, `hi = min(b, j)`.
  Equal to CASE+ built on the cut trajectories (test) and, for the full
  range, to the base (gate). Most trips lie in one or two buckets.

Storage: real and virtual counts per key (dense); `theoretical_bytes` counts
each histogram dense or as (index, value) pairs, whichever is smaller. The
number of keys grows with the number of distinct trajectory spans (a few
times the number of buckets), so the temporal variant's size depends strongly
on the grid and the bucket count.

**Tuning on the validation workload** (`traj_eval/tune.py`, never the test
set): every grid point re-runs the method on the validation benchmark's multi
queries under the experiment id `<experiment>_tuning` and writes
`tuning.csv` (coverage, smoothed Q-error, zero estimates, serialized and
theoretical bytes, build seconds). The chosen `unit_size_m` / `time_buckets`
are then committed to the configs.

```bash
# the pinned upstream checkout (once)
git clone https://github.com/YJ960/TraHist /work_beegfs/sunip956/master_thesis_trajectories/external/trahist
git -C /work_beegfs/sunip956/master_thesis_trajectories/external/trahist checkout 870e8ce6db0e6363eeca80dce193623034c858ab
REPO_ROOT=$PWD STAGE=tune evaluation/trajectory_pol_porto/slurm/submit_p6.sh   # sweeps
REPO_ROOT=$PWD STAGE=run  evaluation/trajectory_pol_porto/slurm/submit_p6.sh   # test runs + comparison
```

**Selection (decided 2026-11-05).** Two settings per method, both from the
validation sweeps (`traj_eval/select_tuned.py`): `budget` = the most accurate
setting whose serialized size fits the 1x budget (the own model's
inference-only size, measured with `scripts/own_model_size.py`: the training
checkpoint re-serialized without its optimizer state), and `best` = the most
accurate setting regardless of size. "Most accurate" = lowest geometric mean
of the validation smoothed Q-error over the answered queries (`q_gmean`,
decided 2026-11-06: one number over the whole distribution instead of one
quantile; it mostly reflects the typical query, so it can prefer a better
median over a better tail - TraHist 25 m, p50 1.65 / p99 11,111 - and p99/max
are reported with every pick), ties by p90, p99, then size. Tuning files written before `q_gmean` existed get it from the run's
`predictions.csv`. Both are evaluated on the test set, so the thesis can show
each method at equal size and at its best, with the cost of the latter.

**Picks (POL v3, 2026-11-06, budget 872,606,460 bytes).** TraHist 25 m,
TraHist-Temporal 25 m / 3200 buckets, CASE+ 25 m: budget = best. CASE+-Temporal:
budget 200 m / 3200 (`case_plus_temporal.yaml`, 288 MB), best 100 m / 3200
(`case_plus_temporal_best.yaml`, 1.17 GB, 1.34x the budget). The values and
their validation errors are noted in each config.

`own_model_size.py --breakdown` additionally reports the pickled size of every
checkpoint field and the largest `metadata.columns` entries (domain lengths and
bytes), to see which part of the inference-only size is model weights and
which is schema payload.

The in-process environment needs pandas and scipy (upstream imports them);
`prepare` records their versions and fails early without them.

## P7 own model

Decided 2026-11-09: the model as trained, configuration
`pol_50m_traj_dedup_single_anchor_mbr_segments_first_20k` (table order
segments-trips-agents), each run's `checkpoint_best.pt`, three training seeds.
Seed 0 is the existing run (job 22578159, step 19,750); seeds 1 and 2 differ
only in `training.seed`, `predicate_generation.seed` and the output directory.
The code that trained them is pinned on branch
`model/pol-segments-first-training-state` (056493b = `ae1a52d` plus the three
uncommitted files of the training checkout; all 84 non-cache files under
`model/src` and `model/configs` match the training tree's git blob hashes).
`configs/pol_v3/own_model.yaml` reads
`$THESIS_WORK_ROOT/p7_models/pol_v3/seed_<s>/{checkpoint_best.pt,training_config.yaml}`.

`traj_eval/own_model_translate.py` maps a v3 row onto the model's query record
without changing the model code. Closed intervals become `start <= U` and
`end >= L` (the adapter's own `temporal_overlap` mode is half-open). Every query is
answered, on one of three routes:

| route | queries | estimate |
|---|---|---|
| `distinct` | multi joining segments | matching segments x `traj_dedup_factor` |
| `join` | multi on trips/agents only | join rows (= distinct trips: the other tables are on the one side) |
| `rows` | single | table rows |

Spatial windows are answered with the segment bounding box
(`approx:segment_mbr`); spatio-temporal queries also test space and time
separately (`approx:decoupled_space_time`). The segment-MBR overcount statistic
measures what that costs on the benchmark itself. `--clamp table:attribute=lo:hi`
(Porto `origin_stand`: NULL modelled as 0, domain 1..63) intersects every
condition with the domain so the NULL value never matches. `test_p7_own_model.py`
evaluates the translated tokens exactly on the fixture data: every non-spatial
fixture query reproduces its label, every spatial one the MBR count.

The bridge (`scripts/own_model_traj_bridge.py`) writes, next to the exchange
CSVs, `<workload>_predictions_components.csv` (route, matching-segment estimate,
`traj_dedup_factor`, forward calls) and its provenance (checkpoint sha256,
training config sha256, the file the `model` package was imported from).
`build` reports the inference-only size (`own_model_size.py`) as
`serialized_bytes`. `own_model_no_dedup.yaml` reports the matching-segment
estimate on `distinct` routes (ablation without the dedup head).

Error split (`traj_eval/segment_truth.py`, then `traj_eval/own_model_error_split.py`):
`D_hat/D_exact = (M_hat/M) * (a_hat/a) * (D/D_exact)` with `M`, `D` the true
matching segments and trajectories under the semantics the model answers
(MBR, space and time separate) and `a = D/M`: the learned segment count, the
learned dedup factor and the semantic approximation, as additive log10 terms.
`a_hat` is taken from the reported estimate (`D_hat / M_hat`), so the ablation
counts with factor 1; the components CSV records the `estimate_kind`, and runs of
different kinds are not pooled. The configs name the checkpoint directory in
their commands (recorded in every run manifest); `submit_p7.sh STAGE=run`
refuses a `P7_MODELS` that differs from it.

```bash
STAGE=check MODEL_ROOT=<training-state worktree> slurm/submit_p7.sh   # repo vs training-state code: identical estimates
STAGE=truth slurm/submit_p7.sh                                         # matching-segment truth, 16 shards
STAGE=run   slurm/submit_p7.sh                                         # seeds 0-2 of own_model and own_model_no_dedup
PROFILES="cpu_1core gpu_single_query" CPU_CONFIGS=own_model GPU_CONFIGS=own_model \
  TIMING_SBATCH_ARGS="--nodelist=<EPYC 7313 node>" slurm/submit_timing.sh
```

## Memory budget

One level: 1× the own model's inference-only serialized size per dataset
(decided 2026-10-26). Every adapter reports `serialized_bytes`, and
`artifact_metadata` adds `budget_bytes` and `within_budget` when the config
sets `adapter.budget_bytes`. POL v3 (2026-11-06): 872,606,460 bytes, the
inference-only size of `pol_50m_traj_dedup_single_anchor_mbr_segments_first_20k`
step 20000 (`own_model_size.py`): 57.1 MB weights and buffers plus 815.2 MB of
value domains that inference needs to map range literals to encoded ids
(`segments:t_e` 405 MB, `segments:t_s` 372 MB, `trips:end_time` 36 MB). The
sampling ladder runs without a budget; a `traj_sampling` config with
`budget_bytes` can now follow.

## Timing

The standardized profiles of the core apply: `cpu_1core` for the in-process
estimators (P3), `cpu_postgres_2core` for the planner (P4, as for PostgreSQL
on IMDB). Accuracy runs time one unguarded pass (`timing.warmup_passes: 0`,
`repetitions: 1`); reported latency comes only from the `time` stage, which
uses `timing.timing_warmup_passes` (1) and `timing.timing_repetitions` (3)
from `_base.yaml`. Timing keys never change estimates or the accuracy hash,
and the `time` stage checks that its estimates equal the accuracy run's.

The jobs use the shared launcher
(`../job_light_imdb_non_trajectory/scripts/timing_launch.py --cli-module
evaluation.trajectory_pol_porto.traj_eval.cli`) and this pipeline's hardware
classes (`configs/timing_hardware.json`, exported as
`JOBLIGHT_TIMING_HARDWARE_SPEC`). On the cluster:

```bash
REPO_ROOT=$PWD AFTER_JOB=<P4 accuracy job> evaluation/trajectory_pol_porto/slurm/submit_timing.sh
```

The hardware class is fixed to the CPU of the first POL table (AMD EPYC
7313, node n152), so later methods are timed on the same model; a job on
another node fails before timing. `TIMING_SBATCH_ARGS` passes extra sbatch
arguments (e.g. `--nodelist=n152` or a `--constraint`), `CPU_CONFIGS` /
`PG_CONFIGS` choose the configs, `CPU_AFTER_JOB` makes the CPU timing wait for
accuracy jobs, and `COMPARE_WITH` adds already timed configs to the final
comparison.

- `timing_cpu_1core.sbatch`: one plan with every P3 config and seed
  (`traj_eval/timing_plan.py` picks the latest complete accuracy run of each,
  as `aggregate` does), all on one node, so one CPU model for the whole
  table. About 19 h, almost all of it the 1M-trip sample (≈ 0.66 s per query
  × 8,000 queries × 4 passes × 3 seeds).
- `timing_cpu_postgres_2core.sbatch`: the three planner configs. The launcher
  starts the POL server inside the step, pinned to the second core
  (autovacuum off), and pins the client to the first; it runs under the
  database job name `pol_mobilitydb` with `--dependency=singleton` and
  `afterok` on `AFTER_JOB`. Minutes.
- Then an aggregate-and-compare job writes
  `results/traj_phase1_pol_v3/timed_comparison/`.

Plans are written when each job starts, so the jobs can be submitted before
their accuracy runs exist. `timing_summary.json` records the passes that ran
(`detail["passes"]` of the adapter). The first timing runs (bc4cb9b, jobs
22579613/22579614) predate this and show the accuracy defaults (0 warm-up,
1 repetition) although they ran 1 + 3 passes; their latency files hold
exactly three rows per answered query. The timing Python is `geo-mlp` (threadpoolctl)
with the IMDB timing package directory on `PYTHONPATH` (psycopg).

## Tests

```bash
python -m pytest -q evaluation/trajectory_pol_porto/tests
# the TraHist gates need the pinned upstream checkout:
TRAJ_EVAL_TRAHIST_SOURCE=/path/to/TraHist python -m pytest -q evaluation/trajectory_pol_porto/tests/test_p6_trahist.py
```

The fixtures build a real v3 benchmark on the synthetic POL data with the v3
generator itself, so a change of the v3 record schema breaks these tests
rather than the cluster runs.
