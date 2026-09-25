# JOB-light IMDB Non-Trajectory Evaluation

This directory contains the evaluation framework and results for learned
cardinality estimation on the relational IMDB/JOB-light benchmark. It is
deliberately separate from the POL trajectory evaluation and does not cover
trajectory-specific data, predicates, correction heads, or metrics.

The purpose of this README is to define the common reporting contract used by
all evaluated methods and the execution contract used by their adapters.

## Phase-one methods

| Method | Adapter | Primary protocol |
| --- | --- | --- |
| PostgreSQL 16.10 | Native `postgres` adapter | Isolated `PGDATA`, default statistics target 100, no extended statistics, top-level `Plan Rows` from `EXPLAIN (FORMAT JSON)` after rewriting `COUNT(*)` to `SELECT 1` |
| MSCN | Subprocess `mscn` adapter | Published SetConv JOB-light recipe: 100k training queries, 100 epochs, 1,000 sample bits, batch 1,024, hidden size 256 |
| MSCN ranges | Subprocess `mscn` adapter | Separate deterministic and test-disjoint 100k range workload, complete-domain ranks for strings, and `=/< />/<=/>=` operators |
| DeepDB | Subprocess `deepdb` adapter | Native `imdb-light` RDC ensemble, samples `10M/10M/1M/1M/1M`, budget factor 5, at most three tables |
| DeepDB JOB-light-ranges adapted | Subprocess `deepdb` adapter, `protocol: adapted` | Project-owned extension of `imdb-light` modeling the six columns JOB-light-ranges filters on; string columns as PostgreSQL-collation ranks; separate preprocessing and a newly trained ensemble with the published budgets |
| NeuroCard | Subprocess `neurocard` adapter | Native JOB-light and JOB-light-ranges ResMADE configurations, native factorized sampling, 8,000 progressive samples |
| DistJoin | Subprocess `distjoin` adapter | Published IMDB configuration and upstream dynamic sampler/ANPM implementation |
| FOJ sampling | Native `foj_sampling` adapter | Independent seeded Exact Weight full-outer-join subsets at 1k, 10k, 100k, 1M, and 7,168,000 rows |
| Own approach | Subprocess `own_model` adapter | Arbitrary named config/checkpoint/ablation with the same result contract |

Duet is intentionally deferred to a separate multi-table adaptation study.
Pinned source URLs and revisions are in `sources.lock.yaml`; source checkouts,
environments, datasets, checkpoints, and generated results are not versioned.

The FOJ adapter requires the immutable upstream NeuroCard sampler cache by
default. Missing JCT or primary-key index files fail immediately instead of
silently starting Ray-based cache construction inside an evaluation job.

## Running the pipeline

Every YAML config supplies a stable `experiment_id`, `method_id`, `variant_id`,
adjustable `display_name`, seed list, workload paths, source information,
resource profile, and adapter settings. Generated runs live under:

```text
results/<experiment_id>/<method_id>/<variant_id>/seed_<seed>/<timestamp>-<config_hash>/
```

Run all configured seeds, or execute resumable stages explicitly:

```bash
python -m evaluation.job_light_imdb_non_trajectory.joblight_eval.cli run \
  --config evaluation/job_light_imdb_non_trajectory/configs/postgres.yaml

python -m evaluation.job_light_imdb_non_trajectory.joblight_eval.cli prepare \
  --config CONFIG --seed 0
python -m evaluation.job_light_imdb_non_trajectory.joblight_eval.cli smoke \
  --config CONFIG --seed 0 --run-directory RUN_DIRECTORY
python -m evaluation.job_light_imdb_non_trajectory.joblight_eval.cli build \
  --config CONFIG --seed 0 --run-directory RUN_DIRECTORY
python -m evaluation.job_light_imdb_non_trajectory.joblight_eval.cli evaluate \
  --config CONFIG --seed 0 --run-directory RUN_DIRECTORY
python -m evaluation.job_light_imdb_non_trajectory.joblight_eval.cli summarize \
  --config CONFIG --seed 0 --run-directory RUN_DIRECTORY
python -m evaluation.job_light_imdb_non_trajectory.joblight_eval.cli time \
  --config CONFIG --seed 0 --run-directory RUN_DIRECTORY --profile cpu_1core
```

`time` is the profile-controlled latency measurement described under
"Standardized timing profiles"; on the cluster it is launched through the
`slurm/timing_*.sbatch` scripts rather than directly.

Aggregate the latest complete seed runs with:

```bash
python -m evaluation.job_light_imdb_non_trajectory.joblight_eval.cli aggregate \
  --config CONFIG
```

Combine completed method/variant aggregates into the cross-method JSON, CSV,
and Markdown comparison:

```bash
python -m evaluation.job_light_imdb_non_trajectory.joblight_eval.cli compare \
  --aggregate-json METHOD_A/aggregate/comparison.json \
  --aggregate-json METHOD_B/aggregate/comparison.json \
  --output results/joblight_phase1/comparison
```

Create the shared dataset checksum manifest once, outside every method's
preprocessing timer:

```bash
python evaluation/job_light_imdb_non_trajectory/scripts/create_dataset_manifest.py \
  --dataset-root /path/to/imdb-csv \
  --output /path/to/imdb_dataset_manifest.json
```

The PostgreSQL config initializes a private cluster, loads `schematext.sql`
and every shared CSV, creates the versioned JOB-light indexes, runs `ANALYZE`
with target 100, asserts that no extended statistics exist, and records the
`pg_statistic` footprint. Existing initialized data is resumed via its own
manifest.

The MSCN-ranges support module deterministically generates a test-disjoint
100k-query training workload from complete column domains, exposes an exact
PostgreSQL labeling hook, regenerates fixed per-table sample bitmaps, and uses
order-preserving complete-domain ranks for numeric and string literals.

Subprocess adapters deliberately isolate legacy Python and PyTorch stacks.
Their evaluation command writes canonical prediction and latency CSV files to
the supplied exchange paths; the common runner owns scoring and reporting.

NeuroCard has concrete normal and range-model configurations in
`configs/neurocard_job_light.yaml` and
`configs/neurocard_job_light_ranges.yaml`. The bridge imports the pinned
upstream ResMADE, lossless factorization, Exact Weight sampler, and progressive
sampling implementation directly. It bypasses only Ray/Tune orchestration and
online W&B logging, which are not part of the estimator mathematics. Smoke
evaluation uses the upstream pretrained checkpoints; production builds retrain
the corresponding published configuration for each seed. Preparation records
the checkout status, tracked-diff hash and patch, and sampler extension hash so
cluster compatibility changes cannot be hidden behind the pinned commit SHA.

MSCN's published normal JOB-light recipe is concrete in
`configs/mscn_job_light.yaml`. The compatibility bridge imports the pinned
upstream `SetConv`, encoders, bitmap format, and Q-error loss, and adds reusable
checkpoints plus the common timing and resource contract. Its Python 3.7
universal-newline file mode is adapted for current Python without changing the
parsed data.

## Adapted DeepDB JOB-light-ranges schema

The native DeepDB `imdb-light` schema evaluates 70/70 JOB-light queries but
only 197/1000 JOB-light-ranges queries: the other 803 filter on columns that
upstream `gen_job_light_imdb_schema()` deliberately marks irrelevant
(`title.phonetic_code` 710 predicate occurrences, `cast_info.nr_order` 123,
`title.season_nr` 79, `title.episode_nr` 74, `title.series_years` 24,
`title.imdb_index` 15). The native result stays unchanged and is reported
separately (`configs/deepdb_job_light.yaml`, variant
`imdb_light_rdc_published`, which reports the 803 queries as unsupported).

`configs/deepdb_job_light_ranges_adapted.yaml` defines the explicitly
non-native variant `deepdb / imdb_light_ranges_adapted`, displayed as
"DeepDB JOB-light-ranges adapted" and carrying `protocol: adapted` plus an
`adaptation` description through run manifests, summaries, aggregates, and the
cross-method comparison (which gains *Display name* and *Protocol* columns;
configs without `protocol` are `native`). It is a genuine adapted baseline,
not an evaluator patch: the added columns change DeepDB's preprocessing, RDC
statistics, ensemble selection, and learned SPNs.

- **Schema** (`joblight_eval/deepdb_ranges.py`): a project-owned copy of the
  upstream generator with the same six tables, attribute order, and five
  `child.movie_id = title.id` relationships; only the six columns above leave
  `irrelevant_attributes`. They are also listed in `no_compression`: DeepDB
  replaces NULL by a sentinel and subtracts its mass inside ranges, but
  histogram compression of leaves with more than 10,000 distinct values
  (`phonetic_code` has 23,259, `episode_nr` 14,906) would erase the sentinel
  and count NULL rows inside ranges.
- **Ordered string encoding**: SPN leaves only support `<, <=, >, >=` on
  numeric columns, so `title.phonetic_code`, `title.imdb_index`, and
  `title.series_years` become dense integer ranks `0..n-1` of their complete
  non-NULL domain, ordered by PostgreSQL itself with
  `ORDER BY column COLLATE "C"`. `C` (byte order) is the collation under
  which the published JOB-light-ranges labels are reproduced; a linguistic
  default such as `en_US.UTF-8` orders `series_years` values like
  `1995-????` differently and fails to reproduce many
  `series_years`-range labels (see validation below). The collation is a
  `prepare-shared --collation` option (`column` uses the column's own
  collation) and is recorded together with the database locale and server
  version. Python never emulates a collation: PostgreSQL computes each
  workload literal's `bisect_left` (`#values < literal`) and `bisect_right`
  (`#values <= literal`).
- **Literal rewriting**: `= v` becomes the exact rank; `>= v` becomes
  `rank >= bisect_left`, `> v` becomes `rank >= bisect_right`, `<= v`
  becomes `rank < bisect_right`, `< v` becomes `rank < bisect_left`. A missing
  equality literal, a range outside the domain, or contradictory bounds is an
  explicit empty predicate result (estimate 0, diagnostic
  `empty_predicate:...`). NULL has no rank and stays NULL (NaN) in the CSV,
  so it never satisfies a rewritten predicate, exactly as in SQL. Textual
  literals are recovered from the workload line so numeric-looking strings
  such as `imdb_index = '1'` are not converted to numbers. Latency rows use
  scope `rank_literal_rewrite_and_native_deepdb_inference`: the per-query
  rewrite is timed together with DeepDB inference; SQL parsing stays outside
  the timed region as in the native bridge.
- **Adapted dataset** (`deepdb_ranges_adapted_shared`, separate from the
  native `deepdb_shared`): `rank_domains.json`; headerless CSVs where the
  five child tables are byte copies and `title` is exported from PostgreSQL
  in source row order with ranked string columns, written in a dialect that
  DeepDB's reader parses exactly (all strings quoted, backslashes escaped).
  This matters: DeepDB's backslash `escapechar` also applies outside quotes,
  so on the IMDB snapshot the native reader shifts every field of title id
  2522636 (unquoted `\Frag'ile\`) and alters the title text of seven more
  rows; the adapted export records these differences
  (`native_reader_title_mismatches`) and verifies that DeepDB's reader
  recovers every PostgreSQL value of the ranked title file. Stage records and
  `shared_preprocessing_manifest.json` hold SHA-256 checksums and sizes of
  every CSV, HDF, and domain file, per-stage conversion and HDF times, the
  columns DeepDB kept, and `native_shared_preprocessing_reused: false`.
- **Ensemble**: HDF, sampled HDF, RDC statistics, and SPN ensemble are all
  rebuilt with the published sample sizes `10M/10M/1M/1M/1M`, budget factor
  5, and at most three tables; the native ensemble cannot be reused because
  its modeled columns and distributions differ.

Execution order on the cluster (all launchers live in `slurm/` and source
`_deepdb_ranges_adapted_common.sh`):

1. `deepdb_ranges_adapted_prepare.sbatch`: rank domains and ranked CSVs, then
   `validate` before the HDF stage, then HDF/sampled HDF, then `validate`
   again. Validation requires 1000/1000 JOB-light-ranges and 70/70 JOB-light
   queries to rewrite and parse with DeepDB's parser against modeled
   columns; compares every distinct ranked-string predicate and every
   query's full title-filter conjunction between PostgreSQL (original
   literals) and the adapted CSV as DeepDB reads it (rank predicates); does
   the same for `cast_info.nr_order` predicates; reports how many workload
   literals would move under code-point order; and re-counts every workload
   query with a string range (`LABEL_CHECK_LIMIT=-1`, the default) against
   the published labels, under the rank collation (all must match) and, as
   evidence, under the database default collation. Reports are written
   to `validation/{pre,post}_hdf_validation.json`; the job fails if any
   check fails.
2. `adapter_smoke.sbatch` with `CONFIG=.../deepdb_job_light_ranges_adapted.yaml`
   runs the two-query smoke: a synthetic fixture trained with native DeepDB
   code, one query with a ranked `phonetic_code` range and one with
   `cast_info.nr_order` and `title.season_nr`, plus rewrite-and-parse of the
   first real workload query on `phonetic_code` and on another added column.
3. `deepdb_ranges_adapted_validate_ensemble.sbatch`: bounded real-ensemble
   validation (real adapted HDF, samples `100k/100k/10k/10k/10k`, all 1,000
   queries); `bounded_validation.json` must report `passed: true`. It is a
   gate, not a reported result.
4. `deepdb_ranges_adapted_seeds.sbatch` (array `0-2%1`): prepare, smoke,
   build, evaluate, and summarize seeds 0, 1, 2 after checking both
   validation gates; then `cli aggregate --config
   configs/deepdb_job_light_ranges_adapted.yaml`.

Legacy dependency stacks are isolated under `environments/`. DeepDB uses the
recorded Python 3.8/SPFlow environment, while DistJoin layers only its missing
packages over the shared PyTorch runtime. Its native sampler is rebuilt with
`slurm/distjoin_sampler_build.sbatch`, which loads GCC 12 before compiling the
extension.

Each complete run contains `resolved_config.json`, `run_manifest.json`,
`predictions.csv`, `latency.csv`, `summary.json`, `build_metrics.json`,
`resource_metrics.json`, and raw command logs. The aggregate emits JSON, CSV,
and Markdown. Incomplete or unsupported queries remain explicit status rows.
On the CAU cluster, `slurm/adapter_smoke.sbatch` uses the production model
interpreter plus the isolated evaluation package layer and persists
`smoke_metrics.json` before any long build may start. NeuroCard production
training uses `slurm/neurocard_build.sbatch`; that staged launcher requires an
existing smoke-passed run directory and explicitly requests the GPU partition
and one GPU. GPU-native evaluation uses `slurm/gpu_evaluate.sbatch`, which
refuses to run without an allocated GPU, prints the actual accelerator model,
and performs the staged `evaluate` and `summarize` steps. MSCN, NeuroCard, and
DistJoin use this path because their reported inference evaluations use GPUs.
PostgreSQL, DeepDB, and FOJ sampling remain CPU evaluations. The own-model
recipe records GPU timing as primary and CPU timing as a supplementary profile
from the same checkpoint.

## Evaluation principles

- Evaluate every method on the same query files and true cardinalities.
- Preserve raw cardinality estimates in all result artifacts.
- Report raw and smoothed Q-error separately.
- Measure latency after model loading and warm-up unless explicitly labeled as
  cold-start latency.
- Keep preprocessing, training, and evaluation time separate.
- Record enough run metadata to reproduce every reported number.
- Report missing, failed, unsupported, or zero estimates explicitly rather than
  silently excluding them.

## Accuracy

For a query with true cardinality `t` and estimated cardinality `e`, raw
Q-error is:

```text
q_error(t, e) = max(e / t, t / e)
```

Raw zero-valued cases must be handled explicitly in the result artifact and
must not be silently replaced before the raw metric is computed.
The implementation uses the existing `1e-12` denominator guard for raw
Q-error only; it never overwrites the stored estimate. Percentiles use NumPy's
default linear interpolation convention.

The smoothed metric uses a floor of one for evaluation only:

```text
t_smooth = max(t, 1)
e_smooth = max(e, 1)
smoothed_q_error(t, e) = max(e_smooth / t_smooth,
                             t_smooth / e_smooth)
```

This smoothing does not modify the estimator output. Each evaluation reports
the following statistics for both raw and smoothed Q-error:

| Statistic | Meaning |
| --- | --- |
| `p50` | Median query Q-error |
| `p90` | 90th percentile |
| `p95` | 95th percentile |
| `p99` | 99th percentile |
| `max` | Worst query Q-error |

The report must additionally include:

- number and fraction of estimates below `1`;
- number of estimates below `0.1` and below `0.01`;
- number of exactly zero estimates;
- total, successfully evaluated, unsupported, and failed query counts;
- per-query true cardinality, raw estimate, raw Q-error, and smoothed Q-error.

## Range support

Accuracy must be reported separately for these workloads:

1. **JOB-light:** the normal workload without the extended range-query set.
2. **JOB-light-ranges:** the workload used to test one-sided and two-sided
   range-predicate support.

For each workload, report raw and smoothed Q-error at `p50`, `p90`, `p95`,
`p99`, and `max`, together with all sub-1 and zero-estimate counts. Do not
combine the two workloads into a single headline distribution.

The workload specification must record:

- query-file name and checksum;
- number of queries;
- predicate/operator counts;
- number of equality, one-sided range, and two-sided range queries;
- true-cardinality range and true-cardinality distribution;
- true-cardinality source and generation procedure.

## Inference efficiency

Measure steady-state, single-query inference after a documented warm-up. Use
the same timing scope for every method and state whether query parsing and
predicate encoding are included.

Required latency statistics, reported in milliseconds per query:

- mean;
- `p50`;
- `p95`;
- `p99`.

Also report throughput in queries per second:

```text
throughput = number_of_timed_queries / total_timed_wall_seconds
```

Record the warm-up query count, timed repetitions, batch size, execution
device, CPU thread count, and whether synchronization was required for GPU
timing. Model-only latency and end-to-end latency may both be reported, but
must be clearly distinguished.

### Standardized timing profiles

Reported latency comes only from profile-controlled, timing-only runs. The
`evaluate` stage still produces the predictions used for accuracy (and records
its own evaluate-stage timing as context), but that timing does not control
threads, pinning, or node class and is labelled as such in every report.

| Profile | Resources | Report table |
| --- | --- | --- |
| `cpu_1core` | one physical core (SMT sibling reserved, unused); BLAS/OpenMP/NumExpr/Numba/PyTorch intra-op threads = 1, PyTorch inter-op threads = 1; CUDA hidden | primary CPU latency, every method |
| `cpu_postgres_2core` | PostgreSQL only: client pinned to one core, server (postmaster and backend) to a second; autovacuum off, planner settings unchanged and recorded | PostgreSQL CPU latency |
| `gpu_single_query` | one L40S, four pinned support cores, CPU thread pools = 1, synchronized CUDA timing | GPU latency, GPU-capable methods |
| `cpu_fullnode_exclusive` | exclusive node, pools sized to the physical core count | supplementary scalability only, never mixed with `cpu_1core` |

**Latency definition.** Per query, starting from the canonical, already parsed
workload query: method-specific encoding, estimation, and conversion of the
result to a Python float. Workload-file parsing, model loading, and result
writing are excluded; evaluation is sequential, one query at a time, after the
configured warm-up. Where a method exposes a meaningful internal boundary the
optional `model_core_ms` column is reported separately (DeepDB: 
`SPNEnsemble.cardinality`; MSCN: the SetConv forward pass on transferred
tensors). DeepDB's SQL rendering and `parse_query` are its predicate encoding
and are therefore inside the timed region; PostgreSQL's timed region is SQL
rendering, `EXPLAIN` planning, and the `Plan Rows` conversion.

**Caches.** No cache of query-dependent results may survive between timed
queries. DeepDB keeps `SPNEnsemble.cached_expecation_vals` across queries,
keyed by a factor hash that omits the SPN; with ten repetitions every timed
call was a cache lookup, and cross-query collisions made estimates depend on
query order. Both DeepDB bridges now clear it before every query (outside the
timer). In a sandbox check on a 5% IMDB subset the adapted DeepDB mean latency
went from 0.22 ms (cache lookups) to 19.9 ms under `cpu_1core`, of which
17.9 ms is `SPNEnsemble.cardinality` itself. Query-independent, model-derived state may stay
warm and is documented in the guard report: DistJoin's upstream `use_cache`
(unfiltered join-key distributions) and FOJ sampling's per-table-subset
inverse-fanout weights.

**Determinism.** Every external command now runs with
`PYTHONHASHSEED=<experiment seed>` (unless a config sets it). Python's
per-process hash randomization changes set iteration order, which DeepDB's
factor/SPN selection depends on: identical models gave estimates up to 6.5x
apart between processes before this change.

**How it works.**

1. Each config declares `timing.primary_profile`, `timing.profiles`, and
   `timing.estimate_consistency` (`deterministic` with tolerances, or
   `stochastic` for NeuroCard's progressive sampling), and external adapters
   declare `adapter.timing_commands.<profile>`.
2. `cli time --config C --seed S --run-directory RUN --profile P` re-measures an
   evaluated run without rebuilding. It refuses if the run's resolved config
   differs from `C` outside timing-only keys (`timing`, `resources`,
   `adapter.timing_commands`, `adapter.supplementary_evaluate_commands`,
   `experiment.display_name`; `config.TIMING_ONLY_CONFIG_PATHS`) unless each
   difference is passed via `--allow-config-drift` (recorded), and refuses an
   allocation that does not match the profile or the declared hardware class.
   The hash of the config without these keys is the *accuracy config hash*
   (`accuracy_config_hash` in manifests and summaries); staged stages
   (`evaluate`, `summarize`, ...) of an existing run also accept a config that
   differs only in timing-only keys and log it under
   `timing_only_config_updates` in the run manifest.
3. The runner exports the profile's thread-limit environment before the
   measuring process starts. The measuring process (bridge, eval runner, or the
   in-process PostgreSQL/FOJ adapter) loads `joblight_eval/timing_guard.py`,
   re-applies the limits (threadpoolctl, PyTorch intra/inter-op), and verifies
   before and after the timed region. Hard checks: thread-limit variables,
   threadpoolctl BLAS pools, PyTorch threads, CPU affinity in physical cores,
   SMT sibling ownership by the Slurm job cgroup, CUDA visibility, unexpected
   child processes, the declared CPU/GPU model, and for PostgreSQL the
   backend's pinning and disjointness from the client. The raw OS thread count (with thread names) and the
   CPU-time/wall-time ratio are recorded and only warn.
4. Results go to `RUN/timing/<profile>/<utc>-<hash>/` (`timing_manifest.json`,
   `timing_summary.json`, `latency.csv` with a `profile` column,
   `predictions.csv`, per-workload guard reports, logs). The timing run's
   estimates are compared with the accuracy run; a deterministic method that
   disagrees fails unless `--allow-estimate-drift` is given.
5. `aggregate` groups seeds by method, variant, and accuracy config hash
   (recomputed from each run's `resolved_config.json`), so a pre-profile seed-0
   run aggregates with later seeds; the full hashes and the timing-only keys in
   which the seed configs differ are listed in the aggregate. `aggregate` and
   `compare` add one table per profile (with CPU/GPU model and the estimate
   consistency), keep the evaluate-stage table separately labelled, and write
   `timing.<profile>.*` CSV columns. Profiles missing for some seeds are listed
   as incomplete instead of being averaged.
6. Hardware classes are enforced, not only recorded. `configs/timing_hardware.json`
   declares one CPU model per profile and the GPU model of `gpu_single_query`
   (`NVIDIA L40S`), shared by every method. The launcher and the runner check
   the node before anything is measured, every measuring process checks again
   (`cpu_model_matches`, `gpu_model_matches`), and the guard reports are
   checked afterwards. The CPU models are intentionally left empty: timing
   refuses to run until they are declared, and the error prints the model
   detected on the allocated node. Values match case-insensitively; use
   `regex:<pattern>` for a family, or `any` to disable one check explicitly.
   `aggregate` rejects seeds timed on different hardware, and `compare`
   rejects a profile table whose methods were timed on different hardware;
   `--allow-hardware-mismatch` writes the report anyway with the table marked
   **NOT COMPARABLE**.
7. Model-core latency is shown only with complete coverage. Its query coverage
   (`model_core.coverage_fraction_min`, observation counts) is aggregated;
   when any seed lacks a model-core value for some timed queries (e.g. native
   DeepDB's unsupported queries) the mean is suppressed and the table shows
   `suppressed: coverage x%`.

**Launching.** `slurm/timing_{cpu_1core,cpu_postgres_2core,gpu_single_query,cpu_fullnode_exclusive}.sbatch`
run `scripts/timing_launch.py` under `srun --cpu-bind=cores` (with
`--hint=nomultithread` and an explicit `--cpus-per-task`, because Slurm >= 22.05
does not pass the batch value to `srun`). The launcher checks the step's CPU
set, splits it into physical cores, and runs every entry of a plan file
(`configs/timing_plan.example.txt`) back to back on the same node; for
PostgreSQL it starts the server pinned to the second core and refuses if a
server is already running outside the allocation. Fix the node class at
submission, e.g. `PLAN=plan.tsv sbatch --constraint=<feature> slurm/timing_cpu_1core.sbatch`,
and for the GPU profile request the L40S explicitly
(`sbatch --gres=gpu:<L40S gres type>:1 ...` or its `--constraint`; the names
are cluster specific, see `slurm_hint` in `configs/timing_hardware.json`). A
wrong node fails in the pre-flight check within seconds.

**Shared PostgreSQL data directory.** `PGDATA` is on BeeGFS and visible from
every node, but `pg_ctl status` and PostgreSQL's `postmaster.pid` check only
see local processes, so a server on another node looks stopped. Every launcher
(`timing_launch.py`, `postgres_admin.py`, and the sbatch scripts through
`slurm/_pg_lock.sh`) therefore starts, uses, and stops the server only while
holding `scripts/pg_cluster_lock.py`'s lock: an atomic `mkdir` of
`.<pgdata>.joblight-lock` beside `PGDATA` with an `owner.json` (host, PID and
start time, Slurm job, token). A lock is broken only when its holder is
provably gone (same host and PID dead, or its Slurm job no longer in
`squeue`); an existing `postmaster.pid` is accepted only when left by such a
dead holder, otherwise the launcher refuses (exit 4) and explains how to
verify and override (`JOBLIGHT_PG_ASSUME_STALE_POSTMASTER=1`). Jobs that find
the lock held fail fast (exit 3) unless `JOBLIGHT_PG_LOCK_WAIT_SECONDS` is set;
`pg_cluster_lock.py status --pgdata ...` shows the holder. Child processes
inherit `JOBLIGHT_PG_LOCK_TOKEN` and re-enter the lock. If a server cannot be
stopped the lock is kept until the job ends (Slurm then kills the server).
Every interpreter that measures needs `threadpoolctl` (present in the DeepDB
environment through scikit-learn; install `threadpoolctl==3.5.0` into the
learned-gpu environment and into `python_packages` for the geo-mlp CLI, which
measures PostgreSQL and FOJ sampling in-process). Setting
`JOBLIGHT_TIMING_ALLOW_UNPINNED=1` permits local debugging on an unpinned
allocation, but such results are marked not reportable.

**Reusing completed runs.** Accuracy runs and checkpoints are reused; only
timing is rerun, and seeds whose configs differ only in timing-only keys
aggregate together. DeepDB (native and adapted) additionally needs its
`evaluate` and `summarize` stages rerun once in the existing run directories,
because the expectation-cache and hash-seed fixes change a small number of its
estimates (no rebuild).

### Evaluate-stage device context (hardware-aligned)

The following policy governs the evaluate-stage timing only; it documents the
execution class of each method's published setup and is not the standardized
comparison above.


The headline device follows the method's published evaluation protocol rather
than forcing every method onto CPU. Every latency row records `device` and
`device_name`; summaries preserve all profiles under `inference_by_device` and
identify one `timing_protocol.primary_device`. CUDA is synchronized immediately
before and after every timed query. Reports include the actual cluster device
and the paper's reference hardware, since measurements on different GPU models
are not hardware-normalized.

| Method | Headline device | Published/reference setup |
| --- | --- | --- |
| PostgreSQL 16.10 | CPU | Native planner; no GPU inference path |
| MSCN | GPU | Original MSCN reports AWS ml.p2.xlarge with CUDA; the NeuroCard comparison reruns MSCN on an NVIDIA V100 |
| DeepDB | CPU | NeuroCard's matched comparison explicitly runs DeepDB on CPU |
| NeuroCard | GPU | AWS EC2, NVIDIA V100, 32 vCPUs |
| DistJoin | GPU | NVIDIA RTX4070Ti 12GB with AMD Ryzen9 7950X3D; the paper reports GPU inference memory |
| FOJ sampling | CPU | Weighted sample scan; no neural accelerator path |
| Own approach | GPU plus CPU supplement | Actual cluster GPU/CPU are recorded for each profile |

Primary sources are the MSCN, NeuroCard, DeepDB, and DistJoin papers and their
official repositories. This policy aligns the execution class, not the
absolute hardware: a cluster GPU result must not be presented as if measured
on the paper's exact V100, AWS GPU instance, or RTX4070Ti.

## Training and build cost

Report wall-clock duration for:

| Metric | Scope |
| --- | --- |
| Preprocessing time | Raw benchmark input to training-ready artifacts |
| Training time | Optimizer/training execution, including validation and checkpointing |
| Total build time | Preprocessing plus training and required model-finalization work |

The report must state whether cached data, domain metadata, sampled tuples, or
precomputed join artifacts were reused. Training specifications must include:

- optimizer steps, epochs, batch size, and nominal sampled tuples;
- early-stopping policy and actual stopping step;
- best/final checkpoint selection rule;
- optimizer, learning rate, and random seeds;
- hardware allocation and accelerator model;
- number of CPU cores and allocated RAM;
- software environment and relevant library versions.

## Storage

Report:

- trainable parameter count;
- total parameter count if different;
- serialized model/checkpoint size in decimal MB (`bytes / 1,000,000`);
- checkpoint format and whether optimizer state is included.

If a training checkpoint contains optimizer or diagnostic state, additionally
report an inference-only serialized size where available.

## Memory

The required memory metric is peak training GPU memory. Report both peak
allocated and peak reserved GPU memory when the framework exposes both, in
MB or GiB with the unit stated explicitly.

Peak inference memory is optional but recommended. Memory reports must state:

- measurement mechanism;
- device;
- batch size;
- whether model loading is included;
- whether the value is process-local or system-wide.

For CPU-only methods, report peak resident set size instead of GPU memory and
label it accordingly.

## Run identity and reproducibility

Every result set must record:

- method and variant name;
- Git commit SHA and dirty-worktree status;
- configuration file and a resolved configuration snapshot;
- dataset/workload version and checksums;
- checkpoint path and checkpoint-selection rule;
- execution date, hostname, and scheduler job ID where applicable;
- random seeds;
- hardware and software environment;
- evaluation command;
- output paths for per-query results and aggregate summaries.

## Planned evaluation extensions

The following dimensions are planned but are not part of the initial
evaluation contract.

### Query-workload shift

Evaluate accuracy when the test-query distribution differs from the training
or tuning workload. The exact shift definitions and grouping statistics will
be specified before experiments are run.

### Data shift and model freshness

Compare stale and retrained models after controlled changes to the underlying
data. In addition to the standard metrics, report retraining or update time
and clearly identify which data snapshot each model observed.

### PostgreSQL workload impact

Evaluate downstream PostgreSQL execution with estimated cardinalities. The
planned measures are end-to-end workload runtime and, where suitable, P-error
or another plan-quality metric. The integration and measurement protocol will
be defined separately.

## Reporting checklist

Each completed method evaluation should provide, at minimum:

- raw and smoothed Q-error `p50/p90/p95/p99/max`;
- sub-1 and zero-estimate diagnostics;
- separate JOB-light and JOB-light-ranges results;
- inference mean/`p50`/`p95`/`p99` latency and throughput;
- preprocessing, training, and total build time;
- parameter count and serialized model size;
- peak training GPU memory;
- complete run identity and reproducibility metadata.
