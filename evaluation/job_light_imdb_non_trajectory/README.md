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
```

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
and one GPU. CPU-comparable evaluation remains a later, separate stage.

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
