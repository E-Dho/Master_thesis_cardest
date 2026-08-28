# Faithful DistJoin-Style ANPM Hypernetwork Report

Branch: `feature/faithful-distjoin-anpm-hypernetwork`

The previous branch retains the additive prefix-offset ablation. This branch
replaces it with a faithful DistJoin-style generated-weight ANPM.

## Checkpoint-Compatible Inference Optimization

Optimization branch: `feature/optimize-factorized-anpm-inference`

The optimized class-space path preserves the trained DistJoin-style checkpoint
and its Q-error. It changes only inference computation:

- Replaced explicit `[batch,D,H]` and `[batch,H,D]` generated matrix
  materialization with exact rank-one contractions.
- Added a query-local factorized probability evaluator.
- Deduplicated repeated factor prefixes during fallback enumeration.
- Added wildcard, equality, one-sided range, and inclusive range scalar-mass
  evaluators for factorized original columns.
- Retained fallback original-domain enumeration for unsupported masks or domain
  orders where encoded-ID intervals do not match predicate semantics.
- Kept deterministic one-backbone-pass inference with no progressive sampling.

### JOB-light Benchmark

Checkpoint:

```text
model/runs/job_light_resmade_distjoin_anpm/checkpoint_step_3500.pt
```

Baseline faithful generated-weight inference:

```text
queries_scored=70
median_q_error=2.722431
p95_q_error=2615.157427
max_q_error=5015.196988
evaluation_wall_seconds=289.689952
total_query_wall_seconds=289.466102
total_model_latency_seconds=289.405067
total_model_forward_calls=87
```

Optimized checkpoint-compatible inference:

```text
queries_scored=70
median_q_error=2.722431
p95_q_error=2615.157427
max_q_error=5015.196988
evaluation_wall_seconds=0.896969
total_query_wall_seconds=0.667440
total_model_latency_seconds=0.611591
total_model_forward_calls=87
```

The speedup came from predicate-specific factorized masses plus rank-one
contractions. Because JOB-light factorized predicates are equality/range cases,
they avoid full original-domain enumeration after this change.

Grouped wall times:

| Group | Baseline Total | Optimized Total | Median Q-error |
| --- | ---: | ---: | ---: |
| Non-factorized normal | `0.519s` | `0.337s` | `1.698` |
| Non-factorized inclusion-exclusion | `0.135s` | `0.137s` | `2.973` |
| Factorized normal | `122.402s` | `0.083s` | `17.420` |
| Factorized inclusion-exclusion | `166.410s` | `0.111s` | `35.003` |

Peak inference memory before and after was not measured by the current
JOB-light evaluator. The production code no longer materializes generated ANPM
matrices; explicit matrix construction remains only in tests as a numerical
reference.

## Native Multi-Predicate Range Inference

Baseline commit before this phase:
`eadc077e011284e0e73eeb5b82030bbba34e1ec7`.

Branch: `feature/native-ranges-live-sampling`.

This phase addresses the JOB-light two-sided range failure caused by external
complete-cardinality subtraction. The previous corrected fixed-fixture run
estimated query 20 by running two separately conditioned complete queries:

```text
true_cardinality = 695701
C_upper_hat      = 144842.47
C_lower_hat      = 211711.12
C_lower_hat > C_upper_hat
clamped_estimate = 0
q_error_epsilon  = 6.957e17
```

The new production evaluation path groups predicates by original column,
normalizes each `ColumnPredicateSet`, returns zero explicitly for
contradictions, runs one ResMADE backbone pass per query, and computes native
interval mass from the resulting original-column distribution. External
full-query subtraction is no longer the production path; it should only be kept
as an explicitly named diagnostic baseline if needed for ablations.

Changed components:

- `model/src/predicates/operators.py`: interval predicate tokens now preserve
  lower/upper inclusivity and serialize stable keys with those flags.
- `model/src/predicates/vocabulary.py`: vocabularies load both legacy 3-field
  token keys and new 5-field interval keys.
- `model/src/predicates/sets.py`: canonical same-column predicate
  normalization and contradiction detection.
- `model/src/model/output_adapter.py`: native interval mass for unfactorized
  and ANPM-factorized original columns, with one query-local evaluator/cache.
- `model/scripts/evaluate_job_light_queries.py`: production JOB-light
  evaluation uses one native model pass and emits tail diagnostics.
- `model/src/evaluation/metrics.py`: supplementary floor-one Q-error.
- `model/tests/test_native_ranges.py` and `model/tests/test_resmade_torch.py`:
  native range normalization, legacy-key compatibility, zero-Q-error reporting,
  and one-backbone-state interval mass coverage.

Native JOB-light evaluation on the corrected fixed-fixture checkpoint:

```text
queries_scored=70
status_counts={ok: 53, ok_native_range: 17}
evaluation_wall_seconds=1.356
total_model_forward_calls=70
median_q_error=3.329
p90_q_error=1218.701
p95_q_error=2740.477
p99_q_error=1745842.050
max_q_error=2423520.177
zero_estimate_count=0
```

Query 20 after native range estimation:

```text
status=ok_native_range
true_cardinality=695701
estimated_cardinality=28724.26
q_error_epsilon=24.22
model_forward_calls=1
zero_estimate=false
```

Backbone call counts improved from `87` to `70` over the full 70-query
JOB-light workload; query 20 changed from two model calls to one. Evaluation
time changed from approximately `0.98s` for the external-subtraction corrected
fixture evaluation to `1.36s` for the native range evaluator.

Normal-query Q-errors were effectively unchanged:

```text
normal_median=2.547
normal_p95=2456.594
normal_max=13212.693
```

Native range query group:

```text
native_range_median=8.715
native_range_p95=1637806.406
native_range_max=2423520.177
```

The catastrophic query-20 monotonicity violation is gone, but the current
native range group still has a long tail. The worst remaining queries are
equality-plus-range cases with tiny nonzero estimates, not clamped zeros.

Live sampler metrics for this phase:

```text
sampling_mode=fixture
nominal_rows_seen=7168000
fresh_sampler_rows=0
fixture_rows_presented=7168000
```

Compositional predicate encoding metrics are not yet available. Training still
uses `predicate_encoding.mode: categorical_legacy`, so the native two-sided
range inference path conditions old checkpoints with a wildcard token for the
ranged column and applies the interval constraint to the output distribution.

Validation checkpoint selection is not yet implemented. The trainer records
interval metrics and periodic checkpoints, but does not yet run held-out
validation NLL/JOB-light evaluation or write `checkpoint_best.pt`.

## Upstream Reference

The implementation was adapted from the ANPM structure in
`GIS-PuppetMaster/DistJoin` `model/made.py`, specifically the
`modulation_embeds`, `modulation_offset_layers_*`, and `logits_for_col` path.
Only the local generated-weight math was adapted; DistJoin's full MADE class was
not copied.

## Implemented Structure

`ANPMColumnDecoder` now owns:

- `previous_factor_embeddings`: embeddings for preceding same-column factor
  values.
- `factor_hypernetworks`: one hypernetwork for every factor after the first.

Each later-factor hypernetwork generates six vectors:

- `first_left`: `[batch, D_k]`
- `first_right`: `[batch, H]`
- `hidden_bias`: `[batch, H]`
- `second_left`: `[batch, H]`
- `second_right`: `[batch, D_k]`
- `logit_bias`: `[batch, D_k]`

The generated low-rank matrices are:

```text
W_1(p) = first_left(p) first_right(p)^T      [batch, D_k, H]
W_2(p) = second_left(p) second_right(p)^T    [batch, H, D_k]
```

The current factor logits are transformed as:

```text
h_k   = ReLU(base_logits_k W_1(p) + hidden_bias(p))
ell_k = ReLU(h_k W_2(p) + logit_bias(p))
```

Factor zero still returns the ResMADE base logits unchanged except for optional
valid-class masking. Training remains teacher-forced and conditions factor `k`
only on `z_<k`. Inference remains deterministic chunked prefix enumeration and
does not introduce progressive sampling.

## Invalid Combinations

`valid_factor_class_mask` computes valid current-factor classes from the
bitwise factorization plan. When `anpm.mask_invalid_combinations: true`,
invalid classes are masked with `-inf` before softmax in both training and
inference. All-invalid rows raise an explicit error.

## Validation

Local validation:

```bash
python3 -m py_compile model/src/model/anpm.py model/src/model/factorization.py \
  model/src/model/output_adapter.py model/src/training/torch_losses.py \
  model/tests/test_resmade_torch.py
python3 -m unittest discover -s model/tests
```

Cluster validation in `/work_beegfs/sunip956/master_thesis_trajectories/Master_thesis_cardest`:

```bash
/work_beegfs/sunip956/micromamba/envs/geo-mlp/bin/python -m unittest model.tests.test_resmade_torch
/work_beegfs/sunip956/micromamba/envs/geo-mlp/bin/python -m unittest discover -s model/tests
/work_beegfs/sunip956/micromamba/envs/geo-mlp/bin/python -m model.scripts.train_resmade \
  --config model/configs/resmade_factorized_smoke.yaml
/work_beegfs/sunip956/micromamba/envs/geo-mlp/bin/python -m model.scripts.train_resmade \
  --config model/configs/job_light_resmade_factorized_smoke.yaml
```

Observed results:

- Focused PyTorch tests: `19` tests passed, `1` CUDA skip.
- Full model tests: `49` tests passed, `1` CUDA skip.
- Synthetic factorized smoke: completed 20 optimizer steps.
- JOB-light factorized smoke: completed 2 optimizer steps.

## Prior Prototype Report

The following notes were retained from the previous implementation report for
planning context.

### Reused Upstream Concepts

- NeuroCard: full-outer-join tuple layout, indicators, fanout semantics, and
  checkpointed column metadata.
- Duet: virtual predicate tokens, predicate-conditioned output heads, per-column
  masks, and sampling-free inference.
- DistJoin: high-bit-to-low-bit lossless factorization, original/factor column
  mappings, previous-factor embedding modulation for ANPM, and chunked
  factorized decoding boundaries.

No upstream code was vendored or copied.

### New Components

- `model/src/data/schema.py`: column metadata and checkpoint metadata.
- `model/src/data/full_join_sampler.py`: synthetic full-outer-join validation
  fixture.
- `model/src/data/complete_domain_preparation.py`: production JOB-light
  metadata preparation from complete base tables and join-key fanout metadata.
- `model/src/predicates`: virtual tokens and domain masks.
- `model/src/training/losses.py`: cumulative inverse weights, weighted cross
  entropy, and effective sample size.
- `model/src/model/output_adapter.py`: unfactorized identity adapter and
  ANPM-backed factorized torch adapter.
- `model/src/model/factorization.py`: immutable bitwise factorization plan,
  row factorization, and factor decode utilities.
- `model/src/model/anpm.py`: local ANPM column decoders with teacher-forced
  factor-prefix conditioning.
- `model/src/model/predicate_made.py`: correctness-first autoregressive
  conditional-table model.
- `model/src/model/resmade.py`: PyTorch predicate-conditioned ResMADE with
  separate predicate input bins and data output bins.
- `model/src/model/masked_layers.py`: NeuroCard-inspired masked linear and
  residual layers.
- `model/src/predicates/vocabulary.py`: per-column predicate-token
  vocabularies.
- `model/src/training/torch_losses.py`: torch weighted per-head CE.
- `model/src/training/resmade_trainer.py`: ResMADE training loop and
  checkpointing.
- `model/src/inference`: one-pass estimator.
- `model/src/evaluation`: exact oracle and q-error metrics.
- `model/scripts/prepare_neurocard_data.py`, `inspect_sampler.py`,
  `train_resmade.py`, and `evaluate_resmade.py`: real-training CLI surface.

### Mathematical Assumptions

- Fanout domains are strictly positive.
- `INV_FANOUT` applies the known reciprocal mask `1/f`.
- Wildcard fanouts contribute factor one.
- Later fanout-head losses are weighted by the cumulative product of earlier
  active inverse fanouts.
- The current fanout never weights its own target loss.
- Factorized ordinary-column losses are summed over factors per row before the
  original-column weight is applied.
- Invalid factor combinations are excluded from factorized inference by
  renormalizing over valid original IDs.

### Test Coverage

Tests cover:

- Logit slicing and softmax normalization.
- Ordinary and indicator predicate masks.
- Reciprocal fanout masks.
- Expected inverse fanout versus inverse expected fanout.
- Cumulative loss weights and wildcard exclusion.
- Weighted cross entropy.
- Checkpoint ordering/domain preservation.
- Factorization default, explicit strategy validation, round-trips, invalid
  tuples, original-row preservation, and output-width reduction on a large
  synthetic domain.
- ResMADE tests are present and skip when PyTorch is not installed. They cover
  output width, separate input/output bins, per-column softmax, masking leakage,
  CPU backward, checkpointing, factor-head masks, ANPM prefix dependence,
  grouped factor loss gradients, factorized adapter normalization, factorized
  checkpoint reload, and CUDA smoke when available.
- Exact two-fanout reweighted marginal.
- Synthetic full-outer-join oracle cases.
- Deterministic training smoke and finite one-pass estimates.
- Complete-domain JOB-light preparation tests for sample-size independence,
  OOD literal classification, explicit indicators, positive complete fanouts,
  manifest validation, and factorization activation.

### Limitations

- Local verification skipped torch-specific tests because PyTorch is not
  installed in this environment.
- The NeuroCard sampler is integrated as a manifest-backed adapter boundary;
  the full upstream Rust/index Exact Weight runtime is not vendored here.
- Direct input-output connections are rejected in factorized mode.
- Indicators and fanouts remain atomic.
- Factorized inference uses chunk enumeration rather than prefix dynamic
  programming.
- Full benchmark-scale JOB-light factorized training still needs cluster
  scheduling and a PyTorch environment.

### Factorized Output Widths

The synthetic unit test with a 5000-value domain and 8-bit factors reduces one
flat 5000-way head to factor heads `(32, 256)`, reducing that column's output
width to 288. JOB-light widths are dataset-metadata dependent and should be
measured with:

```bash
python3 -m model.scripts.inspect_sampler \
  --config model/configs/job_light_resmade_factorized_anpm.yaml
```

The old smoke-domain JOB-light manifest was inferred from one 512-row sample.
It reported `original_output_width=1768` and `factorized_output_width=1768`,
so no ordinary column crossed `minimum_domain_size=2048`.

After rebuilding domains from the complete JOB-light base tables and join
metadata, the cluster manifest reports:

```text
metadata_source=complete_base_tables_and_join_metadata
sample_rows=512
sample_encoding_ood_values=0
original_output_width=374732
factorized_output_width=9915
factorized_output_reduction_ratio=0.026458909300513433
schema_hash=8bcf9422d17f586c66f46030d209ff29d29c9a176be9fa22dc50a544b3f41000
factorization_hash=f8ceebd78df761d74ea4ad91e2c4c7e54b1b15b1ae6eeea007348679ff2f635f
```

Old versus complete domain sizes:

| Column | Old Smoke Size | Complete Size |
| --- | ---: | ---: |
| `cast_info:role_id` | 11 | 12 |
| `movie_companies:company_id` | 350 | 234998 |
| `movie_companies:company_type_id` | 2 | 3 |
| `movie_info:info_type_id` | 52 | 72 |
| `movie_keyword:keyword_id` | 475 | 134171 |
| `title:kind_id` | 5 | 8 |
| `title:production_year` | 53 | 134 |
| `movie_info_idx:info_type_id` | 4 | 6 |
| `__in_title` | 1 | 2 |
| `__in_cast_info` | 1 | 2 |
| `__in_movie_companies` | 1 | 2 |
| `__in_movie_info` | 1 | 2 |
| `__in_movie_keyword` | 1 | 2 |
| `__in_movie_info_idx` | 1 | 2 |
| `__fanout_cast_info` | 234 | 1741 |
| `__fanout_movie_companies` | 59 | 94 |
| `__fanout_movie_info` | 281 | 2937 |
| `__fanout_movie_keyword` | 234 | 540 |
| `__fanout_movie_info_idx` | 2 | 4 |

Selected factorized columns:

| Column | Original Domain | Factors | Factor Domains | Invalid Combos |
| --- | ---: | ---: | --- | ---: |
| `movie_companies:company_id` | 234998 | 2 | `(128, 2048)` | 27146 |
| `movie_keyword:keyword_id` | 134171 | 2 | `(128, 2048)` | 127973 |

Fanout domains are dense positive ranges with neutral value `1`:
`__fanout_cast_info=1..1741`, `__fanout_movie_companies=1..94`,
`__fanout_movie_info=1..2937`, `__fanout_movie_keyword=1..540`, and
`__fanout_movie_info_idx=1..4`.

### OOD Literal Recheck

The previous smoke-domain query evaluation had 18
`zero_due_to_missing_domain` query rows. Parsing those rows yields 19 missing
literal occurrences and 7 unique literal predicates. Against the complete
manifest, all formerly missing equality literals are present in the complete
domain; the remaining production-year cases are range thresholds and no longer
require domain membership.

```text
movie_info_idx:info_type_id = 113 -> present_in_complete_domain
movie_keyword:keyword_id = 117 -> present_in_complete_domain
movie_keyword:keyword_id = 8200 -> present_in_complete_domain
movie_keyword:keyword_id = 398 -> present_in_complete_domain
movie_companies:company_id = 22956 -> present_in_complete_domain
movie_keyword:keyword_id = 7084 -> present_in_complete_domain
title:production_year > 2014 -> range_threshold_not_required_to_be_domain_member
```

The cluster `preparation_stats.json` was updated with
`ood_evaluation_literals=7` and `ood_evaluation_literal_occurrences=19`.

### ANPM Extension Point

The precise extension point is:

```text
model/src/model/output_adapter.py
```

`ANPMFactorizedOutputAdapter` decodes factorized torch outputs back to
original-column predicate factors. A future optimization can replace chunked
enumeration with prefix dynamic programming behind the same adapter API.

## Distinct Trajectory Query Correction

Branch: `feature/distinct-trajectory-query-correction`

Base `main` SHA: `1c64bc0ab95182fad53ca4c874c1f94e5d92cc2f`

This branch adds the first POL distinct-trajectory ablation on top of the
existing segment-level estimator. The inspected model metadata continues to use
the original-column ordering:

```text
DATA columns -> INDICATOR columns -> FANOUT columns
```

Factorized output heads retain the original column degree through
`OutputHeadSpec.source_column_index`, so the terminal correction can be added
after all existing original columns without reordering any data/indicator/fanout
columns.

The new optional output-only scalar head is named `traj_dedup_factor`. It has
logical degree `N` for `N` existing predicate columns. When enabled, hidden
degrees are sampled over `0..N-1`, allowing the terminal head to condition on
every existing predicate token, including the final fanout token. Existing AR
heads keep their original strict `< source_column_index` masks, so no current or
future predicate leakage is introduced.

Training target semantics:

```text
sample one FOJ anchor tuple s
generate the ordinary row-satisfied Duet context Q
compute local m_traj(s)(Q) inside the anchor trajectory only
target = 1 / m_traj(s)(Q)
```

The target provider boundary is `TrajectoryMultiplicityProvider`; the concrete
CSR-style implementation is `TrajectorySegmentIndex`. It stores encoded segment
rows grouped by trajectory id and evaluates the same `GeneratedTrainingContext`
semantics used by ordinary training. For POL, the inspected query schema uses
`trips.trip_id` as the entity key, so the development config sets:

```text
trajectory_key: trip_id
```

The loss is weighted MSE with:

```text
w_traj = rho * product_{r: T_r = INV_FANOUT} 1 / f_r
```

This differs from ordinary per-column WCE because `traj_dedup_factor` is after
all fanout columns, so all active inverse fanouts contribute. The current rare
auxiliary objective is unchanged; trajectory supervision is generated only from
ordinary main-batch contexts.

Inference adds `DistinctTrajectoryEstimate`:

```text
matching_segment_estimate = existing one-pass segment estimator
distinct_trajectory_estimate = matching_segment_estimate * traj_dedup_factor
```

`TorchDistributionModel.predict_column_factors_and_traj_dedup()` obtains both
ordinary factors and the correction factor from one backbone call.

Known limitation: this is deliberately the single-anchor query-only ablation.
Tuple/FOJ and static fanout corrections are applied, but the additional
row-first query-generator factor `G(Q | s)` is not corrected. Exact global
distinct counts are reserved for evaluation, not training labels.
