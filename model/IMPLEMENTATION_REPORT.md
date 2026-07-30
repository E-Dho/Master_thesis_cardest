# Implementation Report

## Reused Upstream Concepts

- NeuroCard: full-outer-join tuple layout, indicators, fanout semantics, and
  checkpointed column metadata.
- Duet: virtual predicate tokens, predicate-conditioned output heads, per-column
  masks, and sampling-free inference.
- DistJoin: high-bit-to-low-bit lossless factorization, original/factor column
  mappings, previous-factor embedding modulation for ANPM, and chunked
  factorized decoding boundaries.

No upstream code was vendored or copied.

## New Components

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

## Mathematical Assumptions

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

## Test Coverage

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

## Limitations

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

## Factorized Output Widths

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

## OOD Literal Recheck

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

## ANPM Extension Point

The precise extension point is:

```text
model/src/model/output_adapter.py
```

`ANPMFactorizedOutputAdapter` now decodes factorized torch outputs back to
original-column predicate factors. A future optimization can replace chunked
enumeration with prefix dynamic programming behind the same adapter API.

## Local Verification

```text
python3 -m compileall -q model/src model/scripts model/tests
python3 -m unittest discover -s model/tests
```

The local unittest run passed 30 tests with 1 PyTorch-dependent test module
skipped.

Cluster verification in the `geo-mlp` environment passed the PyTorch suite:

```text
python -m unittest discover -s model/tests
# Ran 40 tests in 16.184s; OK (skipped=1)
```

Cluster complete-domain preparation was executed with:

```text
python -m model.scripts.prepare_neurocard_data \
  --config model/configs/job_light_resmade_factorized_anpm.yaml \
  --rebuild-domains \
  --sample-rows 512
```

Cluster inspection was executed with:

```text
python -m model.scripts.inspect_sampler \
  --config model/configs/job_light_resmade_factorized_anpm.yaml \
  --sample-rows 2
```

The short JOB-light factorized smoke run was executed with:

```text
python -m model.scripts.train_resmade \
  --config model/configs/job_light_resmade_factorized_smoke.yaml
```

It completed 2 optimizer steps with:

```text
checkpoint=model/runs/job_light_resmade_factorized_smoke/checkpoint_step_2.pt
parameter_count=19228539
parameter_size_bytes=76914156
backbone_parameter_count=19083067
anpm_parameter_count=145472
training_seconds=5.371661
total_sampled_tuples=128
nominal_rows_seen=128
output_width_original=374732
output_width_factorized=9915
```

Last-step fanout effective sample sizes were:

```text
__fanout_cast_info=64.000000
__fanout_movie_companies=39.459285
__fanout_movie_info=17.112084
__fanout_movie_keyword=2.779912
__fanout_movie_info_idx=2.147076
```

The synthetic factorized ANPM smoke run completed 20 optimizer steps with:

```text
parameter_count=19821
backbone_parameter_count=18807
anpm_parameter_count=1014
training_seconds=0.469950
total_sampled_tuples=640
```

The paired evaluation used one backbone forward call and reported:

```text
backbone_forward_seconds=0.00881743
anpm_decode_seconds=0.01548140
model_forward_calls=1
```
