# Implementation Report

## Reused Upstream Concepts

- NeuroCard: full-outer-join tuple layout, indicators, fanout semantics, and
  checkpointed column metadata.
- Duet: virtual predicate tokens, predicate-conditioned output heads, per-column
  masks, and sampling-free inference.
- DistJoin: only the future extension point for ANPM/factorized output adapters.

No upstream code was vendored or copied.

## New Components

- `model/src/data/schema.py`: column metadata and checkpoint metadata.
- `model/src/data/full_join_sampler.py`: synthetic full-outer-join validation
  fixture.
- `model/src/predicates`: virtual tokens and domain masks.
- `model/src/training/losses.py`: cumulative inverse weights, weighted cross
  entropy, and effective sample size.
- `model/src/model/output_adapter.py`: unfactorized identity adapter and ANPM
  placeholder.
- `model/src/model/predicate_made.py`: correctness-first autoregressive
  conditional-table model.
- `model/src/inference`: one-pass estimator.
- `model/src/evaluation`: exact oracle and q-error metrics.

## Mathematical Assumptions

- Fanout domains are strictly positive.
- `INV_FANOUT` applies the known reciprocal mask `1/f`.
- Wildcard fanouts contribute factor one.
- Later fanout-head losses are weighted by the cumulative product of earlier
  active inverse fanouts.
- The current fanout never weights its own target loss.

## Test Coverage

Tests cover:

- Logit slicing and softmax normalization.
- Ordinary and indicator predicate masks.
- Reciprocal fanout masks.
- Expected inverse fanout versus inverse expected fanout.
- Cumulative loss weights and wildcard exclusion.
- Weighted cross entropy.
- Checkpoint ordering/domain preservation.
- Factorization default and explicit failure.
- Exact two-fanout reweighted marginal.
- Synthetic full-outer-join oracle cases.
- Deterministic training smoke and finite one-pass estimates.

## Limitations

- The trainable backend is currently an empirical conditional table, not a
  PyTorch MADE/ResMADE.
- No real NeuroCard sampler integration yet.
- No JOB-light or trajectory workload integration yet.
- No ANPM or column factorization.
- No large-scale performance optimization.

## ANPM Extension Point

The precise extension point is:

```text
model/src/model/output_adapter.py
```

Replace or extend `ANPMFactorizedOutputAdapter` and route it through
configuration validation once factorized columns, reconstruction metadata, and
tests are implemented.

