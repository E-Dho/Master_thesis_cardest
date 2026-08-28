# Attribution

This milestone was designed after inspecting the public repository surfaces of:

- NeuroCard: https://github.com/neurocard/neurocard
- Duet: https://github.com/GIS-PuppetMaster/Duet
- DistJoin: https://github.com/GIS-PuppetMaster/DistJoin

## NeuroCard

Adapted concepts:

- Full-outer-join modeling distribution.
- Table-presence indicator columns.
- Categorical fanout columns.
- Fixed column ordering stored with checkpoints.
- Future sampler/model boundaries compatible with MADE/ResMADE-style density
  estimators.
- Masked autoregressive layers, fixed ordering, residual MADE structure, and
  direct input-output masking ideas from `neurocard/made.py`.
- Exact Weight full-outer-join sampler boundary from
  `neurocard/factorized_sampler.py`, including the role of join-count/index
  artifacts. NeuroCard's `FactorizedSampler` name refers to Exact Weight join
  sampling, not this project's disabled lossless column factorization.

No NeuroCard source code is copied into this repository. The NeuroCard GitHub
repository page identifies Apache-2.0 licensing.

## Duet

Adapted concepts:

- Predicate-conditioned autoregressive heads.
- Virtual query tokens.
- Predicate masks over categorical domains.
- Per-column output distributions and one-pass selectivity/cardinality
  inference.

No Duet source code is copied into this repository.

## DistJoin

Adapted concepts:

- Lossless high-bit-to-low-bit factorization of large categorical dictionary
  IDs from `common.py`.
- Original-column to factor-column mappings and factor metadata.
- Previous-factor embedding modulation for later factor logits from
  `model/made.py`.
- Chunked valid-value enumeration boundary inspired by DistJoin's factorized
  estimator paths.

No DistJoin source files are vendored. The ANPM implementation is a local,
project-shaped adaptation that preserves this repository's original-column
predicate interface.
