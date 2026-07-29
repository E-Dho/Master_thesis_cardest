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

- Future-facing factorized output adapter boundary.
- ANPM is explicitly not implemented in this milestone.

No DistJoin source code is copied into this repository.

