# Implementation Plan

1. Inspect upstream repository structure.
   - NeuroCard: use as conceptual base for schema, full-outer joins, indicators,
     fanouts, samplers, estimators, and MADE/ResMADE boundaries.
   - Duet: use as conceptual base for virtual predicates, per-column output
     slices, predicate-conditioned inference, and training/evaluation flow.
   - DistJoin: inspect only to preserve an ANPM/factorization adapter boundary.
2. Implement exact synthetic oracle first.
3. Implement virtual predicate tokens and masks.
4. Implement `INV_FANOUT` reciprocal masks.
5. Implement cumulative per-head inverse fanout weights.
6. Implement weighted cross entropy and monitoring statistics.
7. Add a correctness-first predicate-conditioned autoregressive model.
8. Implement one-pass inference.
9. Add configuration validation and checkpoint metadata.
10. Add unit and integration tests.
11. Add scripts for synthetic training, evaluation, and exact oracle checks.
12. Document math, limitations, attribution, and ANPM extension point.

The production NeuroCard sampler and PyTorch MADE backend are intentionally left
as the next engineering step.

