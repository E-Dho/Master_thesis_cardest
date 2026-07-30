# Implementation Plan

## Initial Predicate-Conditioned ResMADE Milestone

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

## Lossless Factorization And ANPM Milestone

1. Preserve the original query-facing schema and predicate vocabularies.
2. Add immutable factorization plans that map original columns to model output
   heads without changing `ModelMetadata.columns`.
3. Implement deterministic high-bit-to-low-bit lossless factorization over the
   full original dictionary domain stored in metadata.
4. Generalize ResMADE so output heads can be either original columns or factors,
   while all factor heads of one original column share the same outer
   autoregressive degree.
5. Reject factorized direct input-output connections until dedicated masks are
   tested.
6. Add per-column ANPM decoders adapted from DistJoin's previous-factor
   embedding modulation idea.
7. Train factorized columns with teacher-forced factor losses grouped back to
   original-column loss terms before applying row weights.
8. Keep indicators and `INV_FANOUT` columns atomic and preserve cumulative
   inverse-fanout weighting unchanged.
9. Decode factorized distributions behind the output-adapter boundary using
   chunked valid-ID enumeration and original-domain predicate masks.
10. Store factorization and ANPM metadata in checkpoints and keep legacy
    unfactorized checkpoints loadable.
11. Add synthetic round-trip, leakage, ANPM, loss, checkpoint, and inference
    tests before any long JOB-light training run.
12. Document the implemented math, commands, attribution, and current
    limitations for the next research step.
