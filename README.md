# Master Thesis CardEst

This repository contains the practical implementation for the Master's Thesis of Eric Hotho on the topic of "Learned Cardinalities Estimation on Range Queries".

The project focuses on trajectory-data cardinality estimation: generating POL trajectory datasets, loading them into MobilityDB/PostgreSQL, generating labeled range-query workloads, and later training/evaluating learned estimators against baseline models.

## Directory Structure

- `dataset_generation/`: scripts and notes for POL simulation runs, trajectory extraction, MobilityDB setup, and database loading.
- `query_generation/`: config-driven workload generator for standard, temporal, spatial, and spatio-temporal COUNT queries.
- `model/`: model implementation and test results for the propossed solution.
- `baselines/`: planned implementation area for classical and learned baseline estimators.

Generated datasets, database files, query run outputs, visualizations, logs, and POL source checkouts are intentionally excluded from Git. POL is treated as an external dependency rather than vendored in this repository.

## AI Assistance

AI assistance, primarily Codex, was used to draft and iterate on parts of the code, scripts, documentation, and repository organization. The repository should therefore be treated like any AI-assisted codebase: important logic, experiments, and results need human review and validation.
