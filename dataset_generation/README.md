# Dataset Generation

This directory contains the practical dataset pipeline for the thesis benchmark.

## Contents

- `pol_runs/`: Slurm scripts and run helpers for POL trajectory simulations.
- `mobilitydb_loader/`: scripts for building/running MobilityDB on the CAU cluster and loading POL outputs into PostgreSQL/MobilityDB.

Generated raw POL logs, staging files, database directories, and local POL source mirrors are ignored by Git. The tracked files here are only the reproducible scripts and documentation needed to regenerate or load the benchmark data.

## Cluster Layout

The existing CAU cluster workspace uses:

- Home root: `/zfshome/sunip956/master_thesis_trajectories`
- Work root: `/work_beegfs/sunip956/master_thesis_trajectories`

The repository-side dataset scripts now live below:

```text
dataset_generation/
  mobilitydb_loader/
  pol_runs/
```

When syncing this repository to the cluster, keep that relative structure intact so the Slurm scripts can source `dataset_generation/mobilitydb_loader/common.sh`.
