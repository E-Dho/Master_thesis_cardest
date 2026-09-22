# Experiment configurations

`experiment.template.yaml` documents the versioned interface. Copy it to a
method/variant-specific config, keep the identity slugs stable, and set an
arbitrary human-readable `display_name`.

The native PostgreSQL and FOJ adapters consume their settings directly.
MSCN, DeepDB, NeuroCard, DistJoin, and the project's model execute through the
subprocess contract because their Python/PyTorch environments conflict. Their
`prepare_command`, `smoke_command`, `build_command`, and `evaluate_command`
are supplied by the environment-specific launcher. The common runner exports
the seed and timing policy and substitutes these fields:

```text
{seed} {run_directory} {exchange_directory} {workload_id} {query_limit}
{queries_csv} {queries_sql} {predictions_csv} {latency_csv}
{method_id} {variant_id}
```

The evaluation subprocess must write `query_id,estimated_cardinality` and may
write `status,diagnostic`. It writes timing observations as
`query_id,repetition,latency_ms,scope`. Missing prediction rows are converted
to explicit failures. An optional `artifact_manifest.json` reports parameter
count, inference-only serialized size, full checkpoint size, and framework
memory measurements.

Every long external build should set `require_smoke_before_build: true`. The
adapter then refuses to build unless a successful `smoke_command` is present.
Published budgets and immutable source revisions are recorded in
`baseline_recipes.yaml` and `../sources.lock.yaml`.

NeuroCard is fully instantiated by `neurocard_job_light.yaml` and
`neurocard_job_light_ranges.yaml`; `neurocard_recipe.yaml.example` remains a
minimal recipe reference. The two concrete variants train separate native
ResMADE models because upstream JOB-light and JOB-light-ranges use different
column projections and factorization widths.

Run their `prepare` and `smoke` stages with `../slurm/adapter_smoke.sbatch`,
then pass the resulting run directory to `../slurm/neurocard_build.sbatch`.
The latter refuses to start without CUDA and cannot accidentally execute a
production build on a CPU-only `base` node.

`../slurm/gpu_build.sbatch` provides the same staged CUDA guard for other
learned baselines, including MSCN and DistJoin.

Environment specifications live in `../environments`. They intentionally do
not install DeepDB's 2019 dependency lock into the modern shared environment,
and they do not replace the shared PyTorch build used to compile DistJoin's
native sampler.

The published normal-workload MSCN baseline is fully instantiated by
`mscn_job_light.yaml`. Its bridge retains the upstream `SetConv`, feature
encoding, 100k-query workload, 1,000 materialized sample bits, and Q-error
training objective while adding deterministic seeds, checkpoints, timing, and
resource artifacts. The separately adapted ranges model remains isolated from
this native recipe.

`postgres_16_10_smoke.yaml` and `foj_sampling_smoke.yaml` are disposable,
row/sample-limited end-to-end fixtures for cluster validation. They are not
headline baseline configurations.
