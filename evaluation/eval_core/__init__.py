"""Dataset-agnostic evaluation core shared by the IMDB and trajectory pipelines.

Each pipeline contributes adapters and workload formats through
:mod:`evaluation.eval_core.plugins`; everything else (configs, run
directories, metrics, timing profiles, aggregation, comparison) is common.
"""
