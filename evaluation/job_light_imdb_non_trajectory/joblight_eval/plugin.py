"""IMDB / JOB-light pipeline plugin for :mod:`evaluation.eval_core.plugins`."""

from __future__ import annotations

from .adapters.registry import ADAPTERS
from .workloads import JOBLIGHT_CSV_FORMAT

WORKLOAD_FORMATS = {JOBLIGHT_CSV_FORMAT.name: JOBLIGHT_CSV_FORMAT}
IN_PROCESS_TIMING_ADAPTERS = ("postgres", "foj_sampling")
