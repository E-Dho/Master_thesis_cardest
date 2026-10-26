"""Trajectory (POL / Porto) pipeline plugin for :mod:`evaluation.eval_core.plugins`.

Adapters are added here as they are implemented (P3 onwards).
"""

from __future__ import annotations

from .strata import TRAJECTORY_SUMMARY_EXTENSION
from .workloads import TRAJ_V3_JSONL_FORMAT

ADAPTERS: dict = {}
WORKLOAD_FORMATS = {TRAJ_V3_JSONL_FORMAT.name: TRAJ_V3_JSONL_FORMAT}
IN_PROCESS_TIMING_ADAPTERS: tuple = ()
SUMMARY_EXTENSIONS = {TRAJECTORY_SUMMARY_EXTENSION.schema: TRAJECTORY_SUMMARY_EXTENSION}
