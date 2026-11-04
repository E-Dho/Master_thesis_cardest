"""Trajectory (POL / Porto) pipeline plugin for :mod:`evaluation.eval_core.plugins`."""

from __future__ import annotations

from .strata import TRAJECTORY_SUMMARY_EXTENSION
from .workloads import TRAJ_V3_JSONL_FORMAT

def _exact_mbr():
    from .adapters.exact_mbr import ExactMbrAdapter

    return ExactMbrAdapter


def _trajectory_sampling():
    from .adapters.sampling import TrajectorySamplingAdapter

    return TrajectorySamplingAdapter


def _mobilitydb():
    from .adapters.mobilitydb import MobilityDbAdapter

    return MobilityDbAdapter


def _mscn():
    from .adapters.mscn import MscnAdapter

    return MscnAdapter


def _trahist():
    from .adapters.trahist import TraHistAdapter

    return TraHistAdapter


def _case_plus():
    from .adapters.case_plus import CasePlusAdapter

    return CasePlusAdapter


ADAPTERS = {"traj_exact_mbr": _exact_mbr, "traj_sampling": _trajectory_sampling,
            "traj_mobilitydb": _mobilitydb, "traj_mscn": _mscn, "traj_trahist": _trahist,
            "traj_case_plus": _case_plus}
WORKLOAD_FORMATS = {TRAJ_V3_JSONL_FORMAT.name: TRAJ_V3_JSONL_FORMAT}
IN_PROCESS_TIMING_ADAPTERS = ("traj_exact_mbr", "traj_sampling", "traj_mobilitydb", "traj_mscn", "traj_trahist",
                              "traj_case_plus")
SUMMARY_EXTENSIONS = {TRAJECTORY_SUMMARY_EXTENSION.schema: TRAJECTORY_SUMMARY_EXTENSION}
