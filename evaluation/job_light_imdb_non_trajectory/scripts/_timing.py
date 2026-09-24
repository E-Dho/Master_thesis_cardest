"""Load the standalone timing guard by file path from any method environment.

Bridges run in isolated interpreters (DeepDB's Python 3.8 stack, the
PyTorch environments) where the evaluation package is not necessarily
importable; ``joblight_eval/timing_guard.py`` depends only on the standard
library, so it is loaded directly from its file.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


MODULE_NAME = "joblight_timing_guard"


def load_timing_guard():
    if MODULE_NAME in sys.modules:
        return sys.modules[MODULE_NAME]
    path = Path(__file__).resolve().parents[1] / "joblight_eval" / "timing_guard.py"
    spec = importlib.util.spec_from_file_location(MODULE_NAME, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[MODULE_NAME] = module
    spec.loader.exec_module(module)
    return module


def session_from_environment(role: str, device: str | None = None):
    """Timing session when a profile-controlled timing run is active, else None."""
    session = load_timing_guard().TimingSession.from_environment(role=role)
    if session is not None and device is not None and session.profile.device != device:
        raise ValueError(
            f"timing profile {session.profile.profile_id} measures on "
            f"{session.profile.device}, but the command requested device {device}"
        )
    return session


def measured(session, label: str = "timed_region"):
    """Context manager measuring ``label`` when a session is active."""
    import contextlib

    return session.measure(label) if session is not None else contextlib.nullcontext()


def timing_helper():  # pragma: no cover - trivial accessor used by bridges
    return sys.modules[__name__]
