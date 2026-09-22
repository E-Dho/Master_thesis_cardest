from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


@dataclass(frozen=True)
class CommandResult:
    command: tuple[str, ...]
    returncode: int
    wall_seconds: float
    peak_rss_bytes: int | None


def run_logged_command(
    command: str | list[str],
    *,
    cwd: Path,
    env: Mapping[str, str | None],
    stdout_path: Path,
    stderr_path: Path,
) -> CommandResult:
    argv = shlex.split(command) if isinstance(command, str) else list(command)
    if not argv:
        raise ValueError("command must not be empty")
    merged_env = os.environ.copy()
    for key, value in env.items():
        if value is None:
            merged_env.pop(str(key), None)
        else:
            merged_env[str(key)] = str(value)
    wrapper = Path(__file__).with_name("_resource_wrapper.py")
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    stderr_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix="joblight-resource-",
        suffix=".json",
        dir=stdout_path.parent,
        delete=False,
    ) as metrics_handle:
        metrics_path = Path(metrics_handle.name)
    start = time.perf_counter()
    with stdout_path.open("a", encoding="utf-8") as stdout, stderr_path.open(
        "a", encoding="utf-8"
    ) as stderr:
        completed = subprocess.run(
            [sys.executable, str(wrapper), str(metrics_path), *argv],
            cwd=cwd,
            env=merged_env,
            stdout=stdout,
            stderr=stderr,
            check=False,
        )
    elapsed = time.perf_counter() - start
    if completed.returncode:
        metrics_path.unlink(missing_ok=True)
        raise RuntimeError(
            f"resource wrapper failed with exit code {completed.returncode}; "
            f"see {stderr_path}"
        )
    try:
        measurement = json.loads(metrics_path.read_text(encoding="utf-8"))
    finally:
        metrics_path.unlink(missing_ok=True)
    returncode = int(measurement["returncode"])
    peak = int(measurement["peak_rss_bytes"])
    if returncode:
        raise RuntimeError(
            f"command failed with exit code {returncode}; see {stderr_path}"
        )
    return CommandResult(tuple(argv), returncode, elapsed, peak)
