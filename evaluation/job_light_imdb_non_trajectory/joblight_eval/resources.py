from __future__ import annotations

import os
import resource
import shlex
import subprocess
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
    # Snapshot RUSAGE_CHILDREN before the subprocess so we capture only this
    # command's contribution.  ru_maxrss is a session-cumulative high-water mark,
    # not a per-process value, so we compute a delta.
    before = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    start = time.perf_counter()
    with stdout_path.open("a", encoding="utf-8") as stdout, stderr_path.open(
        "a", encoding="utf-8"
    ) as stderr:
        completed = subprocess.run(
            argv,
            cwd=cwd,
            env=merged_env,
            stdout=stdout,
            stderr=stderr,
            check=False,
        )
    elapsed = time.perf_counter() - start
    after = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    # macOS reports bytes; Linux reports KiB.
    is_darwin = os.uname().sysname == "Darwin"
    scale = 1 if is_darwin else 1024
    # Delta gives a per-command approximation; may undercount if the session-
    # wide high-water mark was already set by an earlier, larger subprocess.
    peak = int(max(after - before, 0) * scale) or None
    if completed.returncode:
        raise RuntimeError(
            f"command failed with exit code {completed.returncode}; see {stderr_path}"
        )
    return CommandResult(tuple(argv), completed.returncode, elapsed, peak)
