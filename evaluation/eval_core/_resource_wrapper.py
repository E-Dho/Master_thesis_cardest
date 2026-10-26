#!/usr/bin/env python3
"""Run one command in a fresh process and record that command's child usage."""

from __future__ import annotations

import json
import os
import resource
import subprocess
import sys
from pathlib import Path


def main() -> int:
    if len(sys.argv) < 3:
        raise SystemExit("usage: _resource_wrapper.py METRICS_PATH COMMAND [ARG ...]")
    metrics_path = Path(sys.argv[1])
    completed = subprocess.run(sys.argv[2:], check=False)
    usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    scale = 1 if os.uname().sysname == "Darwin" else 1024
    metrics_path.write_text(
        json.dumps(
            {
                "returncode": completed.returncode,
                "peak_rss_bytes": int(usage.ru_maxrss * scale),
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
