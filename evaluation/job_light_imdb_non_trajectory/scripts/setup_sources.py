from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from model.src.config import load_simple_yaml


def main() -> int:
    parser = argparse.ArgumentParser(description="Clone pinned JOB-light baseline sources")
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--source", action="append", help="limit setup to a source name")
    args = parser.parse_args()

    payload = load_simple_yaml(args.lock)
    if int(payload.get("schema_version", 0)) != 1:
        raise ValueError("unsupported source lock schema")
    sources = payload.get("sources")
    if not isinstance(sources, dict):
        raise ValueError("sources must be a mapping")
    selected = set(args.source or sources)
    unknown = selected - set(sources)
    if unknown:
        raise ValueError(f"unknown sources: {sorted(unknown)}")
    args.destination.mkdir(parents=True, exist_ok=True)
    for name in sorted(selected):
        spec = sources[name]
        target = args.destination / name
        if not target.exists():
            subprocess.run(["git", "clone", spec["url"], str(target)], check=True)
        subprocess.run(["git", "-C", str(target), "fetch", "origin", spec["commit"]], check=True)
        subprocess.run(["git", "-C", str(target), "checkout", "--detach", spec["commit"]], check=True)
        observed = subprocess.check_output(
            ["git", "-C", str(target), "rev-parse", "HEAD"], text=True
        ).strip()
        if observed != spec["commit"]:
            raise RuntimeError(f"{name}: expected {spec['commit']}, observed {observed}")
        print(f"{name}: {observed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
