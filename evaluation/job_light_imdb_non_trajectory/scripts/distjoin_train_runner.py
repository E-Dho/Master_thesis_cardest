#!/usr/bin/env python3
"""Invoke pinned DistJoin while allowing a smoke-only table subset."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--exp-mark", required=True)
    parser.add_argument("--tables", nargs="*")
    args = parser.parse_args()
    source = args.source_root.resolve()
    sys.path.insert(0, str(source))
    sys.argv = [str(source / "train.py"), "--config", "IMDB", "--exp_mark", args.exp_mark]
    import train

    _install_missing_tesseract_transformer(train)
    _configure_train_module(train, Path.cwd() / "Configs" / "IMDB" / "IMDB.yaml", args.exp_mark)

    if args.tables:
        original = train.JoinOrderBenchmark.GetJobLightJoinKeys()
        selected = {table: original[table] for table in args.tables}
        train.JoinOrderBenchmark.GetJobLightJoinKeys = staticmethod(lambda: selected)
    train.TrainTask(seed=train.config_seed)
    return 0


def _configure_train_module(train, config_path: Path, experiment_mark: str) -> None:
    """Initialize globals that upstream defines only in its __main__ block."""

    raw_config = _yaml_module().safe_load(config_path.read_text(encoding="utf-8"))
    train.args = argparse.Namespace(config="IMDB", exp_mark=experiment_mark)
    train.config_name = "IMDB"
    train.raw_config = raw_config
    train.config_seed = int(raw_config["seed"])
    train.config_excludes = raw_config["excludes"]
    train.config = raw_config["train"]


def _yaml_module():
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("DistJoin runner requires PyYAML") from exc
    return yaml


def _install_missing_tesseract_transformer(module) -> None:
    """Repair an optional upstream class reference left undefined in IMDB mode."""

    if not hasattr(module, "TesseractTransformer"):
        module.TesseractTransformer = type("TesseractTransformer", (), {})


if __name__ == "__main__":
    raise SystemExit(main())
