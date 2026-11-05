"""Inference-only serialized size of own-model checkpoints (the 1x memory budget).

A training checkpoint (``model/src/model/checkpoint.py``) holds the model
weights, the schemas and vocabularies needed at inference, and the optimizer
state. The inference-only size is the same payload serialized with
``optimizer_state_dict`` set to ``None`` (as the IMDB contract asks: "report an
inference-only serialized size where available"). Read-only: checkpoints are
never modified; the reduced payload is serialized into memory only.

    python evaluation/trajectory_pol_porto/scripts/own_model_size.py \\
        --checkpoint RUN/checkpoint_step_N.pt [--checkpoint ...] --output sizes.json
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def measure(path: Path) -> Dict[str, Any]:
    import torch

    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # older PyTorch
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict) or "model_state_dict" not in payload:
        raise ValueError(f"{path} is not a ResMADE training checkpoint (no model_state_dict)")
    state = payload["model_state_dict"]
    tensors = [value for value in state.values() if hasattr(value, "numel")]
    reduced = dict(payload)
    reduced["optimizer_state_dict"] = None
    buffer = io.BytesIO()
    torch.save(reduced, buffer)
    trajectory = payload.get("trajectory_distinct") or {}
    metadata = payload.get("metadata") or {}
    return {
        "checkpoint": str(path),
        "checkpoint_bytes": path.stat().st_size,
        "checkpoint_sha256": _sha256(path),
        "has_optimizer_state": payload.get("optimizer_state_dict") is not None,
        "inference_only_bytes": buffer.getbuffer().nbytes,
        "parameters": int(sum(t.numel() for t in tensors)),
        "parameter_bytes": int(sum(t.numel() * t.element_size() for t in tensors)),
        "step": payload.get("step"),
        "epoch": payload.get("epoch"),
        "trajectory_distinct_enabled": bool(trajectory.get("enabled", False)),
        "trajectory_target_semantics_version": payload.get("trajectory_target_semantics_version"),
        "tables": sorted((metadata.get("tables") or {}).keys()) if isinstance(metadata.get("tables"), dict) else None,
        "torch": torch.__version__,
    }


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", action="append", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    rows = [measure(path) for path in args.checkpoint]
    text = json.dumps(rows, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
