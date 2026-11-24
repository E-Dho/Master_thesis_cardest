"""Select the training interpreter from the config, and refuse to train without CUDA.

The Slurm scripts start ``model.scripts.train_resmade`` with ``$PYTHON``, which
defaults to the ``geo-mlp`` environment, whose PyTorch is a CPU-only build. The
evaluated POL/Porto models were trained under ``learned-gpu`` (PyTorch 2.5.1 +
CUDA 12.1) by overriding ``PYTHON`` at submission; jobs submitted without the
override trained on the CPU without notice (``device: auto``).

``runtime.python`` in the training config names the interpreter to train with:
``train_resmade`` re-executes itself under it (same arguments, same environment
plus a marker against loops) before anything heavy is imported, so already
queued jobs pick the interpreter up from their checkout's config.
``runtime.require_cuda: true`` stops the run when CUDA is unavailable, instead of
falling back to the CPU.
"""

from __future__ import annotations

import os
import sys
from typing import Any, Sequence

REEXEC_MARKER = "RESMADE_RUNTIME_PYTHON"


def runtime_settings(config: dict[str, Any]) -> tuple[str | None, bool]:
    runtime = config.get("runtime") or {}
    if not isinstance(runtime, dict):
        raise ValueError("runtime must be a mapping")
    python = runtime.get("python")
    python = None if python in (None, "", "null") else os.path.abspath(os.path.expanduser(str(python)))
    require_cuda = runtime.get("require_cuda", False)
    if isinstance(require_cuda, str):
        require_cuda = require_cuda.strip().lower() == "true"
    return python, bool(require_cuda)


def reexec_target(config: dict[str, Any], *, executable: str, environ: dict[str, str]) -> str | None:
    """The interpreter to re-execute under, or None to continue in this process."""

    target, _ = runtime_settings(config)
    if target is None:
        return None
    if os.path.abspath(executable) == target:
        return None
    if environ.get(REEXEC_MARKER) == target:
        raise SystemExit(
            f"runtime.python={target}: already re-executed, but running under {executable}; refusing to loop"
        )
    if not (os.path.isfile(target) and os.access(target, os.X_OK)):
        raise SystemExit(f"runtime.python={target} is not an executable file")
    return target


def maybe_reexec(config: dict[str, Any], module: str, argv: Sequence[str]) -> None:
    target = reexec_target(config, executable=sys.executable, environ=dict(os.environ))
    if target is None:
        return
    print(f"runtime_python_reexec={target} (from {sys.executable})", flush=True)
    environment = dict(os.environ)
    environment[REEXEC_MARKER] = target
    os.execve(target, [target, "-m", module, *argv], environment)


def report_and_check_cuda(config: dict[str, Any]) -> None:
    """Print the interpreter and CUDA state; stop if CUDA is required but missing."""

    import torch

    _, require_cuda = runtime_settings(config)
    available = bool(torch.cuda.is_available())
    name = torch.cuda.get_device_name(0) if available else "none"
    print(
        f"runtime_python={sys.executable} torch={torch.__version__} torch_cuda_build={torch.version.cuda} "
        f"torch_cuda_available={available} cuda_device={name}",
        flush=True,
    )
    if require_cuda and not available:
        raise SystemExit(
            f"runtime.require_cuda=true but CUDA is not available under {sys.executable} "
            f"(torch {torch.__version__}, CUDA build {torch.version.cuda}); refusing to train on the CPU"
        )
