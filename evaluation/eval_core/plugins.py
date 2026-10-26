"""Pipeline plugins: which adapters and workload formats exist.

The core knows no dataset. Each evaluation pipeline ships one plugin module
that exposes three module-level names:

``ADAPTERS``
    ``{adapter_type: Adapter subclass}``; the value may also be a zero-argument
    callable returning the class, so heavy imports stay lazy. ``external``, the
    generic subprocess adapter, is built into the core.
``WORKLOAD_FORMATS``
    ``{format_name: WorkloadFormat instance}``.
``IN_PROCESS_TIMING_ADAPTERS``
    adapter types that measure latency in-process and therefore need no
    ``adapter.timing_commands``.
``SUMMARY_EXTENSIONS`` (optional)
    ``{schema: handler}`` for the ``extension`` block a workload format may add
    to ``summary.json``; the handler aggregates it across seeds
    (``aggregate``), turns it into table rows (``rows``) and Markdown
    (``markdown``).

Plugins are imported lazily on first use, so importing the core never pulls in
a pipeline (and the IMDB pipeline keeps working without the trajectory one).
A name defined by two plugins is an error rather than a silent override.
"""

from __future__ import annotations

import importlib
from functools import lru_cache
from typing import Any

PLUGIN_MODULES = (
    "evaluation.job_light_imdb_non_trajectory.joblight_eval.plugin",
    "evaluation.trajectory_pol_porto.traj_eval.plugin",
)


@lru_cache(maxsize=None)
def _plugins() -> tuple[Any, ...]:
    modules = []
    for name in PLUGIN_MODULES:
        try:
            modules.append(importlib.import_module(name))
        except ModuleNotFoundError as exc:
            # Only a missing plugin package is tolerated; an import error inside
            # an existing plugin must surface.
            if exc.name is None or not name.startswith(exc.name):
                raise
    return tuple(modules)


def _builtin_adapters() -> dict[str, Any]:
    from .adapters.external import ExternalCommandAdapter

    return {"external": ExternalCommandAdapter}


def _merged(attribute: str) -> dict[str, Any]:
    merged: dict[str, Any] = _builtin_adapters() if attribute == "ADAPTERS" else {}
    owners: dict[str, str] = {key: "evaluation.eval_core" for key in merged}
    for module in _plugins():
        for key, value in getattr(module, attribute, {}).items():
            if key in merged:
                raise ValueError(
                    f"{attribute} entry {key!r} is defined by both {owners[key]} and "
                    f"{module.__name__}"
                )
            merged[key] = value
            owners[key] = module.__name__
    return merged


def adapter_class(adapter_type: str) -> type:
    adapters = _merged("ADAPTERS")
    try:
        value = adapters[adapter_type]
    except KeyError as exc:
        raise ValueError(f"unknown adapter type {adapter_type!r}") from exc
    return value if isinstance(value, type) else value()


def workload_format(name: str) -> Any:
    formats = _merged("WORKLOAD_FORMATS")
    try:
        return formats[name]
    except KeyError as exc:
        raise ValueError(
            f"unknown workload format {name!r}; known: {sorted(formats)}"
        ) from exc


def in_process_timing_adapters() -> frozenset[str]:
    names: set[str] = set()
    for module in _plugins():
        names.update(getattr(module, "IN_PROCESS_TIMING_ADAPTERS", ()))
    return frozenset(names)


def summary_extension_handler(schema: str) -> Any:
    handlers = _merged("SUMMARY_EXTENSIONS")
    try:
        return handlers[schema]
    except KeyError as exc:
        raise ValueError(f"no plugin handles summary extension schema {schema!r}") from exc
