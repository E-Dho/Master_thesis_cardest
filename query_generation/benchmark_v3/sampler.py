"""Deterministic per-slot query sampling (SEMANTICS.md, sections 3 and 5)."""

from __future__ import annotations

import hashlib
import math
import random
from datetime import datetime, timedelta
from typing import Any, Dict, List, Sequence, Tuple

from query_generation.query_generator import (
    Category,
    ConfigError,
    eligible_tables,
    is_ordered,
    parse_timestamp,
)

from . import SEMANTICS_VERSION
from .config import Settings
from .pools import Pools
from .query_model import (
    from_clause,
    label_kind,
    label_sql,
    predicate_hash,
    predicate_sql,
    timestamp_text,
)


def slot_seed(base_seed: int, category_key: str, index: int, attempt: int) -> int:
    payload = f"v3:{base_seed}:{category_key}:{index}:{attempt}".encode("utf-8")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big")


def slot_id(category_key: str, index: int) -> str:
    return f"{category_key}#{index:04d}"


def quotas(total: int, probabilities: Sequence[float]) -> List[int]:
    """Largest-remainder rounding of ``total * p`` (ties go to the earlier entry)."""

    raw = [total * p for p in probabilities]
    floors = [int(math.floor(value)) for value in raw]
    missing = total - sum(floors)
    order = sorted(range(len(raw)), key=lambda i: (-(raw[i] - floors[i]), i))
    for i in order[:missing]:
        floors[i] += 1
    return floors


class Sampler:
    def __init__(self, settings: Settings, pools: Pools, base_seed: int):
        self.settings = settings
        self.pools = pools
        self.base_seed = int(base_seed)
        self._addon_targets: Dict[str, Dict[int, int]] = {}

    def addon_target(self, category: Category, index: int) -> int:
        """Add-on count of a slot, fixed across attempts.

        The slots of a category are permuted with a seeded RNG and assigned
        add-on counts by exact quotas, so the finished benchmark has exactly
        the configured split even though candidates with add-ons are rejected
        as empty more often.
        """

        if category.relation != "multi" or category.dimension == "standard":
            return 0
        if category.key not in self._addon_targets:
            total = self.settings.count_for(category)
            counts, probabilities = self.settings.addon_distribution
            order = random.Random(f"v3-addons:{self.base_seed}:{category.key}").sample(range(total), total)
            targets: Dict[int, int] = {}
            position = 0
            for count, quota in zip(counts, quotas(total, probabilities)):
                for slot in order[position : position + quota]:
                    targets[slot] = count
                position += quota
            self._addon_targets[category.key] = targets
        return self._addon_targets[category.key][index]

    # -- public ------------------------------------------------------------------
    def candidate(self, category: Category, index: int, attempt: int) -> Dict[str, Any]:
        if not 0 <= index < self.settings.count_for(category):
            raise ConfigError(f"slot {index} is outside the grid of {category.key}")
        seed = slot_seed(self.base_seed, category.key, index, attempt)
        rng = random.Random(seed)
        tables = self._choose_tables(rng, category)
        core = self._core_components(rng, category, tables)
        unbounded = self._choose_unbounded(rng, category, core)
        predicates = [
            self._render_core(rng, component, unbounded.get(position, set()))
            for position, component in enumerate(core)
        ]
        target = self.addon_target(category, index)
        addons = self._addons(rng, tables, target)
        for predicate in predicates:
            predicate["role"] = "core"
        predicates.extend(addons)
        for predicate in predicates:
            predicate["sql"] = predicate_sql(self.settings, predicate)

        _, joins = from_clause(self.settings, tables)
        return {
            "semantics_version": SEMANTICS_VERSION,
            "slot_id": slot_id(category.key, index),
            "category": {
                "dimension": category.dimension,
                "interval": category.interval,
                "relation": category.relation,
                "key": category.key,
                "category_index": int(index),
            },
            "tables": list(tables),
            "joins": joins,
            "predicates": predicates,
            "core_predicates": len(predicates) - len(addons),
            "addon_predicates": len(addons),
            "addon_target": target,
            "label_kind": label_kind(category.relation),
            "sql": label_sql(self.settings, category.relation, tables, predicates),
            "predicate_hash": predicate_hash(category.key, tables, predicates),
            "slot": {
                "base_seed": self.base_seed,
                "index": int(index),
                "attempt": int(attempt),
                "seed": seed,
            },
            "config_sha256": self.settings.sha256,
            "pools_sha256": self.pools.sha256,
            "grid": dict(self.settings.grid),
        }

    # -- tables and components -----------------------------------------------------
    def _choose_tables(self, rng: random.Random, category: Category) -> Tuple[str, ...]:
        candidates = eligible_tables(self.settings.base, category)
        if not candidates:
            raise ConfigError(f"no eligible tables for {category.key}")
        if category.relation == "single":
            return tuple(rng.choice(candidates))
        sizes = sorted({len(candidate) for candidate in candidates})
        target = rng.randint(2, len(self.settings.base["tables"]))
        sized = [candidate for candidate in candidates if len(candidate) == target]
        if not sized:
            nearest = min(sizes, key=lambda size: abs(size - target))
            sized = [candidate for candidate in candidates if len(candidate) == nearest]
        return tuple(rng.choice(sized))

    def _core_components(self, rng, category: Category, tables) -> List[Tuple[str, str, Dict[str, Any]]]:
        seg = self.settings.segments
        dimension = category.dimension
        if dimension == "standard":
            attrs = self.settings.attrs(tables, "standard")
            for _ in range(200):
                count = rng.randint(1, len(attrs))
                selected = rng.sample(attrs, count)
                if category.interval == "unbounded" and not any(is_ordered(attr) for _, attr in selected):
                    continue
                return [("standard", table, attr) for table, attr in selected]
            raise ConfigError(f"cannot sample standard attributes for {category.key} on {tables}")
        if dimension == "temporal":
            table, attr = rng.choice(self.settings.attrs(tables, "temporal"))
            return [("temporal", table, attr)]
        if seg.table not in tables:
            raise ConfigError(f"{category.key} requires {seg.table} but got {tables}")
        geometry = self.settings.attr(seg.table, seg.spatial_attribute)
        if dimension == "spatial":
            return [("spatial", seg.table, geometry)]
        if dimension == "spatio_temporal":
            return [("spatio_temporal", seg.table, geometry)]
        raise ConfigError(f"unknown dimension {dimension!r}")

    def _choose_unbounded(self, rng, category: Category, core) -> Dict[int, set]:
        """Which ordered parts are one-sided; same rule as the v2 generator."""

        if category.interval != "unbounded":
            return {}
        parts: List[Tuple[int, str]] = []
        for position, (kind, _, attr) in enumerate(core):
            if kind == "spatio_temporal":
                parts.extend([(position, "spatial"), (position, "temporal")])
            elif kind in {"temporal", "spatial"} or is_ordered(attr):
                parts.append((position, kind))
        if not parts:
            raise ConfigError(f"{category.key} needs an ordered component")
        if rng.random() < 0.5:
            chosen = set(parts)
        else:
            first = rng.choice(parts)
            chosen = {first}
            for part in parts:
                if part != first and rng.random() < 0.5:
                    chosen.add(part)
        result: Dict[int, set] = {}
        for position, part in chosen:
            result.setdefault(position, set()).add(part)
        return result

    # -- rendering -------------------------------------------------------------------
    def _render_core(self, rng, component, unbounded_parts: set) -> Dict[str, Any]:
        kind, table, attr = component
        if kind == "standard":
            return self._scalar(rng, table, attr, unbounded=bool(unbounded_parts))
        if kind == "temporal":
            center, source = self._temporal_center(rng, table, attr)
            predicate = self._base(table, attr, "temporal", center_source=source)
            predicate.update(self._interval(rng, attr, center, "temporal" in unbounded_parts))
            return predicate
        if kind == "spatial":
            center, source = self._spatial_center(rng)
            predicate = self._base(table, attr, "spatial", center_source=source)
            predicate.update(self._window(rng, center, "spatial" in unbounded_parts))
            return predicate
        if kind == "spatio_temporal":
            (point, moment), source = self._joint_center(rng)
            predicate = self._base(table, attr, "spatio_temporal", center_source=source)
            window = self._window(rng, point, "spatial" in unbounded_parts)
            interval = self._interval(
                rng,
                self.settings.attr(table, self.settings.segments.temporal_attribute),
                moment,
                "temporal" in unbounded_parts,
            )
            predicate.update(window)
            predicate.update(
                {
                    "lower": interval["lower"],
                    "upper": interval["upper"],
                    "temporal_mode": interval["mode"],
                    "temporal_width_source": interval.get("width_source"),
                    "temporal_width_seconds": interval.get("width_seconds"),
                    "spatial_mode": window["mode"],
                    "mode": "unbounded" if unbounded_parts else "range",
                }
            )
            return predicate
        raise ConfigError(f"unknown component {kind!r}")

    def _base(self, table: str, attr: Dict[str, Any], kind: str, **extra: Any) -> Dict[str, Any]:
        predicate = {
            "kind": kind,
            "table": table,
            "attribute": attr["name"],
            "dimension": attr["dimension"],
            "type": attr["type"],
        }
        predicate.update(extra)
        return predicate

    def _scalar(self, rng, table, attr, *, unbounded: bool, role_addon: bool = False) -> Dict[str, Any]:
        if attr["type"] == "nominal":
            value = rng.choice(list(attr["values"]))
            return self._base(table, attr, "nominal_eq", value=value, mode="equality")
        center, source = self._scalar_center(rng, table, attr)
        if unbounded and not role_addon:
            op = rng.choice(("<", "<=", ">", ">="))
            value = math.floor(center) if attr["type"] == "integer" else center
            return self._base(
                table, attr, "scalar_unbounded", operator=op, value=value,
                mode="unbounded", center_source=source,
            )
        domain = attr["domain"]
        width, width_source = sample_width(rng, float(domain["max"]) - float(domain["min"]))
        lower = max(float(domain["min"]), center - width / 2.0)
        upper = min(float(domain["max"]), center + width / 2.0)
        if attr["type"] == "integer":
            lower, upper = math.floor(lower), math.ceil(upper)
        return self._base(
            table, attr, "scalar_range", lower=lower, upper=upper, mode="range",
            center_source=source, width_source=width_source,
        )

    def _addons(self, rng, tables, wanted: int) -> List[Dict[str, Any]]:
        if wanted == 0:
            return []
        candidates = self.settings.attrs(tables, "standard")
        if len(candidates) < wanted:
            raise ConfigError(f"{wanted} add-ons requested but {tables} offer {len(candidates)} standard attributes")
        addons = []
        for table, attr in rng.sample(candidates, wanted):
            predicate = self._scalar(rng, table, attr, unbounded=False, role_addon=True)
            predicate["role"] = "addon"
            addons.append(predicate)
        return addons

    # -- centers ---------------------------------------------------------------------
    def _live(self, rng) -> bool:
        return rng.random() < self.settings.center_live_probability

    def _scalar_center(self, rng, table, attr) -> Tuple[float, str]:
        values = self.pools.scalar_values(table, attr["name"])
        if values and self._live(rng):
            return float(rng.choice(values)), "pool_row"
        domain = attr["domain"]
        return rng.uniform(float(domain["min"]), float(domain["max"])), "domain"

    def _temporal_center(self, rng, table, attr) -> Tuple[datetime, str]:
        intervals = self.pools.intervals(table, attr["name"], self.settings.segments.table)
        if intervals and self._live(rng):
            start, end = rng.choice(intervals)
            return start + (end - start) * rng.random(), "pool_row"
        return self._domain_time(rng), "domain"

    def _spatial_center(self, rng) -> Tuple[Tuple[float, float], str]:
        segments = self.pools.segments(self.settings.segments.table)
        if segments and self._live(rng):
            sx, sy, ex, ey, _, _ = rng.choice(segments)
            u = rng.random()
            return (sx + u * (ex - sx), sy + u * (ey - sy)), "pool_row"
        return self._domain_point(rng), "domain"

    def _joint_center(self, rng) -> Tuple[Tuple[Tuple[float, float], datetime], str]:
        """One pool segment at one u gives a place and the time an object was there."""

        segments = self.pools.segments(self.settings.segments.table)
        if segments and self._live(rng):
            sx, sy, ex, ey, ts, te = rng.choice(segments)
            u = rng.random()
            point = (sx + u * (ex - sx), sy + u * (ey - sy))
            return (point, ts + (te - ts) * u), "pool_row_joint"
        return (self._domain_point(rng), self._domain_time(rng)), "domain"

    def _geometry_domain(self) -> Dict[str, float]:
        seg = self.settings.segments
        return {key: float(value) for key, value in self.settings.attr(seg.table, seg.spatial_attribute)["domain"].items()}

    def _time_domain(self) -> Tuple[datetime, datetime]:
        seg = self.settings.segments
        domain = self.settings.attr(seg.table, seg.temporal_attribute)["domain"]
        return parse_timestamp(domain["min"]), parse_timestamp(domain["max"])

    def _domain_point(self, rng) -> Tuple[float, float]:
        d = self._geometry_domain()
        return rng.uniform(d["min_x"], d["max_x"]), rng.uniform(d["min_y"], d["max_y"])

    def _domain_time(self, rng) -> datetime:
        lower, upper = self._time_domain()
        return lower + timedelta(seconds=rng.uniform(0.0, (upper - lower).total_seconds()))

    # -- ranges ------------------------------------------------------------------------
    def _interval(self, rng, attr, center: datetime, unbounded: bool) -> Dict[str, Any]:
        domain_min = parse_timestamp(attr["domain"]["min"])
        domain_max = parse_timestamp(attr["domain"]["max"])
        center = min(max(center, domain_min), domain_max)
        if unbounded:
            if rng.random() < 0.5:
                return {"lower": None, "upper": timestamp_text(center), "mode": "unbounded"}
            return {"lower": timestamp_text(center), "upper": None, "mode": "unbounded"}
        span = (domain_max - domain_min).total_seconds()
        width = log_uniform(rng, self.settings.temporal_min_seconds, span)
        lower = max(domain_min, center - timedelta(seconds=width / 2.0))
        upper = min(domain_max, center + timedelta(seconds=width / 2.0))
        return {
            "lower": timestamp_text(lower),
            "upper": timestamp_text(upper),
            "mode": "range",
            "width_source": "log_uniform",
            "width_seconds": width,
        }

    def _window(self, rng, center: Tuple[float, float], unbounded: bool) -> Dict[str, Any]:
        d = self._geometry_domain()
        cx = min(max(center[0], d["min_x"]), d["max_x"])
        cy = min(max(center[1], d["min_y"]), d["max_y"])
        if unbounded:
            margin = self.settings.unbounded_spatial_margin
            bounds = {
                "min_x": d["min_x"] - margin,
                "min_y": d["min_y"] - margin,
                "max_x": d["max_x"] + margin,
                "max_y": d["max_y"] + margin,
            }
            side = rng.choice(("x_lt", "x_gt", "y_lt", "y_gt"))
            bounds[{"x_lt": "max_x", "x_gt": "min_x", "y_lt": "max_y", "y_gt": "min_y"}[side]] = (
                cx if side.startswith("x") else cy
            )
            bounds.update({"mode": "unbounded", "unbounded_side": side})
            return bounds
        width = log_uniform(rng, self.settings.spatial_min_width, d["max_x"] - d["min_x"])
        height = log_uniform(rng, self.settings.spatial_min_width, d["max_y"] - d["min_y"])
        return {
            "min_x": max(d["min_x"], cx - width / 2.0),
            "min_y": max(d["min_y"], cy - height / 2.0),
            "max_x": min(d["max_x"], cx + width / 2.0),
            "max_y": min(d["max_y"], cy + height / 2.0),
            "mode": "range",
            "width_source": "log_uniform",
            "width_x_meters": width,
            "width_y_meters": height,
        }


def log_uniform(rng: random.Random, low: float, high: float) -> float:
    """exp(U(log low, log high)); the full range collapses to ``low`` if high < low."""

    high = max(high, low)
    return math.exp(rng.uniform(math.log(low), math.log(high)))


def sample_width(rng: random.Random, size: float) -> Tuple[float, str]:
    """v2 width rule for scalar ranges: 50 % uniform over the domain, 50 % exponential (mean size/10)."""

    if size <= 0:
        raise ConfigError("domain size must be positive")
    if rng.random() < 0.5:
        return max(size * 1e-9, rng.uniform(0, size)), "uniform"
    return min(size, max(size * 1e-9, rng.expovariate(10.0 / size))), "exponential"

