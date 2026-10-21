"""Loading and validating the v3 benchmark configuration."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from query_generation.query_generator import (
    DIMENSIONS,
    INTERVALS,
    RELATIONS,
    Category,
    ConfigError,
    eligible_tables,
    exclusive_attribute_groups,
    attribute_value_dependencies,
    load_config,
    parse_timestamp,
)

from . import SEMANTICS_VERSION

SEGMENT_COLUMNS = ("sx", "sy", "ex", "ey", "ts", "te")


def _sha256_json(payload: Any) -> str:
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class SegmentSpec:
    table: str
    alias: str
    columns: Dict[str, str]
    geometry_expression: str
    spatial_attribute: str
    temporal_attribute: str
    trip_key: str


@dataclass(frozen=True)
class Settings:
    path: Path
    v3: Dict[str, Any]
    base: Dict[str, Any]
    base_path: Path
    grid: Dict[str, int] = field(default_factory=dict)

    # -- identity -----------------------------------------------------------
    @property
    def semantics_version(self) -> str:
        return SEMANTICS_VERSION

    @property
    def sha256(self) -> str:
        """Hash over everything that changes what a query or label means."""

        return _sha256_json({"v3": self.v3, "base": self.base, "grid": self.grid})

    # -- base config access --------------------------------------------------
    @property
    def srid(self) -> int:
        return int(self.base["srid"])

    @property
    def entity_table(self) -> str:
        return str(self.base["entity"]["table"])

    @property
    def entity_expression(self) -> str:
        entity = self.base["entity"]
        if "expression" in entity:
            return str(entity["expression"])
        return f"{self.table(entity['table'])['alias']}.{entity['key']}"

    def table(self, table_id: str) -> Dict[str, Any]:
        return self.base["tables"][table_id]

    def attr(self, table_id: str, name: str) -> Dict[str, Any]:
        for attr in self.table(table_id)["attributes"]:
            if attr["name"] == name:
                return attr
        raise ConfigError(f"unknown attribute {table_id}.{name}")

    def attrs(self, table_ids, dimension: str) -> List[Tuple[str, Dict[str, Any]]]:
        return [
            (table_id, attr)
            for table_id in table_ids
            for attr in self.table(table_id)["attributes"]
            if attr["dimension"] == dimension
        ]

    # -- v3 settings -----------------------------------------------------------
    @property
    def segments(self) -> SegmentSpec:
        spec = self.v3["segments"]
        return SegmentSpec(
            table=spec["table"],
            alias=self.table(spec["table"])["alias"],
            columns=dict(spec["columns"]),
            geometry_expression=self.attr(spec["table"], spec["spatial_attribute"])["expression"],
            spatial_attribute=spec["spatial_attribute"],
            temporal_attribute=spec["temporal_attribute"],
            trip_key=spec["trip_key"],
        )

    @property
    def center_live_probability(self) -> float:
        return float(self.v3["center_live_probability"])

    @property
    def temporal_min_seconds(self) -> float:
        return float(self.v3["temporal_width"]["min_seconds"])

    @property
    def addon_distribution(self) -> Tuple[List[int], List[float]]:
        addons = self.v3["attribute_addons"]
        return [int(v) for v in addons["counts"]], [float(v) for v in addons["probabilities"]]

    @property
    def unbounded_spatial_margin(self) -> float:
        return float(self.v3["unbounded_spatial_margin"])

    @property
    def max_attempts(self) -> int:
        return int(self.v3["max_attempts"])

    @property
    def pool_size(self) -> int:
        return int(self.v3["pools"]["size"])

    @property
    def pool_oversample(self) -> float:
        return float(self.v3["pools"]["oversample"])

    def categories(self) -> List[Category]:
        """All 16 categories in canonical order (dimension, interval, relation)."""

        return [
            Category(dimension, interval, relation)
            for dimension in DIMENSIONS
            for interval in INTERVALS
            for relation in RELATIONS
        ]

    def count_for(self, category: Category) -> int:
        return int(self.grid[category.relation])

    @property
    def total_queries(self) -> int:
        return sum(self.count_for(category) for category in self.categories())


def load_settings(
    path: Path,
    *,
    single_count: Optional[int] = None,
    multi_count: Optional[int] = None,
) -> Settings:
    path = Path(path)
    v3 = json.loads(path.read_text(encoding="utf-8"))
    if v3.get("semantics_version") != SEMANTICS_VERSION:
        raise ConfigError(
            f"{path} declares semantics {v3.get('semantics_version')!r}, expected {SEMANTICS_VERSION!r}"
        )
    base_path = (path.parent / v3["base_config"]).resolve()
    base = load_config(base_path)
    grid = {
        "single": int(single_count if single_count is not None else v3["grid"]["single"]),
        "multi": int(multi_count if multi_count is not None else v3["grid"]["multi"]),
    }
    settings = Settings(path=path, v3=v3, base=base, base_path=base_path, grid=grid)
    validate_settings(settings)
    return settings


def validate_settings(settings: Settings) -> None:
    base = settings.base
    if exclusive_attribute_groups(base) or attribute_value_dependencies(base):
        raise ConfigError("v3 does not support exclusive attribute groups or value dependencies")
    if any(value <= 0 for value in settings.grid.values()):
        raise ConfigError(f"grid counts must be positive: {settings.grid}")

    spec = settings.v3["segments"]
    missing = [key for key in SEGMENT_COLUMNS if key not in spec.get("columns", {})]
    if missing:
        raise ConfigError(f"segments.columns is missing {missing}")
    seg = settings.segments
    if settings.attr(seg.table, seg.spatial_attribute)["type"] != "geometry":
        raise ConfigError("segments.spatial_attribute must be a geometry attribute")
    if settings.attr(seg.table, seg.temporal_attribute)["type"] != "temporal_interval":
        raise ConfigError("segments.temporal_attribute must be a temporal_interval attribute")

    for table_id, table in base["tables"].items():
        for attr in table["attributes"]:
            if attr["type"] == "geometry" and table_id != seg.table:
                raise ConfigError(
                    f"{table_id}.{attr['name']}: v3 allows spatial attributes only on {seg.table}"
                )
            if attr["type"] == "temporal_interval":
                parse_timestamp(attr["domain"]["min"])
                parse_timestamp(attr["domain"]["max"])

    counts, probabilities = settings.addon_distribution
    if len(counts) != len(probabilities) or abs(sum(probabilities) - 1.0) > 1e-9:
        raise ConfigError("attribute_addons counts and probabilities must align and sum to 1")
    if settings.temporal_min_seconds <= 0:
        raise ConfigError("temporal_width.min_seconds must be positive")
    if not 0.0 <= settings.center_live_probability <= 1.0:
        raise ConfigError("center_live_probability must be in [0, 1]")

    for category in settings.categories():
        if not eligible_tables(base, category):
            raise ConfigError(f"category {category.key} has no eligible tables")
