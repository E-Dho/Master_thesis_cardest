"""v3 trajectory workloads (``benchmark.jsonl``) for the evaluation core.

A v3 benchmark file holds both label kinds: single-table queries labelled with
``COUNT(*)`` (``label_kind = row_count``) and multi-table queries labelled with
``COUNT(DISTINCT trip_id)`` (``distinct_trips``). A workload of this pipeline is
one *selection* of that file, e.g. ``select: {relation: multi}``, so the two
label kinds are scored and reported separately and never pooled.

Query identity: ``query_id`` is the 0-based position of a row within the
selection, in file order. That keeps the core's integer contract; the v3
``query_id``, ``slot_id`` and ``predicate_hash`` stay on every record and in
:func:`write_query_index`. Bridges must select with :func:`select_rows` so they
see exactly the same ids.

Everything is read from the structured predicates; the rendered SQL is never
parsed (v3 rule, ``query_generation/benchmark_v3/SEMANTICS.md``).

Import-time dependencies are the standard library only, so legacy bridge
environments can import :func:`select_rows`.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

if TYPE_CHECKING:  # pragma: no cover
    from evaluation.eval_core.config import WorkloadConfig

FORMAT_NAME = "traj_v3_jsonl"
SUPPORTED_SEMANTICS = ("pol_exact_st_v3",)
#: keys a ``select`` mapping may use, and where each value lives in a row
SELECT_KEYS = {
    "relation": ("category", "relation"),
    "dimension": ("category", "dimension"),
    "interval": ("category", "interval"),
    "category": ("category", "key"),
    "label_kind": ("label_kind",),
}
LABEL_FIELD = {"row_count": "join_cardinality", "distinct_trips": "entity_cardinality"}
SPATIAL_KINDS = ("spatial", "spatio_temporal")
TEMPORAL_KINDS = ("temporal", "spatio_temporal")


class WorkloadError(ValueError):
    """The workload file or its selection violates the v3 contract."""


@dataclass(frozen=True)
class TrajQueryRecord:
    workload: str
    query_id: int
    true_cardinality: int
    source_query_id: str
    slot_id: str
    predicate_hash: str
    category: str
    dimension: str
    interval: str
    relation: str
    label_kind: str
    tables: Tuple[str, ...]
    predicates: Tuple[Mapping[str, Any], ...]
    addon_count: int
    features: Mapping[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# selection
# ---------------------------------------------------------------------------


def normalize_select(select: Optional[Mapping[str, Any]]) -> Tuple[Tuple[str, Any], ...]:
    """Validated, order-independent form of a ``select`` mapping."""

    if not select:
        return ()
    if not isinstance(select, Mapping):
        raise WorkloadError(f"select must be a mapping, got {select!r}")
    unknown = sorted(set(select) - set(SELECT_KEYS))
    if unknown:
        raise WorkloadError(f"unknown select keys {unknown}; allowed: {sorted(SELECT_KEYS)}")
    pairs = []
    for key in sorted(select):
        value = select[key]
        if isinstance(value, (list, tuple)):
            value = tuple(str(item) for item in value)
        else:
            value = str(value)
        pairs.append((key, value))
    return tuple(pairs)


def _row_value(row: Mapping[str, Any], key: str) -> Any:
    value: Any = row
    for part in SELECT_KEYS[key]:
        value = value[part]
    return value


def _matches(row: Mapping[str, Any], select: Sequence[Tuple[str, Any]]) -> bool:
    for key, wanted in select:
        value = str(_row_value(row, key))
        if isinstance(wanted, tuple):
            if value not in wanted:
                return False
        elif value != wanted:
            return False
    return True


def read_rows(path: str | Path) -> List[Dict[str, Any]]:
    rows = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise WorkloadError(f"{path}:{line_number}: invalid JSON ({exc})") from exc
    return rows


def select_rows(
    path: str | Path, select: Optional[Mapping[str, Any] | Sequence[Tuple[str, Any]]] = None
) -> List[Dict[str, Any]]:
    """The rows of one workload, in file order; position = ``query_id``."""

    pairs = select if isinstance(select, tuple) else normalize_select(select)  # type: ignore[arg-type]
    return [row for row in read_rows(path) if _matches(row, pairs)]


# ---------------------------------------------------------------------------
# records and features
# ---------------------------------------------------------------------------


def _check_row(row: Mapping[str, Any], position: int) -> None:
    where = f"row {row.get('query_id', position)}"
    version = row.get("semantics_version")
    if version not in SUPPORTED_SEMANTICS:
        raise WorkloadError(f"{where}: semantics_version {version!r} not in {SUPPORTED_SEMANTICS}")
    label_kind = row.get("label_kind")
    if label_kind not in LABEL_FIELD:
        raise WorkloadError(f"{where}: unknown label_kind {label_kind!r}")
    cardinality = row.get("cardinality")
    if not isinstance(cardinality, int) or isinstance(cardinality, bool) or cardinality < 0:
        raise WorkloadError(f"{where}: cardinality must be a non-negative integer")
    if row.get(LABEL_FIELD[label_kind]) != cardinality:
        raise WorkloadError(
            f"{where}: cardinality {cardinality} differs from {LABEL_FIELD[label_kind]} "
            f"{row.get(LABEL_FIELD[label_kind])}"
        )
    expected_kind = "distinct_trips" if row["category"]["relation"] == "multi" else "row_count"
    if label_kind != expected_kind:
        raise WorkloadError(f"{where}: {row['category']['relation']} query with label {label_kind}")


_EPOCH = datetime(1970, 1, 1)


def _seconds(text: str) -> float:
    """Seconds since 1970-01-01 of a stored naive timestamp, without any time zone.

    v3 timestamps are ``timestamp without time zone`` values compared as stored
    (SEMANTICS.md section 1). ``datetime.timestamp()`` would interpret them in
    the machine's local zone and add or drop an hour across a DST change.
    """
    value = datetime.fromisoformat(str(text))
    if value.tzinfo is not None:
        raise WorkloadError(f"v3 timestamps are naive; got {text!r}")
    return (value - _EPOCH).total_seconds()


class Domains:
    """Domain extents used for relative features (optional)."""

    def __init__(self, spatial: Optional[Tuple[float, float, float, float]],
                 temporal: Mapping[str, Tuple[float, float]],
                 segment_time_key: Optional[str] = None):
        self.spatial = spatial
        #: ``"table.attribute"`` -> (min, max) epoch seconds of every temporal attribute
        self.temporal = dict(temporal)
        #: temporal attribute of the segment rows (domain of spatio-temporal intervals)
        self.segment_time_key = segment_time_key

    @classmethod
    def from_benchmark_config(cls, path: str | Path) -> "Domains":
        from query_generation.benchmark_v3.config import load_settings

        settings = load_settings(Path(path))
        segments = settings.segments
        geometry = settings.attr(segments.table, segments.spatial_attribute)["domain"]
        spatial = (
            float(geometry["min_x"]), float(geometry["min_y"]),
            float(geometry["max_x"]), float(geometry["max_y"]),
        )
        temporal = {}
        for table_id, table in settings.base["tables"].items():
            for attribute in table["attributes"]:
                if attribute.get("type") == "temporal_interval":
                    domain = attribute["domain"]
                    temporal[f"{table_id}.{attribute['name']}"] = (
                        _seconds(domain["min"]), _seconds(domain["max"])
                    )
        return cls(spatial, temporal, f"{segments.table}.{segments.temporal_attribute}")


def query_features(row: Mapping[str, Any], domains: Optional[Domains] = None) -> Dict[str, Any]:
    """Per-query features for stratified reporting (all from structured predicates)."""

    predicates = row["predicates"]
    features: Dict[str, Any] = {
        "category": row["category"]["key"],
        "dimension": row["category"]["dimension"],
        "interval": row["category"]["interval"],
        "relation": row["category"]["relation"],
        "label_kind": row["label_kind"],
        "addon_count": int(row.get("addon_predicates") or 0),
        "table_count": len(row["tables"]),
        "true_cardinality": int(row["cardinality"]),
        "true_cardinality_log10": math.log10(row["cardinality"]) if row["cardinality"] > 0 else None,
        "predicate_kinds": sorted(Counter(p["kind"] for p in predicates).items()),
    }
    spatial = [p for p in predicates if p["kind"] in SPATIAL_KINDS]
    temporal = [p for p in predicates if p["kind"] in TEMPORAL_KINDS]
    features["window_area_fraction"] = None
    features["window_width_x"] = None
    features["window_width_y"] = None
    if spatial:
        p = spatial[0]
        x0, y0, x1, y1 = (float(p[k]) for k in ("min_x", "min_y", "max_x", "max_y"))
        if domains is not None and domains.spatial is not None:
            dx0, dy0, dx1, dy1 = domains.spatial
            # clip to the data domain: unbounded windows extend far beyond it
            cx = max(0.0, min(x1, dx1) - max(x0, dx0))
            cy = max(0.0, min(y1, dy1) - max(y0, dy0))
            features["window_width_x"], features["window_width_y"] = cx, cy
            area = (dx1 - dx0) * (dy1 - dy0)
            features["window_area_fraction"] = (cx * cy) / area if area > 0 else None
        else:
            features["window_width_x"], features["window_width_y"] = x1 - x0, y1 - y0
    features["interval_seconds"] = None
    features["interval_fraction"] = None
    features["interval_one_sided"] = None
    if temporal:
        p = temporal[0]
        lower, upper = p.get("lower"), p.get("upper")
        features["interval_one_sided"] = lower is None or upper is None
        if p["kind"] == "spatio_temporal":
            key = domains.segment_time_key if domains is not None else None
        else:
            key = f"{p['table']}.{p['attribute']}"
        extent = domains.temporal.get(key) if domains is not None and key else None
        if extent is not None:
            lo = extent[0] if lower is None else max(extent[0], _seconds(lower))
            hi = extent[1] if upper is None else min(extent[1], _seconds(upper))
            length = max(0.0, hi - lo)
            features["interval_seconds"] = length
            span = extent[1] - extent[0]
            features["interval_fraction"] = length / span if span > 0 else None
        elif lower is not None and upper is not None:
            features["interval_seconds"] = _seconds(upper) - _seconds(lower)
    return features


def records_from_rows(
    rows: Sequence[Mapping[str, Any]], workload_id: str, domains: Optional[Domains] = None
) -> List[TrajQueryRecord]:
    records = []
    seen_hashes: set = set()
    for position, row in enumerate(rows):
        _check_row(row, position)
        digest = row["predicate_hash"]
        if digest in seen_hashes:
            raise WorkloadError(f"duplicate predicate_hash {digest} in workload {workload_id}")
        seen_hashes.add(digest)
        category = row["category"]
        records.append(
            TrajQueryRecord(
                workload=workload_id,
                query_id=position,
                true_cardinality=int(row["cardinality"]),
                source_query_id=str(row.get("query_id", "")),
                slot_id=str(row.get("slot_id", "")),
                predicate_hash=str(digest),
                category=str(category["key"]),
                dimension=str(category["dimension"]),
                interval=str(category["interval"]),
                relation=str(category["relation"]),
                label_kind=str(row["label_kind"]),
                tables=tuple(row["tables"]),
                predicates=tuple(row["predicates"]),
                addon_count=int(row.get("addon_predicates") or 0),
                features=query_features(row, domains),
            )
        )
    return records


def load_traj_workload(
    path: str | Path,
    workload_id: str,
    select: Optional[Mapping[str, Any] | Sequence[Tuple[str, Any]]] = None,
    domains: Optional[Domains] = None,
) -> List[TrajQueryRecord]:
    return records_from_rows(select_rows(path, select), workload_id, domains)


def write_query_index(records: Iterable[TrajQueryRecord], path: str | Path) -> None:
    """``query_id`` -> v3 identity, for joining predictions back to the benchmark."""

    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["query_id", "source_query_id", "slot_id", "predicate_hash", "category",
                         "label_kind", "true_cardinality"])
        for record in records:
            writer.writerow([record.query_id, record.source_query_id, record.slot_id,
                             record.predicate_hash, record.category, record.label_kind,
                             record.true_cardinality])


def workload_statistics(records: Sequence[TrajQueryRecord]) -> Dict[str, Any]:
    truths = sorted(record.true_cardinality for record in records)

    def pick(q: float) -> Optional[float]:
        if not truths:
            return None
        # linear interpolation, as numpy's default percentile
        position = q * (len(truths) - 1)
        low = int(math.floor(position))
        high = min(low + 1, len(truths) - 1)
        return float(truths[low] + (truths[high] - truths[low]) * (position - low))

    kinds = Counter(p["kind"] for record in records for p in record.predicates)
    roles = Counter(f"{p.get('role', 'core')}:{p['kind']}" for record in records for p in record.predicates)
    return {
        "query_count": len(records),
        "category_counts": dict(sorted(Counter(r.category for r in records).items())),
        "label_kind_counts": dict(sorted(Counter(r.label_kind for r in records).items())),
        "predicate_kind_counts": dict(sorted(kinds.items())),
        "predicate_role_counts": dict(sorted(roles.items())),
        "addon_count_distribution": {
            str(k): v for k, v in sorted(Counter(r.addon_count for r in records).items())
        },
        "table_set_counts": dict(sorted(Counter("+".join(r.tables) for r in records).items())),
        "true_cardinality_min": truths[0] if truths else None,
        "true_cardinality_max": truths[-1] if truths else None,
        "true_cardinality_percentiles": {
            "p50": pick(0.5), "p90": pick(0.9), "p95": pick(0.95), "p99": pick(0.99),
            "max": truths[-1] if truths else None,
        },
        "zero_true_cardinality_count": sum(value == 0 for value in truths),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# workload format for evaluation.eval_core
# ---------------------------------------------------------------------------


class TrajV3JsonlFormat:
    """``format: traj_v3_jsonl`` workloads.

    YAML keys: ``queries_jsonl`` (required), ``select`` (mapping, see
    :data:`SELECT_KEYS`), ``expected_query_count`` (counts *selected* rows),
    and ``benchmark_config`` (optional v3 config JSON; enables the
    domain-relative features ``window_area_fraction`` and ``interval_fraction``).
    """

    name = FORMAT_NAME

    def parse_config(self, workload_id, value, resolve_path) -> "WorkloadConfig":
        from evaluation.eval_core.config import WorkloadConfig

        if value.get("queries_jsonl") in (None, ""):
            raise ValueError(f"workloads.{workload_id}.queries_jsonl is required for {self.name}")
        options = []
        if value.get("benchmark_config") not in (None, ""):
            options.append(("benchmark_config", str(resolve_path(value["benchmark_config"]))))
        return WorkloadConfig(
            workload_id=workload_id,
            queries_csv=None,
            expected_query_count=(
                None
                if value.get("expected_query_count") in (None, "")
                else int(value["expected_query_count"])
            ),
            format=self.name,
            queries_jsonl=resolve_path(value["queries_jsonl"]),
            select=normalize_select(value.get("select")),
            options=tuple(options),
        )

    def _domains(self, workload: "WorkloadConfig") -> Optional[Domains]:
        config = dict(workload.options).get("benchmark_config")
        return Domains.from_benchmark_config(config) if config else None

    def validate(self, workload: "WorkloadConfig") -> None:
        path = workload.queries_jsonl
        if path is None or not path.exists():
            raise FileNotFoundError(path)
        records = self.load(workload)
        if workload.expected_query_count is not None and len(records) != workload.expected_query_count:
            raise ValueError(
                f"{workload.workload_id} expected {workload.expected_query_count} "
                f"queries, observed {len(records)}"
            )
        if not records:
            raise ValueError(f"{workload.workload_id} selects no queries")

    def load(self, workload: "WorkloadConfig") -> List[TrajQueryRecord]:
        return load_traj_workload(
            workload.queries_jsonl, workload.workload_id, workload.select, self._domains(workload)
        )

    def metadata(self, workload: "WorkloadConfig", queries) -> Dict[str, Any]:
        records = list(queries)
        versions = sorted({row.get("semantics_version") for row in select_rows(workload.queries_jsonl, workload.select)})
        return {
            "format": self.name,
            "queries_jsonl": str(workload.queries_jsonl),
            "queries_jsonl_sha256": _sha256(workload.queries_jsonl),
            "select": {key: (list(value) if isinstance(value, tuple) else value) for key, value in workload.select},
            "semantics_versions": versions,
            "benchmark_config": dict(workload.options).get("benchmark_config"),
            "query_id_rule": "0-based position within the selection, in file order",
            **workload_statistics(records),
        }

    def placeholders(self, workload: "WorkloadConfig") -> Dict[str, str]:
        return {
            "queries_jsonl": str(workload.queries_jsonl),
            "workload_select_json": json.dumps(
                {key: (list(value) if isinstance(value, tuple) else value) for key, value in workload.select},
                sort_keys=True,
            ),
        }

    def summary_extension(self, workload, queries, predictions) -> Dict[str, Any]:
        from .strata import trajectory_summary

        return trajectory_summary(workload.workload_id, list(queries), list(predictions))


TRAJ_V3_JSONL_FORMAT = TrajV3JsonlFormat()
