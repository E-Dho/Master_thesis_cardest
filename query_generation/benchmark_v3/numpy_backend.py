"""Numpy label backend: evaluates structured predicates with ``exact.py``."""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np

from query_generation.query_generator import ConfigError, parse_timestamp

from . import exact
from .config import Settings

JOIN_CONDITION = re.compile(r"^\s*(\w+)\.(\w+)\s*=\s*(\w+)\.(\w+)\s*$")


class NumpyBackend:
    """Evaluates structured predicates with ``exact.py`` on in-memory tables.

    ``data`` maps table ids to dicts of equal-length arrays. Scalar and nominal
    attributes use their attribute name; a temporal attribute ``name`` uses
    ``name.start`` / ``name.end`` (int64 microseconds); the segments table also
    carries ``sx, sy, ex, ey, ts, te`` (``ts``/``te`` in microseconds).
    """

    def __init__(self, settings: Settings, data: Dict[str, Dict[str, np.ndarray]]):
        self.settings = settings
        self.data = data
        self.join_keys = self._parse_joins(settings)

    @staticmethod
    def _parse_joins(settings: Settings) -> Dict[Tuple[str, str], Tuple[str, str]]:
        alias_to_table = {table["alias"]: table_id for table_id, table in settings.base["tables"].items()}
        keys: Dict[Tuple[str, str], Tuple[str, str]] = {}
        for join in settings.base["joins"]:
            match = JOIN_CONDITION.match(join["condition"])
            if not match:
                raise ConfigError(f"unsupported join condition {join['condition']!r}")
            left_alias, left_col, right_alias, right_col = match.groups()
            left, right = alias_to_table[left_alias], alias_to_table[right_alias]
            keys[(left, right)] = (left_col, right_col)
            keys[(right, left)] = (right_col, left_col)
        return keys

    def _rows(self, table_id: str) -> int:
        columns = self.data[table_id]
        return len(next(iter(columns.values())))

    def predicate_mask(self, predicate: Dict[str, Any]) -> np.ndarray:
        table = self.data[predicate["table"]]
        kind = predicate["kind"]
        name = predicate["attribute"]
        if kind == "nominal_eq":
            return np.asarray(table[name]) == predicate["value"]
        if kind == "scalar_range":
            values = np.asarray(table[name])
            return (values >= predicate["lower"]) & (values <= predicate["upper"])
        if kind == "scalar_unbounded":
            values = np.asarray(table[name])
            op, value = predicate["operator"], predicate["value"]
            return {
                "<": values < value,
                "<=": values <= value,
                ">": values > value,
                ">=": values >= value,
            }[op]
        lower_us = _bound_us(predicate.get("lower"))
        upper_us = _bound_us(predicate.get("upper"))
        if kind == "temporal":
            return exact.temporal_mask(table[f"{name}.start"], table[f"{name}.end"], lower_us, upper_us)
        window = (predicate["min_x"], predicate["min_y"], predicate["max_x"], predicate["max_y"])
        if kind == "spatial":
            return exact.spatial_mask(table["sx"], table["sy"], table["ex"], table["ey"], window)
        if kind == "spatio_temporal_decoupled":
            return exact.decoupled_mask(
                table["sx"], table["sy"], table["ex"], table["ey"], table["ts"], table["te"],
                window, lower_us, upper_us,
            )
        if kind == "spatio_temporal":
            return exact.spatio_temporal_mask(
                table["sx"], table["sy"], table["ex"], table["ey"], table["ts"], table["te"],
                window, lower_us, upper_us,
            )
        raise ConfigError(f"unknown predicate kind {kind!r}")

    def table_mask(self, table_id: str, predicates: Iterable[Dict[str, Any]]) -> np.ndarray:
        mask = np.ones(self._rows(table_id), dtype=bool)
        for predicate in predicates:
            if predicate["table"] == table_id:
                mask &= self.predicate_mask(predicate)
        return mask

    def count(self, record: Dict[str, Any]) -> int:
        predicates = record["predicates"]
        tables = record["tables"]
        relation = record["category"]["relation"]
        if relation == "single":
            if len(tables) != 1:
                raise ConfigError("single queries use exactly one table")
            return int(self.table_mask(tables[0], predicates).sum())

        entity = self.settings.entity_table
        entity_key = self.settings.base["entity"]["key"]
        mask = self.table_mask(entity, predicates)
        for other in tables:
            if other == entity:
                continue
            if (other, entity) not in self.join_keys:
                raise ConfigError(f"numpy backend needs {other} to join {entity} directly")
            other_col, entity_col = self.join_keys[(other, entity)]
            other_mask = self.table_mask(other, predicates)
            matching = np.asarray(self.data[other][other_col])[other_mask]
            mask &= np.isin(np.asarray(self.data[entity][entity_col]), matching)
        ids = np.asarray(self.data[entity][entity_key])[mask]
        return int(np.unique(ids).size)


def _bound_us(value: Optional[str]) -> Optional[int]:
    return None if value is None else exact.to_us(parse_timestamp(value))
