"""MSCN (Kipf et al., CIDR 2019) on v3 trajectory queries, in process.

Architecture and training follow the published MSCN
(``andreaskipf/learnedcardinalities``, ``mscn/model.py`` and ``train.py``):
three set modules (samples, predicates, joins), each a two-layer ReLU MLP
whose outputs are averaged over the set elements, an output MLP with a
sigmoid on min-max normalised log cardinalities, the q-error loss on the
unnormalised values, Adam (lr 1e-3), 100 epochs, batch 1024, 256 hidden
units, 1,000 materialised samples per table. One model is trained on both
label kinds (row counts of single queries and distinct trips of multi
queries); the join set tells them apart (single queries have none).

Adapted part (``protocol: adapted``): the featurisation of v3 predicates.
Upstream encodes ``column op value``; a v3 predicate becomes

- one-hot of its ``table.attribute``,
- one-hot of its kind (nominal equality, scalar range, one-sided scalar per
  operator, temporal interval, spatial window, spatio-temporal window),
- one-hot of the nominal value (nominal equality only),
- six bound slots ``lower, upper, min_x, max_x, min_y, max_y`` normalised by
  the attribute's domain from the v3 config, plus six presence bits.

A sample element is the one-hot table plus the bitmap of the 1,000 sampled
rows of that table that satisfy the query's predicates on it, evaluated with
the v3 ``NumpyBackend`` (the exact label semantics, including the
segment-coupled spatio-temporal test) on rows sampled from the P2 snapshot.

Training data (``adapter.training_jsonl``): a v3 benchmark generated with
another seed that excludes the test and validation queries; ``build``
refuses any predicate hash shared with an evaluated workload. The model, its
normalisation and the samples are stored in ``model.npz`` (all count toward
``serialized_bytes``). ``adapter.validation_jsonl`` is reported, never used
for selection. ``adapter.training_relations`` ([single, multi] by default;
[multi] is the decided fallback variant, which reports the other relation as
unsupported).
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .base import Estimate, InProcessTrajectoryAdapter

KINDS = ("nominal_eq", "scalar_range", "scalar_unbounded:<", "scalar_unbounded:<=", "scalar_unbounded:>",
         "scalar_unbounded:>=", "temporal", "spatial", "spatio_temporal")
SLOTS = ("lower", "upper", "min_x", "max_x", "min_y", "max_y")
NO_JOIN = "<none>"
DEFAULT_RECIPE = {"sample_bits": 1000, "epochs": 100, "batch_size": 1024, "hidden_size": 256,
                  "learning_rate": 0.001}


def _to_us(text: str) -> int:
    from query_generation.benchmark_v3 import exact
    from query_generation.query_generator import parse_timestamp

    return int(exact.to_us(parse_timestamp(str(text))))


class Featurizer:
    """v3 query -> (sample, predicate, join) set encodings, from config-derived vocabularies."""

    def __init__(self, settings: Any, samples: Mapping[str, Mapping[str, np.ndarray]], sample_bits: int):
        from query_generation.benchmark_v3.numpy_backend import NumpyBackend

        self.settings = settings
        self.sample_bits = int(sample_bits)
        self.tables = list(settings.base["tables"])
        self.joins = [join["condition"] for join in settings.base["joins"]] + [NO_JOIN]
        self.keys: List[str] = []
        self.nominal: List[str] = []
        self.domains: Dict[str, Tuple[float, float]] = {}
        for table_id in self.tables:
            for attribute in settings.table(table_id)["attributes"]:
                key = f"{table_id}.{attribute['name']}"
                self.keys.append(key)
                kind, domain = attribute["type"], attribute.get("domain") or {}
                if kind == "nominal":
                    self.nominal += [f"{key}={value}" for value in attribute["values"]]
                elif kind == "temporal_interval":
                    self.domains[key] = (float(_to_us(domain["min"])), float(_to_us(domain["max"])))
                elif kind == "geometry":
                    self.domains[f"{key}:x"] = (float(domain["min_x"]), float(domain["max_x"]))
                    self.domains[f"{key}:y"] = (float(domain["min_y"]), float(domain["max_y"]))
                else:
                    self.domains[key] = (float(domain["min"]), float(domain["max"]))
        self.key_index = {key: i for i, key in enumerate(self.keys)}
        self.nominal_index = {value: i for i, value in enumerate(self.nominal)}
        self.sample_features = len(self.tables) + self.sample_bits
        self.predicate_features = len(self.keys) + len(KINDS) + len(self.nominal) + 2 * len(SLOTS)
        self.join_features = len(self.joins)
        self.backend = NumpyBackend(settings, {t: dict(c) for t, c in samples.items()})
        self.sample_rows = {t: len(next(iter(c.values()))) if c else 0 for t, c in samples.items()}

    def vocabulary(self) -> Dict[str, Any]:
        return {"tables": self.tables, "joins": self.joins, "keys": self.keys, "kinds": list(KINDS),
                "nominal": self.nominal, "slots": list(SLOTS), "sample_bits": self.sample_bits,
                "domains": {key: list(value) for key, value in self.domains.items()}}

    def _norm(self, key: str, value: float) -> float:
        low, high = self.domains[key]
        return 0.0 if high <= low else float(min(1.0, max(0.0, (value - low) / (high - low))))

    def predicate_vector(self, predicate: Mapping[str, Any]) -> np.ndarray:
        vector = np.zeros(self.predicate_features, dtype=np.float32)
        key = f"{predicate['table']}.{predicate['attribute']}"
        kind = predicate["kind"]
        vector[self.key_index[key]] = 1.0
        kind_key = f"scalar_unbounded:{predicate['operator']}" if kind == "scalar_unbounded" else kind
        offset = len(self.keys)
        vector[offset + KINDS.index(kind_key)] = 1.0
        offset += len(KINDS)
        values: Dict[str, float] = {}
        if kind == "nominal_eq":
            vector[offset + self.nominal_index[f"{key}={predicate['value']}"]] = 1.0
        elif kind == "scalar_range":
            values = {"lower": self._norm(key, float(predicate["lower"])),
                      "upper": self._norm(key, float(predicate["upper"]))}
        elif kind == "scalar_unbounded":
            slot = "lower" if predicate["operator"] in (">", ">=") else "upper"
            values = {slot: self._norm(key, float(predicate["value"]))}
        else:
            if kind in ("temporal", "spatio_temporal"):
                time_key = key if kind == "temporal" else self._segment_time_key()
                for slot in ("lower", "upper"):
                    if predicate.get(slot) is not None:
                        values[slot] = self._norm(time_key, float(_to_us(predicate[slot])))
            if kind in ("spatial", "spatio_temporal"):
                space = self._segment_space_key()
                values.update(min_x=self._norm(f"{space}:x", float(predicate["min_x"])),
                              max_x=self._norm(f"{space}:x", float(predicate["max_x"])),
                              min_y=self._norm(f"{space}:y", float(predicate["min_y"])),
                              max_y=self._norm(f"{space}:y", float(predicate["max_y"])))
        offset += len(self.nominal)
        for index, slot in enumerate(SLOTS):
            if slot in values:
                vector[offset + index] = values[slot]
                vector[offset + len(SLOTS) + index] = 1.0
        return vector

    def _segment_time_key(self) -> str:
        return f"{self.settings.segments.table}.{self.settings.segments.temporal_attribute}"

    def _segment_space_key(self) -> str:
        return f"{self.settings.segments.table}.{self.settings.segments.spatial_attribute}"

    def encode(self, tables: Sequence[str], predicates: Sequence[Mapping[str, Any]]
               ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        from query_generation.benchmark_v3.query_model import from_clause

        samples = np.zeros((len(tables), self.sample_features), dtype=np.float32)
        for row, table_id in enumerate(tables):
            samples[row, self.tables.index(table_id)] = 1.0
            mask = self.backend.table_mask(table_id, list(predicates))
            samples[row, len(self.tables):len(self.tables) + mask.size] = mask.astype(np.float32)
        if predicates:
            predicate_set = np.stack([self.predicate_vector(p) for p in predicates])
        else:
            predicate_set = np.zeros((1, self.predicate_features), dtype=np.float32)
        joins = [join["condition"] for join in from_clause(self.settings, list(tables))[1]] or [NO_JOIN]
        join_set = np.zeros((len(joins), self.join_features), dtype=np.float32)
        for row, join in enumerate(joins):
            join_set[row, self.joins.index(join)] = 1.0
        return samples, predicate_set, join_set


def _torch():
    import torch

    return torch


def make_model(featurizer: Featurizer, hidden: int):
    torch = _torch()
    nn = torch.nn

    class SetConv(nn.Module):
        """``mscn/model.py`` of the published code."""

        def __init__(self, sample_feats, predicate_feats, join_feats, hid_units):
            super().__init__()
            self.sample_mlp1 = nn.Linear(sample_feats, hid_units)
            self.sample_mlp2 = nn.Linear(hid_units, hid_units)
            self.predicate_mlp1 = nn.Linear(predicate_feats, hid_units)
            self.predicate_mlp2 = nn.Linear(hid_units, hid_units)
            self.join_mlp1 = nn.Linear(join_feats, hid_units)
            self.join_mlp2 = nn.Linear(hid_units, hid_units)
            self.out_mlp1 = nn.Linear(hid_units * 3, hid_units)
            self.out_mlp2 = nn.Linear(hid_units, 1)

        def forward(self, samples, predicates, joins, sample_mask, predicate_mask, join_mask):
            relu = torch.nn.functional.relu
            hid_sample = relu(self.sample_mlp2(relu(self.sample_mlp1(samples)))) * sample_mask
            hid_sample = torch.sum(hid_sample, dim=1, keepdim=False) / sample_mask.sum(1, keepdim=False)
            hid_predicate = relu(self.predicate_mlp2(relu(self.predicate_mlp1(predicates)))) * predicate_mask
            hid_predicate = torch.sum(hid_predicate, dim=1, keepdim=False) / predicate_mask.sum(1, keepdim=False)
            hid_join = relu(self.join_mlp2(relu(self.join_mlp1(joins)))) * join_mask
            hid_join = torch.sum(hid_join, dim=1, keepdim=False) / join_mask.sum(1, keepdim=False)
            hid = torch.cat((hid_sample, hid_predicate, hid_join), 1)
            return torch.sigmoid(self.out_mlp2(relu(self.out_mlp1(hid))))

    return SetConv(featurizer.sample_features, featurizer.predicate_features, featurizer.join_features, hidden)


def pad(sets: Sequence[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """Upstream ``make_dataset``: zero-pad each set to the batch maximum, with masks."""
    width = max(len(item) for item in sets)
    features = sets[0].shape[1]
    out = np.zeros((len(sets), width, features), dtype=np.float32)
    mask = np.zeros((len(sets), width, 1), dtype=np.float32)
    for index, item in enumerate(sets):
        out[index, :len(item)] = item
        mask[index, :len(item)] = 1.0
    return out, mask


def _row_fields(row: Mapping[str, Any]) -> Tuple[List[str], List[Mapping[str, Any]], str, int]:
    relation = row["category"]["relation"] if "category" in row else row["relation"]
    return list(row["tables"]), list(row["predicates"]), relation, int(row["cardinality"])


def read_rows(path: Path, relations: Iterable[str]) -> List[Dict[str, Any]]:
    wanted = set(relations)
    with Path(path).open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    return [row for row in rows if row["category"]["relation"] in wanted]


class MscnModel:
    def __init__(self, featurizer: Featurizer, network: Any, label_min: float, label_max: float):
        self.featurizer = featurizer
        self.network = network
        self.label_min = float(label_min)
        self.label_max = float(label_max)

    def unnormalize(self, value):
        return np.exp(value * (self.label_max - self.label_min) + self.label_min)

    def predict(self, encoded: Sequence[Tuple[np.ndarray, np.ndarray, np.ndarray]], batch: int = 1024) -> np.ndarray:
        torch = _torch()
        out: List[np.ndarray] = []
        self.network.eval()
        with torch.no_grad():
            for start in range(0, len(encoded), batch):
                part = encoded[start:start + batch]
                tensors = [torch.from_numpy(array) for pair in
                           (pad([e[0] for e in part]), pad([e[1] for e in part]), pad([e[2] for e in part]))
                           for array in pair]
                samples, sample_mask, predicates, predicate_mask, joins, join_mask = tensors
                values = self.network(samples, predicates, joins, sample_mask, predicate_mask, join_mask)
                out.append(values.numpy().reshape(-1).astype(np.float64))
        return self.unnormalize(np.concatenate(out)) if out else np.zeros(0)

    def estimate(self, record: Any) -> Estimate:
        encoded = self.featurizer.encode(list(record.tables), list(record.predicates))
        return "ok", float(self.predict([encoded])[0]), ""


def q_errors(estimates: np.ndarray, truths: np.ndarray) -> np.ndarray:
    estimates = np.maximum(np.asarray(estimates, dtype=np.float64), 1e-12)
    truths = np.maximum(np.asarray(truths, dtype=np.float64), 1e-12)
    return np.maximum(estimates / truths, truths / estimates)


def train(featurizer: Featurizer, rows: Sequence[Mapping[str, Any]], *, seed: int, recipe: Mapping[str, Any],
          validation: Sequence[Mapping[str, Any]] = (), device: str = "cpu", log=print
          ) -> Tuple[MscnModel, Dict[str, Any]]:
    """Upstream ``train.py``: normalised log labels, q-error loss, Adam, fixed epochs."""
    torch = _torch()
    torch.manual_seed(int(seed))
    generator = torch.Generator().manual_seed(int(seed))
    started = time.perf_counter()
    encoded = []
    labels = []
    for row in rows:
        tables, predicates, _, cardinality = _row_fields(row)
        encoded.append(featurizer.encode(tables, predicates))
        labels.append(cardinality)
    featurize_seconds = time.perf_counter() - started
    log_labels = np.log(np.maximum(np.asarray(labels, dtype=np.float64), 1.0))
    label_min, label_max = float(log_labels.min()), float(log_labels.max())
    if label_max <= label_min:
        label_max = label_min + 1.0
    targets = (log_labels - label_min) / (label_max - label_min)
    network = make_model(featurizer, int(recipe["hidden_size"])).to(device)
    optimizer = torch.optim.Adam(network.parameters(), lr=float(recipe["learning_rate"]))
    span = torch.tensor(label_max - label_min, dtype=torch.float64, device=device)
    low = torch.tensor(label_min, dtype=torch.float64, device=device)
    batch = int(recipe["batch_size"])
    epochs = int(recipe["epochs"])
    losses: List[float] = []
    training_started = time.perf_counter()
    for epoch in range(epochs):
        network.train()
        order = torch.randperm(len(encoded), generator=generator).tolist()
        loss_sum = 0.0
        for start in range(0, len(order), batch):
            index = order[start:start + batch]
            part = [encoded[i] for i in index]
            tensors = [torch.from_numpy(array).to(device) for pair in
                       (pad([e[0] for e in part]), pad([e[1] for e in part]), pad([e[2] for e in part]))
                       for array in pair]
            samples, sample_mask, predicates, predicate_mask, joins, join_mask = tensors
            target = torch.from_numpy(targets[index]).to(device)
            optimizer.zero_grad()
            output = network(samples, predicates, joins, sample_mask, predicate_mask, join_mask).reshape(-1)
            prediction = torch.exp(output.double() * span + low)
            truth = torch.exp(target * span + low)
            loss = torch.mean(torch.maximum(prediction / truth, truth / prediction))
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.item()) * len(index)
        losses.append(loss_sum / len(order))
        if epoch == 0 or (epoch + 1) % 10 == 0 or epoch + 1 == epochs:
            log(f"epoch {epoch + 1}/{epochs}: mean q-error loss {losses[-1]:.4f}")
    training_seconds = time.perf_counter() - training_started
    network = network.to("cpu")
    model = MscnModel(featurizer, network, label_min, label_max)
    report: Dict[str, Any] = {
        "training_queries": len(rows), "epochs": epochs, "featurize_seconds": featurize_seconds,
        "training_seconds": training_seconds, "label_min_log": label_min, "label_max_log": label_max,
        "epoch_loss": losses, "device": device,
    }
    train_q = q_errors(model.predict(encoded), np.asarray(labels))
    report["training_q_error"] = {name: float(np.percentile(train_q, q)) for name, q in
                                  (("p50", 50), ("p90", 90), ("p99", 99), ("max", 100))}
    if validation:
        val_encoded = [featurizer.encode(*_row_fields(row)[:2]) for row in validation]
        val_truth = np.asarray([_row_fields(row)[3] for row in validation])
        val_q = q_errors(model.predict(val_encoded), val_truth)
        report["validation_q_error"] = {name: float(np.percentile(val_q, q)) for name, q in
                                        (("p50", 50), ("p90", 90), ("p99", 99), ("max", 100))}
        report["validation_queries"] = len(validation)
        by_relation: Dict[str, List[float]] = {}
        for row, value in zip(validation, val_q):
            by_relation.setdefault(_row_fields(row)[2], []).append(float(value))
        report["validation_q_error_by_relation"] = {
            relation: {"queries": len(values), "p50": float(np.percentile(values, 50)),
                       "p99": float(np.percentile(values, 99))} for relation, values in by_relation.items()}
    return model, report


def draw_samples(data: Mapping[str, Mapping[str, np.ndarray]], tables: Sequence[str], bits: int,
                 seed: int) -> Dict[str, Dict[str, np.ndarray]]:
    """``bits`` rows per table, uniformly without replacement (all rows if fewer)."""
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), 0x6D73636E]))  # "mscn"
    samples: Dict[str, Dict[str, np.ndarray]] = {}
    for table_id in tables:
        columns = data[table_id]
        rows = len(next(iter(columns.values())))
        index = np.sort(rng.choice(rows, size=min(bits, rows), replace=False))
        samples[table_id] = {name: np.asarray(values)[index] for name, values in columns.items()}
    return samples


class MscnAdapter(InProcessTrajectoryAdapter):
    latency_scope = "featurization_sample_bitmaps_and_forward_pass"
    timing_role = "in_process_mscn"

    @property
    def recipe(self) -> Dict[str, Any]:
        configured = dict(self.config.adapter.get("recipe") or {})
        return {**DEFAULT_RECIPE, **configured}

    def _relations(self) -> List[str]:
        relations = self.config.adapter.get("training_relations") or ["single", "multi"]
        relations = [str(item) for item in (relations if isinstance(relations, (list, tuple)) else [relations])]
        if not set(relations) <= {"single", "multi"} or not relations:
            raise ValueError("adapter.training_relations must be a subset of [single, multi]")
        return relations

    def _training_rows(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        value = self.config.adapter.get("training_jsonl")
        if not value:
            raise ValueError("adapter.training_jsonl is required")
        rows = read_rows(self._resolve(value), self._relations())
        evaluated = set()
        from evaluation.eval_core.workload_formats import load_queries

        for workload in self.config.workloads:
            evaluated.update(record.predicate_hash for record in load_queries(workload))
        shared = evaluated & {row["predicate_hash"] for row in rows}
        if shared:
            raise ValueError(f"{len(shared)} training queries also occur in an evaluated workload")
        return rows if limit is None else rows[:limit]

    def _validation_rows(self) -> List[Dict[str, Any]]:
        value = self.config.adapter.get("validation_jsonl")
        return read_rows(self._resolve(value), ("single", "multi")) if value else []

    def _fit(self, data, *, limit: Optional[int] = None, recipe: Optional[Mapping[str, Any]] = None,
             validation: bool = True):
        recipe = {**self.recipe, **(recipe or {})}
        samples = draw_samples(data, list(self.settings.base["tables"]), int(recipe["sample_bits"]), self.seed)
        featurizer = Featurizer(self.settings, samples, int(recipe["sample_bits"]))
        threads = self.config.resources.get("cpu_threads")
        if threads:
            _torch().set_num_threads(int(threads))
        model, report = train(featurizer, self._training_rows(limit), seed=self.seed, recipe=recipe,
                              validation=self._validation_rows() if validation else (),
                              device=str(self.config.adapter.get("device", "cpu")),
                              log=lambda message: print(message, flush=True))
        return model, report, samples

    def _build_model(self, data, trips) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        model, report, samples = self._fit(data)
        arrays: Dict[str, np.ndarray] = {f"w/{name}": tensor.detach().cpu().numpy()
                                         for name, tensor in model.network.state_dict().items()}
        for table_id, columns in samples.items():
            for column, values in columns.items():
                arrays[f"sample/{table_id}/{column}"] = np.asarray(values)
        parameters = int(sum(int(np.prod(arrays[name].shape)) for name in arrays if name.startswith("w/")))
        meta = {"method": "mscn", "recipe": self.recipe, "training_relations": self._relations(),
                "label_min_log": model.label_min, "label_max_log": model.label_max,
                "vocabulary": model.featurizer.vocabulary(), "parameter_count": parameters,
                "training": report}
        return arrays, meta

    def _load_model(self, arrays, meta) -> MscnModel:
        torch = _torch()
        samples: Dict[str, Dict[str, np.ndarray]] = {}
        for name, values in arrays.items():
            if name.startswith("sample/"):
                _, table_id, column = name.split("/", 2)
                samples.setdefault(table_id, {})[column] = values
        featurizer = Featurizer(self.settings, samples, int(meta["recipe"]["sample_bits"]))
        # Every input position is defined by the vocabulary (order of keys,
        # kinds, nominal values, joins, tables) and every bound by its domain,
        # so the rebuilt vocabulary must equal the saved one exactly.
        current = json.loads(json.dumps(featurizer.vocabulary()))
        saved = meta["vocabulary"]
        differing = sorted(key for key in set(current) | set(saved) if current.get(key) != saved.get(key))
        if differing:
            raise ValueError(f"the feature vocabulary differs from the model's ({', '.join(differing)}); "
                             "the v3 config changed since the model was built")
        if sorted(samples) != sorted(featurizer.tables):
            raise ValueError(f"model samples cover {sorted(samples)}, the config has {featurizer.tables}")
        network = make_model(featurizer, int(meta["recipe"]["hidden_size"]))
        network.load_state_dict({name[2:]: torch.from_numpy(np.asarray(values))
                                 for name, values in arrays.items() if name.startswith("w/")})
        network.eval()
        return MscnModel(featurizer, network, meta["label_min_log"], meta["label_max_log"])

    def _estimate(self, model: MscnModel, record: Any) -> Estimate:
        if record.relation not in self._relations():
            return "unsupported", None, f"model not trained on {record.relation} queries"
        return model.estimate(record)

    def smoke(self, workloads, query_limit: int = 2):
        """Train for one epoch on 64 training queries, then estimate a few queries."""
        from evaluation.eval_core.adapters.base import StageMetrics
        from evaluation.eval_core.workload_formats import load_queries

        from .base import _peak_rss_bytes

        started = time.perf_counter()
        model, report, _ = self._fit(self._snapshot_data(), limit=64,
                                     recipe={"epochs": 1, "batch_size": 32}, validation=False)
        results = []
        for workload in workloads:
            for record in list(load_queries(workload))[:query_limit]:
                status, estimate, _ = model.estimate(record)
                if not np.isfinite(estimate) or estimate < 0:
                    raise ValueError(f"smoke estimate {estimate} for {workload.workload_id} {record.query_id}")
                results.append({"workload": workload.workload_id, "query_id": record.query_id,
                                "status": status, "estimate": estimate})
        return StageMetrics(wall_seconds=time.perf_counter() - started, peak_rss_bytes=_peak_rss_bytes(),
                            detail={"queries": results, "training_queries": report["training_queries"]})

    def artifact_metadata(self) -> dict:
        meta = super().artifact_metadata()
        meta["parameter_count"] = meta.get("parameter_count") if isinstance(meta.get("parameter_count"), int) else None
        if meta["parameter_count"] is not None:
            meta.pop("parameter_count_reason", None)
        return meta
