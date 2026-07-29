from __future__ import annotations

from pathlib import Path
from typing import Any

from model.src.data.schema import ModelMetadata
from model.src.model.factorization import FactorizationConfig
from model.src.model.resmade import PredicateResMADE, PredicateResMADEConfig
from model.src.predicates.vocabulary import PredicateVocabularies


def save_resmade_checkpoint(
    path: str | Path,
    model: PredicateResMADE,
    optimizer: Any,
    *,
    epoch: int,
    step: int,
    metadata: ModelMetadata,
    predicate_vocabularies: PredicateVocabularies,
    config: dict[str, Any],
    preparation_manifest_id: str | None = None,
) -> None:
    """Persist model state, optimizer state, schemas, vocabularies, and ordering."""

    import torch

    payload = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "epoch": epoch,
        "step": step,
        "resmade_config": model.config.__dict__,
        "model_configuration": config,
        "metadata": metadata.to_json_dict(),
        "predicate_vocabularies": predicate_vocabularies.to_json_dict(),
        "output_slices": metadata.output_slices,
        "join_cardinality": metadata.full_join_cardinality,
        "factorization": FactorizationConfig().__dict__,
        "preparation_manifest_id": preparation_manifest_id,
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_resmade_checkpoint(
    path: str | Path,
    *,
    expected_schema_hash: str | None = None,
    map_location: str = "cpu",
) -> tuple[PredicateResMADE, dict[str, Any]]:
    """Load a ResMADE checkpoint and reject incompatible schema hashes."""

    import torch

    payload = torch.load(path, map_location=map_location)
    metadata = ModelMetadata.from_json_dict(payload["metadata"])
    actual_hash = metadata.schema_hash or metadata.stable_schema_hash()
    if expected_schema_hash is not None and expected_schema_hash != actual_hash:
        raise ValueError(
            f"checkpoint schema hash {actual_hash} does not match expected {expected_schema_hash}"
        )
    resmade_config = PredicateResMADEConfig(
        predicate_input_bins=tuple(payload["resmade_config"]["predicate_input_bins"]),
        data_output_bins=tuple(payload["resmade_config"]["data_output_bins"]),
        hidden_sizes=tuple(payload["resmade_config"]["hidden_sizes"]),
        residual_connections=bool(payload["resmade_config"]["residual_connections"]),
        direct_io_connections=bool(payload["resmade_config"]["direct_io_connections"]),
        activation=payload["resmade_config"]["activation"],
        input_encoding=payload["resmade_config"]["input_encoding"],
        embedding_size=int(payload["resmade_config"]["embedding_size"]),
        residual_dropout=float(payload["resmade_config"]["residual_dropout"]),
        fixed_ordering=bool(payload["resmade_config"]["fixed_ordering"]),
    )
    model = PredicateResMADE(resmade_config)
    model.load_state_dict(payload["model_state_dict"])
    return model, payload

