from __future__ import annotations

from pathlib import Path
from typing import Any

from model.src.data.schema import FactorizationPlan, ModelMetadata
from model.src.model.factorization import factorization_plan_hash
from model.src.model.resmade import PredicateResMADE, PredicateResMADEConfig
from model.src.predicates.vocabulary import PredicateVocabularies


def checkpoint_static_fields(
    *,
    metadata: ModelMetadata,
    predicate_vocabularies: PredicateVocabularies,
) -> dict[str, Any]:
    """The checkpoint fields that depend only on the schema and vocabularies.

    They hold every value domain, so building them (dataclass conversion plus the
    schema hash) dominates a save on large domains; a training run builds them once
    and passes them to every ``save_resmade_checkpoint`` call.
    """

    return {
        "metadata": metadata.to_json_dict(),
        "predicate_vocabularies": predicate_vocabularies.to_json_dict(),
        "output_slices": metadata.model_output_slices,
        "join_cardinality": metadata.full_join_cardinality,
        "factorization_plan": metadata.factorization_plan.to_json_dict(),
        "factorization_hash": factorization_plan_hash(metadata.factorization_plan),
    }


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
    static_payload: dict[str, Any] | None = None,
) -> None:
    """Persist model state, optimizer state, schemas, vocabularies, and ordering.

    ``static_payload``: ``checkpoint_static_fields`` of the same metadata and
    vocabularies, computed once by the caller (the content is identical).
    """

    import torch

    static = static_payload if static_payload is not None else checkpoint_static_fields(
        metadata=metadata, predicate_vocabularies=predicate_vocabularies
    )
    payload = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "epoch": epoch,
        "step": step,
        "resmade_config": model.config.to_json_dict(),
        "model_configuration": config,
        "metadata": static["metadata"],
        "predicate_vocabularies": static["predicate_vocabularies"],
        "output_slices": static["output_slices"],
        "join_cardinality": static["join_cardinality"],
        "factorization": dict(config.get("factorization", {})),
        "factorization_plan": static["factorization_plan"],
        "factorization_hash": static["factorization_hash"],
        "anpm": dict(config.get("anpm", {})),
        "preparation_manifest_id": preparation_manifest_id,
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_resmade_checkpoint(
    path: str | Path,
    *,
    expected_schema_hash: str | None = None,
    expected_factorization_plan: FactorizationPlan | None = None,
    map_location: str = "cpu",
) -> tuple[PredicateResMADE, dict[str, Any]]:
    """Load a ResMADE checkpoint and reject incompatible schema hashes."""

    import torch

    try:
        payload = torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:  # Older PyTorch versions do not expose weights_only.
        payload = torch.load(path, map_location=map_location)
    metadata = ModelMetadata.from_json_dict(payload["metadata"])
    stored_factorization_hash = payload.get("factorization_hash")
    actual_factorization_hash = factorization_plan_hash(metadata.factorization_plan)
    if (
        stored_factorization_hash is not None
        and stored_factorization_hash != actual_factorization_hash
    ):
        raise ValueError(
            "checkpoint factorization metadata hash does not match checkpoint metadata"
        )
    actual_hash = metadata.schema_hash or metadata.stable_schema_hash()
    if expected_schema_hash is not None and expected_schema_hash != actual_hash:
        raise ValueError(
            f"checkpoint schema hash {actual_hash} does not match expected {expected_schema_hash}"
        )
    if expected_factorization_plan is not None:
        expected_hash = factorization_plan_hash(expected_factorization_plan)
        if expected_hash != actual_factorization_hash:
            raise ValueError(
                "checkpoint factorization plan does not match expected plan: "
                f"{actual_factorization_hash} != {expected_hash}"
            )
    resmade_config = PredicateResMADEConfig.from_json_dict(payload["resmade_config"])
    model = PredicateResMADE(resmade_config)
    model.load_state_dict(payload["model_state_dict"])
    return model, payload

