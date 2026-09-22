from __future__ import annotations

from .external import ExternalCommandAdapter


class MscnAdapter(ExternalCommandAdapter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        recipe = self.config.adapter.get("recipe", {})
        expected = {
            "training_queries": 100_000,
            "epochs": 100,
            "sample_bits": 1_000,
            "batch_size": 1_024,
            "hidden_size": 256,
        }
        _require_recipe("MSCN", recipe, expected)
        if self.config.variant_id == "mscn_job_light_ranges_adapted":
            operators = set(recipe.get("operators", []))
            if operators != {"=", "<", ">", "<=", ">="}:
                raise ValueError("adapted MSCN ranges must enable =, <, >, <=, >=")
            if recipe.get("string_encoding") != "complete_domain_rank":
                raise ValueError("adapted MSCN ranges requires complete-domain ranks")


class DeepDbAdapter(ExternalCommandAdapter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        _require_recipe(
            "DeepDB",
            self.config.adapter.get("recipe", {}),
            {
                "sample_sizes": [10_000_000, 10_000_000, 1_000_000, 1_000_000, 1_000_000],
                "budget_factor": 5,
                "max_tables_per_ensemble": 3,
            },
        )


class NeuroCardAdapter(ExternalCommandAdapter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        recipe = self.config.adapter.get("recipe", {})
        _require_recipe("NeuroCard", recipe, {"progressive_samples": 8_000})
        if recipe.get("sampler") != "native_factorized_sampler":
            raise ValueError("NeuroCard must use its native factorized sampler")


class DistJoinAdapter(ExternalCommandAdapter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        recipe = self.config.adapter.get("recipe", {})
        if recipe.get("sampler") != "upstream_dynamic_sampler":
            raise ValueError("DistJoin must use the upstream dynamic sampler")
        if recipe.get("anpm") != "upstream_distjoin":
            raise ValueError("DistJoin must use its own ANPM implementation")


class OwnModelAdapter(ExternalCommandAdapter):
    pass


def _require_recipe(name: str, actual: dict, expected: dict) -> None:
    for key, expected_value in expected.items():
        if actual.get(key) != expected_value:
            raise ValueError(
                f"{name} recipe requires {key}={expected_value!r}; "
                f"received {actual.get(key)!r}"
            )
