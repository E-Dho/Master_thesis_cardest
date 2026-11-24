"""Training-loop overheads removed without changing the trained model.

The same changes as in the trajectory trainer:

- one ``PredicateTrainingContextGenerator`` per run (it was built at every step and
  at every validation), passed to every step and validation;
- literal-occurrence counts, context diagnostics and embedding-gradient coverage
  are computed on the steps that write metrics (every ``validation_interval_steps``);
- checkpoints use the schema/vocabulary fields serialized once per run;
- literal-token statistics count the available tokens once per domain;
- per-phase step timings (``phase_timings`` / ``phase_timing_summary``).

The reference run below is the previous behaviour (a fresh generator and all
diagnostics at every step): losses, metrics apart from timings, and all checkpoints
are identical, with and without the rare auxiliary batch.
"""

from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from model.src.config import load_simple_yaml
from model.src.data.full_join_sampler import SyntheticFullJoinSampleSource
from model.src.data.sample_sources import sample_source_from_config
from model.src.predicates import generation

HAS_TORCH = importlib.util.find_spec("torch") is not None
if HAS_TORCH:
    import torch

    from model.src.model.checkpoint import checkpoint_static_fields, save_resmade_checkpoint
    from model.src.predicates.vocabulary import PredicateVocabularies
    from model.src.training import resmade_trainer
    from model.src.training.resmade_trainer import build_resmade_from_config, train_resmade_sample_source

# Diagnostics that are now computed on metric steps only: the literal-occurrence
# aggregate counts those steps, so it differs by design (as in the trajectory runs).
_DIAGNOSTIC_KEYS = ("predicate_literal_token_stats",)


def _config(output_directory: str, *, rare: bool) -> dict:
    config = load_simple_yaml("model/configs/resmade_smoke.yaml")
    config["model"]["hidden_sizes"] = [16]
    config["model"]["embedding_size"] = 4
    config["training"]["device"] = "cpu"
    config["training"]["batch_size"] = 8
    config["training"]["steps_per_epoch"] = 6
    config["training"]["checkpoint_interval_steps"] = 3
    config["training"]["validation_interval_steps"] = 2
    config["validation"] = {"enabled": True, "interval_steps": 2, "fresh_sampler_batches": 2, "batch_size": 8,
                            "selection_metric": "validation_weighted_nll", "minimize": True}
    config["logging"]["output_directory"] = output_directory
    if rare:
        config["predicate_generation"].update(
            {
                "strategy": "duet_batch_bounds",
                "table_subset_sampling": "neurocard_table_dropout_rooted",
                "enable_native_range_tokens": True,
                "normalize_predicate_probabilities": False,
                "wildcard_probability": 0.2,
                "equality_probability": 0.2,
                "lower_bound_probability": 0.2,
                "upper_bound_probability": 0.2,
                "native_range_probability": 0.2,
                "per_row_contexts": 1,
            }
        )
        config["importance_sampling"] = {"enabled": False}
        config["rare_support"] = {
            "enabled": True,
            "discovery": {"enabled": True, "root_data_only": True, "minimum_expected_context_support": 20,
                          "max_selected_strata": 4, "support_planning_steps": 20},
            "allocation": {"strategy": "support_deficit"},
            "diagnostics": {"enabled": False},
        }
        config["rare_auxiliary"] = {"enabled": True, "batch_size": 4, "beta": 0.25}
    return config


def _source(config: dict) -> object:
    if "rare_auxiliary" in config:
        return sample_source_from_config(config)
    return SyntheticFullJoinSampleSource()


def _metrics(directory: Path) -> list:
    rows = [json.loads(line) for line in (directory / "training_metrics.jsonl").read_text().splitlines()]
    for row in rows:
        for key in [k for k in row if "timing" in k or "per_second" in k or k in _DIAGNOSTIC_KEYS]:
            del row[key]
    return rows


def _equal_payload(left: dict, right: dict) -> bool:
    if left.keys() != right.keys():
        return False
    for key in left:
        if key == "model_state_dict":
            if not all(torch.equal(left[key][name], right[key][name]) for name in left[key]):
                return False
        elif key == "optimizer_state_dict":
            l_state, r_state = left[key]["state"], right[key]["state"]
            if l_state.keys() != r_state.keys():
                return False
            for index in l_state:
                for name, value in l_state[index].items():
                    other = r_state[index][name]
                    if isinstance(value, torch.Tensor):
                        if not torch.equal(value, other):
                            return False
                    elif value != other:
                        return False
        elif key == "model_configuration":
            a, b = dict(left[key]), dict(right[key])
            a["logging"] = b["logging"] = None
            if a != b:
                return False
        elif left[key] != right[key]:
            return False
    return True


@unittest.skipUnless(HAS_TORCH, "PyTorch is not installed")
class TrainingLoopEquivalenceTest(unittest.TestCase):
    def _run_pair(self, *, rare: bool) -> None:
        original_step = resmade_trainer._train_one_batch
        original_validation = resmade_trainer._run_validation

        def previous_step(*args, **kwargs):
            kwargs["context_generator"] = None  # a fresh generator per step
            kwargs["compute_expensive_diagnostics"] = True  # every step
            return original_step(*args, **kwargs)

        def previous_validation(*args, **kwargs):
            kwargs["context_generator"] = None
            return original_validation(*args, **kwargs)

        with tempfile.TemporaryDirectory() as new, tempfile.TemporaryDirectory() as old:
            generators, flags = [], []

            def spy(*args, **kwargs):
                generators.append(kwargs["context_generator"])
                flags.append(kwargs["compute_expensive_diagnostics"])
                return original_step(*args, **kwargs)

            config = _config(new, rare=rare)
            with mock.patch.object(resmade_trainer, "_train_one_batch", spy):
                new_result = train_resmade_sample_source(_source(config), config)
            self.assertTrue(all(generator is generators[0] for generator in generators))
            self.assertEqual(flags, [False, True, False, True, False, True])

            config = _config(old, rare=rare)
            with mock.patch.object(resmade_trainer, "_train_one_batch", previous_step), \
                    mock.patch.object(resmade_trainer, "_run_validation", previous_validation):
                old_result = train_resmade_sample_source(_source(config), config)

            self.assertEqual((new_result.first_loss, new_result.last_loss),
                             (old_result.first_loss, old_result.last_loss))
            self.assertEqual(_metrics(Path(new)), _metrics(Path(old)))
            for name in ("checkpoint_step_3.pt", "checkpoint_step_6.pt", "checkpoint_best.pt"):
                self.assertTrue(_equal_payload(torch.load(Path(new) / name, weights_only=False),
                                               torch.load(Path(old) / name, weights_only=False)), name)
            rows = [json.loads(line) for line in (Path(new) / "training_metrics.jsonl").read_text().splitlines()]
            summary = rows[-1]["phase_timing_summary"]
            for phase in ("batch_sampling", "predicate_context_generation", "main_forward", "backward",
                          "optimizer_step", "diagnostics", "total_optimizer_step"):
                self.assertIn(phase, summary)
            if rare:
                self.assertIn("rare_sampling_context_generation", summary)
                self.assertGreater(rows[-1]["rare_auxiliary"]["auxiliary_loss_scaled"], 0.0)
            self.assertTrue(rows[-1]["predicate_context_diagnostics"])
            self.assertTrue(rows[-1]["predicate_embedding_gradient_coverage"])

    def test_same_training_as_a_generator_and_diagnostics_per_step(self) -> None:
        self._run_pair(rare=False)

    def test_same_training_with_the_rare_auxiliary_batch(self) -> None:
        self._run_pair(rare=True)


@unittest.skipUnless(HAS_TORCH, "PyTorch is not installed")
class CheckpointStaticFieldsTest(unittest.TestCase):
    def test_precomputed_fields_give_the_same_checkpoint(self) -> None:
        source = SyntheticFullJoinSampleSource()
        config = load_simple_yaml("model/configs/resmade_smoke.yaml")
        model = build_resmade_from_config(source.metadata, config)
        vocabularies = PredicateVocabularies.from_metadata(source.metadata)
        optimizer = torch.optim.Adam(model.parameters())
        with tempfile.TemporaryDirectory() as directory:
            paths = [Path(directory) / "a.pt", Path(directory) / "b.pt"]
            arguments = dict(epoch=0, step=1, metadata=source.metadata, predicate_vocabularies=vocabularies,
                             config=config)
            save_resmade_checkpoint(paths[0], model, optimizer, **arguments)
            static = checkpoint_static_fields(metadata=source.metadata, predicate_vocabularies=vocabularies)
            save_resmade_checkpoint(paths[1], model, optimizer, static_payload=static, **arguments)
            self.assertTrue(_equal_payload(torch.load(paths[0], weights_only=False),
                                           torch.load(paths[1], weights_only=False)))


class LiteralTokenStatsTest(unittest.TestCase):
    def test_counts_are_cached_per_domain_and_unchanged(self) -> None:
        metadata = SyntheticFullJoinSampleSource().metadata
        generation._AVAILABLE_LITERAL_TOKEN_COUNTS.clear()
        reference = {}
        with mock.patch.object(generation, "_available_literal_token_count_cached",
                               generation._available_literal_token_count):
            reference = generation.literal_token_stats({}, metadata)
        calls = []
        original = generation._available_literal_token_count

        def counting(domain, op):
            calls.append(op)
            return original(domain, op)

        with mock.patch.object(generation, "_available_literal_token_count", counting):
            first = generation.literal_token_stats({}, metadata)
            count = len(calls)
            second = generation.literal_token_stats({}, metadata)
        self.assertEqual(first, reference)
        self.assertEqual(second, reference)
        self.assertGreater(count, 0)
        self.assertEqual(len(calls), count)  # the second call hit the cache


if __name__ == "__main__":
    unittest.main()
