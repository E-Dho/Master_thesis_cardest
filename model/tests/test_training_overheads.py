"""Training-loop overheads removed without changing results.

- validation reuses the training loop's predicate generator (its caches), and the
  metrics and checkpoints of a run are the same as with a fresh generator per
  validation;
- checkpoints built from precomputed static fields equal those built per save;
- ``validation.save_best_checkpoint: false`` keeps the validation curve but writes
  no ``checkpoint_best.pt``;
- literal-token statistics are computed once per domain.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

from model.src.config import load_simple_yaml
from model.src.data.full_join_sampler import SyntheticFullJoinSampleSource
from model.src.model.checkpoint import checkpoint_static_fields, save_resmade_checkpoint
from model.src.predicates import generation
from model.src.predicates.vocabulary import PredicateVocabularies
from model.src.training import resmade_trainer
from model.src.training.resmade_trainer import build_resmade_from_config, train_resmade_sample_source


def _config(output_directory: str, **validation) -> dict:
    config = load_simple_yaml("model/configs/resmade_smoke.yaml")
    config["model"]["hidden_sizes"] = [16]
    config["model"]["embedding_size"] = 4
    config["training"]["batch_size"] = 8
    config["training"]["steps_per_epoch"] = 6
    config["training"]["checkpoint_interval_steps"] = 3
    config["training"]["validation_interval_steps"] = 2
    config["validation"] = {"enabled": True, "interval_steps": 2, "fresh_sampler_batches": 2, "batch_size": 8,
                            "selection_metric": "validation_weighted_nll", "minimize": True, **validation}
    config["logging"]["output_directory"] = output_directory
    return config


def _metrics(directory: Path) -> list:
    rows = [json.loads(line) for line in (directory / "training_metrics.jsonl").read_text().splitlines()]
    for row in rows:
        for key in [k for k in row if "timing" in k]:
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
            continue
        elif key == "model_configuration":
            a, b = dict(left[key]), dict(right[key])
            a["logging"] = b["logging"] = None
            if a != b:
                return False
        elif left[key] != right[key]:
            return False
    return True


class ValidationGeneratorTest(unittest.TestCase):
    def test_runs_equal_a_fresh_generator_per_validation(self) -> None:
        original = resmade_trainer._run_validation

        def fresh_generator(*args, **kwargs):
            kwargs["context_generator"] = None  # the behaviour before the change
            return original(*args, **kwargs)

        with tempfile.TemporaryDirectory() as reused, tempfile.TemporaryDirectory() as fresh:
            calls = []

            def spy(*args, **kwargs):
                calls.append(kwargs.get("context_generator"))
                return original(*args, **kwargs)

            with mock.patch.object(resmade_trainer, "_run_validation", spy):
                train_resmade_sample_source(SyntheticFullJoinSampleSource(), _config(reused))
            self.assertEqual(len(calls), 3)
            self.assertIsNotNone(calls[0])
            self.assertTrue(all(generator is calls[0] for generator in calls))
            with mock.patch.object(resmade_trainer, "_run_validation", fresh_generator):
                train_resmade_sample_source(SyntheticFullJoinSampleSource(), _config(fresh))
            self.assertEqual(_metrics(Path(reused)), _metrics(Path(fresh)))
            for name in ("checkpoint_step_3.pt", "checkpoint_step_6.pt", "checkpoint_best.pt"):
                self.assertTrue(_equal_payload(torch.load(Path(reused) / name, weights_only=False),
                                               torch.load(Path(fresh) / name, weights_only=False)), name)

    def test_save_best_checkpoint_false_keeps_validation_without_the_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = _config(directory, save_best_checkpoint=False)
            config["training"]["checkpoint_interval_steps"] = 0
            result = train_resmade_sample_source(SyntheticFullJoinSampleSource(), config)
            files = sorted(p.name for p in Path(directory).glob("*.pt"))
            self.assertEqual(files, ["checkpoint_step_6.pt"])  # the final checkpoint only
            self.assertIsNone(result.best_checkpoint_path)
            self.assertEqual(len([r for r in _metrics(Path(directory)) if "validation_nll" in r]), 3)


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
        expected = generation.literal_token_stats({}, metadata)
        with mock.patch.object(generation, "_available_literal_token_count",
                               side_effect=AssertionError("recomputed")):
            self.assertEqual(generation.literal_token_stats({}, metadata), expected)


if __name__ == "__main__":
    unittest.main()
