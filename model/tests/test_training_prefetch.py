"""training.prefetch_workers: the same training with the CPU part in other processes.

A sampler process makes the training loop's sample-source calls in its order
(main batch, then rare batch, of each step), so a buffered live sampler whose main
and conditional draws share random state yields the same rows, and workers
prepare each step's contexts, weights, forced rare contexts and token IDs. Runs with 0 and with 2-3 workers are
compared: losses, metrics apart from timings, the summary's sampler counters and
all checkpoints (model and optimizer state) are identical, with residual dropout
on so that a shifted torch random stream would show.
"""

from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from model.src.data.full_join_sampler import FullJoinBatch, SyntheticFullJoinSampleSource
from model.src.data.sample_sources import sample_source_from_config
from model.tests.test_training_overheads import _config

HAS_TORCH = importlib.util.find_spec("torch") is not None
if HAS_TORCH:
    import torch

    from model.src.training.resmade_trainer import train_resmade_sample_source


class BufferedSource:
    """Live-sampler stand-in: ignores the seed, draws blocks from its own generator."""

    def __init__(self, block_rows: int = 20) -> None:
        self.inner = SyntheticFullJoinSampleSource()
        self.block_rows = block_rows
        self._rng = np.random.default_rng(7)
        self._buffer: np.ndarray | None = None
        self._cursor = 0
        self.fresh_rows_drawn = 0
        self.sampler_run_calls = 0
        self.sample_batches_generated = 0

    @property
    def metadata(self):
        return self.inner.metadata

    @property
    def join_cardinality(self) -> int:
        return self.inner.join_cardinality

    def discard_buffer(self) -> None:
        self._buffer, self._cursor = None, 0

    def batches(self, batch_size: int, *, seed: int = 0) -> FullJoinBatch:
        del seed
        pieces, fresh, remaining = [], 0, batch_size
        while remaining:
            if self._buffer is None or self._cursor >= len(self._buffer):
                rows = self.inner.dataset.encoded_rows
                self._buffer = rows[self._rng.integers(0, len(rows), size=self.block_rows)]
                self._cursor = 0
                fresh += self.block_rows
                self.sampler_run_calls += 1
                self.sample_batches_generated += 1
            take = min(remaining, len(self._buffer) - self._cursor)
            pieces.append(self._buffer[self._cursor:self._cursor + take])
            self._cursor += take
            remaining -= take
        self.fresh_rows_drawn += fresh
        return FullJoinBatch(encoded_values=np.concatenate(pieces), column_metadata=self.metadata.columns,
                             fresh_rows_drawn=fresh, fixture_rows_reused=0)


class DelegatingWrapper:
    """A wrapper like FactorizedMetadataSampleSource: its base's counter as a read-only property."""

    def __init__(self, base) -> None:
        self.base_source = base

    @property
    def metadata(self):
        return self.base_source.metadata

    @property
    def join_cardinality(self) -> int:
        return self.base_source.join_cardinality

    @property
    def sampler_run_calls(self) -> int:
        return self.base_source.sampler_run_calls

    def discard_buffer(self) -> None:
        self.base_source.discard_buffer()

    def batches(self, batch_size: int, *, seed: int = 0) -> FullJoinBatch:
        return self.base_source.batches(batch_size, seed=seed)


class SharedStateRareSource:
    """Main and rare draws share one random state, as in NeuroCard's sampler.

    Main rows come from ``_rng``; each rare draw first consumes ``_rng`` and its
    rows depend on that draw, so the main stream depends on when rare draws happen.
    """

    def __init__(self, rare_source) -> None:
        self.base_source = rare_source
        self._rows = _synthetic_rows(rare_source)
        self._rng = np.random.default_rng(11)
        self.sampler_run_calls = 0

    @property
    def metadata(self):
        return self.base_source.metadata

    @property
    def join_cardinality(self) -> int:
        return self.base_source.join_cardinality

    def discard_buffer(self) -> None:
        pass

    def batches(self, batch_size: int, *, seed: int = 0) -> FullJoinBatch:
        del seed
        self.sampler_run_calls += 1
        rows = self._rows[self._rng.integers(0, len(self._rows), size=batch_size)]
        return FullJoinBatch(encoded_values=rows, column_metadata=self.metadata.columns,
                             fresh_rows_drawn=batch_size, fixture_rows_reused=0)

    def rare_batches(self, batch_size: int, *, seed: int = 0) -> FullJoinBatch:
        shift = int(self._rng.integers(0, 1000))
        return self.base_source.rare_batches(batch_size, seed=seed + shift)


def _synthetic_rows(source) -> np.ndarray:
    current = source
    while not hasattr(current, "dataset"):
        current = current.base_source
    return current.dataset.encoded_rows


class FailingSource(BufferedSource):
    def batches(self, batch_size: int, *, seed: int = 0) -> FullJoinBatch:
        if self.sampler_run_calls >= 2:
            raise RuntimeError("sampler broke")
        return super().batches(batch_size, seed=seed)


def _prefetch_config(directory: str, *, rare: bool, workers: int) -> dict:
    config = _config(directory, rare=rare)
    config["validation"]["enabled"] = False
    config["model"]["residual_dropout"] = 0.2
    config["training"]["steps_per_epoch"] = 11
    config["training"]["checkpoint_interval_steps"] = 4
    if workers:
        config["training"]["prefetch_workers"] = workers
    return config


def _strip(row: dict) -> dict:
    return {k: v for k, v in row.items()
            if "timing" not in k and "per_second" not in k and "seconds" not in k and k != "prefetch_workers"}


@unittest.skipUnless(HAS_TORCH, "PyTorch is not installed")
class PrefetchEquivalenceTest(unittest.TestCase):
    def _compare(self, make_source, *, rare: bool, workers: int) -> None:
        with tempfile.TemporaryDirectory() as serial, tempfile.TemporaryDirectory() as parallel:
            results = []
            for directory, count in ((serial, 0), (parallel, workers)):
                config = _prefetch_config(directory, rare=rare, workers=count)
                source = make_source(config)
                results.append((train_resmade_sample_source(source, config), source))
            (left, left_source), (right, right_source) = results
            self.assertEqual((left.first_loss, left.last_loss), (right.first_loss, right.last_loss))
            rows = [[_strip(json.loads(line)) for line in (Path(d) / "training_metrics.jsonl").read_text().splitlines()]
                    for d in (serial, parallel)]
            self.assertEqual(rows[0], rows[1])
            summaries = [_strip(json.loads((Path(d) / "training_summary.json").read_text())) for d in (serial, parallel)]
            for key in ("fresh_sampler_rows", "fixture_rows_reused", "sampler_run_calls", "rare_support",
                        "rare_auxiliary", "predicate_token_coverage", "generated_predicate_contexts"):
                if key in summaries[0]:
                    value = summaries[0][key]
                    if key == "rare_support":
                        value = {k: v for k, v in value.items() if "seconds" not in k}
                        other = {k: v for k, v in summaries[1][key].items() if "seconds" not in k}
                    else:
                        other = summaries[1][key]
                    self.assertEqual(value, other, key)
            for name in ("fresh_rows_drawn", "sampler_run_calls", "rare_rows_drawn", "selected_stratum_sample_count"):
                self.assertEqual(getattr(left_source, name, None), getattr(right_source, name, None), name)
            for name in sorted(p.name for p in Path(serial).glob("*.pt")):
                a = torch.load(Path(serial) / name, weights_only=False)
                b = torch.load(Path(parallel) / name, weights_only=False)
                for key in a["model_state_dict"]:
                    self.assertTrue(torch.equal(a["model_state_dict"][key], b["model_state_dict"][key]), (name, key))
                for index, state in a["optimizer_state_dict"]["state"].items():
                    for key, value in state.items():
                        if isinstance(value, torch.Tensor):
                            self.assertTrue(torch.equal(value, b["optimizer_state_dict"]["state"][index][key]))
            parallel_rows = [json.loads(line) for line in (Path(parallel) / "training_metrics.jsonl").read_text().splitlines()]
            self.assertEqual(parallel_rows[-1]["prefetch_workers"], workers)
            self.assertIn("predicate_context_generation", parallel_rows[-1]["worker_phase_timing_summary"])
            self.assertIn("prefetch_wait", parallel_rows[-1]["phase_timing_summary"])

    def test_buffered_live_like_source(self) -> None:
        self._compare(lambda config: BufferedSource(), rare=False, workers=2)

    def test_rare_auxiliary_batch(self) -> None:
        self._compare(sample_source_from_config, rare=True, workers=3)

    def test_counters_behind_read_only_wrapper_properties(self) -> None:
        self._compare(lambda config: DelegatingWrapper(BufferedSource()), rare=False, workers=2)

    def test_main_and_rare_draws_sharing_random_state(self) -> None:
        self._compare(lambda config: SharedStateRareSource(sample_source_from_config(config)), rare=True, workers=3)

    def test_sampler_errors_reach_the_training_process(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = _prefetch_config(directory, rare=False, workers=2)
            with self.assertRaisesRegex(RuntimeError, "sampler broke"):
                train_resmade_sample_source(FailingSource(), config)

    def test_live_sampler_with_validation_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = _prefetch_config(directory, rare=False, workers=2)
            config["validation"]["enabled"] = True
            with self.assertRaisesRegex(ValueError, "validation.enabled: false"):
                train_resmade_sample_source(BufferedSource(), config)


if __name__ == "__main__":
    unittest.main()
