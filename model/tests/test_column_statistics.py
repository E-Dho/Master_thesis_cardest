from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from model.scripts import compute_column_ordering

from model.src.data.column_statistics import (
    bin_columns,
    data_column_indices,
    marginal_entropies,
    ordering_from_scores,
    pairwise_mutual_information,
    reversed_ordering,
)
from model.src.data.complete_domain_preparation import (
    build_complete_metadata,
    build_manifest_payload,
    encode_sample_dataframe,
    preparation_stats,
    write_prepared_artifacts,
)
from model.src.data.schema import ColumnKind, ColumnMetadata, ModelMetadata
from model.src.model.factorization import FactorizationConfig
from model.tests.test_column_ordering import _sample_frame, _tables

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def _metadata(domains):
    return ModelMetadata(
        columns=tuple(
            ColumnMetadata(name, ColumnKind.DATA, tuple(range(size)), table="t")
            for name, size in domains
        ),
        full_join_cardinality=1.0,
        join_root="t",
        join_tables=("t",),
        join_edges=(),
    )


class MarginalEntropyTest(unittest.TestCase):
    """Entropy is measured over the whole domain, not over what was sampled."""

    def test_a_constant_column_has_no_entropy(self) -> None:
        metadata = _metadata([("t:a", 4)])
        rows = np.zeros((100, 1), dtype=np.int64)
        self.assertEqual(marginal_entropies(rows, metadata), (0.0,))

    def test_a_uniform_column_reaches_log_of_its_support(self) -> None:
        metadata = _metadata([("t:a", 4)])
        rows = np.tile(np.arange(4), 25).reshape(-1, 1)
        self.assertAlmostEqual(marginal_entropies(rows, metadata)[0], 2.0)

    def test_unseen_domain_values_do_not_add_entropy(self) -> None:
        # A value the sample never shows has probability zero, and the
        # convention 0 log 0 = 0 is what keeps a wide, sparsely sampled column
        # from scoring high for being wide.
        wide = _metadata([("t:a", 1024)])
        narrow = _metadata([("t:a", 4)])
        rows = np.tile(np.arange(4), 25).reshape(-1, 1)
        self.assertAlmostEqual(
            marginal_entropies(rows, wide)[0], marginal_entropies(rows, narrow)[0]
        )

    def test_rows_that_do_not_match_the_schema_are_refused(self) -> None:
        metadata = _metadata([("t:a", 4), ("t:b", 4)])
        with self.assertRaisesRegex(ValueError, "columns"):
            marginal_entropies(np.zeros((10, 1), dtype=np.int64), metadata)

    def test_an_empty_sample_is_refused(self) -> None:
        metadata = _metadata([("t:a", 4)])
        with self.assertRaisesRegex(ValueError, "no rows"):
            marginal_entropies(np.zeros((0, 1), dtype=np.int64), metadata)


class BinningTest(unittest.TestCase):
    """Binning exists so a 134k-value column does not estimate noise."""

    def test_a_column_that_fits_is_not_pooled(self) -> None:
        metadata = _metadata([("t:a", 8)])
        rows = np.tile(np.arange(8), 10).reshape(-1, 1)
        binned = bin_columns(rows, metadata, max_bins=64)
        self.assertEqual(binned.bin_counts[0], 8)
        self.assertEqual(len(np.unique(binned.assignments[:, 0])), 8)

    def test_the_bin_count_is_what_was_used_not_what_was_reserved(self) -> None:
        # A column whose observed values all fit still needs a fallback bin for
        # the domain values the sample never showed, and nothing lands in it.
        # Counting the lookup's range instead reported one bin too many for
        # every partially observed column -- four values seen in a hundred-value
        # domain were reported as five bins.
        metadata = _metadata([("t:a", 100)])
        rows = np.tile(np.arange(4), 25).reshape(-1, 1)
        binned = bin_columns(rows, metadata, max_bins=64)
        self.assertEqual(binned.bin_counts[0], 4)
        self.assertEqual(len(np.unique(binned.assignments[:, 0])), 4)

    def test_a_wide_column_is_capped_at_max_bins(self) -> None:
        metadata = _metadata([("t:a", 1000)])
        rng = np.random.default_rng(0)
        rows = rng.integers(0, 1000, size=(5000, 1))
        binned = bin_columns(rows, metadata, max_bins=16)
        self.assertEqual(binned.bin_counts[0], 16)

    def test_the_most_common_values_keep_their_own_bins(self) -> None:
        # The tail shares a bin; the head must not, or the mass structure
        # mutual information reads would be destroyed.
        metadata = _metadata([("t:a", 10)])
        column = [0] * 50 + [1] * 30 + [2] * 10 + list(range(3, 10))
        rows = np.asarray(column, dtype=np.int64).reshape(-1, 1)
        binned = bin_columns(rows, metadata, max_bins=4)
        assignments = binned.assignments[:, 0]
        self.assertEqual(assignments[0], 0)
        self.assertEqual(assignments[50], 1)
        self.assertEqual(assignments[80], 2)
        self.assertEqual(len(set(assignments[90:].tolist())), 1)
        self.assertEqual(assignments[90], 3)

    def test_too_few_bins_is_refused(self) -> None:
        metadata = _metadata([("t:a", 8)])
        rows = np.zeros((10, 1), dtype=np.int64)
        with self.assertRaisesRegex(ValueError, "at least two"):
            bin_columns(rows, metadata, max_bins=1)


class MutualInformationTest(unittest.TestCase):
    """The statistic has to separate dependence from independence."""

    def rows(self, pairs):
        return np.asarray(pairs, dtype=np.int64)

    def test_independent_columns_score_zero(self) -> None:
        metadata = _metadata([("t:a", 2), ("t:b", 2)])
        rows = self.rows([[a, b] for a in range(2) for b in range(2)] * 50)
        binned = bin_columns(rows, metadata, max_bins=8)
        matrix = pairwise_mutual_information(binned, (0, 1))
        self.assertAlmostEqual(matrix[0, 1], 0.0, places=9)

    def test_a_copied_column_scores_its_entropy(self) -> None:
        metadata = _metadata([("t:a", 4), ("t:b", 4)])
        values = np.tile(np.arange(4), 50)
        rows = np.stack([values, values], axis=1)
        binned = bin_columns(rows, metadata, max_bins=8)
        matrix = pairwise_mutual_information(binned, (0, 1))
        self.assertAlmostEqual(matrix[0, 1], 2.0, places=9)

    def test_the_matrix_is_symmetric_with_a_zero_diagonal(self) -> None:
        metadata = _metadata([("t:a", 4), ("t:b", 4), ("t:c", 4)])
        rng = np.random.default_rng(1)
        rows = rng.integers(0, 4, size=(400, 3))
        binned = bin_columns(rows, metadata, max_bins=8)
        matrix = pairwise_mutual_information(binned, (0, 1, 2))
        np.testing.assert_allclose(matrix, matrix.T)
        np.testing.assert_allclose(np.diag(matrix), np.zeros(3))

    def test_it_is_never_negative(self) -> None:
        # Floating-point sums can land just below zero for independent
        # columns, and a negative coupling score would sort a column to the
        # wrong end.
        metadata = _metadata([("t:a", 2), ("t:b", 2)])
        rows = self.rows([[a, b] for a in range(2) for b in range(2)] * 50)
        binned = bin_columns(rows, metadata, max_bins=8)
        self.assertGreaterEqual(pairwise_mutual_information(binned, (0, 1))[0, 1], 0.0)

    def test_independent_columns_of_unequal_width_score_exactly_zero(self) -> None:
        # Two things at once, both of which need the bin counts to differ.
        # The joint histogram is flattened as left * right_bins + right, and
        # the transposed spelling only coincides with it when the two columns
        # have the same number of bins -- every other test here does.  And on
        # this particular independent table the unclamped sum lands at
        # -4.4e-16, so the nonnegativity clamp is what keeps a coupling score
        # from sorting a column to the wrong end on nothing but rounding.
        counts_left = [23, 9, 48, 29, 10, 14, 49, 14, 10]
        counts_right = [21, 15]
        left, right = [], []
        for first, left_count in enumerate(counts_left):
            for second, right_count in enumerate(counts_right):
                left.extend([first] * (left_count * right_count))
                right.extend([second] * (left_count * right_count))
        metadata = _metadata([("t:a", len(counts_left)), ("t:b", len(counts_right))])
        rows = np.stack([np.asarray(left), np.asarray(right)], axis=1)
        binned = bin_columns(rows, metadata, max_bins=64)
        self.assertEqual(binned.bin_counts, (9, 2))
        self.assertEqual(pairwise_mutual_information(binned, (0, 1))[0, 1], 0.0)

    def test_relabelling_a_column_does_not_change_it(self) -> None:
        # Frequency-rank binning is used for every column, ordinal or not,
        # because mutual information depends on the partition and not on the
        # labels.  If that were false the binning would need semantics.
        metadata = _metadata([("t:a", 4), ("t:b", 4)])
        rng = np.random.default_rng(2)
        rows = rng.integers(0, 4, size=(2000, 2))
        relabelled = rows.copy()
        relabelled[:, 0] = np.asarray([3, 1, 0, 2])[rows[:, 0]]
        first = pairwise_mutual_information(bin_columns(rows, metadata), (0, 1))
        second = pairwise_mutual_information(bin_columns(relabelled, metadata), (0, 1))
        self.assertAlmostEqual(first[0, 1], second[0, 1], places=9)


class OrderingTest(unittest.TestCase):
    """Only the scored columns move, and only among their own positions."""

    def metadata(self):
        return ModelMetadata(
            columns=(
                ColumnMetadata("t:a", ColumnKind.DATA, (0, 1), table="t"),
                ColumnMetadata("t:b", ColumnKind.DATA, (0, 1), table="t"),
                ColumnMetadata("t:c", ColumnKind.DATA, (0, 1), table="t"),
                ColumnMetadata("__in_t", ColumnKind.INDICATOR, (0, 1), table="t"),
                ColumnMetadata("__fanout_t", ColumnKind.FANOUT, (1, 2), table="t"),
            ),
            full_join_cardinality=1.0,
            join_root="t",
            join_tables=("t",),
            join_edges=(),
        )

    def test_the_highest_score_goes_first(self) -> None:
        order = ordering_from_scores(self.metadata(), {0: 1.0, 1: 3.0, 2: 2.0})
        self.assertEqual(order[:3], ["t:b", "t:c", "t:a"])

    def test_reversing_inverts_the_scored_slots_only(self) -> None:
        metadata = self.metadata()
        order = ordering_from_scores(metadata, {0: 1.0, 1: 3.0, 2: 2.0})
        flipped = reversed_ordering(order, (0, 1, 2))
        self.assertEqual(flipped[:3], ["t:a", "t:c", "t:b"])
        self.assertEqual(flipped[3:], ["__in_t", "__fanout_t"])

    def test_reversing_is_exact_even_when_the_statistic_ties(self) -> None:
        # Ranking the same scores ascending would keep tied columns in their
        # relative order, so the two arms would stop being opposites exactly
        # where the statistic had nothing to say.
        metadata = self.metadata()
        order = ordering_from_scores(metadata, {0: 1.0, 1: 1.0, 2: 1.0})
        flipped = reversed_ordering(order, (0, 1, 2))
        self.assertEqual(flipped[:3], list(reversed(order[:3])))

    def test_a_slot_outside_the_order_is_refused(self) -> None:
        with self.assertRaisesRegex(ValueError, "outside the order"):
            reversed_ordering(["a", "b"], (0, 5))

    def test_the_join_structure_keeps_its_positions(self) -> None:
        # An arm has to differ from the control in the DATA order alone; moving
        # the indicator or fanout blocks would confound the comparison.
        order = ordering_from_scores(self.metadata(), {0: 1.0, 1: 3.0, 2: 2.0})
        self.assertEqual(order[3:], ["__in_t", "__fanout_t"])

    def test_it_is_always_a_permutation(self) -> None:
        order = ordering_from_scores(self.metadata(), {0: 1.0, 1: 3.0, 2: 2.0})
        self.assertEqual(
            sorted(order), sorted(column.name for column in self.metadata().columns)
        )

    def test_a_flat_statistic_reproduces_the_control(self) -> None:
        # Ties break on current position, so a statistic that says nothing
        # leaves the order alone rather than shuffling it arbitrarily.
        metadata = self.metadata()
        order = ordering_from_scores(metadata, {0: 1.0, 1: 1.0, 2: 1.0})
        self.assertEqual(order, [column.name for column in metadata.columns])

    def test_scored_columns_move_only_among_their_own_positions(self) -> None:
        # Every other case here has the scored columns sitting at 0..k, where
        # writing into the first k slots and writing into the scored slots are
        # the same thing.  Interleaved, they are not.
        metadata = ModelMetadata(
            columns=(
                ColumnMetadata("t:a", ColumnKind.DATA, (0, 1), table="t"),
                ColumnMetadata("__in_t", ColumnKind.INDICATOR, (0, 1), table="t"),
                ColumnMetadata("t:b", ColumnKind.DATA, (0, 1), table="t"),
                ColumnMetadata("__fanout_t", ColumnKind.FANOUT, (1, 2), table="t"),
                ColumnMetadata("t:c", ColumnKind.DATA, (0, 1), table="t"),
            ),
            full_join_cardinality=1.0,
            join_root="t",
            join_tables=("t",),
            join_edges=(),
        )
        order = ordering_from_scores(metadata, {0: 1.0, 2: 3.0, 4: 2.0})
        self.assertEqual(order, ["t:b", "__in_t", "t:c", "__fanout_t", "t:a"])

    def test_only_data_columns_are_scored(self) -> None:
        metadata = self.metadata()
        self.assertEqual(data_column_indices(metadata), (0, 1, 2))


class DrawRowsTest(unittest.TestCase):
    """The sample has to be the size that was asked for."""

    def source(self, width, per_batch):
        class Oversized:
            def __init__(self):
                self.seeds = []

            def batches(self, batch_size, *, seed=0):
                self.seeds.append(seed)
                return types.SimpleNamespace(
                    encoded_values=np.zeros((per_batch, width), dtype=np.int64),
                    fresh_rows_drawn=0,
                    fixture_rows_reused=per_batch,
                )

        return Oversized()

    def test_each_batch_is_drawn_with_its_own_seed(self) -> None:
        # The fixture and synthetic sources build a fresh default_rng(seed) per
        # call, so one seed for every batch makes a large sample a handful of
        # identical draws while the report still states the large number.
        source = self.source(3, 10)
        compute_column_ordering.draw_rows(source, 40, 10, base_seed=7)
        self.assertEqual(source.seeds, [7, 8, 9, 10])

    def test_the_seeds_start_from_the_requested_one(self) -> None:
        source = self.source(3, 10)
        compute_column_ordering.draw_rows(source, 20, 10)
        self.assertEqual(source.seeds, [0, 1])

    def test_it_reports_how_much_of_the_sample_was_fresh(self) -> None:
        # A fixture source resamples a fixed set with replacement, so a large
        # sample is not a large number of distinct rows, and the statistics
        # should say which they were estimated from.
        drawn = compute_column_ordering.draw_rows(self.source(3, 10), 30, 10)
        self.assertEqual(drawn.fresh_rows_drawn, 0)
        self.assertEqual(drawn.fixture_rows_reused, 30)

    def test_a_sampler_that_over_delivers_is_trimmed(self) -> None:
        # Both shipped sources return exactly what they are asked for, so
        # nothing in the pipeline exercises this -- but a sample larger than
        # the one recorded in the provenance is a statistic nobody can
        # reproduce.
        drawn = compute_column_ordering.draw_rows(self.source(3, 40), 100, 32)
        self.assertEqual(drawn.rows.shape, (100, 3))

    def test_a_sampler_that_returns_nothing_is_refused(self) -> None:
        with self.assertRaisesRegex(SystemExit, "no rows"):
            compute_column_ordering.draw_rows(self.source(3, 0), 10, 5)

    def test_a_nonpositive_row_count_is_refused(self) -> None:
        with self.assertRaisesRegex(SystemExit, "--rows"):
            compute_column_ordering.draw_rows(self.source(3, 10), 0, 5)


class BuildOrderingsTest(unittest.TestCase):
    def test_a_schema_with_no_data_columns_is_refused(self) -> None:
        metadata = ModelMetadata(
            columns=(
                ColumnMetadata("__in_t", ColumnKind.INDICATOR, (0, 1), table="t"),
                ColumnMetadata("__fanout_t", ColumnKind.FANOUT, (1, 2), table="t"),
            ),
            full_join_cardinality=1.0,
            join_root="t",
            join_tables=("t",),
            join_edges=(),
        )
        rows = np.zeros((10, 2), dtype=np.int64)
        with self.assertRaisesRegex(SystemExit, "no DATA columns"):
            compute_column_ordering.build_orderings(metadata, rows, 8)


class ComputeOrderingCommandTest(unittest.TestCase):
    """End to end, against a fixture sampler rather than a live NeuroCard."""

    def prepared(self, root: Path) -> Path:
        tables, spec = _tables()
        metadata = build_complete_metadata(tables, spec)
        frame = pd.concat([_sample_frame([10, 12, 9999, 13])] * 25, ignore_index=True)
        encoded = encode_sample_dataframe(frame, metadata)
        directory = root / "prepared"
        write_prepared_artifacts(
            prepared_directory=directory,
            manifest_payload=build_manifest_payload(
                metadata=metadata, spec=spec, sample_rows=len(frame)
            ),
            encoded_rows=encoded.encoded_rows,
            stats=preparation_stats(
                metadata=metadata,
                encoded_sample=encoded,
                factorization_config=FactorizationConfig(enabled=False),
            ),
        )
        return directory

    def config(self, root: Path, prepared: Path) -> Path:
        path = root / "config.yaml"
        path.write_text(
            "\n".join(
                [
                    "dataset:",
                    "  type: neurocard_full_join",
                    f"  prepared_directory: {prepared}",
                    "  sampling_mode: fixture",
                    "model:",
                    "  type: predicate_resmade",
                    "  hidden_sizes: [8]",
                    "training:",
                    "  batch_size: 2",
                    "  learning_rate: 0.1",
                    "  epochs: 1",
                    "  steps_per_epoch: 1",
                    "inference:",
                    "  progressive_sampling: false",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        return path

    def run_script(self, config: Path, output: Path, *extra: str):
        return subprocess.run(
            [
                sys.executable,
                "-B",
                "-m",
                "model.scripts.compute_column_ordering",
                "--config",
                str(config),
                "--output-directory",
                str(output),
                "--rows",
                "100",
                *extra,
            ],
            capture_output=True,
            text=True,
            cwd=REPOSITORY_ROOT,
        )

    def test_it_writes_one_file_per_criterion(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "orderings"
            result = self.run_script(self.config(root, self.prepared(root)), output)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(
                sorted(entry.name for entry in output.iterdir()),
                [
                    "column_statistics.json",
                    "coupling.json",
                    "coupling_reversed.json",
                    "domain_size.json",
                    "entropy.json",
                    "entropy_reversed.json",
                ],
            )

    def test_every_order_is_a_permutation_permute_accepts(self) -> None:
        # The output's only job is to be read by permute_prepared_columns, so
        # it has to satisfy that script's contract, not merely look like a list.
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            prepared = self.prepared(root)
            output = root / "orderings"
            self.run_script(self.config(root, prepared), output)
            expected = sorted(
                column["name"]
                for column in json.loads(
                    (prepared / "manifest.json").read_text(encoding="utf-8")
                )["metadata"]["columns"]
            )
            for name in (
                "coupling",
                "coupling_reversed",
                "entropy",
                "entropy_reversed",
                "domain_size",
            ):
                payload = json.loads((output / f"{name}.json").read_text(encoding="utf-8"))
                self.assertEqual(sorted(payload["order"]), expected)
                self.assertEqual(payload["criterion"], name)

    def test_the_two_coupling_arms_are_opposites(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "orderings"
            self.run_script(self.config(root, self.prepared(root)), output)
            forward = json.loads((output / "coupling.json").read_text(encoding="utf-8"))
            backward = json.loads(
                (output / "coupling_reversed.json").read_text(encoding="utf-8")
            )
            data_forward = [name for name in forward["order"] if ":" in name]
            data_backward = [name for name in backward["order"] if ":" in name]
            self.assertEqual(data_forward, list(reversed(data_backward)))

    def test_the_entropy_arms_are_opposites(self) -> None:
        # Both directions are generated because both readings of the mechanism
        # are defensible: an early column's predicate conditions every later
        # factor, while a late column's own factor is the one conditioned.
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "orderings"
            self.run_script(self.config(root, self.prepared(root)), output)
            forward = json.loads((output / "entropy.json").read_text(encoding="utf-8"))
            backward = json.loads(
                (output / "entropy_reversed.json").read_text(encoding="utf-8")
            )
            data_forward = [name for name in forward["order"] if ":" in name]
            data_backward = [name for name in backward["order"] if ":" in name]
            self.assertEqual(data_forward, list(reversed(data_backward)))

    def test_the_sampler_seed_is_honoured_and_reproducible(self) -> None:
        # Same seed, same figures: an arm has to be traceable to a sample
        # someone else can draw again.  Different seeds, different figures:
        # otherwise the flag records a provenance the run did not have.
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = self.config(root, self.prepared(root))
            figures = []
            seeds = []
            for seed in (0, 0, 1, 2, 3):
                output = root / f"orderings_{len(figures)}"
                result = self.run_script(
                    config, output, "--sampler-seed", str(seed), "--rows", "300"
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                payload = json.loads(
                    (output / "column_statistics.json").read_text(encoding="utf-8")
                )
                seeds.append(payload["sampler_seed"])
                # The provenance records the seed, so whole-file equality would
                # differ even when the sample behind it did not.  What has to
                # move is the figures.
                figures.append(
                    json.dumps(
                        [
                            payload["marginal_entropy_bits"],
                            payload["total_mutual_information_bits"],
                        ],
                        sort_keys=True,
                    )
                )
            self.assertEqual(seeds, [0, 0, 1, 2, 3])
            self.assertEqual(figures[0], figures[1])
            self.assertGreater(len(set(figures)), 1)

    def test_the_statistics_record_what_chose_the_order(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "orderings"
            self.run_script(self.config(root, self.prepared(root)), output, "--max-bins", "8")
            statistics = json.loads(
                (output / "column_statistics.json").read_text(encoding="utf-8")
            )
            self.assertEqual(statistics["rows"], 100)
            self.assertEqual(statistics["max_bins"], 8)
            self.assertEqual(statistics["fresh_rows_drawn"], 0)
            self.assertEqual(statistics["fixture_rows_reused"], 100)
            self.assertIn("title:production_year", statistics["marginal_entropy_bits"])
            self.assertIn(
                "title:production_year", statistics["total_mutual_information_bits"]
            )
            self.assertNotIn("__in_title", statistics["marginal_entropy_bits"])

    def test_the_draw_ignores_rare_support(self) -> None:
        # Statistics describe the join.  Rare support reweights the stream
        # towards the regions it targets, so an order chosen under it would be
        # chosen from a distribution no arm actually trains on.
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            prepared = self.prepared(root)
            config = self.config(root, prepared)
            config.write_text(
                config.read_text(encoding="utf-8")
                + "\n".join(
                    [
                        "rare_support:",
                        "  enabled: true",
                        "  discovery:",
                        "    enabled: true",
                        "    root_data_only: true",
                        "    max_selected_strata: 2",
                        "  allocation:",
                        "    strategy: support_deficit",
                        "rare_auxiliary:",
                        "  enabled: true",
                        "  batch_size: 2",
                        "  beta: 0.25",
                        "",
                    ]
                ),
                encoding="utf-8",
            )
            output = root / "orderings"
            result = self.run_script(config, output)
            self.assertEqual(result.returncode, 0, result.stderr)
            statistics = json.loads(
                (output / "column_statistics.json").read_text(encoding="utf-8")
            )
            self.assertNotIn("RareSupport", statistics["source"])

    def test_the_ordering_only_moves_data_columns(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            prepared = self.prepared(root)
            output = root / "orderings"
            self.run_script(self.config(root, prepared), output)
            manifest = json.loads(
                (prepared / "manifest.json").read_text(encoding="utf-8")
            )
            kinds = {
                column["name"]: column["kind"] for column in manifest["metadata"]["columns"]
            }
            original = [column["name"] for column in manifest["metadata"]["columns"]]
            order = json.loads((output / "coupling.json").read_text(encoding="utf-8"))["order"]
            for position, name in enumerate(order):
                if kinds[name] != "data":
                    self.assertEqual(name, original[position])


if __name__ == "__main__":
    unittest.main()
