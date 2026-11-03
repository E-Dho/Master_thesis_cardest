from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from model.src.data.column_ordering import (
    PERMUTED_COLUMN_ORDER,
    column_permutation,
    permute_encoded_rows,
    permute_metadata_columns,
)
from model.src.data.schema import FactorizationPlan
from model.src.data.complete_domain_preparation import (
    CompleteDomainSpec,
    build_complete_metadata,
    build_manifest_payload,
    encode_sample_dataframe,
    preparation_stats,
    validate_prepared_manifest,
    write_prepared_artifacts,
)
from model.src.model.factorization import (
    FactorizationConfig,
    apply_factorization_to_metadata,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def _tables() -> tuple[dict[str, pd.DataFrame], CompleteDomainSpec]:
    tables = {
        "title": pd.DataFrame(
            {
                "id": [1, 2, 3],
                "kind_id": [1, 2, 3],
                "production_year": [1990, 1991, 2000],
            }
        ),
        "movie_keyword": pd.DataFrame(
            {
                "movie_id": [1, 1, 2, 2, 2],
                "keyword_id": [10, 9999, 12, 13, 14],
            }
        ),
    }
    spec = CompleteDomainSpec(
        join_tables=("title", "movie_keyword"),
        join_root="title",
        join_keys={"title": ("id",), "movie_keyword": ("movie_id",)},
        join_cardinality=5,
    )
    return tables, spec


def _sample_frame(keyword_values: list[object]) -> pd.DataFrame:
    rows = []
    for row_index, keyword in enumerate(keyword_values):
        present = not (isinstance(keyword, float) and np.isnan(keyword))
        rows.append(
            {
                "title:kind_id": 1 + (row_index % 3),
                "title:production_year": [1990, 1991, 2000][row_index % 3],
                "movie_keyword:keyword_id": keyword,
                "__in_title": 1,
                "__in_movie_keyword": 1 if present else np.nan,
                "__fanout_movie_keyword": 2 if row_index == 0 else 1,
            }
        )
    return pd.DataFrame(rows)


def _metadata():
    tables, spec = _tables()
    return build_complete_metadata(tables, spec)


def _rotated(names):
    """Rotate by one.

    Reversal is its own inverse, so a permutation applied in the wrong
    direction reproduces it exactly and every reversal-based assertion passes
    on the inverted code.  A rotation does not, so it is what the direction of
    the mapping is actually tested with.
    """

    names = list(names)
    return names[1:] + names[:1]


class ColumnPermutationTest(unittest.TestCase):
    """A permutation renames nothing and loses nothing; it only moves columns."""

    def test_permuting_by_the_current_order_changes_nothing(self) -> None:
        # The identity is the control arm's own order.  If it moved the hash,
        # an arm could differ from the control by the act of being rewritten
        # rather than by its ordering.
        metadata = _metadata()
        order = [column.name for column in metadata.columns]
        permuted = permute_metadata_columns(metadata, order)
        self.assertEqual(permuted.stable_schema_hash(), metadata.stable_schema_hash())
        self.assertEqual(permuted.column_order, metadata.column_order)

    def test_a_real_permutation_is_a_different_schema(self) -> None:
        metadata = _metadata()
        order = _rotated(column.name for column in metadata.columns)
        permuted = permute_metadata_columns(metadata, order)
        self.assertNotEqual(permuted.stable_schema_hash(), metadata.stable_schema_hash())
        self.assertEqual([column.name for column in permuted.columns], order)
        self.assertEqual(permuted.column_order, PERMUTED_COLUMN_ORDER)

    def test_each_column_carries_its_own_domain(self) -> None:
        # The point of the ablation is to move a column, not to re-domain it.
        metadata = _metadata()
        order = _rotated(column.name for column in metadata.columns)
        permuted = permute_metadata_columns(metadata, order)
        for column in permuted.columns:
            original = metadata.columns[metadata.column_index(column.name)]
            self.assertEqual(column.domain, original.domain)
            self.assertEqual(column.kind, original.kind)
            self.assertEqual(column.table, original.table)

    def test_the_join_description_survives(self) -> None:
        metadata = _metadata()
        order = list(reversed([column.name for column in metadata.columns]))
        permuted = permute_metadata_columns(metadata, order)
        self.assertEqual(permuted.join_root, metadata.join_root)
        self.assertEqual(permuted.join_tables, metadata.join_tables)
        self.assertEqual(permuted.join_edges, metadata.join_edges)
        self.assertEqual(
            permuted.full_join_cardinality, metadata.full_join_cardinality
        )

    def test_the_plan_is_reset_rather_than_carried(self) -> None:
        # A disabled plan can still carry one output head per column, in column
        # order, each naming its source index; carrying it across a permutation
        # would leave every head pointing at the column that used to be there.
        metadata = apply_factorization_to_metadata(
            _metadata(), FactorizationConfig(enabled=False)
        )
        self.assertTrue(metadata.factorization_plan.output_head_specs)
        permuted = permute_metadata_columns(
            metadata, _rotated(column.name for column in metadata.columns)
        )
        self.assertEqual(permuted.factorization_plan, FactorizationPlan())

    def test_a_repeated_column_is_refused(self) -> None:
        metadata = _metadata()
        names = [column.name for column in metadata.columns]
        with self.assertRaisesRegex(ValueError, "repeats"):
            column_permutation(metadata, [names[0]] + names[:-1])

    def test_a_dropped_column_is_refused(self) -> None:
        metadata = _metadata()
        names = [column.name for column in metadata.columns]
        with self.assertRaisesRegex(ValueError, "missing"):
            column_permutation(metadata, names[:-1])

    def test_an_unknown_column_is_refused(self) -> None:
        metadata = _metadata()
        names = [column.name for column in metadata.columns]
        with self.assertRaisesRegex(ValueError, "unexpected"):
            column_permutation(metadata, names[:-1] + ["title:not_a_column"])

    def test_a_built_factorization_plan_is_refused(self) -> None:
        # The plan's output heads carry source column indices, so permuting
        # under one would leave every head pointing at the wrong column.
        metadata = apply_factorization_to_metadata(
            _metadata(),
            FactorizationConfig(
                enabled=True,
                strategy="bitwise_lossless",
                word_size_bits=2,
                minimum_domain_size=3,
            ),
        )
        self.assertTrue(metadata.factorization_plan.enabled)
        order = list(reversed([column.name for column in metadata.columns]))
        with self.assertRaisesRegex(ValueError, "factorization plan"):
            permute_metadata_columns(metadata, order)

    def test_a_plan_is_rebuildable_for_the_new_order(self) -> None:
        # Refusing an enabled plan is only acceptable because the runtime builds
        # one from the config after the permutation; the same columns must be
        # chosen for factorization whichever order they sit in.
        metadata = _metadata()
        config = FactorizationConfig(
            enabled=True,
            strategy="bitwise_lossless",
            word_size_bits=2,
            minimum_domain_size=3,
        )
        order = list(reversed([column.name for column in metadata.columns]))
        plain = apply_factorization_to_metadata(metadata, config)
        moved = apply_factorization_to_metadata(
            permute_metadata_columns(metadata, order), config
        )
        self.assertEqual(
            sorted(
                metadata.columns[item.original_column_index].name
                for item in plain.factorization_plan.original_column_factorizations
            ),
            sorted(
                moved.columns[item.original_column_index].name
                for item in moved.factorization_plan.original_column_factorizations
            ),
        )


class AutoregressiveOrderTest(unittest.TestCase):
    """Permuting the manifest is what moves the model's conditioning order.

    ResMADE takes its input and hidden degrees from column position, so column
    i is modelled as P(col_i in R_i | predicates on columns < i).  The ablation
    has no other lever: nothing reads schema.column_order, and every consumer
    takes its index from enumerate(metadata.columns).  If the widths the model
    is built from did not follow a permuted manifest, the arms would differ in
    name only.
    """

    def test_the_model_widths_follow_the_manifest(self) -> None:
        metadata = _metadata()
        order = _rotated(column.name for column in metadata.columns)
        permuted = permute_metadata_columns(metadata, order)
        positions = [metadata.column_index(name) for name in order]
        self.assertEqual(
            permuted.data_output_bins,
            tuple(metadata.data_output_bins[index] for index in positions),
        )
        self.assertEqual(
            permuted.predicate_input_bins,
            tuple(metadata.predicate_input_bins[index] for index in positions),
        )

    def test_the_output_slices_follow_the_manifest(self) -> None:
        # Each head's slice into the flat output is derived from the running
        # widths, so a reordered schema has to re-cut them rather than keep the
        # control's offsets.
        metadata = _metadata()
        order = _rotated(column.name for column in metadata.columns)
        permuted = permute_metadata_columns(metadata, order)
        widths = [stop - start for start, stop in permuted.output_slices]
        self.assertEqual(tuple(widths), permuted.data_output_bins)
        self.assertEqual(
            permuted.output_slices[-1][1], metadata.output_slices[-1][1]
        )


class EncodedRowPermutationTest(unittest.TestCase):
    """Rows have to move with the manifest, or the fixture means something else."""

    def rows(self):
        metadata = _metadata()
        encoded = encode_sample_dataframe(_sample_frame([10, 12, 9999]), metadata)
        return metadata, encoded.encoded_rows

    def test_every_column_keeps_its_values(self) -> None:
        metadata, rows = self.rows()
        order = _rotated(column.name for column in metadata.columns)
        permutation = column_permutation(metadata, order)
        moved = permute_encoded_rows(rows, permutation)
        permuted = permute_metadata_columns(metadata, order)
        for new_index, column in enumerate(permuted.columns):
            old_index = metadata.column_index(column.name)
            np.testing.assert_array_equal(moved[:, new_index], rows[:, old_index])

    def test_the_decoded_row_is_unchanged(self) -> None:
        # The real invariant: a row means the same thing afterwards.  Encoded
        # ids are positions in a column's domain, so decoding through the
        # permuted metadata must reproduce the original name-to-value mapping.
        metadata, rows = self.rows()
        order = _rotated(column.name for column in metadata.columns)
        permuted = permute_metadata_columns(metadata, order)
        moved = permute_encoded_rows(rows, column_permutation(metadata, order))
        for row_index in range(rows.shape[0]):
            before = {
                column.name: column.domain[int(rows[row_index, index])]
                for index, column in enumerate(metadata.columns)
            }
            after = {
                column.name: column.domain[int(moved[row_index, index])]
                for index, column in enumerate(permuted.columns)
            }
            self.assertEqual(before, after)

    def test_the_identity_leaves_the_rows_alone(self) -> None:
        metadata, rows = self.rows()
        order = [column.name for column in metadata.columns]
        moved = permute_encoded_rows(rows, column_permutation(metadata, order))
        np.testing.assert_array_equal(moved, rows)

    def test_a_width_mismatch_is_refused(self) -> None:
        metadata, rows = self.rows()
        order = [column.name for column in metadata.columns]
        permutation = column_permutation(metadata, order)
        with self.assertRaisesRegex(ValueError, "columns"):
            permute_encoded_rows(rows[:, :-1], permutation)

    def test_one_dimensional_rows_are_refused(self) -> None:
        metadata, rows = self.rows()
        order = [column.name for column in metadata.columns]
        with self.assertRaisesRegex(ValueError, "two-dimensional"):
            permute_encoded_rows(rows[0], column_permutation(metadata, order))


class PermutePreparedDirectoryTest(unittest.TestCase):
    """The script's output has to be an ordinary prepared directory."""

    def prepared(self, root: Path) -> Path:
        tables, spec = _tables()
        metadata = build_complete_metadata(tables, spec)
        encoded = encode_sample_dataframe(_sample_frame([10, 12, 9999, 13]), metadata)
        directory = root / "source"
        write_prepared_artifacts(
            prepared_directory=directory,
            manifest_payload=build_manifest_payload(
                metadata=metadata, spec=spec, sample_rows=4
            ),
            encoded_rows=encoded.encoded_rows,
            stats=preparation_stats(
                metadata=metadata,
                encoded_sample=encoded,
                factorization_config=FactorizationConfig(enabled=False),
            ),
        )
        return directory

    def write_order(self, path: Path, order, **provenance) -> Path:
        payload = {"order": list(order), **provenance} if provenance else list(order)
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def run_script(self, prepared: Path, output: Path, order: Path, *extra: str):
        return subprocess.run(
            [
                sys.executable,
                "-B",
                "-m",
                "model.scripts.permute_prepared_columns",
                "--prepared",
                str(prepared),
                "--output",
                str(output),
                "--order",
                str(order),
                *extra,
            ],
            capture_output=True,
            text=True,
            cwd=REPOSITORY_ROOT,
        )

    def test_the_output_validates_and_carries_the_new_order(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            prepared = self.prepared(root)
            source = validate_prepared_manifest(prepared)
            names = [column.name for column in source.columns]
            order = list(reversed(names))
            output = root / "ordB"
            result = self.run_script(
                prepared, output, self.write_order(root / "order.json", order)
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            written = validate_prepared_manifest(output)
            self.assertEqual([column.name for column in written.columns], order)

    def test_the_rows_follow_the_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            prepared = self.prepared(root)
            source = validate_prepared_manifest(prepared)
            order = _rotated(column.name for column in source.columns)
            output = root / "ordB"
            self.run_script(
                prepared, output, self.write_order(root / "order.json", order)
            )
            before = np.load(prepared / "sample_rows.npy")
            after = np.load(output / "sample_rows.npy")
            written = validate_prepared_manifest(output)
            for new_index, column in enumerate(written.columns):
                old_index = source.column_index(column.name)
                np.testing.assert_array_equal(after[:, new_index], before[:, old_index])

    def test_the_identity_reproduces_the_source_manifest(self) -> None:
        # The sharpest check available: rewriting a directory by its own order
        # has to give the same schema back, rows included.
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            prepared = self.prepared(root)
            source = validate_prepared_manifest(prepared)
            order = [column.name for column in source.columns]
            output = root / "ordA"
            result = self.run_script(
                prepared, output, self.write_order(root / "order.json", order)
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(
                json.loads((output / "manifest.json").read_text(encoding="utf-8")),
                json.loads((prepared / "manifest.json").read_text(encoding="utf-8")),
            )
            np.testing.assert_array_equal(
                np.load(output / "sample_rows.npy"),
                np.load(prepared / "sample_rows.npy"),
            )

    def test_the_provenance_records_how_the_order_was_chosen(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            prepared = self.prepared(root)
            source = validate_prepared_manifest(prepared)
            order = list(reversed([column.name for column in source.columns]))
            output = root / "ordB"
            self.run_script(
                prepared,
                output,
                self.write_order(
                    root / "order.json", order, criterion="pairwise_mutual_information"
                ),
            )
            provenance = json.loads(
                (output / "column_permutation.json").read_text(encoding="utf-8")
            )
            self.assertEqual(provenance["column_order"], order)
            self.assertEqual(
                provenance["source_column_order"],
                [column.name for column in source.columns],
            )
            self.assertNotEqual(
                provenance["schema_hash"], provenance["source_schema_hash"]
            )
            self.assertEqual(
                provenance["order_provenance"]["criterion"],
                "pairwise_mutual_information",
            )
            self.assertEqual(provenance["sample_rows"], 4)

    def test_the_written_manifest_carries_its_own_hash(self) -> None:
        # checkpoint.py and this script both read metadata.schema_hash in
        # preference to recomputing it, so a manifest that kept the source's
        # hash would label the arm as the control wherever that field is used.
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            prepared = self.prepared(root)
            source = validate_prepared_manifest(prepared)
            order = _rotated(column.name for column in source.columns)
            output = root / "ordB"
            self.run_script(
                prepared, output, self.write_order(root / "order.json", order)
            )
            manifest = json.loads(
                (output / "manifest.json").read_text(encoding="utf-8")
            )
            written = validate_prepared_manifest(output)
            self.assertEqual(
                manifest["metadata"]["schema_hash"], written.stable_schema_hash()
            )
            self.assertNotEqual(
                manifest["metadata"]["schema_hash"],
                source.schema_hash or source.stable_schema_hash(),
            )

    def test_companion_files_are_carried_over(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            prepared = self.prepared(root)
            source = validate_prepared_manifest(prepared)
            order = list(reversed([column.name for column in source.columns]))
            output = root / "ordB"
            self.run_script(
                prepared, output, self.write_order(root / "order.json", order)
            )
            # preparation_stats.json is written beside the manifest and is not
            # column-ordered, so it has to arrive unchanged rather than go
            # missing and leave the arm without its preparation record.
            self.assertEqual(
                (output / "preparation_stats.json").read_text(encoding="utf-8"),
                (prepared / "preparation_stats.json").read_text(encoding="utf-8"),
            )

    def test_force_does_not_leave_the_previous_arm_rows_behind(self) -> None:
        # A source without sample_rows.npy forced into a directory that has one
        # would otherwise publish the earlier arm's rows under this arm's
        # manifest, and the two orders are exactly what the ablation compares.
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            prepared = self.prepared(root)
            source = validate_prepared_manifest(prepared)
            (prepared / "sample_rows.npy").unlink()
            output = root / "ordB"
            output.mkdir()
            np.save(output / "sample_rows.npy", np.zeros((4, len(source.columns)), dtype=np.int64))
            order = _rotated(column.name for column in source.columns)
            result = self.run_script(
                prepared,
                output,
                self.write_order(root / "order.json", order),
                "--force",
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertFalse((output / "sample_rows.npy").exists())

    def test_writing_over_the_source_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            prepared = self.prepared(root)
            source = validate_prepared_manifest(prepared)
            order = [column.name for column in source.columns]
            result = self.run_script(
                prepared, prepared, self.write_order(root / "order.json", order)
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("must differ", result.stderr)

    def test_a_non_empty_output_needs_force(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            prepared = self.prepared(root)
            source = validate_prepared_manifest(prepared)
            order = list(reversed([column.name for column in source.columns]))
            output = root / "ordB"
            output.mkdir()
            (output / "left_over.json").write_text("{}", encoding="utf-8")
            order_path = self.write_order(root / "order.json", order)
            refused = self.run_script(prepared, output, order_path)
            self.assertNotEqual(refused.returncode, 0)
            self.assertIn("--force", refused.stderr)
            forced = self.run_script(prepared, output, order_path, "--force")
            self.assertEqual(forced.returncode, 0, forced.stderr)

    def test_an_order_that_is_not_a_permutation_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            prepared = self.prepared(root)
            source = validate_prepared_manifest(prepared)
            order = [column.name for column in source.columns][:-1]
            result = self.run_script(
                prepared, root / "ordB", self.write_order(root / "order.json", order)
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("missing", result.stderr)

    def test_an_order_file_of_the_wrong_shape_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            prepared = self.prepared(root)
            order_path = root / "order.json"
            order_path.write_text(json.dumps({"columns": ["a"]}), encoding="utf-8")
            result = self.run_script(prepared, root / "ordB", order_path)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("must hold a JSON list", result.stderr)


if __name__ == "__main__":
    unittest.main()
