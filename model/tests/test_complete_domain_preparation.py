from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from model.src.data.complete_domain_preparation import (
    COMPLETE_METADATA_SOURCE,
    CompleteDomainSpec,
    build_complete_metadata,
    build_manifest_payload,
    classify_query_literal,
    encode_sample_dataframe,
    preparation_stats,
    validate_prepared_manifest,
    write_prepared_artifacts,
)
from model.src.data.column_ordering import (
    column_permutation,
    permute_encoded_rows,
    permute_metadata_columns,
)
from model.src.data.full_join_sampler import OUTER_MISSING
from model.src.model.factorization import (
    FactorizationConfig,
    apply_factorization_to_metadata,
    factorization_plan_hash,
)
from model.src.predicates.operators import PredicateOp


class CompleteDomainPreparationTest(unittest.TestCase):
    def test_complete_domain_contains_value_absent_from_sample(self) -> None:
        tables, spec = _tiny_tables()
        metadata = build_complete_metadata(tables, spec)
        keyword = metadata.columns[metadata.column_index("movie_keyword:keyword_id")]
        self.assertIn(9999, keyword.domain)
        sample = _sample_frame([10, 12])
        sampled_only = set(sample["movie_keyword:keyword_id"].dropna().astype(int).tolist())
        self.assertNotIn(9999, sampled_only)
        encoded = encode_sample_dataframe(sample, metadata)
        self.assertEqual(len(encoded.issues), 0)
        self.assertEqual(encoded.encoded_rows.shape, (2, len(metadata.columns)))

    def test_sample_size_independence_for_domains_and_factorization(self) -> None:
        tables, spec = _tiny_tables()
        metadata = build_complete_metadata(tables, spec)
        factor_config = FactorizationConfig(
            enabled=True,
            strategy="bitwise_lossless",
            word_size_bits=2,
            minimum_domain_size=3,
        )
        plan_hash = factorization_plan_hash(
            apply_factorization_to_metadata(metadata, factor_config).factorization_plan
        )
        domains = [column.domain for column in metadata.columns]
        for size in (1, 2, 4):
            sample = _sample_frame([10, 12, np.nan, 14][:size])
            encoded = encode_sample_dataframe(sample, metadata)
            stats = preparation_stats(
                metadata=metadata,
                encoded_sample=encoded,
                factorization_config=factor_config,
            )
            self.assertEqual([column.domain for column in metadata.columns], domains)
            self.assertEqual(stats["sample_row_count"], size)
            self.assertEqual(stats["sample_encoding_ood_values"], 0)
            self.assertEqual(stats["factorization_hash"], plan_hash)

    def test_ood_query_literals_distinguish_equality_and_ranges(self) -> None:
        metadata = build_complete_metadata(*_tiny_tables())
        present = classify_query_literal(
            metadata,
            "movie_keyword:keyword_id",
            PredicateOp.EQUAL,
            9999,
        )
        absent = classify_query_literal(
            metadata,
            "movie_keyword:keyword_id",
            PredicateOp.EQUAL,
            777777,
        )
        threshold = classify_query_literal(
            metadata,
            "title:production_year",
            PredicateOp.GREATER_EQUAL,
            1995,
        )
        self.assertEqual(present.category, "present_in_complete_domain")
        self.assertEqual(absent.category, "genuinely_absent_from_complete_dataset")
        self.assertEqual(
            threshold.category,
            "range_threshold_not_required_to_be_domain_member",
        )

    def test_factorization_activates_for_complete_large_domain(self) -> None:
        title = pd.DataFrame({"id": list(range(3000)), "kind_id": list(range(3000))})
        tables = {"title": title}
        spec = CompleteDomainSpec(
            join_tables=("title",),
            join_root="title",
            join_keys={"title": ("id",)},
            join_cardinality=3000,
        )
        metadata = build_complete_metadata(tables, spec)
        factorized = apply_factorization_to_metadata(
            metadata,
            FactorizationConfig(
                enabled=True,
                strategy="bitwise_lossless",
                word_size_bits=8,
                minimum_domain_size=2048,
            ),
        )
        self.assertGreater(
            factorized.factorization_plan.original_output_width,
            factorized.factorization_plan.factorized_output_width,
        )
        self.assertIsNotNone(
            factorized.factorization_plan.factorization_for_column(0)
        )

    def test_sampler_encoding_and_fanout_domains_are_complete_and_positive(self) -> None:
        tables, spec = _tiny_tables()
        metadata = build_complete_metadata(tables, spec)
        fanout = metadata.columns[metadata.column_index("__fanout_movie_keyword")]
        self.assertEqual(fanout.domain, (1, 2, 3))
        for values in ([10, np.nan], [13, 14], [9999, np.nan]):
            encoded = encode_sample_dataframe(_sample_frame(values), metadata)
            self.assertEqual(len(encoded.issues), 0)
        self.assertTrue(all(int(value) > 0 for value in fanout.domain))

    def test_manifest_validation_rejects_sample_inferred_domains(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            directory = Path(tmpdir)
            (directory / "manifest.json").write_text(
                json.dumps(
                    {
                        "source": "neurocard_factorized_sampler_smoke",
                        "rows": 512,
                        "domains_complete": False,
                        "metadata_source": "sample_rows",
                        "metadata": {},
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "domains are not marked complete"):
                validate_prepared_manifest(directory)

    def test_written_manifest_roundtrip_is_complete(self) -> None:
        tables, spec = _tiny_tables()
        metadata = build_complete_metadata(tables, spec)
        encoded = encode_sample_dataframe(_sample_frame([10, np.nan]), metadata)
        manifest = build_manifest_payload(
            metadata=metadata,
            spec=spec,
            sample_rows=encoded.encoded_rows.shape[0],
        )
        stats = preparation_stats(
            metadata=metadata,
            encoded_sample=encoded,
            factorization_config=FactorizationConfig(),
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            artifacts = write_prepared_artifacts(
                prepared_directory=tmpdir,
                manifest_payload=manifest,
                encoded_rows=encoded.encoded_rows,
                stats=stats,
            )
            self.assertEqual(artifacts.metadata.stable_schema_hash(), metadata.stable_schema_hash())
            raw = json.loads(artifacts.manifest_path.read_text(encoding="utf-8"))
            self.assertTrue(raw["domains_complete"])
            self.assertEqual(raw["metadata_source"], COMPLETE_METADATA_SOURCE)


class SampleColumnMatchingTest(unittest.TestCase):
    """The sampler's column order is its own; only the names have to agree.

    The frame arrives in whatever order the sampler emits -- for the live
    source, NeuroCard's _rearrange_columns -- and that equalled the manifest's
    order only because build_complete_metadata happened to build the manifest
    the same way.  A manifest permuted on purpose, to vary the model's
    autoregressive ordering, is the same schema over the same columns, so the
    encoder matches by name.  A frame that carries different columns is a
    different schema and is still refused.
    """

    def test_a_shuffled_frame_encodes_identically(self) -> None:
        tables, spec = _tiny_tables()
        metadata = build_complete_metadata(tables, spec)
        frame = _sample_frame([10, 12, 9999])
        shuffled = frame[list(reversed(list(frame.columns)))]
        self.assertNotEqual(list(shuffled.columns), list(frame.columns))
        np.testing.assert_array_equal(
            encode_sample_dataframe(shuffled, metadata).encoded_rows,
            encode_sample_dataframe(frame, metadata).encoded_rows,
        )

    def test_the_sampler_order_encodes_against_permuted_metadata(self) -> None:
        # The ablation's actual path: the manifest is permuted, the sampler is
        # not.  The encoded rows have to come out in the manifest's order, which
        # is what the fixture permutation produces offline.
        tables, spec = _tiny_tables()
        metadata = build_complete_metadata(tables, spec)
        frame = _sample_frame([10, 12, 9999])
        # Rotated, not reversed: reversal is its own inverse, so a permutation
        # applied in the wrong direction would reproduce itself here.
        names = [column.name for column in metadata.columns]
        order = names[1:] + names[:1]
        permuted = permute_metadata_columns(metadata, order)
        np.testing.assert_array_equal(
            encode_sample_dataframe(frame, permuted).encoded_rows,
            permute_encoded_rows(
                encode_sample_dataframe(frame, metadata).encoded_rows,
                column_permutation(metadata, order),
            ),
        )

    def test_a_missing_column_is_named(self) -> None:
        tables, spec = _tiny_tables()
        metadata = build_complete_metadata(tables, spec)
        frame = _sample_frame([10, 12]).drop(columns=["title:kind_id"])
        with self.assertRaises(ValueError) as caught:
            encode_sample_dataframe(frame, metadata)
        self.assertIn("title:kind_id", str(caught.exception))

    def test_an_extra_column_is_named(self) -> None:
        tables, spec = _tiny_tables()
        metadata = build_complete_metadata(tables, spec)
        frame = _sample_frame([10, 12])
        frame["title:not_modeled"] = 1
        with self.assertRaises(ValueError) as caught:
            encode_sample_dataframe(frame, metadata)
        self.assertIn("title:not_modeled", str(caught.exception))

    def test_a_repeated_column_is_refused(self) -> None:
        # Selecting by name against a duplicated label widens the frame instead
        # of reordering it, which would then misalign every later column.
        tables, spec = _tiny_tables()
        metadata = build_complete_metadata(tables, spec)
        frame = _sample_frame([10, 12])
        doubled = pd.concat([frame, frame[["title:kind_id"]]], axis=1)
        with self.assertRaisesRegex(ValueError, "repeats"):
            encode_sample_dataframe(doubled, metadata)


def _tiny_tables() -> tuple[dict[str, pd.DataFrame], CompleteDomainSpec]:
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


if __name__ == "__main__":
    unittest.main()
