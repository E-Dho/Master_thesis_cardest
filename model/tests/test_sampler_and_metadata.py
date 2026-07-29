from __future__ import annotations

import unittest

import numpy as np

from model.src.config import load_simple_yaml, validate_config
from model.src.data.full_join_sampler import (
    SyntheticFullJoinSampleSource,
    canonicalize_fanout_value,
)
from model.src.predicates.vocabulary import PredicateVocabularies


class SamplerMetadataTest(unittest.TestCase):
    def test_metadata_records_separate_input_and_output_bins(self) -> None:
        source = SyntheticFullJoinSampleSource()
        vocabularies = PredicateVocabularies.from_metadata(source.metadata)
        self.assertEqual(source.metadata.data_output_bins[-1], 3)
        self.assertEqual(vocabularies.input_bins[-1], 2)
        self.assertNotEqual(vocabularies.input_bins[-1], source.metadata.data_output_bins[-1])

    def test_sampler_ordering_and_reproducibility(self) -> None:
        source = SyntheticFullJoinSampleSource()
        batch_a = source.batches(4, seed=7)
        batch_b = source.batches(4, seed=7)
        self.assertTrue(np.array_equal(batch_a.encoded_values, batch_b.encoded_values))
        kinds = [column.kind.value for column in batch_a.column_metadata]
        self.assertEqual(kinds, ["data", "data", "data", "indicator", "indicator", "indicator", "fanout", "fanout"])

    def test_indicator_and_fanout_validation(self) -> None:
        source = SyntheticFullJoinSampleSource()
        inspection = source.inspect()
        self.assertGreater(inspection.join_cardinality, 0)
        self.assertIn("I_A", inspection.indicator_frequencies)
        for minimum, maximum in inspection.fanout_min_max.values():
            self.assertGreater(minimum, 0)
            self.assertGreaterEqual(maximum, minimum)

    def test_known_outer_padding_fanout_is_neutral_one(self) -> None:
        self.assertEqual(canonicalize_fanout_value(None, outer_padding=True), 1)
        with self.assertRaises(ValueError):
            canonicalize_fanout_value(0)

    def test_resmade_configs_validate(self) -> None:
        for path in (
            "model/configs/resmade_smoke.yaml",
            "model/configs/resmade_inv_fanout_example.yaml",
            "model/configs/job_light_resmade_inv_fanout.yaml",
        ):
            validate_config(load_simple_yaml(path))


if __name__ == "__main__":
    unittest.main()

