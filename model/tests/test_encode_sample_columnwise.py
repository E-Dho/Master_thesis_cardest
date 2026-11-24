"""encode_sample_dataframe reads columns once; results equal the cell-by-cell version.

The live NeuroCard training encodes every drawn batch with this function. It used
``frame.iloc[row, column]`` per cell; it now reads each column's values once. The
reference below is the previous loop, verbatim; frames with every dtype a sampler
can produce (ints, floats with NaN, objects with None/NaN/strings, nullable Int64
with pd.NA, datetimes with NaT, booleans), in-domain and out-of-domain values, give
the same encoded rows and the same issues, in the same order.
"""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from model.src.data import complete_domain_preparation as prep
from model.src.data.complete_domain_preparation import (
    EncodedSampleRows,
    SampleEncodingIssue,
    build_complete_metadata,
    encode_sample_dataframe,
)
from model.tests.test_complete_domain_preparation import _tiny_tables


def _reference(sample_frame, metadata, *, strict=True) -> EncodedSampleRows:
    """The cell-by-cell implementation before the change."""

    frame = prep._as_dataframe(sample_frame)
    indicator_values_by_table = prep._canonical_indicator_values_by_table(frame, metadata)
    encoded = np.zeros((len(frame), len(metadata.columns)), dtype=np.int64)
    issues = []
    domain_maps = [{value: index for index, value in enumerate(column.domain)} for column in metadata.columns]
    for row_index in range(len(frame)):
        for column_index, column in enumerate(metadata.columns):
            raw_value = frame.iloc[row_index, column_index]
            try:
                value = prep._canonicalize_sample_value(raw_value, column, row_index, indicator_values_by_table)
            except ValueError as exc:
                issues.append(SampleEncodingIssue(row_index, column.name, prep._jsonable(raw_value), str(exc)))
                continue
            encoded_index = domain_maps[column_index].get(value)
            if encoded_index is None:
                issues.append(SampleEncodingIssue(row_index, column.name, value,
                                                  "sampled value is outside the complete manifest domain"))
                continue
            encoded[row_index, column_index] = encoded_index
    if strict and issues:
        raise ValueError("issues")
    return EncodedSampleRows(encoded, tuple(issues))


def _frames(metadata, rng: np.random.Generator):
    names = [column.name for column in metadata.columns]
    rows = 400
    pools = {
        "title:kind_id": [1, 2, 3, 7, np.nan],
        "title:production_year": [1990, 1991, 2000, 1850, np.nan],
        "movie_keyword:keyword_id": [10, 12, 13, 14, 9999, 5, np.nan],
        "__in_title": [0, 1, 1, 1],
        "__in_movie_keyword": [0, 1, 1, np.nan],
        "__fanout_movie_keyword": [1, 2, 3, 3, np.nan, 0],
    }
    draws = {name: rng.choice(np.array(pools[name], dtype=object), size=rows) for name in names}
    # 1) the dtypes pandas infers (float64 where NaN occurs, int64 otherwise)
    yield pd.DataFrame({name: pd.Series(list(draws[name])).infer_objects() for name in names})
    # 2) everything as object, with None mixed in for NaN
    yield pd.DataFrame({name: pd.Series([None if (isinstance(v, float) and np.isnan(v) and i % 2) else v
                                         for i, v in enumerate(draws[name])], dtype=object) for name in names})
    # 3) nullable integers
    yield pd.DataFrame({name: pd.Series([pd.NA if (isinstance(v, float) and np.isnan(v)) else int(v)
                                         for v in draws[name]], dtype="Int64") for name in names})
    # 4) strings, datetimes with NaT and booleans in data columns (out-of-domain values, issues recorded)
    odd = {
        "title:kind_id": pd.Series(rng.choice(np.array(["1", "x", None], dtype=object), size=rows), dtype=object),
        "title:production_year": pd.Series(pd.to_datetime(rng.choice(["2000-01-01", None], size=rows))),
        "movie_keyword:keyword_id": pd.Series(rng.choice([True, False], size=rows)),
    }
    yield pd.DataFrame({name: odd.get(name, pd.Series(list(draws[name])).infer_objects()) for name in names})


class ColumnwiseEncodingTest(unittest.TestCase):
    def test_same_rows_and_issues_as_the_cell_by_cell_loop(self) -> None:
        tables, spec = _tiny_tables()
        metadata = build_complete_metadata(tables, spec)
        rng = np.random.default_rng(0)
        for seed in range(3):
            for case, frame in enumerate(_frames(metadata, np.random.default_rng(seed))):
                expected = _reference(frame, metadata, strict=False)
                observed = encode_sample_dataframe(frame, metadata, strict=False)
                self.assertTrue(np.array_equal(expected.encoded_rows, observed.encoded_rows), (seed, case))
                self.assertEqual(len(expected.issues), len(observed.issues), (seed, case))
                for left, right in zip(expected.issues, observed.issues):
                    self.assertEqual(
                        (left.row_index, left.column_name, repr(left.value), left.reason),
                        (right.row_index, right.column_name, repr(right.value), right.reason),
                        (seed, case),
                    )
                if case < 3:
                    self.assertGreater(len(expected.issues), 0)  # out-of-domain values are covered
        del rng

    def test_strict_mode_raises_on_the_same_first_issue(self) -> None:
        tables, spec = _tiny_tables()
        metadata = build_complete_metadata(tables, spec)
        frame = next(_frames(metadata, np.random.default_rng(5)))
        first = _reference(frame, metadata, strict=False).issues[0]
        with self.assertRaisesRegex(ValueError, f"row={first.row_index}, column={first.column_name}"):
            encode_sample_dataframe(frame, metadata, strict=True)


if __name__ == "__main__":
    unittest.main()
