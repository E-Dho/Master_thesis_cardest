import csv
import json
import tempfile
import unittest
from pathlib import Path

from dataset_generation.porto_taxi_loader.parse_porto_to_staging import (
    STAGING_FORMAT_VERSION,
    InvalidTrace,
    create_candidate_db,
    parse_complete_row,
    parse_polyline,
    reusable_metadata,
    scan_candidates,
    select_candidates,
    sha256_file,
    stable_hash,
    timestamp_text,
    write_staging,
)


FIELDNAMES = [
    "TRIP_ID",
    "CALL_TYPE",
    "ORIGIN_CALL",
    "ORIGIN_STAND",
    "TAXI_ID",
    "TIMESTAMP",
    "DAY_TYPE",
    "MISSING_DATA",
    "POLYLINE",
]


def source_row(**overrides):
    row = {
        "TRIP_ID": "a",
        "CALL_TYPE": "C",
        "ORIGIN_CALL": "",
        "ORIGIN_STAND": "",
        "TAXI_ID": "1",
        "TIMESTAMP": "1",
        "DAY_TYPE": "A",
        "MISSING_DATA": "False",
        "POLYLINE": "[[0, 0], [0, 1], [0, 2]]",
    }
    row.update(overrides)
    return row


def write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


class PortoStagingTest(unittest.TestCase):
    def test_polyline_and_segment_timestamps(self):
        self.assertEqual(parse_polyline("[[-8.6, 41.1], [-8.61, 41.11]]"), [(-8.6, 41.1), (-8.61, 41.11)])
        self.assertEqual(timestamp_text(0, 1), "1970-01-01 00:00:15")

    def test_rejects_invalid_or_incomplete_trace(self):
        with self.assertRaisesRegex(InvalidTrace, "fewer_than_two_points"):
            parse_polyline("[[-8.6, 41.1]]")
        with self.assertRaisesRegex(InvalidTrace, "coordinate_out_of_bounds"):
            parse_polyline("[[181, 41.1], [-8.6, 41.1]]")
        with self.assertRaisesRegex(InvalidTrace, "missing_data"):
            parse_complete_row({"MISSING_DATA": "True"})

    def test_reads_the_day_type_column_the_kaggle_file_actually_has(self):
        # The Kaggle header is DAY_TYPE.  Reading DAYTYPE yielded an empty
        # string for every row, which the daytype CHECK constraint rejects, so
        # COPY aborted on the first trip.
        trace = parse_complete_row(source_row(DAY_TYPE="A"))
        self.assertEqual(trace.day_type, "A")
        self.assertEqual(trace.call_type, "C")
        with self.assertRaisesRegex(InvalidTrace, "invalid_day_type"):
            parse_complete_row(source_row(DAY_TYPE=""))

    def test_rejects_categorical_and_optional_integer_columns(self):
        with self.assertRaisesRegex(InvalidTrace, "invalid_call_type"):
            parse_complete_row(source_row(CALL_TYPE="Z"))
        with self.assertRaisesRegex(InvalidTrace, "invalid_origin_call"):
            parse_complete_row(source_row(ORIGIN_CALL="not-a-number"))
        with self.assertRaisesRegex(InvalidTrace, "invalid_origin_stand"):
            parse_complete_row(source_row(ORIGIN_STAND="12.5"))

    def test_optional_integers_become_the_copy_null_marker(self):
        trace = parse_complete_row(source_row(ORIGIN_CALL="", ORIGIN_STAND="7"))
        self.assertEqual(trace.origin_call, "\\N")
        self.assertEqual(trace.origin_stand, "7")

    def test_a_renamed_source_column_fails_the_scan(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.csv"
            with path.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["TRIP_ID", "TAXI_ID", "TIMESTAMP", "MISSING_DATA", "POLYLINE"])
                writer.writeheader()
            connection = create_candidate_db(Path(directory) / "selection.sqlite")
            try:
                with self.assertRaisesRegex(SystemExit, "CALL_TYPE"):
                    scan_candidates(path, connection, "seed")
            finally:
                connection.close()

    def test_hash_selection_is_stable_and_respects_target(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.csv"
            write_csv(path, [source_row(TRIP_ID=trip_id) for trip_id in ("a", "b", "c")])
            connection = create_candidate_db(Path(directory) / "selection.sqlite")
            try:
                scan_candidates(path, connection, "seed")
                trips, segments = select_candidates(connection, 4)
                written = write_staging(path, connection, Path(directory) / "staging")
            finally:
                connection.close()
        self.assertEqual((trips, segments), (2, 4))
        self.assertEqual(written["trips"], 2)
        self.assertEqual(written["segments"], 4)
        self.assertEqual(written["degenerate_zero_length_segments"], 0)
        self.assertEqual(stable_hash("seed", "a"), stable_hash("seed", "a"))

    def test_zero_length_segments_are_counted_not_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.csv"
            write_csv(path, [source_row(POLYLINE="[[0, 0], [0, 0], [0, 1]]")])
            connection = create_candidate_db(Path(directory) / "selection.sqlite")
            try:
                scan_candidates(path, connection, "seed")
                select_candidates(connection, 10)
                written = write_staging(path, connection, Path(directory) / "staging")
            finally:
                connection.close()
        self.assertEqual(written["segments"], 2)
        self.assertEqual(written["degenerate_zero_length_segments"], 1)

    def test_a_repeated_trip_id_keeps_the_first_row_and_is_counted(self):
        # TRIP_ID is not a key in the Kaggle file: 81 of 1,710,670 rows repeat
        # one and 79 of those differ.  porto.trips declares source_trip_id
        # UNIQUE, so the later rows are dropped, not fatal.
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.csv"
            write_csv(
                path,
                [
                    source_row(TRIP_ID="a", TAXI_ID="1"),
                    source_row(TRIP_ID="a", TAXI_ID="2"),
                    source_row(TRIP_ID="b", TAXI_ID="3"),
                ],
            )
            connection = create_candidate_db(Path(directory) / "selection.sqlite")
            try:
                stats = scan_candidates(path, connection, "seed")
                select_candidates(connection, 100)
                written = write_staging(path, connection, Path(directory) / "staging")
                rows = (Path(directory) / "staging" / "trips.tsv").read_text().splitlines()
            finally:
                connection.close()

        self.assertEqual(stats["duplicate_trip_id"], 1)
        self.assertEqual(stats["eligible_trips"], 2)
        self.assertEqual(written["trips"], 2)
        kept = {row.split("\t")[1]: row.split("\t")[2] for row in rows}
        self.assertEqual(kept, {"a": "1", "b": "3"})

    def test_staging_is_reused_only_for_the_same_input_target_and_seed(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            csv_path = base / "input.csv"
            write_csv(csv_path, [source_row()])
            staging = base / "staging"
            staging.mkdir()
            self.assertIsNone(reusable_metadata(staging, csv_path, 10, "seed"))

            for name in ("trips.tsv", "segments_wgs84.tsv", "taxis.tsv", "selected_trips.tsv"):
                (staging / name).write_text("", encoding="utf-8")
            metadata = {
                "staging_format_version": STAGING_FORMAT_VERSION,
                "input_sha256": sha256_file(csv_path),
                "target_segments": 10,
                "selection_seed": "seed",
                "selected_trips": 1,
            }
            (staging / "load_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
            reused = reusable_metadata(staging, csv_path, 10, "seed")
            self.assertIsNotNone(reused)
            self.assertTrue(reused["reused_existing_staging"])
            self.assertIsNone(reusable_metadata(staging, csv_path, 11, "seed"))
            self.assertIsNone(reusable_metadata(staging, csv_path, 10, "other-seed"))

            write_csv(csv_path, [source_row(), source_row(TRIP_ID="b")])
            self.assertIsNone(reusable_metadata(staging, csv_path, 10, "seed"))

    def test_staging_from_an_older_parser_is_never_reused(self):
        # A directory written before the DAY_TYPE fix holds TSVs whose day types
        # are blank.  Reusing it would load stale semantics or fail the COPY
        # again, so an unversioned or differently versioned directory is
        # re-parsed.
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            csv_path = base / "input.csv"
            write_csv(csv_path, [source_row()])
            staging = base / "staging"
            staging.mkdir()
            for name in ("trips.tsv", "segments_wgs84.tsv", "taxis.tsv", "selected_trips.tsv"):
                (staging / name).write_text("", encoding="utf-8")
            metadata_path = staging / "load_metadata.json"
            unversioned = {
                "input_sha256": sha256_file(csv_path),
                "target_segments": 10,
                "selection_seed": "seed",
            }

            metadata_path.write_text(json.dumps(unversioned), encoding="utf-8")
            self.assertIsNone(reusable_metadata(staging, csv_path, 10, "seed"))

            metadata_path.write_text(
                json.dumps({**unversioned, "staging_format_version": STAGING_FORMAT_VERSION - 1}),
                encoding="utf-8",
            )
            self.assertIsNone(reusable_metadata(staging, csv_path, 10, "seed"))

            metadata_path.write_text(
                json.dumps({**unversioned, "staging_format_version": STAGING_FORMAT_VERSION}),
                encoding="utf-8",
            )
            self.assertIsNotNone(reusable_metadata(staging, csv_path, 10, "seed"))


if __name__ == "__main__":
    unittest.main()
