import csv
import tempfile
import unittest
from pathlib import Path

from dataset_generation.porto_taxi_loader.parse_porto_to_staging import (
    InvalidTrace,
    create_candidate_db,
    parse_complete_row,
    parse_polyline,
    scan_candidates,
    select_candidates,
    stable_hash,
    timestamp_text,
    write_staging,
)


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

    def test_hash_selection_is_stable_and_respects_target(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.csv"
            with path.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["TRIP_ID", "TAXI_ID", "TIMESTAMP", "MISSING_DATA", "POLYLINE"])
                writer.writeheader()
                for trip_id in ("a", "b", "c"):
                    writer.writerow({"TRIP_ID": trip_id, "TAXI_ID": 1, "TIMESTAMP": 1, "MISSING_DATA": "False", "POLYLINE": "[[0, 0], [0, 1], [0, 2]]"})
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
        self.assertEqual(stable_hash("seed", "a"), stable_hash("seed", "a"))


if __name__ == "__main__":
    unittest.main()
