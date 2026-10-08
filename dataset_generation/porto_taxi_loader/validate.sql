-- Every check lands in a temp table so the job can fail on a bad result.
-- Printing `ok` columns to a log is not validation: the previous version of
-- this file always exited 0, so a failed check reached the query-config step
-- unnoticed.
CREATE TEMP TABLE porto_validation (
  check_name text PRIMARY KEY,
  ok boolean NOT NULL,
  observed bigint,
  enforced boolean NOT NULL DEFAULT true
);

INSERT INTO porto_validation (check_name, ok, observed)
SELECT 'taxis_nonempty', COUNT(*) > 0, COUNT(*) FROM porto.taxis;

INSERT INTO porto_validation (check_name, ok, observed)
SELECT 'trips_nonempty', COUNT(*) > 0, COUNT(*) FROM porto.trips;

INSERT INTO porto_validation (check_name, ok, observed)
SELECT 'segments_nonempty', COUNT(*) > 0, COUNT(*) FROM porto.segments;

INSERT INTO porto_validation (check_name, ok, observed)
SELECT 'trip_ids_unique', COUNT(*) = COUNT(DISTINCT trip_id), COUNT(*) FROM porto.trips;

INSERT INTO porto_validation (check_name, ok, observed)
SELECT 'segments_match_trip_counts', COUNT(*) = 0, COUNT(*) FROM (
  SELECT t.trip_id FROM porto.trips t JOIN porto.segments s ON s.trip_id = t.trip_id
  GROUP BY t.trip_id, t.num_of_segments HAVING COUNT(*) <> t.num_of_segments
) invalid;

INSERT INTO porto_validation (check_name, ok, observed)
SELECT 'trip_time_bounds', COUNT(*) = 0, COUNT(*) FROM (
  SELECT t.trip_id FROM porto.trips t JOIN porto.segments s ON s.trip_id = t.trip_id
  GROUP BY t.trip_id, t.start_time, t.end_time HAVING MIN(s.t_s) <> t.start_time OR MAX(s.t_e) <> t.end_time
) invalid;

INSERT INTO porto_validation (check_name, ok, observed)
SELECT 'segment_geometry_present', COUNT(*) = 0, COUNT(*)
FROM porto.segments WHERE segment_geom IS NULL OR segment_tgeom IS NULL;

INSERT INTO porto_validation (check_name, ok, observed)
SELECT 'segment_geometry_srid', COUNT(*) = 0, COUNT(*)
FROM porto.segments WHERE ST_SRID(segment_geom) <> 3763;

-- A segment is exactly one hop of the source polyline, so it must be a
-- two-vertex line.  ST_IsValid is deliberately NOT asserted here: a stationary
-- taxi repeats its position across a 15-second sample, which GEOS reports as
-- "too few points" for the resulting zero-length line.  Around 1.3% of Porto
-- segments are legitimately zero-length, so asserting validity would fail every
-- load.  The rate is recorded below instead.
INSERT INTO porto_validation (check_name, ok, observed)
SELECT 'segment_geometry_two_points', COUNT(*) = 0, COUNT(*)
FROM porto.segments WHERE ST_NPoints(segment_geom) <> 2;

INSERT INTO porto_validation (check_name, ok, observed, enforced)
SELECT 'degenerate_zero_length_segments', true, COUNT(*), false
FROM porto.segments WHERE s_x = e_x AND s_y = e_y;

INSERT INTO porto_validation (check_name, ok, observed, enforced)
SELECT 'degenerate_zero_extent_trips', true, COUNT(*), false
FROM porto.trips WHERE ST_NPoints(ST_RemoveRepeatedPoints(trip_geom)) < 2;

INSERT INTO porto_validation (check_name, ok, observed)
SELECT 'required_indexes', COUNT(*) = 4, COUNT(*)
FROM pg_indexes WHERE schemaname = 'porto' AND indexname IN ('porto_segments_geom_gist_idx', 'porto_segments_tgeom_gist_idx', 'porto_segments_time_idx', 'porto_segments_trip_idx');

SELECT check_name, ok, observed, enforced FROM porto_validation ORDER BY check_name;

-- Fail the job on any enforced check that did not pass.  ON_ERROR_STOP turns
-- the division by zero into a non-zero exit, matching the POL loader's guard.
SELECT
  'porto_validation_guard' AS check_name,
  bool_and(ok) AS ok,
  COUNT(*) FILTER (WHERE NOT ok) AS failed_checks,
  CASE WHEN bool_and(ok) THEN 1 ELSE 1 / 0 END AS fail_if_bad
FROM porto_validation
WHERE enforced;
