SELECT 'taxis_nonempty' AS check_name, COUNT(*) > 0 AS ok, COUNT(*) AS observed FROM porto.taxis;
SELECT 'trips_nonempty' AS check_name, COUNT(*) > 0 AS ok, COUNT(*) AS observed FROM porto.trips;
SELECT 'segments_nonempty' AS check_name, COUNT(*) > 0 AS ok, COUNT(*) AS observed FROM porto.segments;
SELECT 'trip_ids_unique' AS check_name, COUNT(*) = COUNT(DISTINCT trip_id) AS ok, COUNT(*) AS observed FROM porto.trips;
SELECT 'segments_match_trip_counts' AS check_name, COUNT(*) = 0 AS ok, COUNT(*) AS observed FROM (
  SELECT t.trip_id FROM porto.trips t JOIN porto.segments s ON s.trip_id = t.trip_id
  GROUP BY t.trip_id, t.num_of_segments HAVING COUNT(*) <> t.num_of_segments
) invalid;
SELECT 'trip_time_bounds' AS check_name, COUNT(*) = 0 AS ok, COUNT(*) AS observed FROM (
  SELECT t.trip_id FROM porto.trips t JOIN porto.segments s ON s.trip_id = t.trip_id
  GROUP BY t.trip_id, t.start_time, t.end_time HAVING MIN(s.t_s) <> t.start_time OR MAX(s.t_e) <> t.end_time
) invalid;
SELECT 'geometry_srid' AS check_name, COUNT(*) = 0 AS ok, COUNT(*) AS observed FROM porto.segments WHERE ST_SRID(segment_geom) <> 3763;
SELECT 'geometry_valid' AS check_name, COUNT(*) = 0 AS ok, COUNT(*) AS observed FROM porto.segments WHERE NOT ST_IsValid(segment_geom);
SELECT 'required_indexes' AS check_name, COUNT(*) = 4 AS ok, COUNT(*) AS observed
FROM pg_indexes WHERE schemaname = 'porto' AND indexname IN ('porto_segments_geom_gist_idx', 'porto_segments_tgeom_gist_idx', 'porto_segments_time_idx', 'porto_segments_trip_idx');
