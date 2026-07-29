\echo Version checks
SELECT postgis_full_version();
SELECT mobilitydb_version();

\echo Row counts
SELECT 'agents' AS table_name, COUNT(*) AS rows FROM pol.agents
UNION ALL
SELECT 'trips', COUNT(*) FROM pol.trips
UNION ALL
SELECT 'segments', COUNT(*) FROM pol.segments;

\echo Required validation checks
SELECT 'agents_nonempty' AS check_name, COUNT(*) > 0 AS ok, COUNT(*) AS observed FROM pol.agents;
SELECT 'segments_nonempty' AS check_name, COUNT(*) > 0 AS ok, COUNT(*) AS observed FROM pol.segments;
SELECT 'trip_id_unique' AS check_name, COUNT(*) = COUNT(DISTINCT trip_id) AS ok, COUNT(*) AS observed FROM pol.trips;
SELECT 'agent_required_nonnull' AS check_name,
       COUNT(*) = COUNT(age) AND COUNT(*) = COUNT("educationLevel") AND COUNT(*) = COUNT(interest)
       AND COUNT(*) = COUNT(joviality) AND COUNT(*) = COUNT(family_size) AS ok,
       COUNT(*) AS observed
FROM pol.agents;
SELECT 'trip_required_nonnull' AS check_name,
       COUNT(*) = COUNT(agent_id) AND COUNT(*) = COUNT(start_time) AND COUNT(*) = COUNT(end_time)
       AND COUNT(*) = COUNT(num_of_segments) AND COUNT(*) = COUNT(trip_tgeom) AND COUNT(*) = COUNT(trip_geom) AS ok,
       COUNT(*) AS observed
FROM pol.trips;
SELECT 'segment_required_nonnull' AS check_name,
       COUNT(*) = COUNT(s_x) AND COUNT(*) = COUNT(s_y) AND COUNT(*) = COUNT(e_x) AND COUNT(*) = COUNT(e_y)
       AND COUNT(*) = COUNT(t_s) AND COUNT(*) = COUNT(t_e)
       AND COUNT(*) = COUNT(segment_tgeom) AND COUNT(*) = COUNT(segment_geom) AS ok,
       COUNT(*) AS observed
FROM pol.segments;
SELECT 'segment_time_order' AS check_name, COUNT(*) = 0 AS ok, COUNT(*) AS observed
FROM pol.segments
WHERE t_s > t_e;
SELECT 'trip_segment_counts' AS check_name, COUNT(*) = 0 AS ok, COUNT(*) AS observed
FROM (
  SELECT t.trip_id
  FROM pol.trips t
  JOIN pol.segments s ON s.trip_id = t.trip_id
  GROUP BY t.trip_id, t.num_of_segments
  HAVING t.num_of_segments <> COUNT(*)
) bad;
SELECT 'trip_time_bounds' AS check_name, COUNT(*) = 0 AS ok, COUNT(*) AS observed
FROM (
  SELECT t.trip_id
  FROM pol.trips t
  JOIN pol.segments s ON s.trip_id = t.trip_id
  GROUP BY t.trip_id, t.start_time, t.end_time
  HAVING t.start_time <> MIN(s.t_s) OR t.end_time <> MAX(s.t_e)
) bad;

\echo Smoke cardinality queries
SELECT 'trips_time_window' AS query_name, COUNT(*) AS cardinality
FROM pol.trips
WHERE start_time < timestamp '2019-07-02 00:00:00'
  AND end_time >= timestamp '2019-07-01 12:00:00';

SELECT 'segments_spatial_bbox' AS query_name, COUNT(*) AS cardinality
FROM pol.segments
WHERE segment_geom && ST_MakeEnvelope(741000, 3736000, 743000, 3738500, 26916);

SELECT 'trips_demographics_time' AS query_name, COUNT(*) AS cardinality
FROM pol.trips t
JOIN pol.agents a ON a.agent_id = t.agent_id
WHERE a."educationLevel" IN ('Bachelors', 'Graduate')
  AND a.joviality >= 0.5
  AND t.start_time < timestamp '2019-07-08 00:00:00';

SELECT 'trips_spatiotemporal' AS query_name, COUNT(DISTINCT t.trip_id) AS cardinality
FROM pol.trips t
JOIN pol.segments s ON s.trip_id = t.trip_id
WHERE s.t_s < timestamp '2019-07-02 00:00:00'
  AND s.t_e >= timestamp '2019-07-01 00:00:00'
  AND s.segment_geom && ST_MakeEnvelope(741000, 3736000, 743000, 3738500, 26916);
