-- Phase 1 validation for benchmark v3. Read-only; run with psql -X -f.
-- Every check prints one labelled row; the expected value is in the comment.
\pset pager off
\timing on

SELECT 'versions' AS check, version() AS value;
SELECT 'postgis' AS check, postgis_full_version() AS value;
SELECT 'mobilitydb' AS check, mobilitydb_full_version() AS value;
SHOW TimeZone;
SELECT 'server_timezone_setting' AS check, current_setting('TimeZone') AS value;

-- Row counts (expected: 35000 agents, 4304287 trips, 61941301 segments).
SELECT 'rows.agents' AS check, count(*) FROM pol.agents;
SELECT 'rows.trips' AS check, count(*) FROM pol.trips;
SELECT 'rows.segments' AS check, count(*) FROM pol.segments;

-- Data needed by v3 must be complete (expected: 0 everywhere).
SELECT 'segments.null_raw_columns' AS check, count(*) FROM pol.segments
WHERE s_x IS NULL OR s_y IS NULL OR e_x IS NULL OR e_y IS NULL OR t_s IS NULL OR t_e IS NULL OR segment_geom IS NULL;
SELECT 'segments.t_s_after_t_e' AS check, count(*) FROM pol.segments WHERE t_s > t_e;
SELECT 'trips.without_segments' AS check, count(*) FROM pol.trips t
WHERE NOT EXISTS (SELECT 1 FROM pol.segments s WHERE s.trip_id = t.trip_id);

-- segment_geom must be exactly the line between the raw endpoints (expected: 0).
SELECT 'segments.geom_differs_from_raw' AS check, count(*) FROM pol.segments
WHERE ST_X(ST_StartPoint(segment_geom)) <> s_x OR ST_Y(ST_StartPoint(segment_geom)) <> s_y
   OR ST_X(ST_EndPoint(segment_geom)) <> e_x OR ST_Y(ST_EndPoint(segment_geom)) <> e_y;

-- Data domain (expected: inside the domains of pol_v3_config.json, i.e. its
-- domain_overrides; pol_query_config.json is older and narrower).
SELECT 'segments.extent' AS check,
       least(min(s_x), min(e_x)) AS min_x, greatest(max(s_x), max(e_x)) AS max_x,
       least(min(s_y), min(e_y)) AS min_y, greatest(max(s_y), max(e_y)) AS max_y,
       min(t_s) AS min_t, max(t_e) AS max_t
FROM pol.segments;

-- Segment shapes that the exact predicate treats specially.
SELECT 'segments.instants' AS check, count(*) FROM pol.segments WHERE t_s = t_e;
SELECT 'segments.stationary' AS check, count(*) FROM pol.segments WHERE s_x = e_x AND s_y = e_y AND t_s < t_e;
SELECT 'segments.duration_seconds' AS check,
       percentile_disc(ARRAY[0.5, 0.9, 0.99, 1.0]) WITHIN GROUP (ORDER BY EXTRACT(EPOCH FROM t_e - t_s))
FROM pol.segments TABLESAMPLE SYSTEM (1);

-- Trip time equals the span of its segments (expected: 0).
SELECT 'trips.time_differs_from_segments' AS check, count(*) FROM pol.trips t
JOIN (SELECT trip_id, min(t_s) AS lo, max(t_e) AS hi FROM pol.segments GROUP BY trip_id) g USING (trip_id)
WHERE t.start_time <> g.lo OR t.end_time <> g.hi;

-- Consistency of the MobilityDB columns with the raw times (informational:
-- v3 labels do not use tgeom; see SEMANTICS.md, section 6. The pools job
-- measures the size of the deviations).
SELECT 'segment_tgeom.start_differs' AS check, count(*) FROM pol.segments TABLESAMPLE SYSTEM (1)
WHERE startTimestamp(segment_tgeom) <> (t_s AT TIME ZONE 'UTC');

-- Indexes used by the v3 prefilters (expected: segment_geom gist and (t_s, t_e) btree).
SELECT 'indexes' AS check, indexname, indexdef FROM pg_indexes WHERE schemaname = 'pol' ORDER BY indexname;
SELECT 'stats.analyzed' AS check, relname, last_analyze, last_autoanalyze
FROM pg_stat_user_tables WHERE schemaname = 'pol';
