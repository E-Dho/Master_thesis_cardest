\echo Loading agents
COPY pol.agents (agent_id, age, "educationLevel", interest, joviality, family_size)
FROM :agents_file
WITH (FORMAT csv, DELIMITER E'\t', NULL '\N');

\echo Loading trips
COPY pol.trips (trip_id, agent_id, start_time, end_time, num_of_segments)
FROM :trips_file
WITH (FORMAT csv, DELIMITER E'\t', NULL '\N');

\echo Loading segments
COPY pol.segments (trip_id, segment_idx, s_x, s_y, e_x, e_y, t_s, t_e)
FROM :segments_file
WITH (FORMAT csv, DELIMITER E'\t', NULL '\N');

\echo Creating segment geometries and temporal geometries
UPDATE pol.segments
SET
  segment_geom = ST_SetSRID(ST_MakeLine(ST_MakePoint(s_x, s_y), ST_MakePoint(e_x, e_y)), 26916),
  segment_tgeom = CASE
    WHEN t_s = t_e THEN ('SRID=26916;POINT(' || s_x || ' ' || s_y || ')@' || to_char(t_s, 'YYYY-MM-DD"T"HH24:MI:SS.MS'))::tgeompoint
    ELSE ST_SetSRID(
      ST_MakeLine(
        ST_MakePointM(s_x, s_y, EXTRACT(EPOCH FROM t_s)),
        ST_MakePointM(e_x, e_y, EXTRACT(EPOCH FROM t_e))
      ),
      26916
    )::tgeompoint
  END;

\echo Creating trip geometries and temporal geometries
WITH trip_points AS (
  SELECT trip_id, 0::bigint AS ord, s_x AS x, s_y AS y, t_s AS t
  FROM pol.segments
  WHERE segment_idx = 0
  UNION ALL
  SELECT trip_id, (segment_idx + 1)::bigint AS ord, e_x AS x, e_y AS y, t_e AS t
  FROM pol.segments
),
trip_lines AS (
  SELECT
    trip_id,
    ST_SetSRID(ST_MakeLine(ST_MakePoint(x, y) ORDER BY ord), 26916) AS geom,
    ST_SetSRID(ST_MakeLine(ST_MakePointM(x, y, EXTRACT(EPOCH FROM t)) ORDER BY ord), 26916)::tgeompoint AS tgeom
  FROM trip_points
  GROUP BY trip_id
)
UPDATE pol.trips t
SET trip_geom = l.geom,
    trip_tgeom = l.tgeom
FROM trip_lines l
WHERE t.trip_id = l.trip_id;

\echo Creating indexes
CREATE INDEX agents_agent_id_idx ON pol.agents(agent_id);
CREATE INDEX trips_agent_id_idx ON pol.trips(agent_id);
CREATE INDEX trips_time_idx ON pol.trips(start_time, end_time);
CREATE INDEX trips_trip_tgeom_gist_idx ON pol.trips USING gist(trip_tgeom);
CREATE INDEX segments_trip_id_idx ON pol.segments(trip_id);
CREATE INDEX segments_time_idx ON pol.segments(t_s, t_e);
CREATE INDEX segments_segment_tgeom_gist_idx ON pol.segments USING gist(segment_tgeom);
CREATE INDEX segments_segment_geom_gist_idx ON pol.segments USING gist(segment_geom);
ANALYZE pol.agents;
ANALYZE pol.trips;
ANALYZE pol.segments;
