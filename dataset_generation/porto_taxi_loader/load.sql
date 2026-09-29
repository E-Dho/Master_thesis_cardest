COPY porto.taxis (taxi_id)
FROM :taxis_file WITH (FORMAT csv, DELIMITER E'\t', NULL '\N');

COPY porto.trips (trip_id, source_trip_id, taxi_id, call_type, origin_call, origin_stand, start_time, end_time, daytype, num_of_segments)
FROM :trips_file WITH (FORMAT csv, DELIMITER E'\t', NULL '\N');

COPY porto.segment_stage_wgs84 (trip_id, segment_idx, s_lon, s_lat, e_lon, e_lat, t_s, t_e)
FROM :segments_file WITH (FORMAT csv, DELIMITER E'\t', NULL '\N');

INSERT INTO porto.segments (trip_id, segment_idx, s_x, s_y, e_x, e_y, t_s, t_e)
SELECT
  trip_id,
  segment_idx,
  ST_X(ST_Transform(ST_SetSRID(ST_MakePoint(s_lon, s_lat), 4326), 3763)),
  ST_Y(ST_Transform(ST_SetSRID(ST_MakePoint(s_lon, s_lat), 4326), 3763)),
  ST_X(ST_Transform(ST_SetSRID(ST_MakePoint(e_lon, e_lat), 4326), 3763)),
  ST_Y(ST_Transform(ST_SetSRID(ST_MakePoint(e_lon, e_lat), 4326), 3763)),
  t_s,
  t_e
FROM porto.segment_stage_wgs84;
DROP TABLE porto.segment_stage_wgs84;

UPDATE porto.segments
SET segment_geom = ST_SetSRID(ST_MakeLine(ST_MakePoint(s_x, s_y), ST_MakePoint(e_x, e_y)), 3763),
    segment_tgeom = ST_SetSRID(ST_MakeLine(ST_MakePointM(s_x, s_y, EXTRACT(EPOCH FROM t_s)), ST_MakePointM(e_x, e_y, EXTRACT(EPOCH FROM t_e))), 3763)::tgeompoint;

WITH trip_points AS (
  SELECT trip_id, 0::bigint AS ord, s_x AS x, s_y AS y, t_s AS t FROM porto.segments WHERE segment_idx = 0
  UNION ALL
  SELECT trip_id, (segment_idx + 1)::bigint, e_x, e_y, t_e FROM porto.segments
), trip_lines AS (
  SELECT trip_id,
    ST_SetSRID(ST_MakeLine(ST_MakePoint(x, y) ORDER BY ord), 3763) AS geom,
    ST_SetSRID(ST_MakeLine(ST_MakePointM(x, y, EXTRACT(EPOCH FROM t)) ORDER BY ord), 3763)::tgeompoint AS tgeom
  FROM trip_points GROUP BY trip_id
)
UPDATE porto.trips t SET trip_geom = lines.geom, trip_tgeom = lines.tgeom FROM trip_lines lines WHERE t.trip_id = lines.trip_id;

CREATE INDEX porto_taxis_taxi_id_idx ON porto.taxis(taxi_id);
CREATE INDEX porto_trips_taxi_id_idx ON porto.trips(taxi_id);
CREATE INDEX porto_trips_time_idx ON porto.trips(start_time, end_time);
CREATE INDEX porto_trips_num_segments_idx ON porto.trips(num_of_segments);
CREATE INDEX porto_trips_call_type_idx ON porto.trips(call_type);
CREATE INDEX porto_trips_daytype_idx ON porto.trips(daytype);
CREATE INDEX porto_trips_origin_call_idx ON porto.trips(origin_call);
CREATE INDEX porto_trips_origin_stand_idx ON porto.trips(origin_stand);
CREATE INDEX porto_trips_geom_gist_idx ON porto.trips USING gist(trip_geom);
CREATE INDEX porto_trips_tgeom_gist_idx ON porto.trips USING gist(trip_tgeom);
CREATE INDEX porto_segments_trip_idx ON porto.segments(trip_id);
CREATE INDEX porto_segments_time_idx ON porto.segments(t_s, t_e);
CREATE INDEX porto_segments_geom_gist_idx ON porto.segments USING gist(segment_geom);
CREATE INDEX porto_segments_tgeom_gist_idx ON porto.segments USING gist(segment_tgeom);
ANALYZE porto.taxis;
ANALYZE porto.trips;
ANALYZE porto.segments;
