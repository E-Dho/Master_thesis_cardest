DROP SCHEMA IF EXISTS porto CASCADE;
CREATE SCHEMA porto;

CREATE TABLE porto.taxis (
  taxi_id integer PRIMARY KEY
);

CREATE TABLE porto.trips (
  trip_id bigint PRIMARY KEY,
  source_trip_id text UNIQUE NOT NULL,
  taxi_id integer NOT NULL REFERENCES porto.taxis(taxi_id),
  call_type char(1) NOT NULL CHECK (call_type IN ('A', 'B', 'C')),
  origin_call bigint,
  origin_stand integer,
  start_time timestamp without time zone NOT NULL,
  end_time timestamp without time zone NOT NULL,
  daytype char(1) NOT NULL CHECK (daytype IN ('A', 'B', 'C')),
  num_of_segments integer NOT NULL CHECK (num_of_segments > 0),
  trip_tgeom tgeompoint,
  trip_geom geometry(LineString, 3763),
  CHECK (start_time <= end_time)
);

CREATE TABLE porto.segments (
  trip_id bigint NOT NULL REFERENCES porto.trips(trip_id),
  segment_idx integer NOT NULL,
  s_x double precision NOT NULL,
  s_y double precision NOT NULL,
  e_x double precision NOT NULL,
  e_y double precision NOT NULL,
  t_s timestamp without time zone NOT NULL,
  t_e timestamp without time zone NOT NULL,
  segment_tgeom tgeompoint,
  segment_geom geometry(LineString, 3763),
  PRIMARY KEY (trip_id, segment_idx),
  CHECK (t_s < t_e)
);

CREATE UNLOGGED TABLE porto.segment_stage_wgs84 (
  trip_id bigint NOT NULL,
  segment_idx integer NOT NULL,
  s_lon double precision NOT NULL,
  s_lat double precision NOT NULL,
  e_lon double precision NOT NULL,
  e_lat double precision NOT NULL,
  t_s timestamp without time zone NOT NULL,
  t_e timestamp without time zone NOT NULL
);
