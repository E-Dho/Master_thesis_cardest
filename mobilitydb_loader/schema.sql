DROP SCHEMA IF EXISTS pol CASCADE;
CREATE SCHEMA pol;

CREATE TABLE pol.agents (
  agent_id bigint PRIMARY KEY,
  age double precision NOT NULL,
  "educationLevel" text NOT NULL,
  interest text NOT NULL,
  joviality double precision NOT NULL,
  family_size integer NOT NULL
);

CREATE TABLE pol.trips (
  trip_id bigint PRIMARY KEY,
  agent_id bigint NOT NULL REFERENCES pol.agents(agent_id),
  start_time timestamp without time zone NOT NULL,
  end_time timestamp without time zone NOT NULL,
  num_of_segments integer NOT NULL,
  trip_tgeom tgeompoint,
  trip_geom geometry(LineString, 26916),
  CHECK (start_time <= end_time),
  CHECK (num_of_segments > 0)
);

CREATE TABLE pol.segments (
  trip_id bigint NOT NULL REFERENCES pol.trips(trip_id),
  segment_idx integer NOT NULL,
  s_x double precision NOT NULL,
  s_y double precision NOT NULL,
  e_x double precision NOT NULL,
  e_y double precision NOT NULL,
  t_s timestamp without time zone NOT NULL,
  t_e timestamp without time zone NOT NULL,
  segment_tgeom tgeompoint,
  segment_geom geometry(LineString, 26916),
  PRIMARY KEY (trip_id, segment_idx),
  CHECK (t_s <= t_e)
);
