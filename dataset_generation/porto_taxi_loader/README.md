# Porto Taxi Loader

This directory creates an isolated MobilityDB benchmark from the Kaggle Porto taxi `train.csv`.

Place the unmodified source file at:

`/work_beegfs/sunip956/master_thesis_trajectories/datasets/porto_taxi/raw/train.csv`

Submit `load_porto_to_mobilitydb.sbatch`. It retains only `MISSING_DATA=false` traces with two or more valid WGS84 points, deterministically selects full trips up to 50,000,000 segments, loads EPSG:3763 geometry into `porto_taxi_mobilitydb` on port `55433`, validates the database, and writes the runtime query config.

## Source Data Policy

`train.csv` is the unmodified Kaggle file, and three of its properties shape the
loader:

The header column is `DAY_TYPE`, not `DAYTYPE`, and every row carries `A`. It is
loaded as a real column even though it is constant, so queries generated against
it match everything; treat it as a zero-information attribute when reading
results.

`TRIP_ID` is not a key. 81 of 1,710,670 rows repeat one and 79 of those carry
different trip data, so the first occurrence in file order is kept and the rest
are counted as `duplicate_trip_id` in `load_metadata.json`. `porto.trips`
declares `source_trip_id UNIQUE`, which this policy satisfies.

A stationary taxi repeats its position across a 15-second sample, so around 1.3%
of segments are legitimately zero-length, and 26.5% of traces contain at least
one. These are kept: the segment is a real hop of the source polyline and
dropping it would break the uniform `start + 15 * index` timestamp mapping that
`trip_time_bounds` checks. `validate.sql` therefore asserts that every
`segment_geom` is a two-point line with the right SRID rather than asserting
`ST_IsValid`, which GEOS reports false for a zero-length line, and records the
degenerate counts as unenforced observations.

Every other check in `validate.sql` is enforced: the file ends with a guard that
divides by zero when any enforced check failed, so the job stops instead of
proceeding to build a query config over a bad database.

Re-submitting the loader with the same `RUN_ID` reuses an existing staging
directory when the input SHA-256, segment target and selection seed all match,
rather than re-parsing the 1.9 GB CSV.
