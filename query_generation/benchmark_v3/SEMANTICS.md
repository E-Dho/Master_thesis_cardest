# POL benchmark v3: query and label semantics

Version string: `pol_exact_st_v3`. Every query record carries it in
`semantics_version`; the merge step refuses rows with any other value.

This document is the single source of truth for what a v3 label means. The SQL
renderer (`query_model.py`), the numpy reference (`exact.py`) and the
verification job (`verify_sample.py`) all implement exactly these rules.

## 1. Data model

- A **segment** row `(trip_id, segment_idx, s_x, s_y, e_x, e_y, t_s, t_e)`
  describes a movement that is **linear in space and time**: at time
  `t = t_s + u * (t_e - t_s)` with `u in [0, 1]` the object is at
  `(s_x + u * (e_x - s_x), s_y + u * (e_y - s_y))`.
- If `t_s = t_e` the segment is an **instant** (POL has none, but the
  definition and the tests cover them): the object is at `(s_x, s_y)`
  at exactly that time.
- If the start and end position coincide but `t_s < t_e`, the object is
  **stationary** at that position for the whole interval.
- Timestamps are the stored `timestamp without time zone` values, compared as
  they are. No time-zone conversion takes place anywhere in v3, and the
  MobilityDB columns `segment_tgeom` / `trip_tgeom` are **not** used for labels
  (see section 6).
- A **trajectory** is a trip; it consists of its segments.

## 2. Predicates

All windows and intervals are **closed** (boundaries belong to them) and
**non-empty** (`min <= max`, `L <= U`); the SQL renderer and the numpy
reference raise an error otherwise, because the clipping formulas below are
only equivalent to the definition for non-empty intervals.

| Kind | Applies to | Matches when |
|---|---|---|
| `nominal_eq` | any nominal attribute | `value = v` |
| `scalar_range` | numeric / integer attribute | `lower <= value <= upper` |
| `scalar_unbounded` | numeric / integer attribute | `value op v`, `op` in `<, <=, >, >=` |
| `temporal` | `trips.trip_time`, `segments.segment_time` | the row's interval `[start, end]` intersects the query interval `[L, U]` |
| `spatial` | `segments.segment_geom` only | the segment's line (section 1) intersects the closed window `W` |
| `spatio_temporal` | `segments` only | there is a `u in [0, 1]` with time `t(u)` in `[L, U]` **and** position `p(u)` in `W` |

Domains are the measured data domains: `pol_v3_config.json` overrides the
domains of the shared base config with the values the cluster check and the
pilot's pools job measured (the base config's domains came from an earlier,
smaller dataset and cut off up to ~420 m of space and the first hour of
time). The pools job records the data domain of every ordered attribute and
lists any remaining violation.

Unbounded temporal or spatial bounds are represented by `null` (temporal) or
by extending the window by `unbounded_spatial_margin` beyond the data domain
(spatial), which is equivalent to an infinite half-plane because every
segment lies inside the domain.

### 2.1 Exact segment test (Liang-Barsky clipping)

For a segment and a window `W = [x0, x1] x [y0, y1]`, per axis:

- if `e_x = s_x`: the axis allows all of `u in [0, 1]` if `x0 <= s_x <= x1`,
  otherwise nothing (encoded as the empty range `[2, -1]`);
- otherwise `lo_x = min((x0 - s_x) / (e_x - s_x), (x1 - s_x) / (e_x - s_x))`,
  `hi_x = max(...)`.

`u0 = max(0, lo_x, lo_y)`, `u1 = min(1, hi_x, hi_y)`.

- **spatial match**: `u0 <= u1`.
- **spatio-temporal match**, segment with `t_s < t_e`: `u0 <= u1` and the
  parameter range `[u0, u1]` intersects `[uL, uU]`, where
  `uL = seconds(L - t_s) / seconds(t_e - t_s)` and likewise `uU`
  (a missing bound is `-inf` / `+inf`).
- **spatio-temporal match**, instant (`t_s = t_e`): `u0 <= u1` and
  `L <= t_s <= U`.

The SQL renderer and the numpy reference evaluate these formulas with the same
IEEE-754 double operations in the same order, so the verification job can
demand exact agreement instead of a tolerance.

The SQL additionally contains the index-friendly prefilters
`segment_geom && envelope(W)` and `t_s <= U AND t_e >= L`. Both are implied
by the exact test (they can only remove rows the exact test also removes), so
they change speed, not results.

## 3. Queries and labels

A query has a category `dimension.interval.relation`:

- `dimension` in `standard, temporal, spatial, spatio_temporal` defines the
  **core** predicates:
  - `standard`: 1..n standard predicates (n = standard attributes of the
    chosen tables, uniformly drawn count);
  - `temporal`: exactly one `temporal` predicate (on `trips` or `segments`);
  - `spatial`: exactly one `spatial` predicate (on `segments`);
  - `spatio_temporal`: exactly one `spatio_temporal` predicate (on `segments`).
- `interval` in `range, unbounded`: in `unbounded` queries at least one
  ordered core component is one-sided (for `spatio_temporal`, the spatial and
  the temporal part are separate components).
- `relation`:
  - `single`: one table; label = `COUNT(*)` of matching rows
    (`label_kind = "row_count"`).
  - `multi`: a connected join containing `trips` and at least one other
    table; label = `COUNT(DISTINCT t.trip_id)`, the number of matching
    **trajectories** (`label_kind = "distinct_trips"`). `COUNT(*)` is not
    computed for multi queries.

A trajectory matches a multi query when its trip row and its agent row (if
joined) satisfy their predicates and **at least one of its segments** satisfies
all segment predicates simultaneously (segment-coupled semantics).

### 3.1 Attribute add-ons

Non-standard **multi** queries additionally receive 0, 1 or 2 standard
predicates, drawn without replacement from the standard attributes of the
chosen tables. The count is assigned **per slot**, not per candidate: the
slots of a category are permuted with a seeded RNG and given exact quotas
(largest-remainder rounding of 0.50 / 0.25 / 0.25, i.e. 375 / 188 / 187 of
750), and every retry of a slot keeps its count. Otherwise candidates with
add-ons, which are more often empty, would be under-represented after the
positivity rejection. Add-ons are always bounded
(range or equality); the `interval` category refers to the core predicates.
Single queries never receive add-ons.

### 3.2 Restrictions

- At most one spatial window and at most one temporal interval per query.
- No trip-level geometry (`trips.trip_geom`) and no trip-level time in
  spatial or spatio-temporal queries: all spatial and spatio-temporal
  conditions live on one segment row.
- Every label is strictly positive (section 5).

## 4. Grid

| | range | unbounded |
|---|---|---|
| single (per dimension) | 250 | 250 |
| multi (per dimension) | 750 | 750 |

8 x 250 single + 8 x 750 multi = 2,000 + 6,000 = **8,000 queries**.

## 5. Sampling and positivity

- Every slot `(category, index)` draws its candidates from a deterministic
  RNG seeded with `blake2b(base_seed, category, index, attempt)`.
- Centers: with probability 0.9 from a row of a seeded, materialized pool
  (`pools.json`), otherwise uniform over the declared domain.
  - `spatio_temporal`: space and time are drawn **jointly** from one pool
    segment at a uniform `u`, so the center is a place where an object was at
    that time.
  - `spatial`: a uniform point along a pool segment.
  - `temporal`: a uniform time inside a pool row's interval.
- Widths: numeric ranges as in v2 (50 % uniform over the domain, 50 %
  exponential with mean domain/10); **spatial widths log-uniform** per axis
  (independently for x and y) between `min_meters` (10 m) and the extent of
  that axis; **temporal widths log-uniform** between `min_seconds` (60 s)
  and the full temporal domain. The pilot showed why spatial widths need it:
  POL holds about 2.5 million segments per km², so with the v2 rule even the
  smallest spatial window matched 0.46 % of all segments.
- Positivity: every candidate is labelled exactly with one SQL statement.
  A label of 0 rejects the candidate and the slot moves to its next attempt;
  so does a duplicate (same canonical predicate hash as an already accepted
  query of the category). Rejections are logged with their reason, so the
  selection effect is reported per category, and the zero-labelled
  candidates form a free set of known-empty queries. After `max_attempts`
  the slot fails loudly.
- There is no `EXISTS` pre-check: for an empty query `EXISTS` has to inspect
  every candidate row just like `COUNT`, and for a positive query it is
  extra work, so it would never save time.

## 6. Why the labels do not use MobilityDB `tgeompoint`

The exact predicate could be written with MobilityDB
(`eIntersects(atTime(segment_tgeom, span), W)`), but:

1. `segment_tgeom` does not reproduce the stored times exactly. The cluster
   check (job 22321118) found 107,188 mismatches between
   `startTimestamp(segment_tgeom)` and `t_s` in a 1 % sample. The text path
   for instants in `load.sql` cannot be the cause, because POL contains no
   instants; the size of the deviations (rounding of the float epoch in
   the M coordinate vs. a time-zone shift) is measured by the pools job
   (`diagnostics.segment_tgeom_start_offset_seconds`). Either way the raw
   columns are the reference.
2. Function names differ between MobilityDB versions.
3. The closed form on the raw columns is the same formula as the numpy
   reference and the model's segment representation, so verification can be
   exact.

`verify_sample.py` can optionally compare against the MobilityDB formulation
as a diagnostic once the cluster check has confirmed the time-zone handling.
