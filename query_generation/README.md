# Query Generation

This directory contains the config-driven SQL workload generator used to create labeled cardinality-estimation queries.

The generator samples COUNT queries across:

- dimensions: `standard`, `temporal`, `spatial`, `spatio_temporal`
- intervals: `range`, `unbounded`
- relations: `single`, `multi`

## Generate SQL Only

```bash
python3 query_generation/query_generator.py \
  --config query_generation/pol_query_config.json \
  --output queries.jsonl \
  --queries-per-category 500 \
  --seed 1 \
  --no-execute
```

## Generate And Execute

```bash
python3 query_generation/query_generator.py \
  --config query_generation/pol_query_config.json \
  --output queries.jsonl \
  --queries-per-category 500 \
  --seed 1 \
  --execute \
  --host 127.0.0.1 \
  --port 55432 \
  --dbname pol_mobilitydb \
  --user sunip956
```

## Tests

```bash
python3 -m unittest query_generation.test_query_generator
```
