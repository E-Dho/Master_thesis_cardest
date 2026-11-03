"""Write a prepared directory whose columns are in a different order.

The ordering ablation varies one thing: the position of each column in
``ModelMetadata.columns``, which is the model's autoregressive order.  Doing
that by rewriting the prepared artifacts keeps the manifest and
``sample_rows.npy`` consistent by construction, so every arm is an ordinary run
that differs from the control only in ``dataset.prepared_directory``.

    python3 -m model.scripts.permute_prepared_columns \
        --prepared data/neurocard_prepared/job_light_ranges_live_content \
        --output   data/neurocard_prepared/job_light_ranges_live_content_ordB \
        --order    model/orderings/job_light_ranges_mutual_information.json

``--order`` is a JSON list of column names, or a JSON object carrying that list
under "order" plus whatever the ordering was derived from; an object is copied
into the output's provenance record whole, so the arm says how its order was
chosen.  The source directory is never modified.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from model.src.data.column_ordering import (
    column_permutation,
    permute_metadata_columns,
)
from model.src.data.complete_domain_preparation import validate_prepared_manifest

PROVENANCE_NAME = "column_permutation.json"
MANIFEST_NAME = "manifest.json"
SAMPLE_ROWS_NAME = "sample_rows.npy"


def load_order(path: Path) -> tuple[list[str], dict[str, Any]]:
    """Return the requested column order and the provenance it arrived with."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return [str(name) for name in payload], {}
    if isinstance(payload, dict) and isinstance(payload.get("order"), list):
        provenance = {key: value for key, value in payload.items() if key != "order"}
        return [str(name) for name in payload["order"]], provenance
    raise SystemExit(
        f"{path} must hold a JSON list of column names, or an object with an "
        '"order" list'
    )


def permute_sample_rows(source: Path, destination: Path, permutation: tuple[int, ...]) -> int:
    """Copy sample rows column by column so peak memory stays one column wide."""

    rows = np.load(source, mmap_mode="r")
    if rows.ndim != 2 or rows.shape[1] != len(permutation):
        raise SystemExit(
            f"{source} has shape {rows.shape}, which does not match the "
            f"{len(permutation)} columns the manifest declares"
        )
    out = np.lib.format.open_memmap(
        destination, mode="w+", dtype=rows.dtype, shape=rows.shape
    )
    for new_index, old_index in enumerate(permutation):
        out[:, new_index] = rows[:, old_index]
    out.flush()
    del out
    return int(rows.shape[0])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", required=True, help="source prepared directory")
    parser.add_argument("--output", required=True, help="prepared directory to write")
    parser.add_argument(
        "--order",
        required=True,
        help="JSON file holding the new column order, newline-free list or object",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="overwrite a non-empty output directory",
    )
    args = parser.parse_args()

    prepared = Path(args.prepared).resolve()
    output = Path(args.output).resolve()
    if output == prepared:
        raise SystemExit("--output must differ from --prepared; this never edits in place")
    if output.exists() and any(output.iterdir()) and not args.force:
        raise SystemExit(f"{output} is not empty; pass --force to overwrite it")

    # Refuses a source whose rows and manifest already disagree, so a failure
    # here is never mistaken for one this script introduced.
    source_metadata = validate_prepared_manifest(prepared)
    manifest = json.loads((prepared / MANIFEST_NAME).read_text(encoding="utf-8"))

    order, order_provenance = load_order(Path(args.order))
    permutation = column_permutation(source_metadata, order)
    permuted = permute_metadata_columns(source_metadata, order)

    output.mkdir(parents=True, exist_ok=True)
    for entry in sorted(prepared.iterdir()):
        if entry.name in {MANIFEST_NAME, SAMPLE_ROWS_NAME, PROVENANCE_NAME}:
            continue
        destination = output / entry.name
        if entry.is_dir():
            shutil.copytree(entry, destination, dirs_exist_ok=True)
        else:
            # Row-aligned companions (trajectory ids, segment ids) and the
            # preparation statistics are unaffected by a column permutation.
            shutil.copy2(entry, destination)

    manifest["metadata"] = permuted.to_json_dict()
    (output / MANIFEST_NAME).write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )

    sample_rows = prepared / SAMPLE_ROWS_NAME
    written_rows = output / SAMPLE_ROWS_NAME
    row_count = None
    if sample_rows.exists():
        row_count = permute_sample_rows(sample_rows, written_rows, permutation)
    elif written_rows.exists():
        # --force into a directory that held an earlier arm.  Leaving its rows
        # beside a manifest they were never encoded against would give this arm
        # the other arm's data in this arm's column order.
        written_rows.unlink()

    provenance = {
        "source_prepared_directory": str(prepared),
        "source_schema_hash": source_metadata.schema_hash
        or source_metadata.stable_schema_hash(),
        "source_column_order": [column.name for column in source_metadata.columns],
        "schema_hash": permuted.stable_schema_hash(),
        "column_order": [column.name for column in permuted.columns],
        "permutation": list(permutation),
        "sample_rows": row_count,
        "order_provenance": order_provenance,
    }
    (output / PROVENANCE_NAME).write_text(
        json.dumps(provenance, indent=2, sort_keys=True), encoding="utf-8"
    )

    # The same check every sample source runs at startup: each column's encoded
    # values have to lie inside that column's domain.  With domains from 2 to
    # six figures wide, a permutation applied to one of the two artifacts and
    # not the other does not survive this.
    written = validate_prepared_manifest(output)
    moved = sum(
        1
        for new_index, old_index in enumerate(permutation)
        if new_index != old_index
    )
    print(f"wrote {output}")
    print(f"  columns: {len(written.columns)} ({moved} moved)")
    print(f"  schema_hash: {provenance['source_schema_hash']} -> {provenance['schema_hash']}")
    if row_count is not None:
        print(f"  sample_rows: {row_count}")


if __name__ == "__main__":
    main()
