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
chosen.

The source directory is never modified, and the two directories may not overlap
in either direction -- judged by what they are on disk, not by how they are
spelled, since a case-insensitive filesystem gives one directory many names.  ``--force`` empties the output first, so a rewritten arm
is this source under a new order rather than a union with whatever the previous
arm left there.
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


def same_directory(first: Path, second: Path) -> bool:
    """Say whether two paths name the same directory on disk."""

    try:
        return first.samefile(second)
    except OSError:
        # One of them does not exist yet, or cannot be stat'd.  A path that is
        # not there is not the other one.
        return False


def overlaps(first: Path, second: Path) -> bool:
    """Say whether either path is, or contains, the other -- by identity.

    resolve() normalizes symlinks and ".." segments but not case, and not every
    filesystem this runs on is case-sensitive: APFS and NTFS are not by
    default.  "<root>/Source" and "<root>/source" are then one directory
    spelled two ways, which comparing the spellings calls distinct -- and
    --force went on to empty the "output", which was the source, taking
    sample_rows.npy with it and exiting 0.  Stat identity does not care how a
    path is spelled.
    """

    if same_directory(first, second):
        return True
    if any(same_directory(parent, second) for parent in first.parents):
        return True
    return any(same_directory(parent, first) for parent in second.parents)


def clear_directory(directory: Path, *, protecting: Path) -> None:
    """Empty a directory without removing it, so a forced arm is not a union.

    ``protecting`` is re-checked here even though main refuses an overlap
    before reaching this point.  This is the one destructive step in the
    script, what it would destroy is a prepared fixture that costs hours to
    rebuild, and a later change that reorders or weakens the checks above
    should fail loudly rather than delete.
    """

    if overlaps(directory, protecting):
        raise SystemExit(
            f"refusing to empty {directory}: it is, or contains, {protecting}"
        )
    for entry in sorted(directory.iterdir()):
        if entry.is_dir() and not entry.is_symlink():
            shutil.rmtree(entry)
        else:
            entry.unlink()


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
    if overlaps(output, prepared):
        # Equality is not the only way to write into the source.  With --output
        # inside --prepared the arm directory is created in the source and then
        # copied into the next arm; with --prepared inside --output the source
        # ends up nested in the arm it produced.  Neither is a directory any
        # later run could read, and both contradict leaving the source alone.
        raise SystemExit(
            f"--output {output} and --prepared {prepared} overlap; this never "
            "edits in place or writes an arm inside its own source"
        )
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
    if args.force:
        # Everything above this line can still refuse -- a bad order file leaves
        # the output untouched.  From here the arm is written, and it has to be
        # exactly this source under a new order: a companion the previous arm
        # left behind, sample_trajectory_ids.npy above all, would be read as
        # this arm's row provenance.
        clear_directory(output, protecting=prepared)
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
    row_count = None
    if sample_rows.exists():
        row_count = permute_sample_rows(
            sample_rows, output / SAMPLE_ROWS_NAME, permutation
        )

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
