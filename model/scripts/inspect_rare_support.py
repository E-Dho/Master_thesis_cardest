"""Run rare-support discovery against a config's fixture without training.

Discovery is where the conjunctive configuration either works or does not, and
it needs no GPU and no optimizer.  Running it alone turns a failed assumption
from a queue cycle into a couple of minutes on a login node.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    # Running this file by path puts model/scripts on sys.path, not the repo
    # root, so "import model" fails.  The repo convention is
    # "python3 -m model.scripts.<name>" from the root, but the path form is
    # what anyone reads off a README line or shell history, and a
    # ModuleNotFoundError at a cluster prompt costs a round trip to diagnose.
    import sys as _sys
    from pathlib import Path as _Path

    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))

from model.src.config import load_simple_yaml, validate_config
from model.src.data.importance_sampling import RareSupportSampleSource
from model.src.data.sample_sources import sample_source_from_config


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit rare-support discovery for a config without training."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument(
        "--allow-uncovered",
        action="store_true",
        help=(
            "report an uncovered required_extension_column instead of failing, so the "
            "diagnosis can be read without require_conjunctions stopping the run"
        ),
    )
    args = parser.parse_args()

    config = load_simple_yaml(args.config)
    validate_config(config)
    if args.allow_uncovered:
        conjunctive = (
            config.setdefault("rare_support", {})
            .setdefault("discovery", {})
            .setdefault("conjunctive", {})
        )
        conjunctive["require_conjunctions"] = False

    source = sample_source_from_config(config)
    if not isinstance(source, RareSupportSampleSource):
        raise SystemExit(
            "config did not construct an enabled RareSupportSampleSource; "
            "check rare_support.enabled and rare_auxiliary.enabled"
        )

    report: dict[str, Any] = {
        "config": str(args.config),
        "conjunctive": source.conjunctive_summary(),
        "selected_strata": [
            {
                "stratum_id": stratum.stratum_id,
                "depth": stratum.depth,
                "columns": [
                    source.metadata.columns[index].name for index in stratum.column_indices
                ],
                "foj_count": float(stratum.foj_count),
                "support_score": float(stratum.support_score),
                "support_deficit": float(stratum.support_deficit),
                "alpha": float(stratum.alpha),
            }
            for stratum in source.selected_strata
        ],
    }

    status = report["conjunctive"]
    print(f"materialized_rows                  = {status.get('materialized_rows')}")
    print(f"selected_strata                    = {status.get('selected_strata')}")
    print(f"selected_conjunctive_strata        = {status.get('selected_conjunctive_strata')}")
    print(f"selected_depths                    = {status.get('selected_depths')}")
    print(f"scored_extension_candidates        = {status.get('scored_extension_candidates')}")
    print(f"eligible_extension_candidates      = {status.get('eligible_extension_candidates')}")
    print(f"conditional_extension_candidates   = {status.get('conditional_extension_candidates')}")
    print(f"extension_candidates               = {status.get('extension_candidates')}")
    print(f"conjunctive_candidates             = {status.get('conjunctive_candidates')}")
    print(f"required_columns_covered           = {status.get('required_columns_covered')}")
    print(f"required_columns_uncovered         = {status.get('required_columns_uncovered')}")
    print(f"columns_skipped_by_domain_cap      = {status.get('columns_skipped_by_domain_cap')}")
    print(f"columns_excluded_by_name           = {status.get('columns_excluded_by_name')}")
    print(f"candidate_columns                  = {status.get('candidate_columns')}")
    print(f"minimum_stratum_rows               = {status.get('minimum_stratum_rows')}")
    # A floor only applies to materialized rows, so print whether it did.  A
    # configured floor that silently did nothing otherwise reads exactly like
    # a floor that found nothing to prune.
    print(f"row_floor_applied                  = {status.get('row_floor_applied')}")
    print(f"candidates_below_row_floor         = {status.get('candidates_below_row_floor')}")
    print(f"selected_stratum_rows              = {status.get('selected_stratum_rows')}")
    print(f"reason                             = {status.get('reason')}")
    for name, entry in (status.get("required_column_diagnosis") or {}).items():
        print(f"\ndiagnosis for {name}:")
        for key, value in sorted(entry.items()):
            if key == "explanation":
                continue
            print(f"  {key:34s} {value}")
        print(f"  {'explanation':34s} {entry.get('explanation')}")

    columns_by_depth: dict[int, list[str]] = {}
    for stratum in report["selected_strata"]:
        columns_by_depth.setdefault(stratum["depth"], []).extend(stratum["columns"])
    print("\ncolumns appearing in selected strata, by depth:")
    for depth in sorted(columns_by_depth):
        counts: dict[str, int] = {}
        for name in columns_by_depth[depth]:
            counts[name] = counts.get(name, 0) + 1
        print(f"  depth {depth}: {dict(sorted(counts.items(), key=lambda kv: -kv[1]))}")

    output_path = Path(
        args.output
        or Path(config.get("logging", {}).get("output_directory", "model/runs/resmade"))
        / "rare_support_discovery.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(f"\nreport={output_path}")


if __name__ == "__main__":
    main()
