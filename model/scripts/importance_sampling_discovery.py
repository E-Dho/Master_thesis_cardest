from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from model.src.config import load_simple_yaml, validate_config
from model.src.data.importance_sampling import ImportanceSamplingSampleSource
from model.src.data.sample_sources import sample_source_from_config


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build and audit an importance-sampling proposal without training."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--print-json", action="store_true")
    args = parser.parse_args()

    config = load_simple_yaml(args.config)
    validate_config(config)
    source = sample_source_from_config(config)
    if not isinstance(source, ImportanceSamplingSampleSource):
        raise SystemExit("config did not construct an enabled ImportanceSamplingSampleSource")

    summary = source.importance_sampling_summary(actual_optimizer_steps=0)
    report = _discovery_report(summary)
    output_path = Path(
        args.output
        or Path(config.get("logging", {}).get("output_directory", "model/runs/resmade"))
        / "importance_sampling_discovery.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, allow_nan=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    print(f"importance_sampling_discovery={output_path}")
    print(f"selected_strata={report['selected_count']}")
    print(f"alpha_sum={report['alpha_sum']:.12f}")
    print(f"min_probability={report['min_probability']}")
    print(f"max_probability={report['max_probability']}")
    if args.print_json:
        print(json.dumps(report, allow_nan=False, sort_keys=True))


def _discovery_report(summary: dict[str, Any]) -> dict[str, Any]:
    selected = list(summary.get("selected_strata", ()))
    alpha_sum = float(sum(float(stratum.get("alpha", 0.0)) for stratum in selected))
    probabilities = [float(stratum.get("probability", 0.0)) for stratum in selected]
    return {
        "support_planning_steps": summary.get("support_planning_steps"),
        "planned_full_join_sample_count": summary.get("planned_full_join_sample_count"),
        "selected_count": len(selected),
        "alpha_sum": alpha_sum,
        "min_probability": min(probabilities) if probabilities else None,
        "max_probability": max(probabilities) if probabilities else None,
        "proposal_composition": summary.get("proposal_composition", {}),
        "root_mass_diagnostics": summary.get("root_mass_diagnostics", {}),
        "configuration": summary.get("configuration", {}),
        "selected_strata": [
            {
                "rank": rank,
                "stratum_id": stratum.get("stratum_id"),
                "column": stratum.get("column_name"),
                "region_type": stratum.get("region_type"),
                "semantic_type": stratum.get("semantic_type"),
                "value": stratum.get("value"),
                "lower": stratum.get("lower"),
                "upper": stratum.get("upper"),
                "foj_count": stratum.get("foj_count"),
                "P_s": stratum.get("probability"),
                "expected_target_rows": stratum.get("expected_target_rows"),
                "expected_equality_count": stratum.get("expected_equality_count"),
                "expected_lower_count": stratum.get("expected_lower_count"),
                "expected_upper_count": stratum.get("expected_upper_count"),
                "expected_range_support": stratum.get("expected_range_support"),
                "support_score": stratum.get("support_score"),
                "support_deficit": stratum.get("support_deficit"),
                "support_bottleneck": stratum.get("support_bottleneck"),
                "alpha_s": stratum.get("alpha"),
            }
            for rank, stratum in enumerate(selected, start=1)
        ],
    }


if __name__ == "__main__":
    main()
