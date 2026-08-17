from __future__ import annotations

import argparse
import json
import traceback
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

from model.src.config import load_simple_yaml, validate_config
from model.src.data.sample_sources import sample_source_from_config


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Write timestamped flags for live NeuroCard/Ray sampler startup."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir")
    parser.add_argument(
        "--draw-rows",
        type=int,
        default=0,
        help="Optionally draw this many rows after source construction.",
    )
    parser.add_argument("--allow-non-live", action="store_true")
    args = parser.parse_args()

    started = perf_counter()
    output_dir = Path(args.output_dir or "model/runs/live_sampler_startup_check")
    _write_flag(
        output_dir,
        "00_script_started",
        {
            "config": args.config,
            "draw_rows": args.draw_rows,
        },
        started,
    )

    try:
        config = load_simple_yaml(args.config)
        validate_config(config)
        dataset = config.get("dataset", {})
        logging = config.get("logging", {})
        if args.output_dir is None:
            output_dir = Path(logging.get("output_directory", "model/runs/resmade")) / "startup_check"
        sampling_mode = str(dataset.get("sampling_mode", "fixture"))
        if sampling_mode != "live" and not args.allow_non_live:
            raise ValueError(
                "live sampler startup check expects dataset.sampling_mode=live; "
                "pass --allow-non-live to override"
            )
        _write_flag(
            output_dir,
            "01_config_loaded",
            {
                "dataset_name": dataset.get("name"),
                "sampling_mode": sampling_mode,
                "prepared_directory": dataset.get("prepared_directory"),
                "csv_directory": dataset.get("csv_directory"),
                "neurocard_path": dataset.get("neurocard_path"),
                "sampler_batch_size": dataset.get("sampler_batch_size"),
                "training_batch_size": config.get("training", {}).get("batch_size"),
            },
            started,
        )

        source_started = perf_counter()
        source = sample_source_from_config(
            config,
            startup_callback=lambda name, payload: _write_flag(
                output_dir,
                f"02_{name}",
                payload,
                started,
            ),
        )
        _write_flag(
            output_dir,
            "03_live_source_constructed",
            {
                "source_class": type(source).__name__,
                "source_construction_seconds": perf_counter() - source_started,
                "join_cardinality": getattr(source, "join_cardinality", None),
                "column_count": len(getattr(getattr(source, "metadata", None), "columns", ())),
                "sampler_run_calls": getattr(source, "sampler_run_calls", None),
            },
            started,
        )

        if args.draw_rows > 0:
            draw_started = perf_counter()
            batch = source.batches(args.draw_rows, seed=int(config.get("training", {}).get("seed", 0)))
            _write_flag(
                output_dir,
                "04_first_batch_drawn",
                {
                    "draw_requested_rows": args.draw_rows,
                    "encoded_shape": list(batch.encoded_values.shape),
                    "fresh_rows_drawn": batch.fresh_rows_drawn,
                    "fixture_rows_reused": batch.fixture_rows_reused,
                    "draw_seconds": perf_counter() - draw_started,
                    "sampler_run_calls": getattr(source, "sampler_run_calls", None),
                },
                started,
            )

        _write_flag(output_dir, "99_startup_check_complete", {}, started)
    except Exception as exc:
        _write_flag(
            output_dir,
            "98_startup_check_failed",
            {
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            },
            started,
        )
        raise


def _write_flag(
    output_dir: Path,
    name: str,
    payload: dict[str, Any],
    started: float,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{name}.json"
    data = {
        "flag": name,
        "utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": perf_counter() - started,
        **payload,
    }
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)
    print(f"startup_flag[{name}]={path}", flush=True)
    return path


if __name__ == "__main__":
    main()
