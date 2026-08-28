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
    parser.add_argument(
        "--manual-constructor-debug",
        action="store_true",
        help="Run NeuroCard FactorizedSampler constructor steps manually with flags.",
    )
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

        if args.manual_constructor_debug:
            _manual_constructor_debug(config, output_dir, started)
            _write_flag(output_dir, "99_startup_check_complete", {}, started)
            return

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


def _manual_constructor_debug(
    config: dict[str, Any],
    output_dir: Path,
    started: float,
) -> None:
    import sys
    import numpy as np

    from model.src.data.full_join_sampler import _pushd, _resolve_neurocard_package

    dataset = config["dataset"]
    neurocard_path = _resolve_neurocard_package(dataset.get("neurocard_path"))
    csv_directory = Path(dataset["csv_directory"]).resolve()
    sampler_batch_size = int(dataset.get("sampler_batch_size", dataset.get("sample_batch_size", 16384)))
    seed = int(dataset.get("sampler_seed", config.get("training", {}).get("seed", 0)))
    if str(neurocard_path) not in sys.path:
        sys.path.insert(0, str(neurocard_path))

    with _pushd(neurocard_path.parent):
        _write_flag(
            output_dir,
            "10_manual_imports_started",
            {
                "neurocard_path": str(neurocard_path),
                "neurocard_workdir": str(neurocard_path.parent),
            },
            started,
        )
        import datasets  # type: ignore
        import experiments  # type: ignore
        import factorized_sampler  # type: ignore
        import join_utils  # type: ignore
        from factorized_sampler_lib import prepare_utils  # type: ignore

        _write_flag(output_dir, "11_manual_imports_loaded", {}, started)
        cfg = experiments.JOB_LIGHT_BASE
        spec = join_utils.get_join_spec(cfg)
        _write_flag(
            output_dir,
            "12_manual_join_spec_loaded",
            {
                "join_name": getattr(spec, "join_name", None),
                "join_root": getattr(spec, "join_root", None),
                "join_tables": list(getattr(spec, "join_tables", ())),
            },
            started,
        )
        loaded_tables = []
        for table in spec.join_tables:
            _write_flag(output_dir, f"13_load_table_started_{table}", {}, started)
            loaded = datasets.LoadImdb(
                table,
                data_dir=str(csv_directory) + "/",
                use_cols=cfg["use_cols"],
                try_load_parsed=True,
            )
            loaded_tables.append(loaded)
            _write_flag(
                output_dir,
                f"14_load_table_done_{table}",
                {
                    "columns": list(loaded.data.columns),
                    "rows": int(len(loaded.data)),
                },
                started,
            )

        _write_flag(output_dir, "15_prepare_started", {}, started)
        if prepare_utils.check_required_files(spec):
            _write_flag(
                output_dir,
                "15_prepare_cache_hit",
                {
                    "join_name": getattr(spec, "join_name", None),
                },
                started,
            )
        else:
            prepare_utils.prepare(spec)
        _write_flag(output_dir, "16_prepare_done", {}, started)

        dt_actors = []
        for table in loaded_tables:
            _write_flag(output_dir, f"17_data_actor_started_{table.name}", {}, started)
            actor = factorized_sampler.DataTableActor(
                table.name,
                spec.join_keys[table.name],
                table.data,
                spec.join_name,
            )
            dt_actors.append(actor)
            _write_flag(output_dir, f"18_data_actor_done_{table.name}", {}, started)

        jcts = {}
        for table in spec.join_tables:
            _write_flag(output_dir, f"19_load_jct_started_{table}", {}, started)
            jct = factorized_sampler.load_jct(table, spec.join_name)
            jcts[table] = jct
            _write_flag(
                output_dir,
                f"20_load_jct_done_{table}",
                {
                    "rows": int(len(jct)),
                    "columns": list(jct.columns),
                },
                started,
            )

        jct_actors = {}
        for table, jct in jcts.items():
            _write_flag(output_dir, f"21_jct_actor_started_{table}", {}, started)
            actor = factorized_sampler.JoinCountTableActor(table, jct, spec)
            jct_actors[table] = actor
            _write_flag(output_dir, f"22_jct_actor_done_{table}", {}, started)

        ordering = factorized_sampler._make_sampling_table_ordering(loaded_tables, spec.join_root)
        root = spec.join_root
        join_card = jct_actors[root].jct[f"{root}.weight"].sum()
        _write_flag(
            output_dir,
            "23_manual_constructor_equivalent_done",
            {
                "data_actor_count": len(dt_actors),
                "jct_actor_count": len(jct_actors),
                "sampling_tables_ordering": list(ordering),
                "join_cardinality": int(join_card),
                "sampler_batch_size": sampler_batch_size,
                "seed": seed,
            },
            started,
        )
        _ = np.random.default_rng(seed)


if __name__ == "__main__":
    main()
