#!/usr/bin/env python3
"""Record the GPU each own-model training run used, from its Slurm log.

The rare-support-off main runs (model/slurm/job_light_main_rare_off.sbatch) are not
bound to one GPU type, so the training table names the GPU per seed. Each training log
prints ``output_directory=<run>`` (the sbatch script) and ``cuda_device=<name>``
(train_resmade under runtime.require_cuda); a run is attributed to the log that
names it and finished (``end_utc=``).

    python own_model_training_hardware.py --logs '<glob>' --run-directory <run> ... --output <json>
"""

from __future__ import annotations

import argparse
import glob
import json
import re
from pathlib import Path

_OUTPUT = re.compile(r"^output_directory=(.+?)\s*$", re.MULTILINE)
_DEVICE = re.compile(r"cuda_device=(.+?)\s*$", re.MULTILINE)


def read_log(path: Path) -> dict | None:
    text = path.read_text(encoding="utf-8", errors="replace")
    outputs = _OUTPUT.findall(text)
    if not outputs:
        return None
    devices = sorted(set(_DEVICE.findall(text)))
    return {"run_directory": outputs[0], "devices": devices, "finished": "end_utc=" in text, "log": str(path)}


def hardware(run_directories: list[str], logs: list[Path]) -> dict[str, dict]:
    records = [record for record in map(read_log, logs) if record is not None]
    result = {}
    for run in run_directories:
        finished = [r for r in records if r["run_directory"].rstrip("/") == run.rstrip("/") and r["finished"]]
        devices = sorted({device for record in finished for device in record["devices"]})
        if len(finished) != 1 or len(devices) != 1:
            raise SystemExit(f"{run}: expected one finished training log naming one GPU, found "
                             f"{[(r['log'], r['devices']) for r in finished]}")
        result[run] = {"gpu": devices[0], "log": finished[0]["log"]}
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--logs", required=True, help="glob of the training Slurm logs")
    parser.add_argument("--run-directory", action="append", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    result = hardware(args.run_directory, [Path(path) for path in sorted(glob.glob(args.logs))])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    for run, record in result.items():
        print(f"{run}: {record['gpu']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
