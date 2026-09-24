#!/usr/bin/env python3
"""Run a plan of timing-only stages under one profile inside a pinned step.

Launched by the ``slurm/timing_*.sbatch`` scripts through
``srun --cpu-bind=cores``.  It checks the CPU set the step received, splits it
into physical cores, and runs ``cli time`` for every plan entry sequentially
(one query at a time inside each run), so all runs of a plan share the same
node and cores.  For ``cpu_postgres_2core`` it starts the PostgreSQL server
pinned to the second core (autovacuum off; planner settings unchanged, so
estimates stay identical to the accuracy run) and pins the client to the
first.  Plan format, one run per line (``#`` starts a comment)::

    CONFIG_PATH  SEED  RUN_DIRECTORY  [extra cli time arguments ...]
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import _timing  # noqa: E402


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--profile", required=True)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--python", default=sys.executable,
                        help="interpreter of the evaluation CLI")
    parser.add_argument("--report", type=Path)
    parser.add_argument("--allow-config-drift", action="append", default=[])
    parser.add_argument("--allow-estimate-drift", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--pg-bin", type=Path)
    parser.add_argument("--pgdata", type=Path)
    parser.add_argument("--socket-dir", type=Path)
    parser.add_argument("--port", type=int, default=55432)
    parser.add_argument("--database", default="imdb_joblight")
    parser.add_argument("--pg-option", action="append", default=[],
                        help="extra postgres -c option, e.g. log_min_messages=warning")
    parser.add_argument("--pg-log", type=Path,
                        help="server log (default: next to the launch report)")
    args = parser.parse_args(argv)
    # Slurm sends SIGTERM at the time limit; turn it into SystemExit so the
    # pinned PostgreSQL server is stopped and the launch report is written.
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))

    guard = _timing.load_timing_guard()
    profile = guard.get_profile(args.profile)
    cpus = guard.allowed_cpus()
    cores = guard.split_physical_cores(cpus)
    debug_unpinned = os.environ.get(guard.UNPINNED_DEBUG_ENV) == "1"
    client_cpus, server_cpus = _assign_cores(profile, cpus, cores, debug_unpinned)
    entries = _read_plan(args.plan)
    report_path = args.report or args.plan.with_name(
        f"{args.plan.stem}_{args.profile}_{os.environ.get('SLURM_JOB_ID', 'local')}.launch.json"
    )
    report: Dict[str, Any] = {
        "profile": args.profile,
        "plan": str(args.plan.resolve()),
        "step_cpus": guard.format_cpu_list(cpus),
        "physical_cores": cores,
        "client_cpus": guard.format_cpu_list(client_cpus),
        "server_cpus": guard.format_cpu_list(server_cpus) if server_cpus else None,
        "debug_unpinned": debug_unpinned,
        "snapshot": guard.environment_snapshot(),
        "entries": [],
    }
    environment = dict(os.environ)
    environment.update(guard.profile_environment(args.profile, cpus=client_cpus))
    server = None
    exit_code = 0
    try:
        if profile.server_physical_cores:
            if args.pg_log is None:
                args.pg_log = report_path.with_suffix(".postgres.log")
            server = _PinnedPostgres(args, server_cpus)
            if not args.dry_run:
                server.start()
                report["postgres"] = server.describe()
        for entry in entries:
            command = [
                args.python, "-m", "evaluation.job_light_imdb_non_trajectory.joblight_eval.cli",
                "time", "--config", entry["config"], "--seed", entry["seed"],
                "--run-directory", entry["run_directory"], "--profile", args.profile,
                *[item for path in args.allow_config_drift for item in ("--allow-config-drift", path)],
                *(["--allow-estimate-drift"] if args.allow_estimate_drift else []),
                *entry["extra"],
            ]
            result: Dict[str, Any] = {"command": command, **entry}
            if args.dry_run:
                result["status"] = "dry_run"
                print(" ".join(shlex.quote(part) for part in command))
            else:
                started = time.perf_counter()
                completed = subprocess.run(
                    command, env=environment, text=True, stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE, check=False,
                    preexec_fn=_affinity_setter(client_cpus),
                )
                result.update({
                    "returncode": completed.returncode,
                    "status": "complete" if completed.returncode == 0 else "failed",
                    "wall_seconds": time.perf_counter() - started,
                    "timing_directory": (completed.stdout.strip().splitlines() or [""])[-1],
                    "stderr_tail": completed.stderr[-4000:],
                })
                sys.stdout.write(completed.stdout)
                sys.stderr.write(completed.stderr)
                if completed.returncode:
                    exit_code = 1
            report["entries"].append(result)
    finally:
        if server is not None and server.started:
            server.stop()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + "\n",
                               encoding="utf-8")
        print(f"launch_report={report_path}")
    return exit_code


def _assign_cores(profile, cpus: List[int], cores: List[List[int]], debug_unpinned: bool):
    if profile.exclusive_node:
        return cpus, []
    required = int(profile.client_physical_cores) + int(profile.server_physical_cores)
    if len(cores) != required:
        message = (
            f"profile {profile.profile_id} needs exactly {required} physical core(s) in the "
            f"step, received {len(cores)} ({cores}); launch through srun --cpu-bind=cores "
            f"with --cpus-per-task and --hint=nomultithread as in slurm/timing_*.sbatch"
        )
        if not debug_unpinned or len(cores) < required:
            raise SystemExit(message)
        print(f"WARNING (debug, not reportable): {message}", file=sys.stderr)
    client = [cpu for group in cores[: profile.client_physical_cores] for cpu in group]
    server_groups = cores[profile.client_physical_cores: required]
    server = [cpu for group in server_groups for cpu in group]
    return client, server


def _affinity_setter(cpus: List[int]):
    def apply() -> None:
        os.sched_setaffinity(0, set(cpus))

    return apply


def _read_plan(path: Path) -> List[Dict[str, Any]]:
    entries = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        text = line.split("#", 1)[0].strip()
        if not text:
            continue
        parts = shlex.split(text)
        if len(parts) < 3:
            raise SystemExit(f"{path}:{number}: expected CONFIG SEED RUN_DIRECTORY")
        entries.append({"config": parts[0], "seed": parts[1], "run_directory": parts[2],
                        "extra": parts[3:], "plan_line": number})
    if not entries:
        raise SystemExit(f"{path} contains no timing entries")
    return entries


class _PinnedPostgres:
    """PostgreSQL started inside this step with its postmaster pinned to one core."""

    def __init__(self, args, cpus: List[int]) -> None:
        for name in ("pg_bin", "pgdata", "socket_dir"):
            if getattr(args, name) is None:
                raise SystemExit(f"--{name.replace('_', '-')} is required for {args.profile}")
        self.args = args
        self.cpus = cpus
        self.started = False

    def _ctl(self, *extra: str, pinned: bool = False, check: bool = True):
        # No pipes: the postmaster started by ``pg_ctl start`` would inherit and
        # hold them open, blocking this process forever.
        return subprocess.run(
            [str(self.args.pg_bin / "pg_ctl"), "-D", str(self.args.pgdata), *extra],
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            check=check, preexec_fn=_affinity_setter(self.cpus) if pinned else None,
        )

    def start(self) -> None:
        if self._ctl("status", check=False).returncode == 0:
            raise SystemExit(
                "PostgreSQL is already running (probably started by another job); its "
                "backends would not be pinned inside this allocation. Stop it first."
            )
        options = [f"-k {self.args.socket_dir}", f"-p {self.args.port}", "-c listen_addresses=''",
                   "-c autovacuum=off", *[f"-c {option}" for option in self.args.pg_option]]
        self.args.pg_log.parent.mkdir(parents=True, exist_ok=True)
        self._ctl("-o", " ".join(options), "-l", str(self.args.pg_log), "-w", "start", pinned=True)
        self.started = True
        for _ in range(120):
            ready = subprocess.run(
                [str(self.args.pg_bin / "pg_isready"), "-h", str(self.args.socket_dir),
                 "-p", str(self.args.port), "-d", self.args.database],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False,
            )
            if ready.returncode == 0:
                break
            time.sleep(1)
        else:
            raise SystemExit("PostgreSQL did not become ready")
        pid = self.postmaster_pid()
        observed = sorted(os.sched_getaffinity(pid))
        if observed != sorted(self.cpus):
            raise SystemExit(f"postmaster affinity {observed} differs from {self.cpus}")

    def postmaster_pid(self) -> int:
        return int((self.args.pgdata / "postmaster.pid").read_text().splitlines()[0])

    def describe(self) -> Dict[str, Any]:
        pid = self.postmaster_pid()
        return {"postmaster_pid": pid, "postmaster_cpus": sorted(os.sched_getaffinity(pid)),
                "options": ["autovacuum=off", *self.args.pg_option], "log": str(self.args.pg_log)}

    def stop(self) -> None:
        self._ctl("-m", "fast", "-w", "stop", check=False)
        self.started = False


if __name__ == "__main__":
    raise SystemExit(main())
