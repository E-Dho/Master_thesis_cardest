#!/usr/bin/env python3
"""Cluster-visible lock for the shared PostgreSQL data directory.

``PGDATA`` lives on BeeGFS and is visible from every node, but ``pg_ctl status``
and PostgreSQL's own ``postmaster.pid`` check only look for a *local* process:
a server running on another node looks stopped, and a second postmaster could
be started against the same data directory.  Every launcher in this
repository therefore holds this lock for the whole time it starts, uses, and
stops a server:

* the lock is the directory ``<parent>/.<pgdata name>.joblight-lock`` created
  with ``mkdir`` (atomic on BeeGFS/NFS) and an ``owner.json`` naming host,
  holder PID (+ start time), Slurm job, and a random token;
* a holder is stale only when provably dead: same host and PID gone, or its
  Slurm job no longer in ``squeue``; otherwise the lock is treated as held;
* after acquisition an existing ``postmaster.pid`` is accepted only if the
  previous holder was proven dead (its server died with it); an unexplained
  one, or a PostgreSQL process running here without the lock, is refused;
* processes started by a holder inherit ``JOBLIGHT_PG_LOCK_TOKEN`` and
  re-enter the lock without acquiring or releasing it.

Command line (used by ``slurm/_pg_lock.sh``)::

    pg_cluster_lock.py acquire --pgdata P --holder-pid PID [--wait-seconds N]
        -> prints "TOKEN owner|reentrant"; exit 3 held elsewhere, 4 orphaned server
    pg_cluster_lock.py release --pgdata P --token TOKEN
    pg_cluster_lock.py status  --pgdata P
    pg_cluster_lock.py break   --pgdata P [--force]

Standard library only; runs on Python >= 3.6 (system interpreters).
"""

import argparse
import contextlib
import json
import os
import shutil
import socket
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone


TOKEN_ENV = "JOBLIGHT_PG_LOCK_TOKEN"
WAIT_ENV = "JOBLIGHT_PG_LOCK_WAIT_SECONDS"
ASSUME_STALE_ENV = "JOBLIGHT_PG_ASSUME_STALE_POSTMASTER"
OWNER_FILE = "owner.json"
INCOMPLETE_GRACE_SECONDS = 120
FINISHED_SLURM_STATES = {
    "BOOT_FAIL", "CANCELLED", "COMPLETED", "DEADLINE", "FAILED", "NODE_FAIL",
    "OUT_OF_MEMORY", "PREEMPTED", "REVOKED", "TIMEOUT",
}


class LockError(RuntimeError):
    exit_code = 1


class LockHeld(LockError):
    exit_code = 3


class OrphanedServer(LockError):
    exit_code = 4


def hostname():
    return socket.gethostname()


def lock_directory(pgdata):
    path = os.path.realpath(str(pgdata))
    return os.path.join(os.path.dirname(path), "." + os.path.basename(path) + ".joblight-lock")


def process_start_time(pid):
    """Kernel start time of ``pid`` (guards against PID reuse), None if unknown."""
    try:
        with open("/proc/%d/stat" % int(pid)) as handle:
            text = handle.read()
    except (OSError, ValueError):
        return None
    fields = text.rsplit(")", 1)[-1].split()
    return fields[19] if len(fields) > 19 else None


def pid_alive(pid):
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except (OSError, ValueError, TypeError):
        return False
    return True


def slurm_job_alive(job_id, squeue="squeue"):
    """True/False from squeue, None if squeue is unavailable or ambiguous."""
    try:
        completed = subprocess.run(
            [squeue, "-h", "-j", str(job_id), "-o", "%T"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            universal_newlines=True, timeout=60, check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if completed.returncode:
        return False if "invalid job id" in completed.stderr.lower() else None
    states = completed.stdout.split()
    if not states:
        return False
    return any(state.upper() not in FINISHED_SLURM_STATES for state in states)


def read_owner(directory):
    try:
        with open(os.path.join(directory, OWNER_FILE), encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, ValueError):
        return None


def holder_state(owner, *, squeue="squeue"):
    """("alive" | "dead" | "unknown", reason) for the recorded lock holder."""
    if owner.get("host") == hostname():
        pid = owner.get("holder_pid")
        if not pid_alive(pid):
            return "dead", "holder pid %s no longer exists on %s" % (pid, owner.get("host"))
        recorded = owner.get("holder_start_time")
        observed = process_start_time(pid)
        if recorded and observed and recorded != observed:
            return "dead", "holder pid %s was reused by another process" % pid
        return "alive", "holder pid %s is running on this host" % pid
    job = owner.get("slurm_job_id")
    if job:
        alive = slurm_job_alive(job, squeue=squeue)
        if alive is False:
            return "dead", "Slurm job %s of the holder has ended" % job
        if alive:
            return "alive", "Slurm job %s of the holder is still active" % job
        return "unknown", "cannot query Slurm job %s from %s" % (job, hostname())
    return "unknown", "holder on %s has no Slurm job; liveness cannot be checked from %s" % (
        owner.get("host"), hostname())


def _describe(owner):
    if not owner:
        return "an incompletely written lock"
    return "%s (host %s, pid %s, Slurm job %s, since %s)" % (
        owner.get("label") or "holder", owner.get("host"), owner.get("holder_pid"),
        owner.get("slurm_job_id") or "-", owner.get("acquired_at_utc"))


def _remove(directory, expected_token=None):
    """Atomically move the lock aside and delete it; False if it changed meanwhile."""
    target = "%s.removed-%s" % (directory, uuid.uuid4().hex)
    try:
        os.rename(directory, target)
    except FileNotFoundError:
        return False
    if expected_token is not None:
        moved = read_owner(target) or {}
        if moved.get("token") != expected_token:
            # Someone replaced the stale lock between our check and the rename;
            # put the live lock back.
            try:
                os.rename(target, directory)
            except OSError:
                raise LockError("lock %s changed while breaking a stale holder; retry" % directory)
            return False
    shutil.rmtree(target, ignore_errors=True)
    return True


def _is_postgres(pid):
    try:
        with open("/proc/%d/cmdline" % int(pid), "rb") as handle:
            return b"postgres" in handle.read()
    except OSError:
        return not os.path.isdir("/proc")  # without procfs, assume it is


def check_postmaster(pgdata, previous_holder_dead, assume_stale=False):
    """Refuse an unexplained ``postmaster.pid``; return a note for the owner record."""
    path = os.path.join(str(pgdata), "postmaster.pid")
    if not os.path.exists(path):
        return None
    try:
        with open(path) as handle:
            pid = int(handle.readline().strip())
    except (OSError, ValueError):
        pid = None
    if pid is not None and pid_alive(pid) and _is_postgres(pid):
        raise OrphanedServer(
            "PostgreSQL (pid %s) is running on %s against %s without holding the lock; "
            "stop it (pg_ctl -D %s stop) before using a locked launcher" % (pid, hostname(), pgdata, pgdata))
    if previous_holder_dead:
        return "stale postmaster.pid (pid %s) left by a dead lock holder" % pid
    if assume_stale:
        return "postmaster.pid (pid %s) assumed stale via %s=1" % (pid, ASSUME_STALE_ENV)
    raise OrphanedServer(
        "%s exists (pid %s) but no live lock holder explains it: a server may be running on "
        "another node without the lock. Verify that no job uses this PGDATA (squeue -u $USER), "
        "then rerun with %s=1" % (path, pid, ASSUME_STALE_ENV))


def acquire(pgdata, holder_pid=None, label="", wait_seconds=0.0, poll_seconds=15.0,
            assume_stale=None, squeue="squeue"):
    """Acquire (or re-enter) the lock; returns {"token", "mode", "owner", ...}."""
    directory = lock_directory(pgdata)
    holder_pid = int(holder_pid or os.getpid())
    if assume_stale is None:
        assume_stale = os.environ.get(ASSUME_STALE_ENV) == "1"
    inherited = os.environ.get(TOKEN_ENV)
    deadline = time.time() + float(wait_seconds)
    broken = None
    while True:
        owner = read_owner(directory)
        if owner and inherited and owner.get("token") == inherited:
            return {"token": inherited, "mode": "reentrant", "owner": owner, "lock": directory}
        try:
            os.mkdir(directory)
        except FileExistsError:
            owner = read_owner(directory)
            if owner is None:
                try:
                    age = time.time() - os.stat(directory).st_mtime
                except FileNotFoundError:
                    continue
                if age > INCOMPLETE_GRACE_SECONDS:
                    _remove(directory)
                    broken = {"owner": None, "reason": "incomplete lock older than %ds" % age}
                    continue
                state, reason = "unknown", "lock is being written"
            else:
                state, reason = holder_state(owner, squeue=squeue)
                if state == "dead":
                    if _remove(directory, expected_token=owner.get("token")):
                        broken = {"owner": owner, "reason": reason}
                    continue
            broken = None  # a live holder ran after any stale lock we broke
            if time.time() >= deadline:
                raise LockHeld("PostgreSQL data directory %s is locked by %s: %s. Wait for it, "
                               "or set %s to wait" % (pgdata, _describe(owner), reason, WAIT_ENV))
            time.sleep(max(0.1, min(poll_seconds, deadline - time.time())))
            continue
        token = uuid.uuid4().hex
        owner = {
            "schema_version": 1,
            "token": token,
            "pgdata": os.path.realpath(str(pgdata)),
            "host": hostname(),
            "holder_pid": holder_pid,
            "holder_start_time": process_start_time(holder_pid),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_step_id": os.environ.get("SLURM_STEP_ID"),
            "label": label,
            "acquired_at_utc": datetime.now(timezone.utc).isoformat(),
            "broke_stale_lock": broken,
        }
        temporary = os.path.join(directory, OWNER_FILE + ".tmp")
        with open(temporary, "w", encoding="utf-8") as handle:
            json.dump(owner, handle, indent=2, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, os.path.join(directory, OWNER_FILE))
        try:
            dead_previous = bool(broken and broken.get("owner"))
            owner["postmaster_note"] = check_postmaster(pgdata, dead_previous, assume_stale)
        except OrphanedServer:
            _remove(directory, expected_token=token)
            raise
        return {"token": token, "mode": "owner", "owner": owner, "lock": directory}


def release(pgdata, token):
    directory = lock_directory(pgdata)
    owner = read_owner(directory)
    if owner is None:
        raise LockError("no lock to release at %s" % directory)
    if owner.get("token") != token:
        raise LockError("lock %s is held by %s, not by this token" % (directory, _describe(owner)))
    _remove(directory, expected_token=token)


def status(pgdata, squeue="squeue"):
    directory = lock_directory(pgdata)
    owner = read_owner(directory)
    if owner is None:
        return {"lock": directory, "held": os.path.isdir(directory), "owner": None}
    state, reason = holder_state(owner, squeue=squeue)
    return {"lock": directory, "held": True, "owner": owner, "holder_state": state, "reason": reason}


def break_lock(pgdata, force=False, squeue="squeue"):
    info = status(pgdata, squeue=squeue)
    if not info["held"]:
        return info
    if info.get("holder_state") != "dead" and not force:
        raise LockHeld("holder is %s (%s); pass --force only after verifying it is gone" % (
            info.get("holder_state", "unknown"), info.get("reason", "incomplete lock")))
    _remove(info["lock"])
    info["broken"] = True
    return info


@contextlib.contextmanager
def held(pgdata, label="", wait_seconds=None, holder_pid=None):
    """Hold the lock for a ``with`` block (re-entrant under an inherited token)."""
    if wait_seconds is None:
        wait_seconds = float(os.environ.get(WAIT_ENV, "0") or 0)
    info = acquire(pgdata, holder_pid=holder_pid, label=label, wait_seconds=wait_seconds)
    previous = os.environ.get(TOKEN_ENV)
    os.environ[TOKEN_ENV] = info["token"]
    try:
        yield info
    finally:
        if info["mode"] == "owner":
            release(pgdata, info["token"])
        if previous is None:
            os.environ.pop(TOKEN_ENV, None)
        else:
            os.environ[TOKEN_ENV] = previous


def main(argv=None):
    parser = argparse.ArgumentParser(description="cluster-visible PGDATA lock")
    commands = parser.add_subparsers(dest="command")
    commands.required = True
    for name in ("acquire", "release", "status", "break"):
        command = commands.add_parser(name)
        command.add_argument("--pgdata", required=True)
    commands.choices["acquire"].add_argument("--holder-pid", type=int, required=True)
    commands.choices["acquire"].add_argument("--label", default="")
    commands.choices["acquire"].add_argument(
        "--wait-seconds", type=float, default=float(os.environ.get(WAIT_ENV, "0") or 0))
    commands.choices["release"].add_argument("--token", required=True)
    commands.choices["break"].add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.command == "acquire":
            info = acquire(args.pgdata, args.holder_pid, args.label, args.wait_seconds)
            print("%s %s" % (info["token"], info["mode"]))
            sys.stderr.write("pg lock %s: %s (%s)\n" % (
                info["mode"], info["lock"], info["owner"].get("postmaster_note") or "clean"))
        elif args.command == "release":
            release(args.pgdata, args.token)
        elif args.command == "status":
            print(json.dumps(status(args.pgdata), indent=2, sort_keys=True))
        else:
            print(json.dumps(break_lock(args.pgdata, args.force), indent=2, sort_keys=True))
    except LockError as exc:
        sys.stderr.write("pg lock: %s\n" % exc)
        return exc.exit_code
    return 0


if __name__ == "__main__":
    sys.exit(main())
