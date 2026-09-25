from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("pg_cluster_lock", ROOT / "scripts" / "pg_cluster_lock.py")
LOCK = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(LOCK)


def _dead_pid() -> int:
    process = subprocess.Popen([sys.executable, "-c", "pass"])
    process.wait()
    return process.pid


class ClusterLockTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.pgdata = self.root / "pgdata"
        self.pgdata.mkdir()
        self.environment = mock.patch.dict(os.environ, {}, clear=False)
        self.environment.start()
        for name in (LOCK.TOKEN_ENV, LOCK.ASSUME_STALE_ENV, "SLURM_JOB_ID"):
            os.environ.pop(name, None)

    def tearDown(self):
        self.environment.stop()
        self.temporary.cleanup()

    def _squeue(self, output: str, returncode: int = 0, stderr: str = "") -> str:
        script = self.root / "squeue"
        script.write_text(f"#!/bin/sh\nprintf '%s' '{output}'\nprintf '%s' '{stderr}' >&2\nexit {returncode}\n")
        script.chmod(0o755)
        return str(script)

    def _foreign_lock(self, **owner) -> Path:
        directory = Path(LOCK.lock_directory(self.pgdata))
        directory.mkdir()
        (directory / LOCK.OWNER_FILE).write_text(json.dumps({"token": "foreign", **owner}))
        return directory

    def test_exclusive_acquire_and_release(self):
        info = LOCK.acquire(self.pgdata, holder_pid=os.getpid(), label="first")
        self.assertEqual(info["mode"], "owner")
        self.assertTrue(Path(info["lock"]).name.startswith(".pgdata.joblight-lock"))
        with self.assertRaises(LOCK.LockHeld) as caught:
            LOCK.acquire(self.pgdata, holder_pid=os.getpid(), label="second")
        self.assertEqual(caught.exception.exit_code, 3)
        with self.assertRaises(LOCK.LockError):
            LOCK.release(self.pgdata, "wrong-token")
        LOCK.release(self.pgdata, info["token"])
        self.assertFalse(Path(info["lock"]).exists())

    def test_inherited_token_reenters_without_releasing(self):
        with LOCK.held(self.pgdata, label="parent") as parent:
            self.assertEqual(os.environ[LOCK.TOKEN_ENV], parent["token"])
            with LOCK.held(self.pgdata, label="child") as child:
                self.assertEqual(child["mode"], "reentrant")
            self.assertTrue(Path(parent["lock"]).exists())
        self.assertFalse(Path(parent["lock"]).exists())
        self.assertNotIn(LOCK.TOKEN_ENV, os.environ)

    def test_dead_holder_on_this_host_is_broken(self):
        self._foreign_lock(host=LOCK.hostname(), holder_pid=_dead_pid())
        info = LOCK.acquire(self.pgdata, holder_pid=os.getpid())
        self.assertEqual(info["mode"], "owner")
        self.assertIn("no longer exists", info["owner"]["broke_stale_lock"]["reason"])

    def test_holder_on_another_node_is_judged_by_its_slurm_job(self):
        self._foreign_lock(host="other-node", holder_pid=1, slurm_job_id="42")
        with self.assertRaisesRegex(LOCK.LockHeld, "still active"):
            LOCK.acquire(self.pgdata, holder_pid=os.getpid(), squeue=self._squeue("RUNNING"))
        with self.assertRaisesRegex(LOCK.LockHeld, "cannot query"):
            LOCK.acquire(self.pgdata, holder_pid=os.getpid(), squeue=str(self.root / "missing"))
        info = LOCK.acquire(self.pgdata, holder_pid=os.getpid(),
                            squeue=self._squeue("", 1, "slurm_load_jobs error: Invalid job id specified"))
        self.assertIn("has ended", info["owner"]["broke_stale_lock"]["reason"])

    def test_holder_on_another_node_without_job_is_never_assumed_dead(self):
        self._foreign_lock(host="other-node", holder_pid=1)
        with self.assertRaisesRegex(LOCK.LockHeld, "liveness cannot be checked"):
            LOCK.acquire(self.pgdata, holder_pid=os.getpid())

    def test_unexplained_postmaster_pid_is_refused(self):
        (self.pgdata / "postmaster.pid").write_text(f"{_dead_pid()}\n{self.pgdata}\n")
        with self.assertRaises(LOCK.OrphanedServer) as caught:
            LOCK.acquire(self.pgdata, holder_pid=os.getpid())
        self.assertEqual(caught.exception.exit_code, 4)
        self.assertFalse(Path(LOCK.lock_directory(self.pgdata)).exists())
        info = LOCK.acquire(self.pgdata, holder_pid=os.getpid(), assume_stale=True)
        self.assertIn("assumed stale", info["owner"]["postmaster_note"])

    def test_postmaster_pid_of_a_dead_holder_is_accepted(self):
        (self.pgdata / "postmaster.pid").write_text(f"{_dead_pid()}\n")
        self._foreign_lock(host="other-node", holder_pid=1, slurm_job_id="42")
        info = LOCK.acquire(self.pgdata, holder_pid=os.getpid(), squeue=self._squeue(""))
        self.assertIn("dead lock holder", info["owner"]["postmaster_note"])

    def test_local_server_without_lock_is_refused(self):
        (self.pgdata / "postmaster.pid").write_text(f"{os.getpid()}\n")
        with mock.patch.object(LOCK, "_is_postgres", return_value=True):
            with self.assertRaisesRegex(LOCK.OrphanedServer, "without holding the lock"):
                LOCK.acquire(self.pgdata, holder_pid=os.getpid(), assume_stale=True)

    def test_command_line_exit_codes(self):
        script = str(ROOT / "scripts" / "pg_cluster_lock.py")
        first = subprocess.run([sys.executable, script, "acquire", "--pgdata", str(self.pgdata),
                                "--holder-pid", str(os.getpid())], capture_output=True, text=True)
        self.assertEqual(first.returncode, 0, first.stderr)
        token, mode = first.stdout.split()
        self.assertEqual(mode, "owner")
        second = subprocess.run([sys.executable, script, "acquire", "--pgdata", str(self.pgdata),
                                 "--holder-pid", str(os.getpid())], capture_output=True, text=True)
        self.assertEqual(second.returncode, 3)
        status = json.loads(subprocess.run([sys.executable, script, "status", "--pgdata", str(self.pgdata)],
                                           capture_output=True, text=True, check=True).stdout)
        self.assertEqual(status["holder_state"], "alive")
        subprocess.run([sys.executable, script, "release", "--pgdata", str(self.pgdata), "--token", token],
                       check=True)


FAKE_PG_CTL = textwrap.dedent("""\
    #!/bin/bash
    # records calls; "running" state is a marker file next to PGDATA
    echo "$*" >> "$FAKE_PG_LOG"
    marker="$FAKE_PG_STATE"
    for argument in "$@"; do mode=$argument; done
    case "$mode" in
      start) touch "$marker"; exit 0 ;;
      stop) rm -f "$marker"; exit 0 ;;
      status) [[ -e "$marker" ]] && exit 0 || exit 3 ;;
    esac
""")


@unittest.skipUnless(shutil.which("bash"), "needs bash")
class ShellHelperTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.bin = self.root / "bin"
        self.bin.mkdir()
        (self.bin / "pg_ctl").write_text(FAKE_PG_CTL)
        (self.bin / "pg_isready").write_text("#!/bin/sh\nexit 0\n")
        for name in ("pg_ctl", "pg_isready"):
            (self.bin / name).chmod(0o755)
        (self.root / "pgdata").mkdir()

    def tearDown(self):
        self.temporary.cleanup()

    def _run(self, body: str) -> subprocess.CompletedProcess:
        script = textwrap.dedent(f"""\
            set -euo pipefail
            REPO={ROOT.parents[1]}
            PG_BIN={self.bin}
            PGDATA={self.root / 'pgdata'}
            SOCKET={self.root}
            PORT=1
            PG_LOCK_PYTHON={sys.executable}
            source {ROOT / 'slurm' / '_pg_lock.sh'}
            """) + textwrap.dedent(body)
        environment = {key: value for key, value in os.environ.items()
                       if key not in (LOCK.TOKEN_ENV, "SLURM_JOB_ID")}
        environment.update(FAKE_PG_LOG=str(self.root / "calls.log"), FAKE_PG_STATE=str(self.root / "running"))
        return subprocess.run(["bash", "-c", script], capture_output=True, text=True, env=environment)

    def test_start_and_stop_hold_the_lock(self):
        lock = Path(LOCK.lock_directory(self.root / "pgdata"))
        result = self._run(f"""\
            trap stop_postgres_locked EXIT
            start_postgres_locked
            test -d {lock}
            test "$STARTED_POSTGRES" -eq 1
            """)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse(lock.exists())
        calls = (self.root / "calls.log").read_text()
        self.assertIn("start", calls)
        self.assertIn("-m fast -w -t 900 stop", calls)

    def test_start_is_refused_while_another_job_holds_the_lock(self):
        holder = LOCK.acquire(self.root / "pgdata", holder_pid=os.getpid(), label="other job")
        try:
            result = self._run("""\
                trap stop_postgres_locked EXIT
                start_postgres_locked
                """)
        finally:
            LOCK.release(self.root / "pgdata", holder["token"])
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("locked by other job", result.stderr)
        self.assertFalse((self.root / "calls.log").exists())


if __name__ == "__main__":
    unittest.main()
