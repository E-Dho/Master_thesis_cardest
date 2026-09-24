"""Timing profiles and the shared thread/affinity/hardware guard.

Every latency that enters a report is measured under one named profile:

``cpu_1core``
    One physical core (its SMT sibling reserved, not used), every numerical
    library and PyTorch limited to one thread.  The primary latency table.
``cpu_postgres_2core``
    PostgreSQL only: one pinned core for the client, one for the backend.
``gpu_single_query``
    One GPU, pinned support CPUs, PyTorch/BLAS CPU threads limited to one.
``cpu_fullnode_exclusive``
    Optional scalability result on an exclusive node using all physical
    cores; never mixed into the one-core table.

The launcher (``scripts/timing_launch.py`` under ``srun --cpu-bind=cores``)
fixes the CPU set, the runner exports ``profile_environment`` before the
measuring process starts (so thread limits exist before numpy/torch load),
and the measuring process calls ``TimingSession`` to re-apply, verify, and
record everything.  Hard checks: thread-limit variables, threadpoolctl BLAS
pools, PyTorch intra-/inter-op threads, CPU affinity in physical cores, SMT
sibling ownership, CUDA visibility, and unexpected child processes.  The raw
OS thread count and the CPU-time/wall-time ratio are recorded and only warn.

This module is deliberately standalone (standard library only at import
time, Python >= 3.7) because it is loaded by file path from legacy method
environments such as DeepDB's Python 3.8 stack.
"""

from __future__ import annotations

import contextlib
import json
import os
import platform
import socket
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple


GUARD_VERSION = 1
PROFILE_ENV = "JOBLIGHT_TIMING_PROFILE"
REPORT_ENV = "JOBLIGHT_TIMING_ENVIRONMENT_JSON"
COMPUTE_THREADS_ENV = "JOBLIGHT_TIMING_COMPUTE_THREADS"
ALLOW_MISSING_THREADPOOLCTL_ENV = "JOBLIGHT_TIMING_ALLOW_MISSING_THREADPOOLCTL"
LATENCY_DEFINITION = (
    "per query, starting from the canonical already-parsed workload query: "
    "method-specific encoding, estimation, and conversion of the result to a "
    "Python float; workload file parsing, model loading, and result writing "
    "are excluded"
)
THREAD_LIMIT_VARIABLES = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "NUMEXPR_MAX_THREADS",
    "NUMBA_NUM_THREADS",
    "LOKY_MAX_CPU_COUNT",
)
FIXED_VARIABLES = {"MKL_DYNAMIC": "FALSE", "OMP_DYNAMIC": "FALSE"}
RECORDED_VARIABLES = THREAD_LIMIT_VARIABLES + tuple(FIXED_VARIABLES) + (
    "CUDA_VISIBLE_DEVICES", "OMP_PROC_BIND", "OMP_PLACES", "GOMP_CPU_AFFINITY",
    "KMP_AFFINITY", "TORCH_NUM_THREADS", "PYTHONHASHSEED", PROFILE_ENV, COMPUTE_THREADS_ENV,
)
SLURM_VARIABLES = (
    "SLURM_JOB_ID", "SLURM_STEP_ID", "SLURM_ARRAY_JOB_ID", "SLURM_ARRAY_TASK_ID",
    "SLURM_JOB_PARTITION", "SLURM_JOB_NODELIST", "SLURMD_NODENAME",
    "SLURM_CPUS_PER_TASK", "SLURM_CPUS_ON_NODE", "SLURM_JOB_CPUS_PER_NODE",
    "SLURM_CPU_BIND", "SLURM_CPU_BIND_TYPE", "SLURM_CPU_BIND_LIST",
    "SLURM_THREADS_PER_CORE", "SLURM_HINT", "SLURM_TRES_PER_TASK",
    "SLURM_JOB_GPUS", "SLURM_STEP_GPUS", "SLURM_GPUS_ON_NODE",
    "SLURM_JOB_CONSTRAINTS", "SLURM_EXCLUSIVE",
)
CPU_FLAGS_OF_INTEREST = (
    "sse4_2", "avx", "avx2", "fma", "avx512f", "avx512bw", "avx512vl",
    "avx512_vnni", "avx512_bf16", "amx_tile", "amx_bf16",
)
SYSFS_CPU = Path("/sys/devices/system/cpu")
PROC = Path("/proc")
CGROUP_ROOT = Path("/sys/fs/cgroup")


@dataclass(frozen=True)
class TimingProfile:
    profile_id: str
    device: str
    #: physical cores of the measuring (client) process; None: every core of the node
    client_physical_cores: Optional[int]
    #: BLAS/PyTorch intra-op threads; None: number of allocated physical cores
    compute_threads: Optional[int]
    server_physical_cores: int = 0
    exclusive_node: bool = False
    report_table: str = ""
    description: str = ""


PROFILES: Dict[str, TimingProfile] = {
    "cpu_1core": TimingProfile(
        "cpu_1core", "cpu", 1, 1,
        report_table="CPU latency, one physical core",
        description="one physical core, SMT sibling unused, all numerical libraries "
                    "and PyTorch limited to one thread",
    ),
    "cpu_postgres_2core": TimingProfile(
        "cpu_postgres_2core", "cpu", 1, 1, server_physical_cores=1,
        report_table="CPU latency, PostgreSQL client and backend on one core each",
        description="one pinned physical core for the client and one for the "
                    "PostgreSQL backend",
    ),
    "gpu_single_query": TimingProfile(
        "gpu_single_query", "cuda", 4, 1,
        report_table="GPU latency, one GPU with four pinned support cores",
        description="one GPU, four pinned support physical cores, PyTorch and BLAS "
                    "CPU threads limited to one",
    ),
    "cpu_fullnode_exclusive": TimingProfile(
        "cpu_fullnode_exclusive", "cpu", None, None, exclusive_node=True,
        report_table="Supplementary CPU scalability, exclusive full node",
        description="exclusive node, all physical cores, thread pools sized to the "
                    "physical core count; supplementary, never mixed with one-core results",
    ),
}


class TimingGuardError(RuntimeError):
    """A hard timing-protocol check failed."""


# ---------------------------------------------------------------------------
# topology helpers
# ---------------------------------------------------------------------------


def parse_cpu_list(text: str) -> List[int]:
    """Parse the kernel list format, e.g. ``0-3,8,10-11``."""
    cpus: List[int] = []
    for part in text.strip().split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            cpus.extend(range(int(start), int(end) + 1))
        else:
            cpus.append(int(part))
    return sorted(set(cpus))


def format_cpu_list(cpus: Sequence[int]) -> str:
    values = sorted(set(int(cpu) for cpu in cpus))
    ranges: List[str] = []
    start = previous = None
    for cpu in values:
        if start is None:
            start = previous = cpu
        elif cpu == previous + 1:
            previous = cpu
        else:
            ranges.append(str(start) if start == previous else f"{start}-{previous}")
            start = previous = cpu
    if start is not None:
        ranges.append(str(start) if start == previous else f"{start}-{previous}")
    return ",".join(ranges)


def _read(path: Path) -> Optional[str]:
    try:
        return path.read_text(encoding="utf-8", errors="replace").strip()
    except OSError:
        return None


def allowed_cpus(pid: int = 0) -> List[int]:
    if hasattr(os, "sched_getaffinity"):
        return sorted(os.sched_getaffinity(pid))
    return list(range(os.cpu_count() or 1))


def online_cpus(sysfs_cpu: Path = SYSFS_CPU) -> List[int]:
    text = _read(sysfs_cpu / "online")
    return parse_cpu_list(text) if text else list(range(os.cpu_count() or 1))


def core_key(cpu: int, sysfs_cpu: Path = SYSFS_CPU) -> Tuple[int, ...]:
    topology = sysfs_cpu / f"cpu{cpu}" / "topology"
    package = _read(topology / "physical_package_id")
    core = _read(topology / "core_id")
    if package is None or core is None:
        return (-1, cpu)
    return (int(package), int(core))


def thread_siblings(cpu: int, sysfs_cpu: Path = SYSFS_CPU) -> List[int]:
    topology = sysfs_cpu / f"cpu{cpu}" / "topology"
    text = _read(topology / "thread_siblings_list") or _read(topology / "core_cpus_list")
    return parse_cpu_list(text) if text else [cpu]


def physical_cores(cpus: Sequence[int], sysfs_cpu: Path = SYSFS_CPU) -> Dict[Tuple[int, ...], List[int]]:
    cores: Dict[Tuple[int, ...], List[int]] = {}
    for cpu in cpus:
        cores.setdefault(core_key(cpu, sysfs_cpu), []).append(int(cpu))
    return {key: sorted(value) for key, value in sorted(cores.items())}


def split_physical_cores(cpus: Sequence[int], sysfs_cpu: Path = SYSFS_CPU) -> List[List[int]]:
    """Logical CPUs grouped per physical core, ordered by first logical CPU."""
    return sorted(physical_cores(cpus, sysfs_cpu).values(), key=lambda group: group[0])


def cgroup_cpuset(proc: Path = PROC, cgroup_root: Path = CGROUP_ROOT) -> Dict[str, Any]:
    """CPU set and quota of the enclosing Slurm job cgroup (v2 or v1).

    The job-level cgroup (``job_<id>``) is preferred over step or task levels
    because SMT sibling ownership is decided at job allocation granularity.
    """
    text = _read(proc / "self" / "cgroup") or ""
    candidates: List[Tuple[str, Path, Path]] = []
    for line in text.splitlines():
        parts = line.split(":", 2)
        if len(parts) != 3:
            continue
        _, controllers, relative = parts
        relative = relative.lstrip("/")
        if controllers == "":  # cgroup v2
            base = cgroup_root / relative
            for directory in [base, *base.parents]:
                if cgroup_root not in (directory, *directory.parents):
                    break
                candidates.append(("v2", directory, directory / "cpuset.cpus.effective"))
        elif "cpuset" in controllers.split(","):
            base = cgroup_root / "cpuset" / relative
            for directory in [base, *base.parents]:
                if (cgroup_root / "cpuset") not in (directory, *directory.parents):
                    break
                effective = directory / "cpuset.effective_cpus"
                candidates.append(("v1", directory, effective if effective.exists() else directory / "cpuset.cpus"))
    readable = [(version, directory, path) for version, directory, path in candidates if _read(path)]
    if not readable:
        return {"available": False}
    job_level = [item for item in readable if item[1].name.startswith("job_")]
    version, directory, path = job_level[0] if job_level else readable[0]
    quota = _read(directory / "cpu.max") if version == "v2" else None
    return {
        "available": True,
        "cgroup_version": version,
        "path": str(directory),
        "level": "slurm_job" if job_level else "deepest_readable",
        "cpus": parse_cpu_list(_read(path) or ""),
        "cpu_max": quota,
    }


def cpu_model(proc: Path = PROC) -> Dict[str, Any]:
    text = _read(proc / "cpuinfo") or ""
    model = ""
    flags: List[str] = []
    for line in text.splitlines():
        key, _, value = line.partition(":")
        key = key.strip().lower()
        if key == "model name" and not model:
            model = value.strip()
        elif key == "flags" and not flags:
            flags = value.split()
    return {
        "model_name": model or platform.processor() or platform.machine(),
        "flags_of_interest": [flag for flag in CPU_FLAGS_OF_INTEREST if flag in flags],
    }


def node_summary(sysfs_cpu: Path = SYSFS_CPU) -> Dict[str, Any]:
    online = online_cpus(sysfs_cpu)
    governor = _read(sysfs_cpu / f"cpu{online[0]}" / "cpufreq" / "scaling_governor") if online else None
    return {
        "hostname": socket.gethostname(),
        "online_logical_cpus": len(online),
        "online_physical_cores": len(physical_cores(online, sysfs_cpu)),
        "smt_active": _read(sysfs_cpu / "smt" / "active"),
        "scaling_governor": governor,
        "kernel": platform.release(),
        "platform": platform.platform(),
    }


def slurm_environment(environ: Mapping[str, str] = os.environ) -> Dict[str, str]:
    return {name: environ[name] for name in SLURM_VARIABLES if name in environ}


def thread_names(pid: int = 0, proc: Path = PROC) -> Dict[str, int]:
    task_root = proc / ("self" if pid == 0 else str(pid)) / "task"
    names: Counter = Counter()
    try:
        tasks = list(task_root.iterdir())
    except OSError:
        return {}
    for task in tasks:
        names[_read(task / "comm") or "?"] += 1
    return dict(sorted(names.items()))


def child_processes(pid: Optional[int] = None, proc: Path = PROC) -> List[Dict[str, Any]]:
    """Direct and indirect children of ``pid`` (default: this process)."""
    root = os.getpid() if pid is None else pid
    parents: Dict[int, Tuple[int, str]] = {}
    try:
        entries = [entry for entry in proc.iterdir() if entry.name.isdigit()]
    except OSError:
        return []
    for entry in entries:
        stat = _read(entry / "stat")
        if not stat or ")" not in stat:
            continue
        name = stat[stat.find("(") + 1: stat.rfind(")")]
        fields = stat[stat.rfind(")") + 2:].split()
        if len(fields) > 1:
            parents[int(entry.name)] = (int(fields[1]), name)
    descendants: List[Dict[str, Any]] = []
    frontier = [root]
    while frontier:
        current = frontier.pop()
        for child, (parent, name) in parents.items():
            if parent == current:
                descendants.append({"pid": child, "ppid": parent, "name": name})
                frontier.append(child)
    return sorted(descendants, key=lambda item: item["pid"])


def threadpool_snapshot() -> Dict[str, Any]:
    try:
        import threadpoolctl
    except ImportError:
        return {"available": False, "pools": []}
    pools = []
    for info in threadpoolctl.threadpool_info():
        pools.append({
            key: info.get(key)
            for key in ("user_api", "internal_api", "prefix", "version", "num_threads",
                        "threading_layer", "architecture", "filepath")
        })
    return {"available": True, "version": getattr(threadpoolctl, "__version__", ""), "pools": pools}


def torch_snapshot(torch: Any = None) -> Dict[str, Any]:
    torch = torch if torch is not None else sys.modules.get("torch")
    if torch is None:
        return {"imported": False}
    snapshot: Dict[str, Any] = {
        "imported": True,
        "version": str(getattr(torch, "__version__", "")),
        "num_threads": int(torch.get_num_threads()),
        "num_interop_threads": int(torch.get_num_interop_threads()),
    }
    try:
        snapshot["cuda_available"] = bool(torch.cuda.is_available())
        snapshot["cuda_initialized"] = bool(torch.cuda.is_initialized())
        snapshot["cuda_version"] = str(torch.version.cuda)
        if torch.cuda.is_available():
            snapshot["cuda_device_count"] = int(torch.cuda.device_count())
            snapshot["cuda_device_names"] = [
                str(torch.cuda.get_device_name(index)) for index in range(torch.cuda.device_count())
            ]
        backends = getattr(torch, "backends", None)
        if backends is not None and hasattr(backends, "cudnn"):
            snapshot["cudnn_version"] = backends.cudnn.version()
    except Exception as exc:  # pragma: no cover - driver specific
        snapshot["cuda_error"] = f"{type(exc).__name__}: {exc}"
    return snapshot


def numpy_snapshot() -> Dict[str, Any]:
    numpy = sys.modules.get("numpy")
    if numpy is None:
        return {"imported": False}
    return {"imported": True, "version": str(numpy.__version__)}


# ---------------------------------------------------------------------------
# profile environment
# ---------------------------------------------------------------------------


def get_profile(profile_id: str) -> TimingProfile:
    try:
        return PROFILES[profile_id]
    except KeyError as exc:
        raise ValueError(f"unknown timing profile {profile_id!r}; known: {sorted(PROFILES)}") from exc


def resolved_compute_threads(profile: TimingProfile, cpus: Optional[Sequence[int]] = None,
                             sysfs_cpu: Path = SYSFS_CPU) -> int:
    if profile.compute_threads is not None:
        return int(profile.compute_threads)
    return max(1, len(physical_cores(cpus if cpus is not None else allowed_cpus(), sysfs_cpu)))


def profile_environment(profile_id: str, cpus: Optional[Sequence[int]] = None,
                        sysfs_cpu: Path = SYSFS_CPU) -> Dict[str, str]:
    """Environment the runner exports before the measuring process starts."""
    profile = get_profile(profile_id)
    threads = str(resolved_compute_threads(profile, cpus, sysfs_cpu))
    environment = {name: threads for name in THREAD_LIMIT_VARIABLES}
    environment.update(FIXED_VARIABLES)
    environment[PROFILE_ENV] = profile_id
    environment[COMPUTE_THREADS_ENV] = threads
    if profile.device == "cpu":
        environment["CUDA_VISIBLE_DEVICES"] = ""
    return environment


UNPINNED_DEBUG_ENV = "JOBLIGHT_TIMING_ALLOW_UNPINNED"


def launch_violations(profile_id: str, cpus: Optional[Sequence[int]] = None,
                      sysfs_cpu: Path = SYSFS_CPU) -> List[str]:
    """Allocation problems detectable before starting a measuring process."""
    profile = get_profile(profile_id)
    cpus = list(cpus) if cpus is not None else allowed_cpus()
    cores = physical_cores(cpus, sysfs_cpu)
    problems: List[str] = []
    if profile.exclusive_node:
        online = online_cpus(sysfs_cpu)
        if sorted(cpus) != online:
            problems.append(
                f"{profile_id} needs the whole node; affinity {format_cpu_list(cpus)} "
                f"differs from online CPUs {format_cpu_list(online)}"
            )
    elif len(cores) != profile.client_physical_cores:
        problems.append(
            f"{profile_id} needs {profile.client_physical_cores} physical core(s) for the "
            f"measuring process; affinity {format_cpu_list(cpus)} spans {len(cores)}"
        )
    return problems


def environment_snapshot(torch: Any = None) -> Dict[str, Any]:
    cpus = allowed_cpus()
    return {
        "guard_version": GUARD_VERSION,
        "python": sys.version.split()[0],
        "python_executable": sys.executable,
        "pid": os.getpid(),
        "node": node_summary(),
        "cpu": cpu_model(),
        "affinity": {
            "logical_cpus": cpus,
            "logical_cpu_list": format_cpu_list(cpus),
            "physical_cores": [group for group in split_physical_cores(cpus)],
            "thread_siblings": {str(cpu): thread_siblings(cpu) for cpu in cpus},
        },
        "cgroup": cgroup_cpuset(),
        "slurm": slurm_environment(),
        "environment": {name: os.environ.get(name) for name in RECORDED_VARIABLES},
        "threadpools": threadpool_snapshot(),
        "torch": torch_snapshot(torch),
        "numpy": numpy_snapshot(),
        "threads": thread_names(),
    }


# ---------------------------------------------------------------------------
# session
# ---------------------------------------------------------------------------


class TimingSession:
    """Apply, verify, and record the timing profile inside the measuring process."""

    def __init__(self, profile_id: str, report_path: Optional[Path] = None, *,
                 compute_threads: Optional[int] = None, strict: bool = True,
                 role: str = "measuring_process") -> None:
        self.profile = get_profile(profile_id)
        self.report_path = None if report_path is None else Path(report_path)
        configured = compute_threads or os.environ.get(COMPUTE_THREADS_ENV)
        self.compute_threads = int(configured) if configured else resolved_compute_threads(self.profile)
        self.strict = strict
        self.role = role
        self.checks: List[Dict[str, Any]] = []
        self.warnings: List[str] = []
        self.measurements: List[Dict[str, Any]] = []
        self.notes: Dict[str, Any] = {}
        self._limiter: Any = None
        self._torch: Any = None
        self._configured_torch_errors: List[str] = []
        self.started_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    # -- construction ------------------------------------------------------
    @classmethod
    def from_environment(cls, *, role: str = "measuring_process") -> Optional["TimingSession"]:
        profile = os.environ.get(PROFILE_ENV)
        if not profile:
            return None
        report = os.environ.get(REPORT_ENV)
        return cls(profile, Path(report) if report else None, role=role)

    # -- configuration -----------------------------------------------------
    def configure(self, torch: Any = None) -> "TimingSession":
        for name in THREAD_LIMIT_VARIABLES:
            os.environ.setdefault(name, str(self.compute_threads))
        try:
            import threadpoolctl

            self._limiter = threadpoolctl.threadpool_limits(limits=self.compute_threads)
        except ImportError:
            self._limiter = None
        if torch is not None:
            self.configure_torch(torch)
        return self

    def configure_torch(self, torch: Any) -> None:
        self._torch = torch
        torch.set_num_threads(self.compute_threads)
        try:
            if torch.get_num_interop_threads() != 1:
                torch.set_num_interop_threads(1)
        except RuntimeError as exc:
            # Only possible if parallel work already ran; verify() reports it.
            self._configured_torch_errors.append(str(exc))

    def reapply_thread_limits(self) -> None:
        """Re-limit pools loaded after configure() (e.g. lazily imported BLAS)."""
        try:
            import threadpoolctl

            self._limiter = threadpoolctl.threadpool_limits(limits=self.compute_threads)
        except ImportError:
            pass

    def note(self, **values: Any) -> None:
        self.notes.update(values)

    # -- checks ------------------------------------------------------------
    def _check(self, stage: str, name: str, passed: bool, observed: Any, expected: Any,
               *, hard: bool = True) -> None:
        self.checks.append({
            "stage": stage, "name": name, "passed": bool(passed), "hard": hard,
            "observed": observed, "expected": expected,
        })
        if not passed and not hard:
            self.warnings.append(f"{stage}:{name}: observed {observed!r}, expected {expected!r}")

    def verify(self, stage: str) -> Dict[str, Any]:
        profile = self.profile
        threads = self.compute_threads
        self.reapply_thread_limits()
        for name in THREAD_LIMIT_VARIABLES:
            self._check(stage, f"env:{name}", os.environ.get(name) == str(threads),
                        os.environ.get(name), str(threads))

        pools = threadpool_snapshot()
        allow_missing = os.environ.get(ALLOW_MISSING_THREADPOOLCTL_ENV) == "1"
        self._check(stage, "threadpoolctl_available", pools["available"] or allow_missing,
                    pools["available"], True, hard=not allow_missing)
        oversized = [pool for pool in pools["pools"] if (pool.get("num_threads") or 0) > threads]
        self._check(stage, "blas_thread_pools", not oversized,
                    [(pool.get("internal_api"), pool.get("num_threads")) for pool in pools["pools"]],
                    f"<= {threads} threads per pool")

        torch = self._torch or sys.modules.get("torch")
        if torch is not None:
            self._check(stage, "torch_intra_op_threads", torch.get_num_threads() == threads,
                        int(torch.get_num_threads()), threads)
            self._check(stage, "torch_inter_op_threads", torch.get_num_interop_threads() == 1,
                        int(torch.get_num_interop_threads()), 1)
            if self._configured_torch_errors:
                self.warnings.append("set_num_interop_threads: " + "; ".join(self._configured_torch_errors))

        cpus = allowed_cpus()
        cores = physical_cores(cpus)
        node = node_summary()
        if profile.exclusive_node:
            online = online_cpus()
            self._check(stage, "affinity_is_whole_node", sorted(cpus) == online,
                        format_cpu_list(cpus), format_cpu_list(online))
        else:
            self._check(stage, "affinity_physical_cores", len(cores) == profile.client_physical_cores,
                        {"physical_cores": len(cores), "logical_cpus": format_cpu_list(cpus)},
                        profile.client_physical_cores)
        cgroup = cgroup_cpuset()
        if cgroup.get("available"):
            owned = set(cgroup["cpus"])
            foreign = sorted({sibling for cpu in cpus for sibling in thread_siblings(cpu)} - owned)
            self._check(stage, "smt_siblings_reserved", not foreign,
                        {"siblings_outside_job_cgroup": foreign, "cgroup": cgroup["path"]},
                        "every SMT sibling of an allocated core belongs to this job")
        else:
            self._check(stage, "smt_siblings_reserved", False, "cgroup cpuset unreadable",
                        "verifiable cgroup cpuset", hard=False)

        children = child_processes()
        self._check(stage, "no_child_processes", not children, children, [])

        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        if profile.device == "cpu":
            self._check(stage, "cuda_hidden", visible == "", visible, "")
            if torch is not None:
                initialized = bool(torch.cuda.is_initialized())
                self._check(stage, "cuda_not_initialized", not initialized, initialized, False)
        else:
            available = torch is not None and bool(torch.cuda.is_available())
            count = int(torch.cuda.device_count()) if available else 0
            self._check(stage, "single_cuda_device", available and count == 1, count, 1)

        names = thread_names()
        total = sum(names.values())
        expected_threads = threads + 1
        self._check(stage, "os_thread_count", total <= expected_threads, {"total": total, "names": names},
                    f"<= {expected_threads} (warning only)", hard=False)
        failures = [check for check in self.checks if check["stage"] == stage and check["hard"] and not check["passed"]]
        result = {"stage": stage, "passed": not failures, "failures": failures, "node": node["hostname"]}
        if failures and self.strict:
            self.write()
            raise TimingGuardError(
                f"timing profile {profile.profile_id} violated at {stage}: "
                + "; ".join(f"{item['name']} observed {item['observed']!r} expected {item['expected']!r}"
                            for item in failures)
            )
        return result

    def verify_process_affinity(self, pid: int, *, stage: str, label: str,
                                expected_physical_cores: int,
                                disjoint_from: Optional[Sequence[int]] = None) -> List[int]:
        """Verify another process's pinning, e.g. the PostgreSQL backend."""
        cpus = allowed_cpus(pid)
        cores = physical_cores(cpus)
        self._check(stage, f"{label}_physical_cores", len(cores) == expected_physical_cores,
                    {"pid": pid, "logical_cpus": format_cpu_list(cpus), "physical_cores": len(cores)},
                    expected_physical_cores)
        if disjoint_from is not None:
            own_cores = set(physical_cores(disjoint_from))
            overlap = sorted(set(cores) & own_cores)
            self._check(stage, f"{label}_disjoint_from_client", not overlap,
                        [list(key) for key in overlap], [])
        failures = [check for check in self.checks
                    if check["stage"] == stage and check["name"].startswith(label)
                    and check["hard"] and not check["passed"]]
        if failures and self.strict:
            self.write()
            raise TimingGuardError(f"{label} pinning violated: {failures}")
        return cpus

    # -- measurement -------------------------------------------------------
    @contextlib.contextmanager
    def measure(self, label: str = "timed_region") -> Iterator[Dict[str, Any]]:
        record: Dict[str, Any] = {"label": label}
        cpu_started = time.process_time()
        wall_started = time.perf_counter()
        record["threads_before"] = thread_names()
        try:
            yield record
        finally:
            wall = time.perf_counter() - wall_started
            cpu = time.process_time() - cpu_started
            record.update({
                "wall_seconds": wall,
                "process_cpu_seconds": cpu,
                "cpu_to_wall_ratio": (cpu / wall) if wall > 0 else None,
                "threads_after": thread_names(),
                "child_processes_after": child_processes(),
            })
            ratio = record["cpu_to_wall_ratio"]
            limit = float(len(physical_cores(allowed_cpus())))
            if self.profile.device == "cpu" and ratio is not None and ratio > 1.05 * limit:
                self.warnings.append(
                    f"{label}: CPU/wall ratio {ratio:.3f} exceeds {limit:g} allocated core(s)"
                )
            self.measurements.append(record)

    # -- reporting ---------------------------------------------------------
    def report(self) -> Dict[str, Any]:
        hard_failures = [check for check in self.checks if check["hard"] and not check["passed"]]
        return {
            "guard_version": GUARD_VERSION,
            "role": self.role,
            "profile": asdict(self.profile),
            "compute_threads": self.compute_threads,
            "latency_definition": LATENCY_DEFINITION,
            "started_utc": self.started_utc,
            "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "passed": bool(self.checks) and not hard_failures,
            "verified_stages": sorted({check["stage"] for check in self.checks}),
            "hard_failures": hard_failures,
            "warnings": self.warnings,
            "checks": self.checks,
            "measurements": self.measurements,
            "notes": self.notes,
            "snapshot": environment_snapshot(self._torch),
        }

    def write(self, path: Optional[Path] = None) -> Optional[Path]:
        target = Path(path) if path is not None else self.report_path
        if target is None:
            return None
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(target.suffix + ".tmp")
        temporary.write_text(json.dumps(self.report(), indent=2, sort_keys=True, default=str) + "\n",
                             encoding="utf-8")
        temporary.replace(target)
        return target


def load_from_path():  # pragma: no cover - documentation helper
    """Bridges import this file by path; see ``scripts/_timing.py``."""
    return sys.modules[__name__]
