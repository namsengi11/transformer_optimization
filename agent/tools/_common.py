"""Shared plumbing for the measurement tools: provenance stamping, GPU idle
gating, foreign-process detection, and the non-overlap comparison.

Nothing here measures anything. It exists so that every number a tool emits
carries the tree state and GPU conditions it was taken under -- the log's
trap 4 (a branch A/B invalidated by `main` moving mid-measurement) and trap 5
(benchmarks silently run against a GPU at 100% from another session) are both
"the number was real, the context was not".
"""
from __future__ import annotations

import json
import os
import subprocess
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

try:
    from .paths import REPO_ROOT, agent_path
except ImportError:  # direct execution from agent/tools/
    from paths import REPO_ROOT, agent_path

TOOL_CONTRACT_VERSION = "1.0"

# Torch 2.8 on Windows overflows a C long with the 64-bit stream handle when
# inductor's static launcher is on, which kills CUDA-graph capture. Tools that
# capture graphs themselves opt in via this helper; the suite driver does NOT,
# so the compiled bar keeps its normal launcher and stays a fair comparison.
def disable_static_cuda_launcher() -> None:
    os.environ["TORCHINDUCTOR_USE_STATIC_CUDA_LAUNCHER"] = "0"


# --------------------------------------------------------------------------
# provenance
# --------------------------------------------------------------------------

def _git(*args: str) -> str:
    try:
        out = subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True,
                             text=True, timeout=30)
        return out.stdout.strip()
    except Exception:
        return ""


def head_sha() -> str:
    return _git("rev-parse", "HEAD")


def branch_name() -> str:
    return _git("rev-parse", "--abbrev-ref", "HEAD")


def tree_dirty() -> bool:
    return bool(_git("status", "--porcelain"))


def ahead_behind(other: str = "main") -> tuple[int, int] | None:
    """(ahead, behind) of HEAD relative to `other`, or None if unavailable."""
    out = _git("rev-list", "--left-right", "--count", f"{other}...HEAD")
    parts = out.split()
    if len(parts) != 2:
        return None
    behind, ahead = int(parts[0]), int(parts[1])
    return ahead, behind


def stamp(tool: str, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    """Provenance block that every tool's JSON output must carry."""
    gpu = gpu_query()
    s = {
        "tool": tool,
        "tool_contract_version": TOOL_CONTRACT_VERSION,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_head": head_sha(),
        "git_branch": branch_name(),
        "git_tree_dirty": tree_dirty(),
        "gpu": gpu,
    }
    if extra:
        s.update(extra)
    return s


# --------------------------------------------------------------------------
# GPU state
# --------------------------------------------------------------------------

def _nvidia_smi(query: str, extra: Sequence[str] = ()) -> list[list[str]]:
    try:
        out = subprocess.run(
            ["nvidia-smi", f"--query-{query}", "--format=csv,noheader,nounits", *extra],
            capture_output=True, text=True, timeout=20)
    except Exception:
        return []
    if out.returncode != 0:
        return []
    rows = []
    for line in out.stdout.strip().splitlines():
        line = line.strip()
        if line and "not supported" not in line.lower():
            rows.append([c.strip() for c in line.split(",")])
    return rows


def gpu_query() -> dict[str, Any]:
    rows = _nvidia_smi("gpu=name,utilization.gpu,clocks.sm,clocks.max.sm,"
                       "memory.used,memory.total,driver_version")
    if not rows:
        return {"available": False}
    r = rows[0]

    def _i(v: str) -> int | None:
        try:
            return int(float(v))
        except Exception:
            return None

    return {
        "available": True,
        "name": r[0],
        "util_pct": _i(r[1]),
        "clock_sm_mhz": _i(r[2]),
        "clock_sm_max_mhz": _i(r[3]),
        "mem_used_mib": _i(r[4]),
        "mem_total_mib": _i(r[5]),
        "driver": r[6] if len(r) > 6 else None,
    }


def compute_pids() -> set[int]:
    """PIDs of every process currently holding a CUDA context on the GPU."""
    pids = set()
    for row in _nvidia_smi("compute-apps=pid"):
        try:
            pids.add(int(row[0]))
        except Exception:
            pass
    return pids


def process_sm_utilization() -> dict[int, int]:
    """Best-effort per-process SM utilization from ``nvidia-smi pmon``.

    WDDM exposes many idle graphics/compute contexts through compute-apps.
    `pmon` is the available signal that an already-open context actually
    started consuming SM time during a benchmark.
    """
    try:
        out = subprocess.run(
            ["nvidia-smi", "pmon", "-c", "1", "-s", "u"],
            capture_output=True, text=True, timeout=10)
    except Exception:
        return {}
    if out.returncode != 0:
        return {}
    result: dict[int, int] = {}
    for line in out.stdout.splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        fields = line.split()
        if len(fields) < 4:
            continue
        try:
            pid = int(fields[1])
            sm = 0 if fields[3] == "-" else int(fields[3])
        except ValueError:
            continue
        result[pid] = sm
    return result


def _descendants(root: int) -> set[int]:
    """Best-effort process subtree of `root` on Windows, via wmic-free CIM."""
    try:
        ps = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command",
             "Get-CimInstance Win32_Process | "
             "Select-Object ProcessId,ParentProcessId | ConvertTo-Json -Compress"],
            capture_output=True, text=True, timeout=30)
        data = json.loads(ps.stdout or "[]")
    except Exception:
        return {root}
    if isinstance(data, dict):
        data = [data]
    children: dict[int, list[int]] = {}
    for row in data:
        try:
            children.setdefault(int(row["ParentProcessId"]), []).append(int(row["ProcessId"]))
        except Exception:
            pass
    seen, stack = {root}, [root]
    while stack:
        cur = stack.pop()
        for kid in children.get(cur, []):
            if kid not in seen:
                seen.add(kid)
                stack.append(kid)
    return seen


def wait_for_idle(util_threshold: int = 10, hold_s: float = 5.0,
                  timeout_s: float = 900.0, poll_s: float = 1.0,
                  verbose: bool = True) -> dict[str, Any]:
    """Block until the GPU has been below `util_threshold`% for `hold_s`.

    Returns {"idle": bool, ...}. On timeout the caller must REFUSE to measure
    rather than measure anyway -- a busy GPU is the difference between a real
    regression and someone else's benchmark.
    """
    start = time.time()
    below_since: float | None = None
    last_report = 0.0
    while time.time() - start < timeout_s:
        g = gpu_query()
        if not g.get("available"):
            return {"idle": False, "reason": "nvidia-smi unavailable", "gpu": g}
        util = g.get("util_pct")
        now = time.time()
        if util is not None and util < util_threshold:
            below_since = below_since or now
            if now - below_since >= hold_s:
                return {"idle": True, "waited_s": round(now - start, 1), "gpu": g}
        else:
            below_since = None
            if verbose and now - last_report > 15:
                last_report = now
                others = sorted(compute_pids() - _descendants(os.getpid()))
                print(f"[gpu-gate] busy at {util}% (foreign pids: {others or 'unknown'}) "
                      f"-- waited {now - start:.0f}s", flush=True)
        time.sleep(poll_s)
    return {"idle": False, "reason": f"GPU still busy after {timeout_s:.0f}s",
            "gpu": gpu_query()}


class ForeignLoadSampler:
    """Samples GPU utilisation and foreign CUDA processes during a run.

    A case whose measurement overlapped a foreign CUDA process is marked
    untrusted rather than silently reported.
    """

    def __init__(self, poll_s: float = 1.0):
        self.poll_s = poll_s
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.max_util = 0
        self.foreign_pids: set[int] = set()
        self.baseline_foreign_pids: set[int] = set()
        self.baseline_foreign_sm: dict[int, int] = {}
        self.foreign_sm: dict[int, int] = {}

    def _run(self) -> None:
        while not self._stop.is_set():
            g = gpu_query()
            util = g.get("util_pct") or 0
            self.max_util = max(self.max_util, util)
            # Benchmark children are launched after this sampler is created.
            # Recompute the process tree on every sample so those children are
            # never mislabeled as foreign GPU users.
            current_sm = process_sm_utilization()
            own = _descendants(os.getpid())
            for pid, sm in current_sm.items():
                if pid in own:
                    continue
                baseline_sm = self.baseline_foreign_sm.get(pid, 0)
                # Ignore the 1-2% desktop-compositor noise visible at idle;
                # record a foreign context only when it has a material spike.
                if sm >= 5 and sm > baseline_sm + 3:
                    self.foreign_pids.add(pid)
                    self.foreign_sm[pid] = max(self.foreign_sm.get(pid, 0), sm)
            self._stop.wait(self.poll_s)

    def __enter__(self) -> "ForeignLoadSampler":
        self.baseline_foreign_pids = compute_pids() - _descendants(os.getpid())
        baseline_sm = process_sm_utilization()
        own = _descendants(os.getpid())
        self.baseline_foreign_sm = {pid: sm for pid, sm in baseline_sm.items() if pid not in own}
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)

    @property
    def trusted(self) -> bool:
        return not self.foreign_pids

    def report(self) -> dict[str, Any]:
        return {"trusted": self.trusted,
                "max_util_pct": self.max_util,
                "idle_baseline_cuda_pids": sorted(self.baseline_foreign_pids),
                "idle_baseline_sm_util_pct": self.baseline_foreign_sm,
                "foreign_cuda_pids": sorted(self.foreign_pids),
                "foreign_sm_util_pct": self.foreign_sm}


# --------------------------------------------------------------------------
# comparison
# --------------------------------------------------------------------------

def ranges_overlap(a: Sequence[float], b: Sequence[float]) -> bool:
    """True if two sample sets' [min,max] ranges overlap.

    The promotion rule requires NON-overlapping ranges across repeated runs
    before a latency delta counts. Overlapping ranges are noise, whatever the
    medians say -- `07_seq32`-class shapes swing 25% between identical runs.
    """
    if not a or not b:
        return True
    return not (max(a) < min(b) or max(b) < min(a))


def summarize(samples: Sequence[float]) -> dict[str, float | int]:
    ordered = sorted(samples)
    n = len(ordered)
    mid = n // 2
    median = ordered[mid] if n % 2 else (ordered[mid - 1] + ordered[mid]) / 2
    return {"n": n, "min": ordered[0], "median": median, "max": ordered[-1]}


def geomean(values: Iterable[float]) -> float | None:
    vals = [v for v in values if v and v > 0]
    if not vals:
        return None
    logsum = sum(__import__("math").log(v) for v in vals)
    return __import__("math").exp(logsum / len(vals))


def emit(payload: dict[str, Any], as_json: bool, out_path: Path | None = None) -> None:
    text = json.dumps(payload, indent=2, default=str)
    if out_path:
        out_path = agent_path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text)
        print(f"\nwrote {out_path}")
    if as_json:
        print(text)
