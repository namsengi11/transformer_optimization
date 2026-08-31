"""Pinned locations for everything the measurement tools shell out to.

Rediscovering tool paths per run is how two runs end up measured by two
different profilers. Everything is resolved once, here, and every other tool
imports from this module rather than calling `where`/`which` itself.

Neither Nsight tool is on PATH on this machine, so the verified install
locations are pinned explicitly.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
AGENT_ROOT = REPO_ROOT / "agent"
TOOLS_DIR = AGENT_ROOT / "tools"
RESULTS = AGENT_ROOT / "results"
ARTIFACTS = RESULTS / "_artifacts"        # nsys/ncu reports; gitignored
EXPERIMENTS = AGENT_ROOT / "experiments"

PYTHON = sys.executable
HARNESS = REPO_ROOT / "torch_transformer_benchmark.py"
RUN_BENCH = REPO_ROOT / "run_bench.py"

# Verified present 2026-08-31 on this machine.
_NSYS_CANDIDATES = [
    r"C:\Program Files\NVIDIA Corporation\Nsight Systems 2025.1.3\target-windows-x64\nsys.exe",
]
_NCU_CANDIDATES = [
    r"C:\Program Files\NVIDIA Corporation\Nsight Compute 2025.2.1\target\windows-desktop-win7-x64\ncu.exe",
]


def _resolve(candidates: list[str]) -> str | None:
    for cand in candidates:
        if Path(cand).is_file():
            return cand
    return None


NSYS_EXE = _resolve(_NSYS_CANDIDATES)
NCU_EXE = _resolve(_NCU_CANDIDATES)

# nvprof ships with CUDA 12.9 here but is deprecated and does not support
# sm_120. Recorded so nobody "discovers" it again and uses it.
NVPROF_BANNED_REASON = "deprecated, no sm_120 support -- use nsys_trace.py / ncu_profile.py"


def ensure_dirs() -> None:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    EXPERIMENTS.mkdir(parents=True, exist_ok=True)


def agent_path(value: str | Path) -> Path:
    """Resolve a writable output path and require it to stay under agent/."""
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    candidate = candidate.resolve()
    try:
        candidate.relative_to(AGENT_ROOT.resolve())
    except ValueError as exc:
        raise ValueError(f"agent output must stay under {AGENT_ROOT}: {candidate}") from exc
    return candidate


if __name__ == "__main__":
    for k, v in [("REPO_ROOT", REPO_ROOT), ("PYTHON", PYTHON),
                 ("NSYS_EXE", NSYS_EXE), ("NCU_EXE", NCU_EXE)]:
        print(f"{k:12} {v}")
