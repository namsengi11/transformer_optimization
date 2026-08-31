#!/usr/bin/env python3
"""Validate the pinned software, GPU, and profiler environment."""
from __future__ import annotations

import argparse
import ctypes
import importlib.metadata
import json
import platform
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if sys.path and Path(sys.path[0]).resolve() == Path(__file__).resolve().parent:
    sys.path.pop(0)  # prevent agent/tools/profile.py from shadowing stdlib profile
sys.path.insert(0, str(REPO_ROOT))

from agent.tools._common import stamp
from agent.tools.paths import NCU_EXE, NSYS_EXE, NVPROF_BANNED_REASON

EXPECTED_TORCH = "2.8.0+cu129"
EXPECTED_TRITON_WINDOWS = "3.4.0.post21"
EXPECTED_CAPABILITY = (12, 0)
REGISTRY_PATH = r"SYSTEM\CurrentControlSet\Services\nvlddmkm\Global\NVTweak"
REGISTRY_VALUE = "RmProfilingAdminOnly"
COUNTER_FIX = (
    rf"set HKLM\{REGISTRY_PATH}\{REGISTRY_VALUE} (DWORD) to 0 and reboot, "
    "or run the profiling session elevated"
)


def _distribution_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _is_admin() -> bool:
    if platform.system() != "Windows":
        return False
    try:
        return bool(ctypes.windll.shell32.IsUserAnAdmin())
    except Exception:
        return False


def _counter_registry_value() -> int | None:
    if platform.system() != "Windows":
        return None
    try:
        import winreg
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, REGISTRY_PATH) as key:
            value, _ = winreg.QueryValueEx(key, REGISTRY_VALUE)
            return int(value)
    except (FileNotFoundError, OSError, ValueError):
        return None


def inspect() -> dict[str, Any]:
    try:
        import torch
        torch_version = torch.__version__
        cuda_available = torch.cuda.is_available()
        capability = list(torch.cuda.get_device_capability(0)) if cuda_available else None
        gpu_name = torch.cuda.get_device_name(0) if cuda_available else None
    except Exception as exc:
        torch_version = None
        cuda_available = False
        capability = None
        gpu_name = None
        torch_error = repr(exc)
    else:
        torch_error = None

    triton_windows = _distribution_version("triton-windows")
    admin = _is_admin()
    registry_value = _counter_registry_value()
    counters_available = bool(NCU_EXE) and (admin or registry_value == 0)
    checks = {
        "torch_version": torch_version == EXPECTED_TORCH,
        "triton_windows_version": triton_windows == EXPECTED_TRITON_WINDOWS,
        "cuda_available": cuda_available,
        "compute_capability": capability == list(EXPECTED_CAPABILITY),
        "nsys_present": bool(NSYS_EXE),
        "ncu_present": bool(NCU_EXE),
    }
    return {
        "provenance": stamp("agent/tools/env_check.py"),
        "expected": {
            "torch": EXPECTED_TORCH,
            "triton_windows": EXPECTED_TRITON_WINDOWS,
            "compute_capability": list(EXPECTED_CAPABILITY),
        },
        "observed": {
            "python": sys.executable,
            "torch": torch_version,
            "torch_import_error": torch_error,
            "triton_windows": triton_windows,
            "cuda_available": cuda_available,
            "gpu_name": gpu_name,
            "compute_capability": capability,
        },
        "profilers": {
            "nsys": {"available": bool(NSYS_EXE), "path": NSYS_EXE},
            "ncu": {
                "installed": bool(NCU_EXE),
                "path": NCU_EXE,
                "available": counters_available,
                "is_admin": admin,
                "registry_value": registry_value,
                "reason": None if counters_available else "needs elevated GPU counters",
                "fix": None if counters_available else COUNTER_FIX,
            },
            "nvprof": {"available": False, "reason": NVPROF_BANNED_REASON},
        },
        "checks": checks,
        "ok": all(checks.values()),
        "trusted": all(checks.values()),
    }


def _human(payload: dict[str, Any]) -> None:
    observed = payload["observed"]
    print("Environment check")
    print(f"  torch             {observed['torch']} (expected {EXPECTED_TORCH})")
    print(f"  triton-windows    {observed['triton_windows']} (expected {EXPECTED_TRITON_WINDOWS})")
    print(f"  GPU               {observed['gpu_name']}")
    print(f"  compute capability {observed['compute_capability']} (expected {list(EXPECTED_CAPABILITY)})")
    for name, info in payload["profilers"].items():
        state = "available" if info.get("available") else "unavailable"
        print(f"  {name:<17} {state}: {info.get('path') or info.get('reason')}")
    ncu = payload["profilers"]["ncu"]
    if not ncu["available"]:
        print(f"  NCU fix           {ncu['fix']}")
    print(f"  result            {'PASS' if payload['ok'] else 'FAIL'}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    payload = inspect()
    if args.json:
        print(json.dumps(payload, indent=2, default=str))
    else:
        _human(payload)
    return 0 if payload["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
