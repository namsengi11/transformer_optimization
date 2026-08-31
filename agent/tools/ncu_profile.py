#!/usr/bin/env python3
"""Tier-3 Nsight Compute profile with graceful GPU-counter permission gating."""
from __future__ import annotations

import argparse
import csv
import ctypes
import io
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from agent.tools._common import stamp
from agent.tools.paths import ARTIFACTS, NCU_EXE, PYTHON, TOOLS_DIR, agent_path, ensure_dirs

SECTIONS = ("SpeedOfLight", "MemoryWorkloadAnalysis", "Occupancy", "LaunchStats")
REGISTRY_PATH = r"SYSTEM\CurrentControlSet\Services\nvlddmkm\Global\NVTweak"
REGISTRY_VALUE = "RmProfilingAdminOnly"
FIX = (
    rf"set HKLM\{REGISTRY_PATH}\{REGISTRY_VALUE} (DWORD) to 0 and reboot, "
    "or run the profiling session elevated"
)


def _counter_access_likely() -> bool:
    if platform.system() != "Windows":
        return True
    try:
        if ctypes.windll.shell32.IsUserAnAdmin():
            return True
    except Exception:
        pass
    try:
        import winreg
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, REGISTRY_PATH) as key:
            value, _ = winreg.QueryValueEx(key, REGISTRY_VALUE)
            return int(value) == 0
    except (FileNotFoundError, OSError, ValueError):
        return False


def _unavailable(reason: str) -> dict[str, Any]:
    return {
        "provenance": stamp("agent/tools/ncu_profile.py"),
        "available": False,
        "reason": reason,
        "fix": FIX if "counter" in reason.lower() or "permission" in reason.lower() else None,
        "sections": list(SECTIONS),
        "trusted": False,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not NCU_EXE:
        return _unavailable("Nsight Compute executable not found")
    if not _counter_access_likely():
        return _unavailable("needs elevated GPU counters")
    ensure_dirs()
    report = ARTIFACTS / f"{args.tag}.ncu-rep"
    target = [PYTHON, str(TOOLS_DIR / "profile.py"), "--arm", args.arm,
              "--warmup", str(args.warmup), "--iterations", str(args.iterations),
              "--row-limit", "5", "--external-profiler", "--json"]
    if args.case:
        target += ["--case", args.case]
    command = [NCU_EXE, "--target-processes", "all", "--force-overwrite",
               "--export", str(report), "--kernel-name", f"regex:{args.kernel_regex}",
               "--launch-count", str(args.launch_count)]
    for section in SECTIONS:
        command += ["--section", section]
    command += target
    proc = subprocess.run(command, cwd=REPO_ROOT, capture_output=True, text=True,
                          errors="replace", timeout=args.timeout)
    combined = proc.stdout + proc.stderr
    if "ERR_NVGPUCTRPERM" in combined or "permission" in combined.lower() and "counter" in combined.lower():
        return _unavailable("needs elevated GPU counters (ERR_NVGPUCTRPERM)")
    if proc.returncode != 0 or not report.is_file():
        return {
            "provenance": stamp("agent/tools/ncu_profile.py"), "available": True,
            "captured": False, "reason": combined[-5000:], "command": command,
            "sections": list(SECTIONS), "trusted": False,
        }
    import_cmd = [NCU_EXE, "--import", str(report), "--csv", "--page", "raw"]
    imported = subprocess.run(import_cmd, cwd=REPO_ROOT, capture_output=True, text=True,
                              errors="replace", timeout=args.timeout)
    metrics: list[dict[str, str]] = []
    if imported.returncode == 0:
        lines = [line for line in imported.stdout.splitlines() if line and not line.startswith("==")]
        try:
            metrics = list(csv.DictReader(io.StringIO("\n".join(lines))))[:500]
        except csv.Error:
            metrics = []
    return {
        "provenance": stamp("agent/tools/ncu_profile.py", {"invocation": " ".join(sys.argv)}),
        "available": True,
        "captured": True,
        "report": str(report),
        "sections": list(SECTIONS),
        "kernel_regex": args.kernel_regex,
        "launch_count": args.launch_count,
        "metrics": metrics,
        "trusted": True,
    }


def _human(payload: dict[str, Any]) -> None:
    if not payload.get("available"):
        print(f"Nsight Compute unavailable: {payload['reason']}")
        if payload.get("fix"):
            print(f"  fix: {payload['fix']}")
        return
    if not payload.get("captured"):
        print(f"Nsight Compute failed: {payload.get('reason')}")
        return
    print(f"report={payload['report']}")
    print(f"sections={', '.join(payload['sections'])} metrics={len(payload['metrics'])}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--arm", choices=("baseline", "compiled", "optimized"), default="optimized")
    parser.add_argument("--case", default="default:default_fp32")
    parser.add_argument("--kernel-regex", default=".*")
    parser.add_argument("--launch-count", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--output", type=agent_path)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    try:
        payload = run(args)
    except (OSError, RuntimeError, subprocess.TimeoutExpired) as exc:
        payload = _unavailable(repr(exc))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2, default=str))
    if args.json:
        print(json.dumps(payload, indent=2, default=str))
    else:
        _human(payload)
    # Counter permission is an expected degraded mode, not a tool crash.
    return 0 if (payload.get("captured") or payload.get("available") is False) else 2


if __name__ == "__main__":
    raise SystemExit(main())
