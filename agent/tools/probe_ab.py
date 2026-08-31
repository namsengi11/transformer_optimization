#!/usr/bin/env python3
"""Fixed-checkout, one-process-per-arm A/B gate probe."""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from agent.tools._common import ranges_overlap, stamp
from agent.tools.paths import ARTIFACTS, TOOLS_DIR, agent_path


def _git(*args: str) -> str:
    proc = subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or f"git {' '.join(args)} failed")
    return proc.stdout.strip()


def _run_arm(args: argparse.Namespace, label: str, value: str, head: str,
             main_sha: str) -> dict[str, Any]:
    if _git("rev-parse", "HEAD") != head:
        raise RuntimeError("HEAD moved during A/B; refusing result")
    if _git("rev-parse", "main") != main_sha:
        raise RuntimeError("main moved during A/B; rebase and re-measure")
    env = dict(os.environ)
    env[args.flag] = value
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    output = ARTIFACTS / f"{args.tag}-{label}.bench.json"
    cmd = [sys.executable, str(TOOLS_DIR / "bench.py"),
           "--suite", args.suite, "--tag", f"{args.tag}-{label}",
           "--runs", str(args.runs), "--timeout", str(args.timeout),
           "--bar-reps", str(args.bar_reps), "--output", str(output), "--json"]
    for case in args.case or []:
        cmd += ["--case", case]
    proc = subprocess.run(cmd, cwd=REPO_ROOT, env=env, capture_output=True,
                          text=True, timeout=args.timeout * max(4, args.runs * 3))
    if proc.returncode != 0:
        raise RuntimeError(f"arm {label} failed:\n{(proc.stdout + proc.stderr)[-5000:]}")
    if not output.is_file():
        raise RuntimeError(f"arm {label} did not create {output}")
    return json.loads(output.read_text())


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", args.flag):
        raise ValueError("--flag must be one environment variable name")
    if _git("status", "--porcelain"):
        raise RuntimeError("working tree is dirty; commit the fixed checkout before A/B")
    head = _git("rev-parse", "HEAD")
    main_sha = _git("rev-parse", "main")
    if args.pinned_sha and head != args.pinned_sha:
        raise RuntimeError(f"HEAD {head} does not match --pinned-sha {args.pinned_sha}")
    if args.base_sha and main_sha != args.base_sha:
        raise RuntimeError(f"main {main_sha} moved from recorded base {args.base_sha}")
    arm_a = _run_arm(args, "a", args.a_value, head, main_sha)
    arm_b = _run_arm(args, "b", args.b_value, head, main_sha)
    b_cases = {case["name"]: case for case in arm_b["cases"]}
    comparisons = []
    for a_case in arm_a["cases"]:
        b_case = b_cases[a_case["name"]]
        a_summary = a_case["summaries"]["optimized_ms"]
        b_summary = b_case["summaries"]["optimized_ms"]
        a_range = [a_summary["min"], a_summary["max"]]
        b_range = [b_summary["min"], b_summary["max"]]
        overlap = ranges_overlap(a_range, b_range)
        comparisons.append({
            "name": a_case["name"],
            "ranges_overlap": overlap,
            "verdict": "noise" if overlap else (
                "b_faster" if max(b_range) < min(a_range) else "a_faster"
            ),
            "a_median_ms": a_summary["median"],
            "b_median_ms": b_summary["median"],
            "b_delta_pct": 100.0 * (b_summary["median"] / a_summary["median"] - 1.0),
        })
    if _git("rev-parse", "HEAD") != head or _git("rev-parse", "main") != main_sha:
        raise RuntimeError("checkout or main moved before judgment; discarding A/B")
    return {
        "provenance": stamp("agent/tools/probe_ab.py", {"invocation": " ".join(sys.argv)}),
        "fixed_checkout": {"head": head, "main": main_sha},
        "toggle": {"flag": args.flag, "arm_a": args.a_value, "arm_b": args.b_value},
        "arm_a": arm_a,
        "arm_b": arm_b,
        "comparisons": comparisons,
        "trusted": arm_a.get("trusted") and arm_b.get("trusted"),
    }


def _human(payload: dict[str, Any]) -> None:
    toggle = payload["toggle"]
    print(f"{toggle['flag']}: A={toggle['arm_a']} B={toggle['arm_b']} @ {payload['fixed_checkout']['head'][:12]}")
    for row in payload["comparisons"]:
        print(f"{row['name']:<20} {row['verdict']:<8} A={row['a_median_ms']:.4f} ms B={row['b_median_ms']:.4f} ms ({row['b_delta_pct']:+.2f}%)")
    print(f"trusted={payload['trusted']}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=("quick", "default", "full", "user_matrix"), default="quick")
    parser.add_argument("--case", action="append")
    parser.add_argument("--tag", required=True)
    parser.add_argument("--flag", required=True)
    parser.add_argument("--a-value", default="0")
    parser.add_argument("--b-value", default="1")
    parser.add_argument("--pinned-sha")
    parser.add_argument("--base-sha")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--bar-reps", type=int, default=2)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--output", type=agent_path)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    try:
        payload = run(args)
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
        print(f"probe_ab refused: {exc}", file=sys.stderr)
        return 2
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2, default=str))
    if args.json:
        print(json.dumps(payload, indent=2, default=str))
    else:
        _human(payload)
    return 0 if payload["trusted"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
