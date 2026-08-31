#!/usr/bin/env python3
"""Run judged latency/MFU suites with idle gating and repeated ranges."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if sys.path and Path(sys.path[0]).resolve() == Path(__file__).resolve().parent:
    sys.path.pop(0)
sys.path.insert(0, str(REPO_ROOT))

import run_bench
from agent.tools._common import ForeignLoadSampler, geomean, ranges_overlap, stamp, summarize, wait_for_idle
from agent.tools.paths import AGENT_ROOT, RESULTS, RUN_BENCH, agent_path

METRICS = ("optimized_ms", "compiled_baseline_ms", "speedup_vs_compiled", "mfu_optimized")


def _safe_tag(value: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in value)


def _run_case(suite: str, case: str, tag: str, timeout: int,
              bar_reps: int, env: dict[str, str]) -> tuple[dict[str, Any], dict[str, Any]]:
    idle = wait_for_idle(timeout_s=timeout, verbose=False)
    if not idle.get("idle"):
        raise RuntimeError(idle.get("reason", "GPU did not become idle"))
    cmd = [sys.executable, str(RUN_BENCH), "--suite", suite, "--case", case,
           "--tag", tag, "--timeout", str(timeout), "--bar-reps", str(bar_reps)]
    with ForeignLoadSampler() as sampler:
        proc = subprocess.run(cmd, cwd=AGENT_ROOT, env=env, capture_output=True,
                              text=True, errors="replace",
                              timeout=timeout * max(2, bar_reps + 1))
    if proc.returncode != 0:
        tail = (proc.stdout + proc.stderr)[-5000:]
        raise RuntimeError(f"run_bench failed for {case} (exit {proc.returncode}):\n{tail}")
    result_path = RESULTS / f"{_safe_tag(tag)}_{suite}.json"
    if not result_path.is_file():
        raise RuntimeError(f"run_bench did not create {result_path}")
    rows = json.loads(result_path.read_text())
    if len(rows) != 1:
        raise RuntimeError(f"expected one result for {case}, got {len(rows)}")
    load = sampler.report()
    load["idle_gate"] = idle
    return rows[0], load


def _comparison_cases(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {case["name"]: case for case in payload.get("cases", [])}


def run_suite(suite: str, tag: str, runs: int, timeout: int, bar_reps: int,
              selected_cases: list[str] | None, compare: Path | None) -> dict[str, Any]:
    if runs < 3:
        raise ValueError("judged measurements require --runs >= 3")
    cases = list(run_bench.SUITES[suite])
    if selected_cases:
        unknown = sorted(set(selected_cases) - set(cases))
        if unknown:
            raise ValueError(f"unknown cases for {suite}: {', '.join(unknown)}")
        cases = [name for name in cases if name in selected_cases]
    env = dict(os.environ)
    env["TJ_DEBUG_GATE"] = "1"
    env.setdefault("PYTHONHASHSEED", "1234")
    comparison = json.loads(compare.read_text()) if compare else None
    comparison_cases = _comparison_cases(comparison) if comparison else {}

    output_cases: list[dict[str, Any]] = []
    for case in cases:
        raw_runs: list[dict[str, Any]] = []
        loads: list[dict[str, Any]] = []
        for run_no in range(1, runs + 1):
            arm_tag = _safe_tag(f"{tag}-{case}-r{run_no}")
            print(f"[bench] {case} run {run_no}/{runs}", file=sys.stderr, flush=True)
            row, load = _run_case(suite, case, arm_tag, timeout, bar_reps, env)
            raw_runs.append(row)
            loads.append(load)
        summaries = {
            metric: summarize([float(row[metric]) for row in raw_runs if row.get(metric) is not None])
            for metric in METRICS
            if any(row.get(metric) is not None for row in raw_runs)
        }
        statuses = {row.get("status", "EXECUTED") for row in raw_runs}
        preflight_blocked = statuses == {"PREFLIGHT_BLOCKED"}
        case_payload: dict[str, Any] = {
            "name": case,
            "status": "PREFLIGHT_BLOCKED" if preflight_blocked else "EXECUTED",
            "accuracy_pass": (
                None if preflight_blocked
                else all(row.get("accuracy") == "PASS" for row in raw_runs)
            ),
            "trusted": all(load.get("trusted") for load in loads),
            "summaries": summaries,
            "runs": raw_runs,
            "gpu_load": loads,
        }
        old = comparison_cases.get(case)
        if old and "optimized_ms" in summaries:
            old_summary = old.get("summaries", {}).get("optimized_ms")
            if old_summary:
                new_range = [summaries["optimized_ms"]["min"], summaries["optimized_ms"]["max"]]
                old_range = [old_summary["min"], old_summary["max"]]
                overlap = ranges_overlap(new_range, old_range)
                case_payload["comparison"] = {
                    "source": str(compare),
                    "ranges_overlap": overlap,
                    "verdict": "noise" if overlap else (
                        "improved" if max(new_range) < min(old_range) else "regressed"
                    ),
                    "median_delta_pct": 100.0 * (
                        summaries["optimized_ms"]["median"] / old_summary["median"] - 1.0
                    ),
                }
        output_cases.append(case_payload)

    executed_cases = [case for case in output_cases if case["status"] == "EXECUTED"]
    blocked_cases = [case for case in output_cases if case["status"] == "PREFLIGHT_BLOCKED"]
    valid_speedups = [case["summaries"]["speedup_vs_compiled"]["median"] for case in executed_cases
                      if "speedup_vs_compiled" in case["summaries"]]
    valid_mfu = [case["summaries"]["mfu_optimized"]["median"] for case in executed_cases
                 if "mfu_optimized" in case["summaries"]]
    comparisons = [case.get("comparison", {}).get("verdict") for case in output_cases
                   if case.get("comparison")]
    return {
        "provenance": stamp("agent/tools/bench.py", {"invocation": " ".join(sys.argv)}),
        "suite": suite,
        "tag": tag,
        "runs_per_case": runs,
        "cases": output_cases,
        "aggregate": {
            "geomean_speedup_vs_compiled": geomean(valid_speedups),
            "average_mfu_optimized": (sum(valid_mfu) / len(valid_mfu)) if valid_mfu else None,
            "accuracy": (
                f"{sum(c['accuracy_pass'] is True for c in executed_cases)}/"
                f"{len(executed_cases)} PASS"
            ),
            "preflight_blocked": f"{len(blocked_cases)}/{len(output_cases)}",
            "comparison_verdict": (
                "regressed" if "regressed" in comparisons else
                "improved" if "improved" in comparisons else
                "noise" if comparisons else None
            ),
        },
        "trusted": all(case["trusted"] for case in output_cases) and all(
            case["accuracy_pass"] is True for case in executed_cases
        ),
    }


def _human(payload: dict[str, Any]) -> None:
    print(f"suite={payload['suite']} tag={payload['tag']} runs={payload['runs_per_case']}")
    print(f"{'case':<20} {'acc':<5} {'trusted':<8} {'opt ms [min/med/max]':<30} verdict")
    for case in payload["cases"]:
        s = case.get("summaries", {}).get("optimized_ms", {})
        timing = f"{s.get('min', float('nan')):.4f}/{s.get('median', float('nan')):.4f}/{s.get('max', float('nan')):.4f}"
        verdict = case.get("comparison", {}).get("verdict", "-")
        accuracy = "BLOCK" if case.get("status") == "PREFLIGHT_BLOCKED" else str(case["accuracy_pass"])
        print(f"{case['name']:<20} {accuracy:<5} {str(case['trusted']):<8} {timing:<30} {verdict}")
    agg = payload["aggregate"]
    print(f"geomean vs compiled: {agg['geomean_speedup_vs_compiled']}")
    print(f"average MFU: {agg['average_mfu_optimized']}")
    print(f"accuracy: {agg['accuracy']} | trusted={payload['trusted']}")
    print(f"preflight blocked: {agg['preflight_blocked']}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=sorted(run_bench.SUITES), default="default")
    parser.add_argument("--tag", required=True)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--bar-reps", type=int, default=2)
    parser.add_argument("--case", action="append")
    parser.add_argument("--compare", type=Path)
    parser.add_argument("--output", type=agent_path)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    try:
        payload = run_suite(args.suite, args.tag, args.runs, args.timeout,
                            args.bar_reps, args.case, args.compare)
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
        print(f"bench refused: {exc}", file=sys.stderr)
        return 2
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2, default=str))
    if args.json:
        print(json.dumps(payload, indent=2, default=str))
    else:
        _human(payload)
        if args.output:
            print(f"wrote {args.output}")
    return 0 if payload["trusted"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
