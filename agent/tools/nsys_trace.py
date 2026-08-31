#!/usr/bin/env python3
"""Tier-2 Nsight Systems trace with a parsed CUDA timeline and launch gaps."""
from __future__ import annotations

import argparse
import json
import sqlite3
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from agent.tools._common import stamp
from agent.tools.paths import ARTIFACTS, NSYS_EXE, PYTHON, TOOLS_DIR, agent_path, ensure_dirs


def _columns(conn: sqlite3.Connection, table: str) -> set[str]:
    return {row[1] for row in conn.execute(f'PRAGMA table_info("{table}")')}


def _merge_intervals(intervals: Iterable[tuple[int, int]]) -> list[tuple[int, int]]:
    ordered = sorted((start, end) for start, end in intervals if end > start)
    if not ordered:
        return []
    merged: list[tuple[int, int]] = []
    cur_start, cur_end = ordered[0]
    for start, end in ordered[1:]:
        if start <= cur_end:
            cur_end = max(cur_end, end)
        else:
            merged.append((cur_start, cur_end))
            cur_start, cur_end = start, end
    merged.append((cur_start, cur_end))
    return merged


def _union_duration(intervals: Iterable[tuple[int, int]]) -> int:
    return sum(end - start for start, end in _merge_intervals(intervals))


def _intersection_duration(a: list[tuple[int, int]], b: list[tuple[int, int]]) -> int:
    left = _merge_intervals(a)
    right = _merge_intervals(b)
    total = 0
    i = j = 0
    while i < len(left) and j < len(right):
        a_start, a_end = left[i]
        b_start, b_end = right[j]
        total += max(0, min(a_end, b_end) - max(a_start, b_start))
        if a_end <= b_end:
            i += 1
        else:
            j += 1
    return total


def _resolve_string(conn: sqlite3.Connection, value: Any) -> str:
    if isinstance(value, str):
        return value
    try:
        row = conn.execute("SELECT value FROM StringIds WHERE id = ?", (value,)).fetchone()
        return str(row[0]) if row else str(value)
    except sqlite3.Error:
        return str(value)


def parse_sqlite(path: Path, timeline_limit: int) -> dict[str, Any]:
    conn = sqlite3.connect(path)
    try:
        tables = [row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")]
        kernel_tables = [name for name in tables if "KERNEL" in name.upper() and "ENUM" not in name.upper()]
        kernels: list[tuple[int, int, str]] = []
        for table in kernel_tables:
            cols = _columns(conn, table)
            if not {"start", "end"}.issubset(cols):
                continue
            name_col = next((name for name in ("demangledName", "shortName", "name") if name in cols), None)
            select = f'SELECT start, end{", " + name_col if name_col else ""} FROM "{table}"'
            for row in conn.execute(select):
                name = _resolve_string(conn, row[2]) if len(row) > 2 else table
                kernels.append((int(row[0]), int(row[1]), name))
        kernels.sort(key=lambda item: item[0])
        if not kernels:
            return {"parsed": False, "reason": "no CUDA kernel table in Nsight export", "tables": tables}

        base = kernels[0][0]
        gaps = [max(0, kernels[index][0] - kernels[index - 1][1]) for index in range(1, len(kernels))]
        runtime_tables = [name for name in tables if "RUNTIME" in name.upper()]
        runtime_events: list[tuple[int, int, str]] = []
        for table in runtime_tables:
            cols = _columns(conn, table)
            if not {"start", "end"}.issubset(cols):
                continue
            name_col = next((name for name in ("nameId", "name") if name in cols), None)
            if not name_col:
                continue
            for start, end, name_id in conn.execute(f'SELECT start, end, {name_col} FROM "{table}"'):
                runtime_events.append((int(start), int(end), _resolve_string(conn, name_id)))
        gpu_intervals = [(start, end) for start, end, _ in kernels]
        cpu_intervals = [(start, end) for start, end, _ in runtime_events]
        span = max(end for _, end, _ in kernels) - min(start for start, _, _ in kernels)
        gpu_busy = _union_duration(gpu_intervals)
        overlap = _intersection_duration(gpu_intervals, cpu_intervals)
        timeline = []
        previous_end = base
        for start, end, name in kernels[:timeline_limit]:
            timeline.append({
                "name": name,
                "start_us": (start - base) / 1000.0,
                "duration_us": (end - start) / 1000.0,
                "launch_gap_us": max(0, start - previous_end) / 1000.0,
            })
            previous_end = max(previous_end, end)
        graph_apis = sorted({name for _, _, name in runtime_events if "GraphLaunch" in name})
        return {
            "parsed": True,
            "kernel_count": len(kernels),
            "timeline": timeline,
            "launch_gaps_us": {
                "count": len(gaps),
                "min": min(gaps) / 1000.0 if gaps else 0.0,
                "median": sorted(gaps)[len(gaps) // 2] / 1000.0 if gaps else 0.0,
                "max": max(gaps) / 1000.0 if gaps else 0.0,
            },
            "graph_replay": {"confirmed": bool(graph_apis), "apis": graph_apis},
            "overlap": {
                "gpu_busy_fraction_of_span": gpu_busy / span if span else None,
                "cpu_cuda_api_overlap_fraction_of_gpu_busy": min(overlap, gpu_busy) / gpu_busy if gpu_busy else None,
            },
        }
    finally:
        conn.close()


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not NSYS_EXE:
        return {"provenance": stamp("agent/tools/nsys_trace.py"), "available": False,
                "reason": "Nsight Systems executable not found", "trusted": False}
    ensure_dirs()
    stem = ARTIFACTS / args.tag
    rep = stem.with_suffix(".nsys-rep")
    sqlite_path = stem.with_suffix(".sqlite")
    target = [PYTHON, str(TOOLS_DIR / "profile.py"), "--arm", args.arm,
              "--warmup", str(args.warmup), "--iterations", str(args.iterations),
              "--row-limit", "5", "--external-profiler", "--cuda-profiler-range", "--json"]
    if args.case:
        target += ["--case", args.case]
    else:
        target += ["--batch-size", str(args.batch_size), "--seq-len", str(args.seq_len),
                   "--d-model", str(args.d_model), "--heads", str(args.heads),
                   "--ffn-dim", str(args.ffn_dim), "--layers", str(args.layers),
                   "--dtype", args.dtype]
        if args.causal:
            target.append("--causal")
    # Windows Nsight Systems does not expose the Linux `osrt` trace domain.
    # CUDA tracing still records runtime APIs, kernels, graph launches, and
    # their CPU/GPU intervals; NVTX preserves any application ranges.
    cmd = [NSYS_EXE, "profile", "--trace=cuda,nvtx", "--sample=none",
           "--cuda-event-trace=false",
           "--cuda-graph-trace=node",
           "--capture-range=cudaProfilerApi", "--capture-range-end=stop",
           "--force-overwrite=true", f"--output={stem}", *target]
    proc = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True,
                          errors="replace", timeout=args.timeout)
    combined = proc.stdout + proc.stderr
    if proc.returncode != 0 or not rep.is_file():
        return {"provenance": stamp("agent/tools/nsys_trace.py"), "available": True,
                "captured": False, "command": cmd, "reason": combined[-5000:], "trusted": False}
    export_cmd = [NSYS_EXE, "export", "--type=sqlite", "--force-overwrite=true",
                  f"--output={sqlite_path}", str(rep)]
    exported = subprocess.run(export_cmd, cwd=REPO_ROOT, capture_output=True, text=True,
                              errors="replace", timeout=args.timeout)
    if exported.returncode != 0 or not sqlite_path.is_file():
        return {"provenance": stamp("agent/tools/nsys_trace.py"), "available": True,
                "captured": True, "parsed": False, "report": str(rep),
                "reason": (exported.stdout + exported.stderr)[-5000:], "trusted": False}
    parsed = parse_sqlite(sqlite_path, args.timeline_limit)
    return {
        "provenance": stamp("agent/tools/nsys_trace.py", {"invocation": " ".join(sys.argv)}),
        "available": True,
        "captured": True,
        "report": str(rep),
        "sqlite": str(sqlite_path),
        "command": cmd,
        **parsed,
        "trusted": bool(parsed.get("parsed")),
    }


def _human(payload: dict[str, Any]) -> None:
    if not payload.get("available") or not payload.get("captured"):
        print(f"Nsight Systems unavailable: {payload.get('reason')}")
        return
    print(f"report={payload['report']}")
    if not payload.get("parsed"):
        print(f"parse unavailable: {payload.get('reason')}")
        return
    print(f"kernels={payload['kernel_count']} graph replay={payload['graph_replay']['confirmed']}")
    gaps = payload["launch_gaps_us"]
    print(f"launch gaps us min/median/max={gaps['min']:.3f}/{gaps['median']:.3f}/{gaps['max']:.3f}")
    for row in payload["timeline"][:20]:
        print(f"{row['start_us']:>10.3f} us +{row['duration_us']:>9.3f} us gap={row['launch_gap_us']:>8.3f}  {row['name']}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--arm", choices=("baseline", "compiled", "optimized"), default="optimized")
    parser.add_argument("--case")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--ffn-dim", type=int, default=2048)
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="float32")
    parser.add_argument("--causal", action="store_true")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--timeline-limit", type=int, default=200)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--output", type=agent_path)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    try:
        payload = run(args)
    except (OSError, RuntimeError, sqlite3.Error, subprocess.TimeoutExpired) as exc:
        payload = {"provenance": stamp("agent/tools/nsys_trace.py"), "available": bool(NSYS_EXE),
                   "captured": False, "reason": repr(exc), "trusted": False}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2, default=str))
    if args.json:
        print(json.dumps(payload, indent=2, default=str))
    else:
        _human(payload)
    return 0 if payload.get("trusted") else 2


if __name__ == "__main__":
    raise SystemExit(main())
