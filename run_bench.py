#!/usr/bin/env python3
"""Reproducible driver for torch_transformer_benchmark.py.

Runs a fixed suite of shapes/dtypes with the baseline torch.compile'd
(the stated bar: torch.compile + SDPA/flash), parses the reported speedup,
and prints a summary table plus a geometric mean.

Usage:
    python run_bench.py [--suite default|full|quick] [--tag NAME]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path

SCRIPT = Path(__file__).parent / "torch_transformer_benchmark.py"

# name -> extra CLI args. Every case compiles the BASELINE (the bar to beat).
SUITES = {
    "quick": {
        "default_fp32": [],
    },
    "default": {
        # the script's own default configuration
        "default_fp32":      [],
        "default_fp16":      ["--dtype", "float16"],
        "default_bf16":      ["--dtype", "bfloat16"],
        "causal_fp16":       ["--dtype", "float16", "--causal"],
        "padded_fp16":       ["--dtype", "float16", "--padding-ratio", "0.4"],
    },
    "full": {
        "default_fp32":      [],
        "default_fp16":      ["--dtype", "float16"],
        "default_bf16":      ["--dtype", "bfloat16"],
        "causal_fp16":       ["--dtype", "float16", "--causal"],
        "padded_fp16":       ["--dtype", "float16", "--padding-ratio", "0.4"],
        "big_fp16":          ["--dtype", "float16", "--batch-size", "32",
                              "--seq-len", "512"],
        "big_causal_fp16":   ["--dtype", "float16", "--batch-size", "32",
                              "--seq-len", "512", "--causal"],
        "long_fp16":         ["--dtype", "float16", "--batch-size", "4",
                              "--seq-len", "2048"],
        "small_fp16":        ["--dtype", "float16", "--batch-size", "1",
                              "--seq-len", "64"],
        "wide_fp16":         ["--dtype", "float16", "--d-model", "1024",
                              "--heads", "16", "--ffn-dim", "4096",
                              "--layers", "12", "--batch-size", "16",
                              "--seq-len", "256"],
    },
}

SPEEDUP_RE = re.compile(r"speedup\s*:\s*([0-9.]+)x")
BASE_RE = re.compile(r"baseline\s*:\s*median=([0-9.]+) ms")
OPT_RE = re.compile(r"optimized:\s*median=([0-9.]+) ms")
ACC_RE = re.compile(r"summary:\s*(PASS|FAIL)\s*\|\s*max_abs=(\S+)\s*\|\s*max_rel=(\S+)")


def run_case(name: str, extra: list[str], compile_baseline: bool,
             compile_user: bool, timeout: int) -> dict:
    cmd = [sys.executable, str(SCRIPT), *extra]
    if compile_baseline:
        cmd.append("--compile-baseline")
    if compile_user:
        cmd.append("--compile-user")
    cmd.append("--benchmark-on-failure")

    env = dict(os.environ)
    env.setdefault("PYTHONWARNINGS", "ignore")
    t0 = time.time()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              timeout=timeout, env=env)
        out = proc.stdout + proc.stderr
    except subprocess.TimeoutExpired as exc:
        out = (exc.stdout or "") + (exc.stderr or "")
        return {"name": name, "error": "timeout", "log": out[-4000:],
                "wall_s": time.time() - t0}

    def grab(rx, cast=float, group=1):
        m = rx.search(out)
        return cast(m.group(group)) if m else None

    acc = ACC_RE.search(out)
    return {
        "name": name,
        "cmd": " ".join(cmd[1:]),
        "speedup": grab(SPEEDUP_RE),
        "baseline_ms": grab(BASE_RE),
        "optimized_ms": grab(OPT_RE),
        "accuracy": acc.group(1) if acc else None,
        "max_abs": acc.group(2) if acc else None,
        "max_rel": acc.group(3) if acc else None,
        "returncode": proc.returncode,
        "wall_s": round(time.time() - t0, 1),
        "log": out[-4000:] if proc.returncode not in (0, 2) else "",
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="default", choices=sorted(SUITES))
    ap.add_argument("--tag", default=None, help="label for the results file")
    ap.add_argument("--no-compile-baseline", action="store_true",
                    help="do NOT torch.compile the baseline (easier bar)")
    ap.add_argument("--compile-user", action="store_true",
                    help="also wrap the user model in torch.compile externally")
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--case", action="append", default=None,
                    help="run only these case names")
    args = ap.parse_args()

    cases = SUITES[args.suite]
    if args.case:
        cases = {k: v for k, v in cases.items() if k in set(args.case)}

    tag = args.tag or subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        capture_output=True, text=True).stdout.strip() or "run"

    results = []
    for name, extra in cases.items():
        print(f"[run_bench] {name} ...", flush=True)
        r = run_case(name, extra, not args.no_compile_baseline,
                     args.compile_user, args.timeout)
        results.append(r)
        s = r.get("speedup")
        print(f"[run_bench]   speedup={s} acc={r.get('accuracy')} "
              f"base={r.get('baseline_ms')}ms opt={r.get('optimized_ms')}ms "
              f"({r['wall_s']}s)", flush=True)
        if r.get("log"):
            print(r["log"], flush=True)

    ok = [r for r in results if r.get("speedup") and r.get("accuracy") == "PASS"]
    print("\n" + "=" * 78)
    print(f"SUITE={args.suite}  TAG={tag}  "
          f"(baseline {'NOT ' if args.no_compile_baseline else ''}compiled)")
    print("=" * 78)
    hdr = f"{'case':<18} {'acc':<5} {'base_ms':>9} {'opt_ms':>9} {'speedup':>9} {'max_rel':>10}"
    print(hdr)
    print("-" * len(hdr))
    for r in results:
        print(f"{r['name']:<18} {str(r.get('accuracy')):<5} "
              f"{r.get('baseline_ms') or float('nan'):>9.4f} "
              f"{r.get('optimized_ms') or float('nan'):>9.4f} "
              f"{r.get('speedup') or float('nan'):>8.3f}x "
              f"{str(r.get('max_rel')):>10}")
    if ok:
        gmean = statistics.geometric_mean([r["speedup"] for r in ok])
        print("-" * len(hdr))
        print(f"geomean speedup over {len(ok)}/{len(results)} passing cases: {gmean:.3f}x")

    outdir = Path("results")
    outdir.mkdir(exist_ok=True)
    path = outdir / f"{tag.replace('/', '_')}_{args.suite}.json"
    path.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
