#!/usr/bin/env python3
"""Reproducible driver for torch_transformer_benchmark.py.

MEASUREMENT PROTOCOL (why each case is run twice)
-------------------------------------------------
torch.compile changes the BASELINE's own fp16/bf16 numerics by more than the
harness tolerance (measured: bf16 max_abs 0.094, 12% of elements fail; fp16
max_abs 0.0117). So "eager baseline" and "compiled baseline" are two mutually
incompatible references -- no implementation can be within tolerance of both.

We therefore split the two things the harness conflates:

  * CORRECTNESS is judged against the EAGER baseline (the harness's default
    configuration and the true semantic reference)  -> run without
    --compile-baseline.
  * SPEED is judged against the COMPILED baseline (the stated bar:
    torch.compile) -> run with --compile-baseline, and we take only its
    baseline latency.

speedup = compiled_baseline_median_ms / optimized_median_ms

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
    ap.add_argument("--compile-user", action="store_true",
                    help="also wrap the user model in torch.compile externally")
    ap.add_argument("--skip-speed-bar", action="store_true",
                    help="skip the compiled-baseline run (correctness only)")
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
        print(f"[run_bench] {name} (correctness vs eager baseline) ...", flush=True)
        acc_run = run_case(name, extra, False, args.compile_user, args.timeout)

        bar = None
        if not args.skip_speed_bar:
            print(f"[run_bench] {name} (speed bar: compiled baseline) ...", flush=True)
            bar_run = run_case(name, extra, True, args.compile_user, args.timeout)
            bar = bar_run.get("baseline_ms")

        opt_ms = acc_run.get("optimized_ms")
        r = {
            "name": name,
            "cmd": acc_run.get("cmd"),
            "accuracy": acc_run.get("accuracy"),
            "max_abs": acc_run.get("max_abs"),
            "max_rel": acc_run.get("max_rel"),
            "eager_baseline_ms": acc_run.get("baseline_ms"),
            "compiled_baseline_ms": bar,
            "optimized_ms": opt_ms,
            "speedup_vs_compiled": (bar / opt_ms) if (bar and opt_ms) else None,
            "speedup_vs_eager": ((acc_run.get("baseline_ms") or 0) / opt_ms)
                                 if opt_ms else None,
            "wall_s": acc_run["wall_s"],
        }
        results.append(r)
        print(f"[run_bench]   acc={r['accuracy']} opt={opt_ms}ms "
              f"eager_base={r['eager_baseline_ms']}ms compiled_base={bar}ms "
              f"speedup_vs_compiled={r['speedup_vs_compiled']}", flush=True)
        if acc_run.get("log"):
            print(acc_run["log"], flush=True)

    print(chr(10) + "=" * 92)
    print(f"SUITE={args.suite}  TAG={tag}")
    print("correctness vs EAGER baseline | speed vs COMPILED baseline")
    print("=" * 92)
    hdr = (f"{'case':<18} {'acc':<5} {'eager_ms':>9} {'compiled_ms':>12} "
           f"{'opt_ms':>9} {'vs_compiled':>12} {'vs_eager':>10}")
    print(hdr)
    print("-" * len(hdr))
    nan = float("nan")
    for r in results:
        print(f"{r['name']:<18} {str(r['accuracy']):<5} "
              f"{r['eager_baseline_ms'] or nan:>9.4f} "
              f"{r['compiled_baseline_ms'] or nan:>12.4f} "
              f"{r['optimized_ms'] or nan:>9.4f} "
              f"{r['speedup_vs_compiled'] or nan:>11.3f}x "
              f"{r['speedup_vs_eager'] or nan:>9.3f}x")

    ok = [r for r in results
          if r["accuracy"] == "PASS" and r["speedup_vs_compiled"]]
    print("-" * len(hdr))
    if ok:
        g = statistics.geometric_mean([r["speedup_vs_compiled"] for r in ok])
        print(f"geomean speedup vs COMPILED baseline over {len(ok)}/{len(results)} "
              f"passing cases: {g:.3f}x")
    n_pass = sum(1 for r in results if r["accuracy"] == "PASS")
    print(f"accuracy: {n_pass}/{len(results)} cases PASS vs eager baseline")

    outdir = Path("results")
    outdir.mkdir(exist_ok=True)
    path = outdir / f"{tag.replace('/', '_')}_{args.suite}.json"
    path.write_text(json.dumps(results, indent=2))
    print(f"{chr(10)}wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
