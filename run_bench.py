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

MFU (Model FLOPs Utilization)
------------------------------
Every run now also reports MFU: (dense forward FLOPs / measured latency) as
a fraction of this GPU's peak FLOPs/s at the relevant precision. FLOPs are
the DENSE matmul count (QKV + attn QK^T + attn@V + out_proj + FFN; LayerNorm/
softmax/GELU excluded, the standard MFU convention) -- no discount for
causal masking, since this implementation computes the full [S,S] score
matrix and masks it afterward, matching what's actually executed.

Peak FLOPs/s is an EMPIRICALLY MEASURED reference for this exact card, not a
scraped spec figure: web sources for the RTX 5060 Ti's tensor-core TFLOPS
were inconsistent across sites (47/120/200 TFLOPS all cited for the same
SKU). Measured instead via large (2048/4096/8192-square) matmuls, which is
the same methodology used throughout this project's own GEMM-ceiling work:
TF32 ~24.7 TFLOPS, FP16 ~48.8 TFLOPS, BF16 ~49.4 TFLOPS (fp16≈2x TF32,
consistent with tensor cores wasting half their throughput on TF32's wider
mantissa -- a known, generation-independent ratio, which is a sanity check
that these numbers are real). Re-measure with peak_flops.py-style large
square matmuls if this script ever runs on different hardware.

Usage:
    python run_bench.py [--suite default|full|quick|user_matrix] [--tag NAME]
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

# Empirically measured on this RTX 5060 Ti (sm_120) via large square GEMMs
# (2048/4096/8192), NOT a spec-sheet figure -- see module docstring.
PEAK_TFLOPS = {"float32": 24.7, "float16": 48.8, "bfloat16": 49.4}

# torch_transformer_benchmark.py's own argparse defaults, mirrored here so
# FLOP counting is correct for cases that only override a few flags.
_SHAPE_DEFAULTS = {
    "batch_size": 8, "seq_len": 128, "d_model": 512,
    "heads": 8, "ffn_dim": 2048, "layers": 6, "dtype": "float32",
}
_SHAPE_FLAGS = {
    "--batch-size": ("batch_size", int), "--seq-len": ("seq_len", int),
    "--d-model": ("d_model", int), "--heads": ("heads", int),
    "--ffn-dim": ("ffn_dim", int), "--layers": ("layers", int),
    "--dtype": ("dtype", str),
}


def resolve_shape(extra: list[str]) -> dict:
    """Parse the subset of CLI flags that determine FLOP count / dtype out of
    an `extra` args list, falling back to the script's own defaults for
    anything not overridden."""
    shape = dict(_SHAPE_DEFAULTS)
    i = 0
    while i < len(extra):
        flag = extra[i]
        if flag in _SHAPE_FLAGS and i + 1 < len(extra):
            key, cast = _SHAPE_FLAGS[flag]
            shape[key] = cast(extra[i + 1])
            i += 2
        else:
            i += 1
    return shape


def count_dense_flops(shape: dict) -> int:
    """Dense forward-pass matmul FLOPs (2*M*K*N per GEMM), excluding
    LayerNorm/softmax/GELU per the standard MFU convention. No discount for
    causal masking -- the full [S,S] score matrix is computed either way."""
    B, S, D = shape["batch_size"], shape["seq_len"], shape["d_model"]
    ffn = shape["ffn_dim"]
    per_layer = (
        8 * B * S * D * D        # QKV (3x) + out_proj: 4 GEMMs of [.,D]x[D,D]
        + 4 * B * S * S * D      # attn QK^T + attn@V
        + 4 * B * S * D * ffn    # ffn_in + ffn_out
    )
    return shape["layers"] * per_layer


def compute_mfu(shape: dict, latency_ms: float | None) -> float | None:
    if not latency_ms:
        return None
    flops = count_dense_flops(shape)
    peak = PEAK_TFLOPS.get(shape["dtype"])
    if peak is None:
        return None
    achieved_flops_per_s = flops / (latency_ms / 1000.0)
    return achieved_flops_per_s / (peak * 1e12)


def _gpu_total_memory_bytes() -> int | None:
    """Return the first visible GPU's total memory without creating a CUDA context."""
    try:
        proc = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if proc.returncode != 0:
            return None
        first = proc.stdout.splitlines()[0].strip()
        return int(first) * 1024**2
    except (OSError, ValueError, IndexError, subprocess.TimeoutExpired):
        return None


def preflight_case(extra: list[str]) -> dict | None:
    """Reject a shape whose baseline score tensor cannot fit on the GPU.

    BaselineSelfAttention explicitly materializes one fp32/fp16/bf16
    ``[batch, heads, seq_len, seq_len]`` score tensor. This is a strict lower
    bound, not a fitted peak-memory estimate: if that tensor alone exceeds
    total device memory, launching the child can only OOM.
    """
    shape = resolve_shape(extra)
    element_bytes = 4 if shape["dtype"] == "float32" else 2
    score_bytes = (
        shape["batch_size"]
        * shape["heads"]
        * shape["seq_len"]
        * shape["seq_len"]
        * element_bytes
    )
    device_bytes = _gpu_total_memory_bytes()
    if device_bytes is None or score_bytes <= device_bytes:
        return None
    return {
        "status": "PREFLIGHT_BLOCKED",
        "reason": (
            "one dense baseline attention-score tensor exceeds total GPU memory"
        ),
        "score_tensor_bytes": score_bytes,
        "gpu_total_memory_bytes": device_bytes,
    }

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
    # Authoritative user-supplied grading matrix. The source columns are:
    # case_id, batch_size, d_model (QKV dim), heads, seq_len, layers,
    # ffn_dim, causal. dtype is not a source column, so the harness default
    # fp32 applies. Keep the case names aligned with the axis varied by the
    # source row; see agent/docs/USER_MATRIX.md for the canonical table and
    # the 2026-08-31 schema correction.
    "user_matrix": {
        "01_base":        ["--batch-size", "64",    "--seq-len", "128",  "--heads", "4",  "--ffn-dim", "128",    "--layers", "4", "--causal", "--d-model", "128"],
        "02_batch1":      ["--batch-size", "1",      "--seq-len", "128",  "--heads", "4",  "--ffn-dim", "128",    "--layers", "4", "--causal", "--d-model", "128"],
        "03_batch4":      ["--batch-size", "4",      "--seq-len", "128",  "--heads", "4",  "--ffn-dim", "128",    "--layers", "4", "--causal", "--d-model", "128"],
        "04_batch16":     ["--batch-size", "16",     "--seq-len", "128",  "--heads", "4",  "--ffn-dim", "128",    "--layers", "4", "--causal", "--d-model", "128"],
        "05_batch128":    ["--batch-size", "128",    "--seq-len", "128",  "--heads", "4",  "--ffn-dim", "128",    "--layers", "4", "--causal", "--d-model", "128"],
        "06_batch10000":  ["--batch-size", "10000",  "--seq-len", "128",  "--heads", "4",  "--ffn-dim", "128",    "--layers", "4", "--causal", "--d-model", "128"],
        "07_dmodel32":    ["--batch-size", "64",     "--seq-len", "128",  "--heads", "4",  "--ffn-dim", "32",     "--layers", "4", "--causal", "--d-model", "32"],
        "08_dmodel1024":  ["--batch-size", "64",     "--seq-len", "128",  "--heads", "4",  "--ffn-dim", "1024",   "--layers", "4", "--causal", "--d-model", "1024"],
        "09_heads1":      ["--batch-size", "64",     "--seq-len", "128",  "--heads", "1",  "--ffn-dim", "128",    "--layers", "4", "--causal", "--d-model", "128"],
        "10_heads2":      ["--batch-size", "64",     "--seq-len", "128",  "--heads", "2",  "--ffn-dim", "128",    "--layers", "4", "--causal", "--d-model", "128"],
        "11_heads16":     ["--batch-size", "64",     "--seq-len", "128",  "--heads", "16", "--ffn-dim", "128",    "--layers", "4", "--causal", "--d-model", "128"],
        "12_seq32":       ["--batch-size", "64",     "--seq-len", "32",   "--heads", "4",  "--ffn-dim", "128",    "--layers", "4", "--causal", "--d-model", "128"],
        "13_seq1024":     ["--batch-size", "64",     "--seq-len", "1024", "--heads", "4",  "--ffn-dim", "128",    "--layers", "4", "--causal", "--d-model", "128"],
        "14_extreme":     ["--batch-size", "32",     "--seq-len", "100000", "--heads", "16", "--ffn-dim", "1024", "--layers", "2", "--causal", "--d-model", "1024"],
    },
}

SPEEDUP_RE = re.compile(r"speedup\s*:\s*([0-9.]+)x")
BASE_RE = re.compile(r"baseline\s*:\s*median=([0-9.]+) ms")
OPT_RE = re.compile(r"optimized:\s*median=([0-9.]+) ms")
ACC_RE = re.compile(r"summary:\s*(PASS|FAIL)\s*\|\s*max_abs=(\S+)\s*\|\s*max_rel=(\S+)")
FP16_GATE_RE = re.compile(r"-> (ENABLE|DISABLE) fp16-GEMM")


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
    # Surfaces the fp32 fp16-GEMM policy on stderr, so
    # MFU can be computed against the PRECISION ACTUALLY EXECUTED (fp16
    # tensor cores if the gate enabled it) rather than the model's declared
    # dtype -- without this, an fp32 case running the fp16-GEMM path gets
    # compared against TF32's peak, understating the true speedup and
    # producing an impossible >100% MFU. No effect on fp16/bf16 runs (the
    # gate only exists on the fp32 path) or on --compile-baseline runs
    # (the compiled baseline is BaselineTransformer, which never takes this
    # path either).
    env["TJ_DEBUG_GATE"] = "1"
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
    gate = FP16_GATE_RE.search(out)
    return {
        "name": name,
        "cmd": " ".join(cmd[1:]),
        "speedup": grab(SPEEDUP_RE),
        "baseline_ms": grab(BASE_RE),
        "optimized_ms": grab(OPT_RE),
        "accuracy": acc.group(1) if acc else None,
        "max_abs": acc.group(2) if acc else None,
        "max_rel": acc.group(3) if acc else None,
        "fp16_gemm_enabled": (gate.group(1) == "ENABLE") if gate else None,
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
    ap.add_argument("--bar-reps", type=int, default=2,
                    help="runs of the compiled baseline; the fastest is used "
                         "as the bar (conservative)")
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
        preflight = preflight_case(extra)
        if preflight is not None:
            shape = resolve_shape(extra)
            row = {
                "name": name,
                "cmd": " ".join([str(SCRIPT), *extra]),
                "status": preflight["status"],
                "accuracy": None,
                "max_abs": None,
                "max_rel": None,
                "eager_baseline_ms": None,
                "compiled_baseline_ms": None,
                "optimized_ms": None,
                "speedup_vs_compiled": None,
                "speedup_vs_eager": None,
                "mfu_optimized": None,
                "mfu_compiled_baseline": None,
                "fp16_gemm_enabled": None,
                "wall_s": 0.0,
                "shape": shape,
                "preflight": preflight,
            }
            results.append(row)
            print(
                f"[run_bench] {name} PREFLIGHT_BLOCKED: "
                f"{preflight['score_tensor_bytes']} score bytes > "
                f"{preflight['gpu_total_memory_bytes']} GPU bytes",
                flush=True,
            )
            continue
        print(f"[run_bench] {name} (correctness vs eager baseline) ...", flush=True)
        acc_run = run_case(name, extra, False, args.compile_user, args.timeout)

        bar = None
        if not args.skip_speed_bar:
            # The compiled baseline has ~30% run-to-run variance across processes
            # (observed: padded_fp16 bar 2.03 ms vs 1.53 ms on identical code).
            # Take the FASTEST bar over `--bar-reps` runs, which is the
            # conservative choice: it makes our reported speedup a lower bound.
            bars = []
            for rep in range(args.bar_reps):
                print(f"[run_bench] {name} (speed bar: compiled baseline, "
                      f"rep {rep + 1}/{args.bar_reps}) ...", flush=True)
                bar_run = run_case(name, extra, True, args.compile_user, args.timeout)
                if bar_run.get("baseline_ms"):
                    bars.append(bar_run["baseline_ms"])
            bar = min(bars) if bars else None

        opt_ms = acc_run.get("optimized_ms")
        shape = resolve_shape(extra)
        # MFU must reflect the precision ACTUALLY EXECUTED. The fp32
        # fp16-GEMM policy can switch an fp32 case onto
        # fp16 tensor cores; the compiled baseline never takes that path
        # (it's plain BaselineTransformer, always TF32 for fp32 dtype).
        opt_shape = dict(shape)
        if shape["dtype"] == "float32" and acc_run.get("fp16_gemm_enabled"):
            opt_shape["dtype"] = "float16"
        r = {
            "name": name,
            "cmd": acc_run.get("cmd"),
            "status": "EXECUTED",
            "accuracy": acc_run.get("accuracy"),
            "max_abs": acc_run.get("max_abs"),
            "max_rel": acc_run.get("max_rel"),
            "eager_baseline_ms": acc_run.get("baseline_ms"),
            "compiled_baseline_ms": bar,
            "optimized_ms": opt_ms,
            "speedup_vs_compiled": (bar / opt_ms) if (bar and opt_ms) else None,
            "speedup_vs_eager": ((acc_run.get("baseline_ms") or 0) / opt_ms)
                                 if opt_ms else None,
            "mfu_optimized": compute_mfu(opt_shape, opt_ms),
            "mfu_compiled_baseline": compute_mfu(shape, bar),
            "fp16_gemm_enabled": acc_run.get("fp16_gemm_enabled"),
            "wall_s": acc_run["wall_s"],
        }
        results.append(r)
        print(f"[run_bench]   acc={r['accuracy']} opt={opt_ms}ms "
              f"eager_base={r['eager_baseline_ms']}ms compiled_base={bar}ms "
              f"speedup_vs_compiled={r['speedup_vs_compiled']} "
              f"mfu_optimized={r['mfu_optimized']}", flush=True)
        if acc_run.get("log"):
            print(acc_run["log"], flush=True)

    print(chr(10) + "=" * 108)
    print(f"SUITE={args.suite}  TAG={tag}")
    print("correctness vs NAIVE baseline | speed + MFU vs COMPILED baseline")
    print("=" * 108)
    hdr = (f"{'case':<18} {'acc':<5} {'naive_ms':>10} {'compiled_ms':>12} "
           f"{'ours_ms':>10} {'vs_compiled':>12} {'mfu_ours':>9} {'mfu_bar':>9}")
    print(hdr)
    print("-" * len(hdr))
    nan = float("nan")
    for r in results:
        print(f"{r['name']:<18} {str(r['accuracy']):<5} "
              f"{r['eager_baseline_ms'] or nan:>10.4f} "
              f"{r['compiled_baseline_ms'] or nan:>12.4f} "
              f"{r['optimized_ms'] or nan:>10.4f} "
              f"{r['speedup_vs_compiled'] or nan:>11.3f}x "
              f"{100*(r['mfu_optimized'] or nan):>8.2f}% "
              f"{100*(r['mfu_compiled_baseline'] or nan):>8.2f}%")

    ok = [r for r in results
          if r["accuracy"] == "PASS" and r["speedup_vs_compiled"]]
    print("-" * len(hdr))
    if ok:
        g = statistics.geometric_mean([r["speedup_vs_compiled"] for r in ok])
        print(f"geomean LATENCY RATIO (speedup) vs COMPILED baseline over "
              f"{len(ok)}/{len(results)} passing cases: {g:.3f}x")
    mfu_ok = [r["mfu_optimized"] for r in results if r.get("mfu_optimized")]
    if mfu_ok:
        avg_mfu = statistics.fmean(mfu_ok)
        print(f"average MFU (optimized) across {len(mfu_ok)}/{len(results)} "
              f"cases: {100*avg_mfu:.2f}%")
    mfu_bar_ok = [r["mfu_compiled_baseline"] for r in results if r.get("mfu_compiled_baseline")]
    if mfu_bar_ok:
        avg_mfu_bar = statistics.fmean(mfu_bar_ok)
        print(f"average MFU (compiled baseline) across {len(mfu_bar_ok)}/{len(results)} "
              f"cases: {100*avg_mfu_bar:.2f}%")
    executed = [r for r in results if r["status"] == "EXECUTED"]
    n_pass = sum(1 for r in executed if r["accuracy"] == "PASS")
    n_blocked = sum(1 for r in results if r["status"] == "PREFLIGHT_BLOCKED")
    print(f"accuracy: {n_pass}/{len(executed)} executed cases PASS vs eager baseline")
    if n_blocked:
        print(f"preflight: {n_blocked}/{len(results)} cases structurally blocked")

    outdir = Path("results")
    outdir.mkdir(exist_ok=True)
    path = outdir / f"{tag.replace('/', '_')}_{args.suite}.json"
    path.write_text(json.dumps(results, indent=2))
    print(f"{chr(10)}wrote {path}")
    return 0 if n_pass == len(executed) else 2


if __name__ == "__main__":
    raise SystemExit(main())
