#!/usr/bin/env python3
"""Reproducible driver for torch_transformer_benchmark.py.

MEASUREMENT PROTOCOL
--------------------
torch.compile changes the BASELINE's own fp16/bf16 numerics by more than the
harness tolerance (measured: bf16 max_abs 0.094, 12% of elements fail; fp16
max_abs 0.0117). So "eager baseline" and "compiled baseline" are two mutually
incompatible references -- no implementation can be within tolerance of both.

The default report therefore uses one harness run to obtain both requested
latencies against the same inputs and weights:

  * BASELINE is the EAGER original implementation (the harness's default
    configuration and the true semantic reference).
  * SOTA is the fully optimized user implementation.

When ``--speed-bar`` is requested, a separate compiled-baseline run is added
for the historical comparison bar. Its output is never used as the numerical
correctness reference.

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

By default, ``python run_bench.py`` runs the canonical user matrix and reports
the two requested latency measurements for every shape:

  * baseline: naive eager FP32 ``BaselineTransformer``;
  * sota: fully optimized ``UserOptimizedTransformer`` workflow.

The historical compiled-baseline speed bar is available with ``--speed-bar``.

Usage:
    python run_bench.py [--suite user_matrix|default|full|quick] [--tag NAME]
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
    """Select the installed-VRAM streaming protocol before launching a child.

    Step 21 calibrated this estimate with ``agent/tools/capacity_probe.py``.
    The linear term protects input/output/QKV/FFN residency while the
    quadratic term protects dense attention. A B*S-only gate is insufficient:
    equal-token probes ranged from 76 MB to 2.26 GB as S grew.
    """
    shape = resolve_shape(extra)
    element_bytes = 4 if shape["dtype"] == "float32" else 2
    rows = shape["batch_size"] * shape["seq_len"]
    linear_bytes = rows * shape["d_model"] * element_bytes
    ffn_bytes = rows * shape["ffn_dim"] * element_bytes
    score_bytes = (
        shape["batch_size"]
        * shape["heads"]
        * shape["seq_len"]
        * shape["seq_len"]
        * element_bytes
    )
    device_bytes = _gpu_total_memory_bytes()
    estimated_dense_bytes = 6 * linear_bytes + 2 * score_bytes + 2 * ffn_bytes
    budget_bytes = int(device_bytes * 0.85) if device_bytes is not None else None
    if budget_bytes is None or estimated_dense_bytes <= budget_bytes:
        return None
    return {
        "status": "STREAMING_REQUIRED",
        "reason": (
            "estimated dense peak exceeds the calibrated installed-VRAM budget"
        ),
        "rows_b_times_s": rows,
        "linear_tensor_bytes": linear_bytes,
        "score_tensor_bytes": score_bytes,
        "ffn_tensor_bytes": ffn_bytes,
        "estimated_dense_bytes": estimated_dense_bytes,
        "streaming_budget_bytes": budget_bytes,
        "gpu_total_memory_bytes": device_bytes,
    }

# name -> extra CLI args. The direct default compares eager baseline with SOTA;
# --speed-bar optionally adds the historical compiled-baseline measurement.
SUITES = {
    "quick": {
        "default_fp32": [],
    },
    "default": {
        # The script's own default configuration, reduced to the two cases the
        # headline report is about: the naive fp32 reference and the fp16 path
        # this project ships. The bf16/causal/padded variants still exist --
        # they moved to EXTENDED_CASES below and are merged back in with
        # --extended. Nothing was deleted, only hidden by default.
        "default_fp32":      [],
        "default_fp16":      ["--dtype", "float16"],
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

# Dtype and masking variants of the default shape. Excluded from the `default`
# suite's headline report (which is baseline-vs-shipped only) and merged back in
# with --extended. The `full` suite always contains them in its own right.
EXTENDED_CASES = {
    "default_bf16":      ["--dtype", "bfloat16"],
    "causal_fp16":       ["--dtype", "float16", "--causal"],
    "padded_fp16":       ["--dtype", "float16", "--padding-ratio", "0.4"],
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
    ap.add_argument("--suite", default="user_matrix", choices=sorted(SUITES),
                    help="shape suite (default: canonical user_matrix)")
    ap.add_argument("--tag", default=None, help="label for the results file")
    ap.add_argument("--compile-user", action="store_true",
                    help="also wrap the user model in torch.compile externally")
    speed_bar_group = ap.add_mutually_exclusive_group()
    speed_bar_group.add_argument(
        "--speed-bar", action="store_true",
        help="also measure the historical compiled-baseline comparison bar",
    )
    speed_bar_group.add_argument(
        "--skip-speed-bar", action="store_true", help=argparse.SUPPRESS,
    )
    ap.add_argument("--bar-reps", type=int, default=2,
                    help="runs of the compiled baseline; the fastest is used "
                         "as the bar (conservative)")
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--case", action="append", default=None,
                    help="run only these case names")
    ap.add_argument("--extended", action="store_true",
                    help="add the bf16 / causal / padded variants "
                         "(default suite only; `full` already includes them)")
    ap.add_argument("--show-bar", action="store_true",
                    help="also report the compiled-baseline speed bar columns; "
                         "requires --speed-bar (the measurement is always "
                         "written to JSON when enabled)")
    args = ap.parse_args()

    cases = dict(SUITES[args.suite])
    if args.extended and args.suite == "default":
        cases.update(EXTENDED_CASES)
    if args.case:
        cases = {k: v for k, v in cases.items() if k in set(args.case)}

    tag = args.tag or subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        capture_output=True, text=True).stdout.strip() or "run"

    results = []
    for name, extra in cases.items():
        preflight = preflight_case(extra)
        streaming = preflight is not None
        if preflight is not None:
            print(
                f"[run_bench] {name} STREAMING_REQUIRED: "
                f"estimated={preflight['estimated_dense_bytes']} bytes > "
                f"budget={preflight['streaming_budget_bytes']} bytes",
                flush=True,
            )
        reference_name = "query-tiled eager baseline" if streaming else "eager baseline"
        print(f"[run_bench] {name} (correctness vs {reference_name}) ...", flush=True)
        acc_run = run_case(
            name, extra, False, args.compile_user and not streaming, args.timeout
        )

        bar = None
        if args.speed_bar and not args.skip_speed_bar and not streaming:
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
            "status": "STREAMED" if streaming else "EXECUTED",
            "accuracy": acc_run.get("accuracy"),
            "max_abs": acc_run.get("max_abs"),
            "max_rel": acc_run.get("max_rel"),
            "eager_baseline_ms": acc_run.get("baseline_ms"),
            "compiled_baseline_ms": bar,
            "optimized_ms": opt_ms,
            # Clear public name for the fully optimized workflow. Keep
            # optimized_ms as a compatibility key for agent tooling and old
            # result consumers.
            "sota_ms": opt_ms,
            "speedup_vs_compiled": (bar / opt_ms) if (bar and opt_ms) else None,
            "speedup_vs_eager": ((acc_run.get("baseline_ms") or 0) / opt_ms)
                                 if opt_ms else None,
            "mfu_optimized": compute_mfu(opt_shape, opt_ms),
            "mfu_eager_baseline": compute_mfu(shape, acc_run.get("baseline_ms")),
            "mfu_compiled_baseline": compute_mfu(shape, bar),
            "fp16_gemm_enabled": acc_run.get("fp16_gemm_enabled"),
            "wall_s": acc_run["wall_s"],
            "shape": shape,
            "preflight": preflight,
        }
        # Headline improvement, naive baseline -> shipped model.
        #
        # Deliberately no MFU-gain ratio: with the model FLOPs identical on
        # both sides it reduces to the latency ratio scaled by the ratio of the
        # two peaks (exactly speedup/2 whenever the baseline runs TF32 and the
        # optimized path runs fp16 tensor cores), so it restates the latency
        # column rather than adding information. MFU is reported as a LEVEL --
        # mfu_eager_baseline and mfu_optimized -- which is what actually says
        # how close each side runs to its own ceiling.
        r["latency_gain_vs_eager"] = r["speedup_vs_eager"]
        results.append(r)
        print(f"[run_bench]   acc={r['accuracy']} sota={opt_ms}ms "
              f"eager_base={r['eager_baseline_ms']}ms compiled_base={bar}ms "
              f"speedup_vs_compiled={r['speedup_vs_compiled']} "
              f"mfu_optimized={r['mfu_optimized']}", flush=True)
        if acc_run.get("log"):
            print(acc_run["log"], flush=True)

    def fmt_ms(v):
        """Match the reading precision to the magnitude: 3 decimals under
        100 ms, 1 decimal and thousands separators above (a 5,830.1 ms row
        does not need microsecond digits)."""
        if not v:
            return "--"
        return f"{v:,.1f} ms" if v >= 100 else f"{v:.3f} ms"

    def row_label(name, i):
        """Leading digits of a matrix case name ('07_dmodel32' -> '7'); the
        bare name for suites whose cases are not numbered."""
        m = re.match(r"0*(\d+)", name)
        return m.group(1) if m else name

    cols = [("#", 5), ("B", 7), ("S", 6), ("d", 6), ("H", 4), ("L", 4),
            ("ffn", 6), ("accuracy", 9), ("max_abs", 10),
            ("baseline", 12), ("sota", 12), ("speedup", 9), ("mfu", 8)]
    if args.show_bar:
        cols += [("compiled", 12), ("vs_comp", 9)]
    hdr = " ".join(f"{h:>{w}}" for h, w in cols)
    print(chr(10) + "=" * len(hdr))
    print(f"SUITE={args.suite}  TAG={tag}")
    print("baseline = naive eager original (query-tiled equivalent if required) | "
          "sota = fully optimized workflow")
    if not args.speed_bar:
        print("compiled-baseline bar not measured; --speed-bar to include it")
    elif not args.show_bar:
        print("compiled-baseline bar measured and stored in the JSON; "
              "--show-bar to print it")
    print("=" * len(hdr))
    print(hdr)
    print("-" * len(hdr))
    for i, r in enumerate(results, 1):
        s = r["shape"]
        acc = r["accuracy"] or ("STREAMED" if r["status"] == "STREAMED"
                                else "ERR")
        try:
            ma = f"{float(r['max_abs']):.2e}" if r["max_abs"] else "--"
        except (TypeError, ValueError):
            ma = str(r["max_abs"])
        vals = [row_label(r["name"], i), s["batch_size"], s["seq_len"],
                s["d_model"], s["heads"], s["layers"], s["ffn_dim"], acc, ma,
                fmt_ms(r["eager_baseline_ms"]), fmt_ms(r["sota_ms"]),
                (f"{r['latency_gain_vs_eager']:.2f}x"
                 if r["latency_gain_vs_eager"] else "--"),
                (f"{100*r['mfu_optimized']:.2f}%"
                 if r["mfu_optimized"] else "--")]
        if args.show_bar:
            vals += [fmt_ms(r["compiled_baseline_ms"]),
                     (f"{r['speedup_vs_compiled']:.2f}x"
                      if r["speedup_vs_compiled"] else "--")]
        print(" ".join(f"{str(v):>{w}}" for v, (_, w) in zip(vals, cols)))

    print("-" * len(hdr))
    lat_ok = [r for r in results
              if r["accuracy"] == "PASS" and r["latency_gain_vs_eager"]]
    if lat_ok:
        g = statistics.geometric_mean([r["latency_gain_vs_eager"] for r in lat_ok])
        print(f"geomean LATENCY improvement vs naive baseline over "
              f"{len(lat_ok)}/{len(results)} passing cases: {g:.3f}x")
    mfu_ok = [r["mfu_optimized"] for r in results if r.get("mfu_optimized")]
    if mfu_ok:
        print(f"average MFU (ours) across {len(mfu_ok)}/{len(results)} "
              f"cases: {100*statistics.fmean(mfu_ok):.2f}%")
    mfu_base_ok = [r["mfu_eager_baseline"] for r in results
                   if r.get("mfu_eager_baseline")]
    if mfu_base_ok:
        print(f"average MFU (naive baseline) across {len(mfu_base_ok)}/"
              f"{len(results)} cases: {100*statistics.fmean(mfu_base_ok):.2f}%")
    if args.show_bar:
        ok = [r for r in results
              if r["accuracy"] == "PASS" and r["speedup_vs_compiled"]]
        if ok:
            g = statistics.geometric_mean([r["speedup_vs_compiled"] for r in ok])
            print(f"geomean LATENCY RATIO vs COMPILED baseline over "
                  f"{len(ok)}/{len(results)} passing cases: {g:.3f}x")
        mfu_bar_ok = [r["mfu_compiled_baseline"] for r in results
                      if r.get("mfu_compiled_baseline")]
        if mfu_bar_ok:
            print(f"average MFU (compiled baseline) across {len(mfu_bar_ok)}/"
                  f"{len(results)} cases: "
                  f"{100*statistics.fmean(mfu_bar_ok):.2f}%")
    executed = [r for r in results if r["status"] in ("EXECUTED", "STREAMED")]
    n_pass = sum(1 for r in executed if r["accuracy"] == "PASS")
    n_streamed = sum(1 for r in results if r["status"] == "STREAMED")
    print(f"accuracy: {n_pass}/{len(executed)} executed cases PASS vs eager baseline")
    if n_streamed:
        print(f"streaming: {n_streamed}/{len(results)} cases used capacity streaming")

    outdir = SCRIPT.parent / "agent" / "results"
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / f"{tag.replace('/', '_')}_{args.suite}.json"
    path.write_text(json.dumps(results, indent=2))
    print(f"{chr(10)}wrote {path}")
    return 0 if n_pass == len(executed) else 2


if __name__ == "__main__":
    raise SystemExit(main())
