#!/usr/bin/env python3
"""
Run the 14 grading configs on optimized_transformer.UserOptimizedTransformer.

For each config (subprocess-isolated to avoid compile-state pollution):
  - accuracy: optimized vs the EAGER baseline (the semantic reference)
  - speed  : optimized median vs the torch.compile'd baseline median (the bar)
  - MFU    : model matmul FLOPs / opt_latency / co-temporal fp16 GEMM peak
             (fp16 because the optimized fp32 path runs fp16 tensor-core GEMMs)

  python run_matrix.py            # all 14
  python run_matrix.py --worker i # internal
"""
from __future__ import annotations

import json
import statistics as st
import subprocess
import sys

import torch

import benchmark as B

# batch, seq, heads, ffn, layers, d_model  (all causal, fp32)
CONFIGS = [
    ("01_base",       64,    128, 4,  128,    4, 128),
    ("02_batch1",     1,     128, 4,  128,    4, 128),
    ("03_batch4",     4,     128, 4,  128,    4, 128),
    ("04_batch16",    16,    128, 4,  128,    4, 128),
    ("05_batch128",   128,   128, 4,  128,    4, 128),
    ("06_batch10000", 10000, 128, 4,  128,    4, 128),
    ("07_seq32",      64,    32,  4,  128,    4, 32),
    ("08_seq1024",    64,    1024,4,  128,    4, 1024),
    ("09_heads1",     64,    128, 1,  128,    4, 128),
    ("10_heads2",     64,    128, 2,  128,    4, 128),
    ("11_heads16",    64,    128, 16, 128,    4, 128),
    ("12_ffn32",      64,    128, 4,  32,     4, 128),
    ("13_ffn1024",    64,    128, 4,  1024,   4, 128),
    ("14_extreme",    32,    1024,16, 100000, 2, 1024),
]


def model_flops(B_, S, H, ffn, L, d, causal=True):
    proj = 8 * B_ * S * d * d
    attn = 4 * B_ * S * S * d * (0.5 if causal else 1.0)
    mlp = 4 * B_ * S * d * ffn
    return L * (proj + attn + mlp)


def gemm_peak(dtype=torch.float16, n=4096, it=30):
    a = torch.randn(n, n, device="cuda", dtype=dtype)
    b = torch.randn(n, n, device="cuda", dtype=dtype)
    for _ in range(8):
        c = a @ b
    torch.cuda.synchronize()
    ev = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(it)]
    for s, e in ev:
        s.record(); c = a @ b; e.record()
    torch.cuda.synchronize()
    return 2.0 * n**3 / (st.median(s.elapsed_time(e) for s, e in ev) * 1e-3) / 1e12


def worker(i, which):
    """Time ONE model (opt or cbase) with a SINGLE torch.compile in this process.
    Two reduce-overhead compiles in one process corrupt each other's CUDA graphs,
    so opt and cbase are measured in separate subprocesses."""
    import optimized_transformer as OT
    dev = torch.device("cuda")
    torch.manual_seed(1234)
    name, bs, S, H, ffn, L, d = CONFIGS[i]
    cfg = B.TransformerConfig(batch_size=bs, seq_len=S, d_model=d, num_heads=H,
                              ffn_dim=ffn, num_layers=L, causal=True)
    cfg.validate()
    out = {"name": name, "which": which}
    try:
        ref = B.BaselineTransformer(cfg).to(dev, torch.float32).eval()
        x, m = B.generate_random_case(cfg, dev, torch.float32, 7, 0.0, 1.0)
        if which == "cbase":
            model = torch.compile(ref, mode="reduce-overhead")
        else:
            model = OT.UserOptimizedTransformer(cfg).to(dev, torch.float32).eval()
            B.copy_model_weights(ref, model, True)
            with torch.inference_mode():
                model(x, m)  # warm/compile
                worst = 0.0; ok = True
                for t in range(3):
                    xa, ma = B.generate_random_case(cfg, dev, torch.float32, 20 + t, 0.0, 1.0)
                    r = B.compare_outputs(ref(xa, ma), model(xa, ma), 0.02, 0.002)
                    worst = max(worst, r.max_abs_error); ok &= r.passed
            out["acc_ok"] = bool(ok); out["max_abs"] = worst
        B.warmup_model(model, x, m, 20, dev)
        with torch.inference_mode():
            t = st.median(B.benchmark_once(model, x, m, 60, dev))
        out["ms"] = t
        if which == "opt":
            peak = gemm_peak()
            fl = model_flops(bs, S, H, ffn, L, d)
            out["tflops"] = fl / (t * 1e-3) / 1e12
            out["mfu"] = fl / (t * 1e-3) / peak / 1e12 * 100
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {str(e)[:80]}"
    print("RESULT " + json.dumps(out))


def _run(i, which):
    p = subprocess.run([sys.executable, __file__, "--worker", str(i), which],
                       capture_output=True, text=True)
    line = next((l for l in p.stdout.splitlines() if l.startswith("RESULT ")), None)
    return json.loads(line[len("RESULT "):]) if line else {"error": "crash"}


def driver():
    print(f"{'config':14s} {'acc':4s} {'max_abs':>9s} {'opt_ms':>9s} {'cbase_ms':>9s} {'speedup':>8s} {'MFU%':>7s}")
    rows = []
    for i in range(len(CONFIGS)):
        o = _run(i, "opt")            # separate process
        c = _run(i, "cbase")          # separate process
        name = CONFIGS[i][0]
        if "error" in o or "ms" not in o:
            print(f"{name:14s} opt-error: {o.get('error','?')[:55]}"); rows.append({"name": name, **o}); continue
        if "error" in c or "ms" not in c:
            print(f"{name:14s} {'OK' if o.get('acc_ok') else 'FAIL':4s} {o['max_abs']:9.2e} "
                  f"{o['ms']:9.3f} {'cbase-OOM':>9s}  (MFU {o['mfu']:.1f}%)")
            rows.append({"name": name, **o, "speedup": None}); continue
        sp = c["ms"] / o["ms"]
        rows.append({"name": name, "acc_ok": o.get("acc_ok"), "max_abs": o["max_abs"],
                     "opt_ms": o["ms"], "cbase_ms": c["ms"], "speedup": sp, "mfu": o["mfu"]})
        print(f"{name:14s} {'OK' if o.get('acc_ok') else 'FAIL':4s} {o['max_abs']:9.2e} "
              f"{o['ms']:9.3f} {c['ms']:9.3f} {sp:7.2f}x {o['mfu']:6.1f}%")
    good = [r for r in rows if r.get("speedup") and r.get("acc_ok")]
    if good:
        geo = st.geometric_mean([r["speedup"] for r in good])
        print(f"\n{len(good)}/{len(CONFIGS)} pass+timed | geomean speedup vs compiled baseline = {geo:.3f}x | "
              f"mean MFU = {st.mean(r['mfu'] for r in good):.1f}%")


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "--worker":
        worker(int(sys.argv[2]), sys.argv[3])
    else:
        driver()
