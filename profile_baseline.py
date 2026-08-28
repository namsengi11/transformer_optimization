#!/usr/bin/env python3
"""Kernel-level profile of the compiled baseline to find the bottleneck."""
import sys, torch
from torch.profiler import profile, ProfilerActivity
import torch_transformer_benchmark as B

def main():
    dtype = {"float32": torch.float32, "float16": torch.float16,
             "bfloat16": torch.bfloat16}[sys.argv[1] if len(sys.argv) > 1 else "float32"]
    dev = torch.device("cuda")
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    cfg = B.TransformerConfig(8, 128, 512, 8, 2048, 6, False)
    m = B.BaselineTransformer(cfg).to(dev, dtype).eval()
    m = torch.compile(m)
    x, mask = B.generate_random_case(cfg, dev, dtype, 1234, 0.0, 1.0)
    with torch.inference_mode():
        for _ in range(20):
            m(x, mask)
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            for _ in range(20):
                m(x, mask)
            torch.cuda.synchronize()
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=25))
    evs = [e for e in prof.key_averages() if e.device_time_total > 0 and e.key.startswith(("void","triton","sm","cutlass","std::","at::","ampere","cudnn","<unnamed>","flash","_"))]
    tot = sum(e.device_time_total for e in evs)
    print(f"\ntotal device time / iter = {tot/20:.1f} us over {len(evs)} distinct kernels")
    n = sum(e.count for e in evs)
    print(f"kernel launches / iter = {n/20:.1f}")

main()
