#!/usr/bin/env python3
"""Tier-1 torch.profiler breakdown for baseline, compiled, or optimized arms."""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if sys.path and Path(sys.path[0]).resolve() == Path(__file__).resolve().parent:
    sys.path.pop(0)  # avoid shadowing the stdlib profile module
sys.path.insert(0, str(REPO_ROOT))

import torch
from torch.profiler import ProfilerActivity, profile

import run_bench
import torch_transformer_benchmark as B
from agent.tools._common import stamp
from agent.tools.paths import agent_path

DTYPES = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}


def _shape_from_args(args: argparse.Namespace) -> dict[str, Any]:
    shape = {
        "batch_size": args.batch_size,
        "seq_len": args.seq_len,
        "d_model": args.d_model,
        "heads": args.heads,
        "ffn_dim": args.ffn_dim,
        "layers": args.layers,
        "dtype": args.dtype,
        "causal": args.causal,
        "padding_ratio": args.padding_ratio,
    }
    if args.case:
        suite, case = args.case.split(":", 1)
        try:
            extra = run_bench.SUITES[suite][case]
        except KeyError as exc:
            raise ValueError(f"unknown suite case {args.case!r}") from exc
        resolved = run_bench.resolve_shape(extra)
        shape.update(resolved)
        shape["causal"] = "--causal" in extra
        if "--padding-ratio" in extra:
            index = extra.index("--padding-ratio")
            shape["padding_ratio"] = float(extra[index + 1])
    return shape


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    shape = _shape_from_args(args)
    dtype = DTYPES[shape["dtype"]]
    device = torch.device("cuda")
    torch.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    config = B.TransformerConfig(
        shape["batch_size"], shape["seq_len"], shape["d_model"], shape["heads"],
        shape["ffn_dim"], shape["layers"], shape["causal"]
    )
    baseline = B.BaselineTransformer(config)
    optimized = B.UserOptimizedTransformer(config)
    B.copy_model_weights(baseline, optimized, strict=True)
    baseline = baseline.to(device=device, dtype=dtype).eval()
    optimized = optimized.to(device=device, dtype=dtype).eval()
    if args.arm == "baseline":
        model = baseline
    elif args.arm == "compiled":
        model = torch.compile(baseline, mode=args.compile_mode)
    else:
        model = optimized
    x, mask = B.generate_random_case(
        config, device, dtype, 1234, shape["padding_ratio"], 1.0
    )
    with torch.inference_mode():
        for _ in range(args.warmup):
            model(x, mask)
        torch.cuda.synchronize()
        if args.external_profiler:
            if args.cuda_profiler_range:
                torch.cuda.cudart().cudaProfilerStart()
            for _ in range(args.iterations):
                model(x, mask)
            torch.cuda.synchronize()
            if args.cuda_profiler_range:
                torch.cuda.cudart().cudaProfilerStop()
            return {
                "provenance": stamp("agent/tools/profile.py", {"invocation": " ".join(sys.argv)}),
                "arm": args.arm,
                "shape": shape,
                "warmup": args.warmup,
                "iterations": args.iterations,
                "external_profiler_workload": True,
                "trusted": True,
            }
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            for _ in range(args.iterations):
                model(x, mask)
            torch.cuda.synchronize()

    kernels: list[dict[str, Any]] = []
    total_device_us = 0.0
    total_launches = 0
    for event in prof.key_averages():
        if getattr(event, "device_type", None) != torch.autograd.DeviceType.CUDA:
            continue
        device_us = float(getattr(event, "self_device_time_total", 0.0) or 0.0)
        if device_us <= 0:
            continue
        if event.key.startswith(("Memcpy", "Memset", "Activity Buffer")):
            continue
        count = int(event.count)
        total_device_us += device_us
        total_launches += count
        kernels.append({
            "name": event.key,
            "device_time_total_us": device_us,
            "device_time_per_iter_us": device_us / args.iterations,
            "calls_total": count,
            "calls_per_iter": count / args.iterations,
        })
    kernels.sort(key=lambda row: row["device_time_total_us"], reverse=True)
    return {
        "provenance": stamp("agent/tools/profile.py", {"invocation": " ".join(sys.argv)}),
        "arm": args.arm,
        "shape": shape,
        "warmup": args.warmup,
        "iterations": args.iterations,
        "device_time_per_iter_us": total_device_us / args.iterations,
        "launches_per_iter": total_launches / args.iterations,
        "kernels": kernels[:args.row_limit],
        "trusted": True,
    }


def _human(payload: dict[str, Any]) -> None:
    print(f"arm={payload['arm']} shape={payload['shape']}")
    print(f"device time/iter={payload['device_time_per_iter_us']:.2f} us | launches/iter={payload['launches_per_iter']:.1f}")
    print(f"{'device us/iter':>15} {'calls/iter':>10}  kernel")
    for row in payload["kernels"]:
        print(f"{row['device_time_per_iter_us']:>15.2f} {row['calls_per_iter']:>10.2f}  {row['name']}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=("baseline", "compiled", "optimized"), default="compiled")
    parser.add_argument("--case", help="suite:case, for example default:default_fp32")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--ffn-dim", type=int, default=2048)
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--causal", action="store_true")
    parser.add_argument("--dtype", choices=sorted(DTYPES), default="float32")
    parser.add_argument("--padding-ratio", type=float, default=0.0)
    parser.add_argument("--compile-mode", choices=("default", "reduce-overhead", "max-autotune"), default="default")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--row-limit", type=int, default=25)
    parser.add_argument("--external-profiler", action="store_true",
                        help="run the workload without torch.profiler for nsys/ncu")
    parser.add_argument("--cuda-profiler-range", action="store_true",
                        help="bracket external workload with cudaProfilerStart/Stop")
    parser.add_argument("--output", type=agent_path)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    try:
        payload = run(args)
    except (RuntimeError, ValueError, KeyError) as exc:
        print(f"profile refused: {exc}", file=sys.stderr)
        return 2
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2, default=str))
    if args.json:
        print(json.dumps(payload, indent=2, default=str))
    else:
        _human(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
