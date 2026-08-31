#!/usr/bin/env python3
"""CUDA-graph microbenchmark for sanctioned single-kernel targets."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parents[2]
if sys.path and Path(sys.path[0]).resolve() == Path(__file__).resolve().parent:
    sys.path.pop(0)
sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.nn.functional as F

import torch_transformer_benchmark as B
from agent.tools._common import (ForeignLoadSampler, disable_static_cuda_launcher,
                                 ranges_overlap, stamp, summarize, wait_for_idle)
from agent.tools.paths import agent_path

DTYPES = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}


def _time_graph(graph: torch.cuda.CUDAGraph, replays: int) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(replays):
        graph.replay()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000.0 / replays


def _capture(call: Callable[[], torch.Tensor]) -> tuple[torch.cuda.CUDAGraph, torch.Tensor]:
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = call()
    return graph, output


def _short_attn(args: argparse.Namespace) -> tuple[Callable[[], torch.Tensor], dict[str, Any]]:
    dtype = DTYPES[args.dtype]
    torch.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)
    shape = (args.batch_size, args.heads, args.seq_len, args.head_dim)
    q = torch.randn(shape, device="cuda", dtype=dtype)
    k = torch.randn(shape, device="cuda", dtype=dtype)
    v = torch.randn(shape, device="cuda", dtype=dtype)
    scale = args.head_dim ** -0.5

    def call() -> torch.Tensor:
        return B._triton_short_attention(q, k, v, scale, args.causal, False, None)

    # The eager SDPA result is validation only, never used as a timing arm.
    with torch.inference_mode():
        candidate = call()
        reference = F.scaled_dot_product_attention(q, k, v, is_causal=args.causal, scale=scale)
        reference = reference.transpose(1, 2).contiguous().view(
            args.batch_size, args.seq_len, args.heads * args.head_dim
        )
        max_abs = float((candidate - reference).abs().max().item())
    return call, {"shape": list(shape), "causal": args.causal, "max_abs_vs_sdpa": max_abs}


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    idle = wait_for_idle(timeout_s=args.idle_timeout, verbose=False)
    if not idle.get("idle"):
        raise RuntimeError(idle.get("reason", "GPU did not become idle"))
    disable_static_cuda_launcher()
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    if args.target == "short_attn":
        call, target_info = _short_attn(args)
    else:
        raise ValueError(f"unsupported target: {args.target}")

    with ForeignLoadSampler() as sampler:
        with torch.inference_mode():
            for _ in range(args.warmup):
                call()
            torch.cuda.synchronize()
            graph, _ = _capture(call)
            empty_graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(empty_graph):
                # Intentionally capture no device work: this measures graph-replay
                # scheduling overhead without subtracting a real CUDA kernel.
                pass
            torch.cuda.synchronize()
            samples: list[float] = []
            empty_samples: list[float] = []
            for _ in range(args.trials):
                measured = _time_graph(graph, args.replays)
                empty = _time_graph(empty_graph, args.replays)
                empty_samples.append(empty)
                samples.append(max(0.0, measured - empty))
    load = sampler.report()
    load["idle_gate"] = idle

    payload: dict[str, Any] = {
        "provenance": stamp("agent/tools/microbench.py", {"invocation": " ".join(sys.argv)}),
        "target": args.target,
        "target_info": target_info,
        "protocol": {
            "seed": 1234,
            "warmup": args.warmup,
            "trials": args.trials,
            "back_to_back_replays": args.replays,
            "empty_graph_replay_subtracted": True,
        },
        "timing_us": {"samples": samples, "summary": summarize(samples)},
        "empty_graph_us": {"samples": empty_samples, "summary": summarize(empty_samples)},
        "gpu_load": load,
        "trusted": load["trusted"],
    }
    if args.compare:
        previous = json.loads(args.compare.read_text())
        old_samples = previous.get("timing_us", {}).get("samples", [])
        overlap = ranges_overlap(samples, old_samples)
        payload["comparison"] = {
            "source": str(args.compare),
            "ranges_overlap": overlap,
            "verdict": "noise" if overlap else (
                "improved" if max(samples) < min(old_samples) else "regressed"
            ),
        }
    return payload


def _human(payload: dict[str, Any]) -> None:
    summary = payload["timing_us"]["summary"]
    print(f"target={payload['target']} CUDA-graph replay, empty replay subtracted")
    print(f"  kernel us min/median/max: {summary['min']:.3f}/{summary['median']:.3f}/{summary['max']:.3f}")
    print(f"  max_abs vs SDPA: {payload['target_info'].get('max_abs_vs_sdpa')}")
    if "comparison" in payload:
        print(f"  comparison: {payload['comparison']['verdict']}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", choices=("short_attn",), default="short_attn")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--head-dim", type=int, default=32)
    parser.add_argument("--dtype", choices=sorted(DTYPES), default="float32")
    parser.add_argument("--causal", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--replays", type=int, default=100)
    parser.add_argument("--idle-timeout", type=int, default=900)
    parser.add_argument("--compare", type=Path)
    parser.add_argument("--output", type=agent_path)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    if args.trials < 3 or args.replays < 2:
        parser.error("require --trials >= 3 and --replays >= 2")
    try:
        payload = run(args)
    except (RuntimeError, ValueError, OSError, json.JSONDecodeError) as exc:
        print(f"microbench refused: {exc}", file=sys.stderr)
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
