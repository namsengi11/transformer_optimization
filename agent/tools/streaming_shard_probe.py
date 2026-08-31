#!/usr/bin/env python3
"""Measure one optimized batch-shard candidate in an isolated CUDA process."""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import torch

import torch_transformer_benchmark as B
from agent.tools._common import stamp, wait_for_idle
from agent.tools.paths import agent_path


DTYPES = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def run(args: argparse.Namespace) -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    idle = wait_for_idle(timeout_s=args.idle_timeout, verbose=False)
    if not idle.get("idle"):
        raise RuntimeError(idle.get("reason", "GPU did not become idle"))
    device = torch.device("cuda")
    dtype = DTYPES[args.dtype]
    config = B.TransformerConfig(
        args.batch_shard,
        args.seq_len,
        args.d_model,
        args.heads,
        args.ffn_dim,
        args.layers,
        args.causal,
    )
    config.validate()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    baseline = B.QueryTiledBaselineTransformer(config, query_tile_size=1)
    model = B.UserOptimizedTransformer(config)
    B.copy_model_weights(baseline, model, strict=True)
    del baseline
    model = model.to(device=device, dtype=dtype).eval()
    x, mask = B.generate_random_case(config, device, dtype, args.seed, 0.0, 1.0)
    payload = {
        "provenance": stamp(
            "agent/tools/streaming_shard_probe.py",
            {"invocation": " ".join(sys.argv)},
        ),
        "shape": {
            "batch_shard": args.batch_shard,
            "seq_len": args.seq_len,
            "d_model": args.d_model,
            "heads": args.heads,
            "ffn_dim": args.ffn_dim,
            "layers": args.layers,
            "causal": args.causal,
            "dtype": args.dtype,
        },
        "target_vram_fraction": args.target_vram_fraction,
        "idle_gate": idle,
        "total_vram_bytes": torch.cuda.get_device_properties(device).total_memory,
    }
    try:
        with torch.inference_mode():
            for _ in range(args.warmup):
                model(x, mask)
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        samples_ms = []
        with torch.inference_mode():
            for _ in range(args.repeats):
                started = torch.cuda.Event(enable_timing=True)
                ended = torch.cuda.Event(enable_timing=True)
                started.record()
                output = model(x, mask)
                ended.record()
                torch.cuda.synchronize(device)
                samples_ms.append(started.elapsed_time(ended))
                del output
        peak_allocated = torch.cuda.max_memory_allocated(device)
        peak_reserved = torch.cuda.max_memory_reserved(device)
        total = payload["total_vram_bytes"]
        payload.update(
            status=(
                "PASS"
                if max(peak_allocated, peak_reserved)
                <= int(total * args.target_vram_fraction)
                else "OVER_TARGET"
            ),
            latency_ms=statistics.median(samples_ms),
            latency_samples_ms=samples_ms,
            peak_allocated_bytes=peak_allocated,
            peak_reserved_bytes=peak_reserved,
        )
    except (torch.OutOfMemoryError, torch.AcceleratorError) as exc:
        payload.update(
            status="OOM",
            error=str(exc),
            peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(device),
        )
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-shard", type=int, required=True)
    parser.add_argument("--seq-len", type=int, required=True)
    parser.add_argument("--d-model", type=int, required=True)
    parser.add_argument("--heads", type=int, required=True)
    parser.add_argument("--ffn-dim", type=int, required=True)
    parser.add_argument("--layers", type=int, required=True)
    parser.add_argument("--causal", action="store_true")
    parser.add_argument("--dtype", choices=sorted(DTYPES), default="float32")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--target-vram-fraction", type=float, default=0.85)
    parser.add_argument("--idle-timeout", type=int, default=900)
    parser.add_argument("--output", type=agent_path, required=True)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    if not 0.0 < args.target_vram_fraction < 1.0:
        parser.error("--target-vram-fraction must be between 0 and 1")
    if args.warmup < 0 or args.repeats <= 0:
        parser.error("--warmup must be non-negative and --repeats must be positive")
    try:
        payload = run(args)
    except (RuntimeError, ValueError) as exc:
        print(f"streaming shard probe refused: {exc}", file=sys.stderr)
        return 2
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, default=str))
    if args.json:
        print(json.dumps(payload, indent=2, default=str))
    else:
        print(
            f"batch_shard={payload['shape']['batch_shard']} "
            f"status={payload['status']} latency_ms={payload.get('latency_ms')} "
            f"peak_allocated={payload.get('peak_allocated_bytes')} "
            f"peak_reserved={payload.get('peak_reserved_bytes')}"
        )
    return 0 if payload["status"] == "PASS" else 3


if __name__ == "__main__":
    raise SystemExit(main())
