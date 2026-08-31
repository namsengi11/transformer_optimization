#!/usr/bin/env python3
"""Measure one query-tiled reference forward in an isolated CUDA process."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import torch

import torch_transformer_benchmark as B
from agent.tools._common import stamp, wait_for_idle
from agent.tools.paths import agent_path


def run(args: argparse.Namespace) -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    idle = wait_for_idle(timeout_s=args.idle_timeout, verbose=False)
    if not idle.get("idle"):
        raise RuntimeError(idle.get("reason", "GPU did not become idle"))
    device = torch.device("cuda")
    config = B.TransformerConfig(
        args.batch_size,
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
    model = B.QueryTiledBaselineTransformer(
        config, query_tile_size=args.query_tile, ffn_chunk_size=args.ffn_chunk
    ).to(device=device, dtype=torch.float32).eval()
    x, mask = B.generate_random_case(
        config, device, torch.float32, args.seed, 0.0, 1.0
    )
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    started = torch.cuda.Event(enable_timing=True)
    ended = torch.cuda.Event(enable_timing=True)
    payload = {
        "provenance": stamp(
            "agent/tools/streaming_tile_probe.py",
            {"invocation": " ".join(sys.argv)},
        ),
        "shape": {
            "batch_size": args.batch_size,
            "seq_len": args.seq_len,
            "d_model": args.d_model,
            "heads": args.heads,
            "ffn_dim": args.ffn_dim,
            "layers": args.layers,
            "causal": args.causal,
        },
        "query_tile": args.query_tile,
        "target_vram_fraction": args.target_vram_fraction,
        "idle_gate": idle,
        "total_vram_bytes": torch.cuda.get_device_properties(device).total_memory,
    }
    try:
        with torch.inference_mode():
            started.record()
            model(x, mask)
            ended.record()
        torch.cuda.synchronize(device)
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
            latency_ms=started.elapsed_time(ended),
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
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=100000)
    parser.add_argument("--d-model", type=int, default=1024)
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--ffn-dim", type=int, default=1024)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--causal", action="store_true")
    parser.add_argument("--query-tile", type=int, required=True)
    parser.add_argument("--ffn-chunk", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--target-vram-fraction", type=float, default=0.85)
    parser.add_argument("--idle-timeout", type=int, default=900)
    parser.add_argument("--output", type=agent_path, required=True)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    if not 0.0 < args.target_vram_fraction < 1.0:
        parser.error("--target-vram-fraction must be between 0 and 1")
    try:
        payload = run(args)
    except (RuntimeError, ValueError) as exc:
        print(f"streaming tile probe refused: {exc}", file=sys.stderr)
        return 2
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, default=str))
    if args.json:
        print(json.dumps(payload, indent=2, default=str))
    else:
        print(
            f"tile={payload['query_tile']} status={payload['status']} "
            f"latency_ms={payload.get('latency_ms')} "
            f"peak_allocated={payload.get('peak_allocated_bytes')} "
            f"peak_reserved={payload.get('peak_reserved_bytes')}"
        )
    return 0 if payload["status"] == "PASS" else 3


if __name__ == "__main__":
    raise SystemExit(main())
