#!/usr/bin/env python3
"""Measure dense forward peak CUDA memory for one explicit shape.

Run one shape per process so a CUDA OOM cannot contaminate later measurements. The tool
records both observed allocator peaks and the static linear/quadratic terms used by the
long-sequence capacity gate.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import torch

import torch_transformer_benchmark as B
from agent.tools._common import stamp
from agent.tools.paths import agent_path


DTYPES = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def static_terms(config: B.TransformerConfig, element_bytes: int) -> dict[str, int]:
    rows = config.batch_size * config.seq_len
    terms = {
        "rows_b_times_s": rows,
        "input_bytes": rows * config.d_model * element_bytes,
        "output_bytes": rows * config.d_model * element_bytes,
        "ffn_intermediate_bytes": rows * config.ffn_dim * element_bytes,
        "dense_score_bytes": (
            config.batch_size
            * config.num_heads
            * config.seq_len
            * config.seq_len
            * element_bytes
        ),
    }
    terms["estimated_dense_bytes"] = (
        6 * terms["input_bytes"]
        + 2 * terms["dense_score_bytes"]
        + 2 * terms["ffn_intermediate_bytes"]
    )
    return terms


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda")
    dtype = DTYPES[args.dtype]
    config = B.TransformerConfig(
        args.batch_size,
        args.seq_len,
        args.d_model,
        args.heads,
        args.ffn_dim,
        args.layers,
        args.causal,
    )
    props = torch.cuda.get_device_properties(device)
    terms = static_terms(config, torch.empty((), dtype=dtype).element_size())
    payload: dict[str, Any] = {
        "provenance": stamp(
            "agent/tools/capacity_probe.py", {"invocation": " ".join(sys.argv)}
        ),
        "shape": {
            "batch_size": config.batch_size,
            "seq_len": config.seq_len,
            "d_model": config.d_model,
            "heads": config.num_heads,
            "ffn_dim": config.ffn_dim,
            "layers": config.num_layers,
            "causal": config.causal,
            "dtype": args.dtype,
        },
        "device": props.name,
        "total_vram_bytes": props.total_memory,
        "streaming_budget_fraction": 0.65,
        "streaming_budget_bytes": int(props.total_memory * 0.65),
        "static_terms": terms,
        "status": "PENDING",
    }

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    model = B.BaselineTransformer(config).to(device=device, dtype=dtype).eval()
    x, mask = B.generate_random_case(config, device, dtype, args.seed, 0.0, 1.0)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    try:
        with torch.inference_mode():
            model(x, mask)
        torch.cuda.synchronize(device)
        payload.update(
            status="PASS",
            peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(device),
        )
    except torch.OutOfMemoryError as exc:
        payload.update(
            status="OOM",
            error=str(exc),
            peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(device),
        )
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--seq-len", type=int, required=True)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--ffn-dim", type=int, default=128)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--causal", action="store_true")
    parser.add_argument("--dtype", choices=sorted(DTYPES), default="float32")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--output", type=agent_path, required=True)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    try:
        payload = run(args)
    except (RuntimeError, ValueError) as exc:
        print(f"capacity probe refused: {exc}", file=sys.stderr)
        return 2
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, default=str))
    if args.json:
        print(json.dumps(payload, indent=2, default=str))
    else:
        terms = payload["static_terms"]
        print(
            f"status={payload['status']} B*S={terms['rows_b_times_s']} "
            f"score={terms['dense_score_bytes']} "
            f"peak_allocated={payload.get('peak_allocated_bytes')} "
            f"peak_reserved={payload.get('peak_reserved_bytes')}"
        )
    return 0 if payload["status"] == "PASS" else 3


if __name__ == "__main__":
    raise SystemExit(main())
