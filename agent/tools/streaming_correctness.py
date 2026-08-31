#!/usr/bin/env python3
"""Cross-check query-tiled and optimized attention against the dense eager model."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

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


def summarize(result: B.AccuracyResult) -> dict:
    return {
        "passed": result.passed,
        "total_elements": result.total_elements,
        "failed_elements": result.failed_elements,
        "max_abs_error": result.max_abs_error,
        "max_relative_error": result.max_relative_error,
        "mean_abs_error": result.mean_abs_error,
    }


def run(args: argparse.Namespace) -> dict:
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
    config.validate()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    dense = B.BaselineTransformer(config)
    tiled = B.QueryTiledBaselineTransformer(
        config, query_tile_size=args.query_tile, ffn_chunk_size=args.ffn_chunk
    )
    optimized = B.UserOptimizedTransformer(config)
    B.copy_model_weights(dense, tiled, strict=True)
    B.copy_model_weights(dense, optimized, strict=True)
    dense = dense.to(device=device, dtype=dtype).eval()
    tiled = tiled.to(device=device, dtype=dtype).eval()
    optimized = optimized.to(device=device, dtype=dtype).eval()
    x, mask = B.generate_random_case(config, device, dtype, args.seed, 0.0, 1.0)
    with torch.inference_mode():
        reference = dense(x, mask)
        tiled_output = tiled(x, mask)
        optimized_output = optimized(x, mask)
    tiled_result = B.compare_outputs(reference, tiled_output, args.rtol, args.atol)
    optimized_result = B.compare_outputs(
        reference, optimized_output, args.rtol, args.atol
    )
    return {
        "provenance": stamp(
            "agent/tools/streaming_correctness.py",
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
            "dtype": args.dtype,
        },
        "query_tile": args.query_tile,
        "tiled_vs_dense": summarize(tiled_result),
        "optimized_vs_dense": summarize(optimized_result),
        "passed": tiled_result.passed and optimized_result.passed,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--ffn-dim", type=int, default=128)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--query-tile", type=int, default=128)
    parser.add_argument("--ffn-chunk", type=int, default=4096)
    parser.add_argument("--causal", action="store_true")
    parser.add_argument("--dtype", choices=sorted(DTYPES), default="float32")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--atol", type=float, default=0.002)
    parser.add_argument("--rtol", type=float, default=0.02)
    parser.add_argument("--output", type=agent_path, required=True)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    try:
        payload = run(args)
    except (RuntimeError, ValueError) as exc:
        print(f"streaming correctness refused: {exc}", file=sys.stderr)
        return 2
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, default=str))
    if args.json:
        print(json.dumps(payload, indent=2, default=str))
    else:
        print(
            f"passed={payload['passed']} "
            f"tiled={payload['tiled_vs_dense']} "
            f"optimized={payload['optimized_vs_dense']}"
        )
    return 0 if payload["passed"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
