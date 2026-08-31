#!/usr/bin/env python3
"""Classify an operation against this RTX 5060 Ti's measured roofline."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from agent.tools._common import stamp

PEAK_TFLOPS = {"float32": 24.7, "float16": 48.8, "bfloat16": 49.4}
L2_BYTES = 32 * 1024 * 1024
L2_GBPS = 1500.0
DRAM_GBPS = 385.0
CARD = {"sm_count": 36, "shared_memory_kib": 99, "l2_mib": 32}


def classify(flops: float, traffic_bytes: float, working_set_bytes: float,
             dtype: str, measured_us: float | None = None) -> dict[str, Any]:
    if flops <= 0 or traffic_bytes <= 0 or working_set_bytes <= 0:
        raise ValueError("flops, bytes, and working-set-bytes must be positive")
    peak = PEAK_TFLOPS[dtype]
    residency = "L2" if working_set_bytes <= L2_BYTES else "DRAM"
    bandwidth = L2_GBPS if residency == "L2" else DRAM_GBPS
    intensity = flops / traffic_bytes
    memory_roof_tflops = intensity * bandwidth / 1000.0
    attainable_tflops = min(peak, memory_roof_tflops)
    compute_time_us = flops / (peak * 1e12) * 1e6
    memory_time_us = traffic_bytes / (bandwidth * 1e9) * 1e6
    bound = "compute" if compute_time_us >= memory_time_us else f"{residency.lower()}-bandwidth"
    result: dict[str, Any] = {
        "provenance": stamp("agent/tools/roofline.py"),
        "inputs": {
            "flops": flops,
            "traffic_bytes": traffic_bytes,
            "working_set_bytes": working_set_bytes,
            "dtype": dtype,
            "measured_us": measured_us,
        },
        "card": CARD,
        "roof": {
            "residency": residency,
            "bandwidth_gbps": bandwidth,
            "peak_tflops": peak,
            "arithmetic_intensity_flop_per_byte": intensity,
            "memory_roof_tflops": memory_roof_tflops,
            "attainable_tflops": attainable_tflops,
            "compute_time_us": compute_time_us,
            "memory_time_us": memory_time_us,
            "bound": bound,
        },
        "trusted": True,
    }
    if measured_us:
        result["measured"] = {
            "achieved_tflops": flops / (measured_us * 1e-6) / 1e12,
            "effective_bandwidth_gbps": traffic_bytes / (measured_us * 1e-6) / 1e9,
            "fraction_of_attainable_roof": (
                flops / (measured_us * 1e-6) / 1e12 / attainable_tflops
            ),
        }
    return result


def _add_ln(rows: int, d_model: int, dtype: str) -> tuple[float, float, float]:
    element_bytes = 4 if dtype == "float32" else 2
    elements = rows * d_model
    # delta + residual + weight + bias reads, residual + normalized writes.
    traffic = (4 * elements + 2 * d_model) * element_bytes
    # Add, mean, variance, normalize, affine: a conservative 8 FLOPs/element.
    flops = 8 * elements
    working_set = traffic
    return float(flops), float(traffic), float(working_set)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flops", type=float)
    parser.add_argument("--bytes", dest="traffic_bytes", type=float)
    parser.add_argument("--working-set-bytes", type=float)
    parser.add_argument("--dtype", choices=sorted(PEAK_TFLOPS), default="float32")
    parser.add_argument("--measured-us", type=float)
    parser.add_argument("--operation", choices=("add_ln",))
    parser.add_argument("--rows", type=int)
    parser.add_argument("--d-model", type=int)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    if args.operation == "add_ln":
        if not args.rows or not args.d_model:
            parser.error("--operation add_ln requires --rows and --d-model")
        flops, traffic, working_set = _add_ln(args.rows, args.d_model, args.dtype)
    else:
        if args.flops is None or args.traffic_bytes is None or args.working_set_bytes is None:
            parser.error("provide --flops, --bytes, and --working-set-bytes")
        flops, traffic, working_set = args.flops, args.traffic_bytes, args.working_set_bytes

    payload = classify(flops, traffic, working_set, args.dtype, args.measured_us)
    if args.json:
        print(json.dumps(payload, indent=2, default=str))
    else:
        roof = payload["roof"]
        print(f"roofline: {roof['bound']} bound against {roof['residency']} ({roof['bandwidth_gbps']:.0f} GB/s)")
        print(f"  arithmetic intensity {roof['arithmetic_intensity_flop_per_byte']:.3f} FLOP/byte")
        print(f"  attainable roof      {roof['attainable_tflops']:.3f} TFLOP/s")
        if "measured" in payload:
            measured = payload["measured"]
            print(f"  achieved             {measured['achieved_tflops']:.3f} TFLOP/s ({100*measured['fraction_of_attainable_roof']:.1f}% roof)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
