#!/usr/bin/env python3
"""Verify that run_bench.py resolves the canonical user matrix exactly."""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import run_bench


# case -> (batch_size, d_model, heads, seq_len, layers, ffn_dim, causal)
EXPECTED = {
    "01_base": (64, 128, 4, 128, 4, 128, True),
    "02_batch1": (1, 128, 4, 128, 4, 128, True),
    "03_batch4": (4, 128, 4, 128, 4, 128, True),
    "04_batch16": (16, 128, 4, 128, 4, 128, True),
    "05_batch128": (128, 128, 4, 128, 4, 128, True),
    "06_batch10000": (10000, 128, 4, 128, 4, 128, True),
    "07_dmodel32": (64, 32, 4, 128, 4, 32, True),
    "08_dmodel1024": (64, 1024, 4, 128, 4, 1024, True),
    "09_heads1": (64, 128, 1, 128, 4, 128, True),
    "10_heads2": (64, 128, 2, 128, 4, 128, True),
    "11_heads16": (64, 128, 16, 128, 4, 128, True),
    "12_seq32": (64, 128, 4, 32, 4, 128, True),
    "13_seq1024": (64, 128, 4, 1024, 4, 128, True),
    "14_extreme": (32, 1024, 16, 100000, 2, 1024, True),
}


def main() -> int:
    cases = run_bench.SUITES["user_matrix"]
    if list(cases) != list(EXPECTED):
        raise AssertionError(
            f"case names/order differ: actual={list(cases)!r}, expected={list(EXPECTED)!r}"
        )

    for name, extra in cases.items():
        shape = run_bench.resolve_shape(extra)
        actual = (
            shape["batch_size"],
            shape["d_model"],
            shape["heads"],
            shape["seq_len"],
            shape["layers"],
            shape["ffn_dim"],
            "--causal" in extra,
        )
        if actual != EXPECTED[name]:
            raise AssertionError(f"{name}: actual={actual!r}, expected={EXPECTED[name]!r}")
        print(name, *actual)

    print("PASS: all 14 user_matrix cases match agent/docs/USER_MATRIX.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
