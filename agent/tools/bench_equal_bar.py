"""Rerun ``user_matrix`` with an equal reduce-overhead compiled bar.

This sanctioned diagnostic isolates how much of the reported speedup comes
from the compile-mode choice. Its ``run_bench.py`` results are deliberately
routed to ``agent/results/``.
"""
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
AGENT_ROOT = REPO_ROOT / "agent"
if sys.path and Path(sys.path[0]).resolve() == Path(__file__).resolve().parent:
    sys.path.pop(0)
sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("TORCHINDUCTOR_USE_STATIC_CUDA_LAUNCHER", "0")
import run_bench

ro_cases = {name: list(extra) + ["--compile-mode", "reduce-overhead"]
            for name, extra in run_bench.SUITES["user_matrix"].items()}
run_bench.SUITES["user_matrix_ro"] = ro_cases

sys.argv = ["run_bench.py", "--suite", "user_matrix_ro", "--tag", "main_reduce_overhead_bar"]
os.chdir(AGENT_ROOT)
raise SystemExit(run_bench.main())
