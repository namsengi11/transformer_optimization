# Step 23 result: general 85%-VRAM streaming autotuner

- Outcome: MERGED
- Branch: `opt/23-streaming-autotune`
- Base SHA: `ba933aa7a8f1b0db398898ecd9416510c936b5f5`
- Measured SHA: `d115f513fb999cc4ba2ff5a3b06cd59a86b72a2b`
- Completed UTC: 2026-08-31T21:28:00Z
- GPU time: about 8 minutes of isolated probes plus correctness runs

## Verdict

Promote. Capacity streaming now uses an 85% installed-VRAM ceiling for every shape,
not an exact benchmark predicate. Reference batch/query tiles and optimized batch shards
are selected independently. The optimized shard stops growing at the calibrated 8192-token
throughput region because filling more memory made row 14 slower.

## Accuracy

- Final canonical row 14 (`B=32,S=100000,D=1024,H=16,F=1024,L=2`) passed
  3,276,800,000/3,276,800,000 elements with `max_abs=0.000951409` at unchanged
  `atol=0.002, rtol=0.02`.
- A split-shard long case (`B=2,S=7000,D=1024,H=64`) selected reference batch 1 and
  optimized batch 2 and passed 14,336,000/14,336,000 elements (`max_abs=0.000523984`).
- A separate canonical-dimension `B=1` check also passed 102,400,000 elements with
  `max_abs=0.000811517`.

## Latency, MFU, and memory

The final row-14 optimized batch-1 probe measured transfer-inclusive samples of
1931.0662, 1910.7597, and 1911.8657 ms (median 1911.8657 ms), or 90.50% MFU against the
project's measured 48.8-TFLOP/s FP16 peak. Peak allocated/reserved memory was
5,026,916,352/5,813,305,344 bytes (34.0% reserved).

The final query-256 reference measured 39,090.332 ms with
8,251,843,584/9,531,555,840 bytes allocated/reserved (55.7% reserved). Optimized batch 2
was capacity-safe but slower per element including transfers (1.983 versus 1.943 seconds),
so it was rejected. Batch 3 reserved 90.2% and was rejected on capacity.

## Query-tile sweep

For row-14 dimensions: query 256 passed at 55.7% reserved; 384 passed at 72.5%; 448
passed at 82.2%; and 512 reached 91.8% and was rejected. On the independent
`S=7000,H=64` shape: 2048 passed and was fastest, 2304 passed but was slower, 2560 OOMed
on a transient 3.14 GiB allocation, and 2880 reached 91.9%. This cross-shape evidence led
to power-of-two candidates and a shape-independent 1.50x reference workspace reserve.

## Evidence

Clean-checkout artifacts are under
`agent/results/worktrees/step23-measure/agent/experiments/23-streaming-autotune/`:
`correctness-row14-full-final.json`, `correctness-row14-shard-final.json`,
`correctness-split-final.json`, `row14-opt-final.json`, `row14-ref-final.json`,
`surrogate-s7000-ref-q2048.json`, and `surrogate-s7000-ref-final.json`. Final probes used the idle gate and report
`git_tree_dirty=false`.

Compile, canonical matrix schema, a small streamed case, the split long case, and the
canonical-dimension shard correctness check all passed. Dense-path latency is excluded
because below-gate kernels are unchanged. The unrelated working-tree edit to `run_bench.py`
is not part of this experiment.
