# Canonical user benchmark matrix

This is the authoritative shape schema supplied by the user on 2026-08-31. The column order
is explicit and must not be inferred:

`case_id, batch_size, d_model (QKV dim), heads, seq_len, layers, ffn_dim, causal`

All rows use the harness-default `float32` dtype.

| Case | Benchmark name | Batch | d_model | Heads | Seq len | Layers | FFN dim | Causal |
|---:|---|---:|---:|---:|---:|---:|---:|:---:|
| 1 | `01_base` | 64 | 128 | 4 | 128 | 4 | 128 | true |
| 2 | `02_batch1` | 1 | 128 | 4 | 128 | 4 | 128 | true |
| 3 | `03_batch4` | 4 | 128 | 4 | 128 | 4 | 128 | true |
| 4 | `04_batch16` | 16 | 128 | 4 | 128 | 4 | 128 | true |
| 5 | `05_batch128` | 128 | 128 | 4 | 128 | 4 | 128 | true |
| 6 | `06_batch10000` | 10000 | 128 | 4 | 128 | 4 | 128 | true |
| 7 | `07_dmodel32` | 64 | 32 | 4 | 128 | 4 | 32 | true |
| 8 | `08_dmodel1024` | 64 | 1024 | 4 | 128 | 4 | 1024 | true |
| 9 | `09_heads1` | 64 | 128 | 1 | 128 | 4 | 128 | true |
| 10 | `10_heads2` | 64 | 128 | 2 | 128 | 4 | 128 | true |
| 11 | `11_heads16` | 64 | 128 | 16 | 128 | 4 | 128 | true |
| 12 | `12_seq32` | 64 | 128 | 4 | 32 | 4 | 128 | true |
| 13 | `13_seq1024` | 64 | 128 | 4 | 1024 | 4 | 128 | true |
| 14 | `14_extreme` | 32 | 1024 | 16 | 100000 | 2 | 1024 | true |

## Execution policy

All 14 rows are executable. Rows whose estimated dense peak fits the calibrated installed-
VRAM budget use the ordinary dense protocol. A row above the budget is labeled `STREAMED`
and uses the capacity protocol described below. On the pinned 16 GiB GPU this applies to
row 14: one fp32 `[batch, heads, seq_len, seq_len]` score tensor alone would require
20.48 TB (about 18.6 TiB), while its input and output are each 13.1 GB.

The gate is general and contains both kinds of pressure:

`estimated_dense_bytes = 6*(B*S*D*element_bytes) + 2*(B*H*S^2*element_bytes) + 2*(B*S*ffn*element_bytes)`

Streaming is selected when that estimate exceeds 85% of installed VRAM. It uses installed
capacity, never current free memory, and does not match a benchmark name or fixed sequence
length. `agent/tools/capacity_probe.py` calibrated the estimate: equal `B*S=8192` shapes
peaked at 76 MB for `B=64,S=128`, 312 MB for `B=8,S=1024`, and 2.26 GB for
`B=1,S=8192`, demonstrating why `B*S` alone is insufficient. The row-6-like
`B=10000,S=128` shape peaked at 10.51 GB and remains on the dense side of the gate.

For a streamed row, inputs remain on the CPU and independent batch shards are transferred
to CUDA. Reference and optimized shards are selected independently: the optimized side uses
the smallest capacity-safe shard that reaches the calibrated 8192-token throughput region,
while the eager reference searches batch/query work units under the same capacity ceiling.
The reference retains all keys and values, so it
still computes every full causal-attention relationship without an `[S,S]` allocation.
Each optimized CPU shard is subdivided for the reference and compared against slices of the
same optimized output. Streaming timing includes transfers and uses explicit long-case counts
(`accuracy_trials=1`, `warmup=0`, `repeats=1`, `rounds=1`). There is no compiled-baseline
ratio for a streamed row until an equivalent compiled streaming bar exists; report its
optimized latency and MFU separately and exclude it from compiled-bar geomeans.

The query-tiled reference no longer receives a fixed percentage of memory. The planner first
subtracts estimated parameters and per-batch resident state from the 85% target, then sizes
the simultaneous score/probability workspace from the remainder. A shape-independent 1.50x
workspace reserve, measured with isolated candidate processes, covers allocator block rounding
and GEMM workspace. Power-of-two query candidates are used because a second long-shape sweep
found query 2048 faster than 2304 while 2560 OOMed; there is no benchmark-name or exact-shape
predicate. On row 14 this selects reference batch 1/query 256 and optimized batch 1. Query 384
measured 12.40 GB reserved but was only 1.7% faster than query 256, and a second long shape showed
that the earlier 1.35x reserve could cross the 85% ceiling. Query 512 measured
15.70 GB (91.8%) and was rejected. Optimized batch 2 is capacity-safe but its end-to-end latency was
1.983 seconds per element versus 1.943 seconds for batch 1, so filling more memory was slower;
batch 3 measured 15.43 GB (90.2%) and was rejected on capacity.

## Schema correction

Before 2026-08-31, `run_bench.py` incorrectly interpreted the unlabeled values as
`batch_size, seq_len, heads, ffn_dim, layers, causal, d_model`. That produced different
shapes for rows 7, 8, 12, 13, and 14. The old names `07_seq32`, `08_seq1024`, `12_ffn32`,
and `13_ffn1024` are retired.

Results produced with those names or that old column interpretation are historical only.
They must not contribute to a corrected-matrix latency aggregate, MFU aggregate, promotion
decision, or before/after comparison. Rows 1-6 and 9-11 were unchanged by the correction,
but any published full-matrix aggregate still requires a fresh run under this schema.

Run `python agent/tools/check_user_matrix.py` after changing the suite. It checks all values,
case names, order, and causal flags against this canonical schema without launching GPU work.
