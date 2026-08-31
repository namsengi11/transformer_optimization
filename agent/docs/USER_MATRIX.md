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

Rows 1-13 are the executable scored matrix. Row 14 remains in the suite definition so the
protocol is complete, but it is structurally infeasible for the dense-attention baseline on
the pinned 16 GiB GPU: one fp32 `[batch, heads, seq_len, seq_len]` score tensor alone would
require 20.48 TB (about 18.6 TiB). The benchmark must report it as `PREFLIGHT_BLOCKED`; it
must not start a child process that attempts the allocation.

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
