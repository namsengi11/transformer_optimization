# Continuous-agent optimization log

This is the fresh durable log for experiments governed by `agent/ORCHESTRATOR.md`. It starts
after the legacy optimization workflow; it does not replace or rewrite that history.

## 1. Legacy boundary

- Historical log: `OPTIMIZATION_LOG.md` at repository root.
- Historical research: `KERNEL_METHODS_ANALYSIS.md` and `SOTA_AGENT_KERNEL_METHODS.md` at
  repository root.
- Historical raw benchmark output: `results/` and the root `*_results.txt` / `14_extreme*.txt`
  files.
- Last legacy numbered step: 17. The first continuous-agent experiment is step 18.

Treat legacy measurements as prior evidence, not as fresh promotion evidence. Reproduce any
number that affects a new decision with `agent/tools/` and store the resulting artifact under
`agent/experiments/` or `agent/results/`.

## 2. Inherited standing

The legacy log reported, at its recorded checkout and environment, 1.207x geomean versus the
compiled bar on the default suite. It also reported 3.764x on the graded matrix and 55.6%
average MFU, but those matrix figures used the pre-2026-08-31 misdecoded shapes for rows 7,
8, 12, 13, and 14. They are invalid for the canonical matrix and must not be used as a
starting hypothesis or comparison baseline. See `agent/docs/USER_MATRIX.md`.

## 3. Agent experiment index

| Step | Outcome | Branch | Claim | Default delta | Matrix delta | Evidence |
|---:|---|---|---|---:|---:|---|
| 18 | DISCARDED | `opt/18-windows-text-decoding` | Loss-tolerant Windows subprocess decoding keeps GPU provenance alive | not run | not run | `agent/experiments/18-windows-text-decoding/quick.json` (untrusted) |
| 20 | DISCARDED | `opt/20-inductor-fused-graph` | Admit a whole-core Inductor candidate behind an accuracy gate | not run | not run | superseded before measurement; branch retained |
| 22 | DISCARDED | `opt/22-inductor-triton-erf` | Make fused FFN composable with whole-core Inductor | not run | not run | superseded before implementation; branch retained |

## 4. Agent rejection index

| Mechanism | Step | Reason | Revisit condition |
|---|---:|---|---|

## 5. Agent measurement notes

Add only durable protocol or environment observations that affect interpretation of more than
one experiment. Put step-specific detail in `agent/experiments/NN-slug/result.md`.

- **2026-08-31 matrix schema correction:** the authoritative columns are `batch_size`,
  `d_model`, `heads`, `seq_len`, `layers`, `ffn_dim`, `causal`. Old artifacts named
  `07_seq32`, `08_seq1024`, `12_ffn32`, or `13_ffn1024` executed different shapes and are
  not corrected-matrix evidence. Rows 1-13 require a fresh baseline; row 14 is retained as
  a structural preflight block. Canonical values: `agent/docs/USER_MATRIX.md`.

## Step records

Append one `### Step NN: title` section per completed experiment, in numeric order. Every
number must name its `agent/tools/` producer and an artifact under `agent/`.

### Step 18: Windows text decoding

The first current-main baseline attempt crashed the foreign-load sampler while strict
`cp1252` decoding processed `nvidia-smi pmon` output. Step 18 added replacement-tolerant
decoding to captured-text subprocesses. Direct import/compile smoke checks passed, and the
fixed-checkout quick suite completed 1/1 eager-reference accuracy with optimized latency
1.0917-1.0958 ms (`agent/tools/bench.py`,
`agent/experiments/18-windows-text-decoding/quick.json`).

The artifact was nevertheless untrusted in all three runs. Each run mislabeled three
short-lived benchmark children at 96-98% SM as foreign because `pmon` was sampled before a
fresh descendant snapshot; children that exited between those calls disappeared from the
tree. The verifier therefore returned `REFUTED`, the implementation stayed unmerged, and a
fresh tooling step must fix both decoding and descendant-attribution races before baseline
measurement.

### Step 20: Inductor-fused graph

The experiment was superseded before measurement by the user's explicit request to make
canonical row 14 and other capacity-bound long sequences executable through streaming.
No latency, MFU, or accuracy claim is made. The implementation remains unmerged on
`opt/20-inductor-fused-graph`; see
`agent/experiments/20-inductor-fused-graph/result.md`.

### Step 22: Inductor Triton compatibility

The experiment was superseded before implementation when the user explicitly prioritized
general long-sequence autotuning and an 85% installed-VRAM target. No numerical or
performance claim is made. The hypothesis and documentation-only result remain on
`opt/22-inductor-triton-erf`; see
`agent/experiments/22-inductor-triton-erf/result.md`.
