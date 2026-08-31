# Optimization lessons

This is rolling agent memory. Update it after every merged or discarded step. Each entry must
state the lesson, evidence, and the rule it creates.

## CUDA-graph timing is mandatory for small kernels

**Lesson:** Eager microbenchmarks rank launch overhead, not the kernel.

**Evidence:** `_short_attn_kernel` measured 47.4 us eagerly and 19.97 us under CUDA-graph
replay. Triton's Python launch cost is comparable to the kernels in the graded shapes.

**How to apply:** Use `agent/tools/microbench.py`, which times back-to-back graph replays and
subtracts an empty graph. Never publish an eager per-kernel timing.

## Working-set residency chooses the bandwidth roof

**Lesson:** An isolated operation that fits in L2 cannot be evaluated against DRAM bandwidth.

**Evidence:** This card has 32 MiB L2 at about 1500 GB/s and a cliff to about 385 GB/s beyond
it. Scoring `add_ln` against DRAM produced the retracted 2022 GB/s and “78% of roof” claim.

**How to apply:** Give FLOPs, traffic bytes, and full working-set bytes to
`agent/tools/roofline.py`; accept its L2/DRAM choice.

## Tiny shapes require ranges, not one median

**Lesson:** A small median delta is not evidence when run ranges overlap.

**Evidence:** The historical, misdecoded case named `07_seq32` varied from roughly 0.064 to
0.327 ms under sync-per-call timing, and autotuning selected four different short-attention
configurations across runs. That case combined `seq_len=32` with `d_model=32`; it is not the
corrected official row 7 and cannot establish a per-case baseline for `07_dmodel32`.

**How to apply:** Use at least three independent `agent/tools/bench.py` runs. Promote latency only
when min/max ranges do not overlap; otherwise record `noise`.

## Benchmark schemas must be explicit

**Lesson:** Never infer a positional benchmark schema from values that happen to produce
constructible models.

**Evidence:** The user matrix was decoded in the wrong column order, changing rows 7, 8, 12,
13, and 14 while leaving enough rows valid to hide the mistake. Historical full-matrix
latency and MFU aggregates therefore described a different workload.

**How to apply:** Use the named columns and canonical table in `agent/docs/USER_MATRIX.md`.
Keep case names aligned with their varied axis, validate all 14 resolved shapes before a
run, and never compare an old retired-name artifact with a corrected-matrix artifact.

## Fixed-checkout A/B prevents moving-main comparisons

**Lesson:** Checking out an old branch and then main is not a valid A/B in this repository.

**Evidence:** A reported 0.913x Triton-GEMM result compared a branch that had become three
commits behind main; it measured another session's work rather than the gate.

**How to apply:** Use `agent/tools/probe_ab.py` on a clean, pinned checkout. Toggle one environment
gate and let the tool reject moved HEAD or main.

## GPU idleness is part of provenance

**Lesson:** A valid implementation can appear regressed when another session owns the GPU.

**Evidence:** Previous benchmarks silently overlapped 100% GPU utilization from another
agent session.

**How to apply:** Only `agent/tools/bench.py` produces judged suite numbers. It requires five
seconds below 10% utilization and marks foreign CUDA PIDs as untrusted.

## A/B arms need fresh processes

**Lesson:** In-process A/B leaks graph, allocator, compiler, and autotune state across arms.

**Evidence:** The optimized model caches probe verdicts, CUDA graphs, fused weights, and
Triton autotune choices by shape.

**How to apply:** `agent/tools/probe_ab.py` launches each arm independently. Do not toggle a gate
inside a live Python process.

## Correctness and speed use different baseline modes

**Lesson:** The eager baseline is the semantic reference; the compiled baseline is only the
speed bar.

**Evidence:** Compiling the fp16/bf16 baseline changes its numerics beyond the harness
tolerance, so no candidate can match both references.

**How to apply:** Require zero eager-reference failures at `atol=0.002`, `rtol=0.02`. Judge
latency with `optimized_ms`; never treat compiled output as correct-by-definition.

## Promotion is MFU-weighted but latency-safe

**Lesson:** MFU can promote a useful efficiency improvement only when latency is flat and
executed precision is unchanged.

**Evidence:** The fp32 calibration gate may actually execute fp16 GEMMs, making a naïve TF32
MFU comparison invalid; the compiled bar also has about 30% cross-process variance.

**How to apply:** Compare `fp16_gemm_enabled` before MFU. Promote a clear non-overlapping
latency win with no suite over 2% worse, or an MFU win with every case within +/-1% latency.

## New kernels need a permanent escape hatch

**Lesson:** Shape support is not enough; a kernel must prove whole-forward correctness before
graph capture and remain disableable after a runtime failure.

**Evidence:** Steps 3, 8, 11, 14, 16, and 17 established the probe/cache/kill-switch pattern.
Tile-local tests miss accumulated six-layer error.

**How to apply:** Implement `_probe_X`, cached `_resolve_X_enabled`, `_X_disabled`, `use_X`
threading, graph-key inclusion, and `TJ_DEBUG_GATE` output. Compare a full `_forward_core`.

## Card facts constrain plausible kernels

**Lesson:** Optimize for the measured RTX 5060 Ti, not a generic CUDA roof.

**Evidence:** Measured peaks are 24.7 TFLOP/s TF32, 48.8 FP16, and 49.4 BF16. The card has 36
SMs, 99 KiB shared memory per SM, 32 MiB L2, about 1500 GB/s L2 bandwidth, and about 385 GB/s
beyond L2. It idles near 442 MHz versus a 3090 MHz maximum. Inductor's 68-SM “big GPU” gate
therefore declines this card.

**How to apply:** Use these constants only through `agent/tools/roofline.py` and `run_bench.py`.
Prefer enough blocks to occupy 36 SMs and use back-to-back work to leave the idle clock.

## Windows toolchain constraints are part of the experiment

**Lesson:** The pinned Windows stack has two known capability gates.

**Evidence:** Required versions are torch `2.8.0+cu129` and
`triton-windows==3.4.0.post21`. The static CUDA launcher can raise an `OverflowError` from a
64-bit stream handle. This build has no flash SDPA. Nsight Compute counters are admin-only
unless `RmProfilingAdminOnly=0`; Nsight Systems works without elevation.

**How to apply:** Start with `agent/tools/env_check.py`. Use Tier 1 and 2 when Tier 3 reports
`available:false`. Do not use `nvprof`. Do not silently change package versions.

## Windows GPU provenance has two independent host-side failure modes

**Lesson:** Loss-tolerant subprocess decoding is necessary but not sufficient; an own child
can also exit between `pmon` sampling and descendant discovery and be mislabeled as foreign.

**Evidence:** Step 18 removed the `cp1252` `UnicodeDecodeError` and completed three quick
runs, but `agent/experiments/18-windows-text-decoding/quick.json` marked all three untrusted
after attributing each run's short-lived 96-98% SM benchmark children to foreign work.

**How to apply:** Use `errors="replace"` for captured diagnostic text, and bracket `pmon`
with before/after process-tree snapshots so either snapshot can identify an own descendant.
Require the end-to-end artifact to remain trusted; a live sampler alone is not enough.

## Priority changes preserve abandoned evidence

**Lesson:** Reprioritizing a serial experiment is not a measurement result.

**Evidence:** Step 20 was stopped before measurement when the user selected general
long-sequence streaming. Its two implementation commits remain isolated on
`opt/20-inductor-fused-graph` and establish neither a win nor a rejection.

**How to apply:** Close the active record, retain its branch, state that no evidence was
collected, and give the replacement mechanism a new experiment number.

Step 22 is a second application: its Inductor/Triton hypothesis was stopped before any code
or measurement when general long-sequence autotuning became the explicit priority. It
remains eligible for a later revisit.

## A VRAM target is a ceiling, not a performance objective

**Lesson:** Choose the fastest work unit below the memory ceiling; do not maximize residency.

**Evidence:** In step 23, row-14 optimized batch 2 fit but was about 2% slower per element
than batch 1 including transfers. Reference query 384 was only about 1.7% faster than 256,
while a different long shape showed that a larger nominally feasible tile could OOM on a
3.14 GiB transient GEMM allocation.

**How to apply:** Gate streaming at 85% of installed VRAM, reserve transient workspace,
prefer measured-efficient tile families, and stop optimized batching once `B*S` reaches the
calibrated throughput region. Validate a second long shape before generalizing a reserve.

## Rejections remain live evidence

**Lesson:** A rejection is scoped to its measured mechanism and can be overturned only by new
measurements.

**Evidence:** Whole-model low precision fails accuracy; error-compensated fp16 costs two
GEMMs; folded attention scaling diverges in fp16; fused softmax accumulates too much error;
KV caching does not apply to full-sequence stateless forwards; fp8 exceeds the 2% tolerance
and its activation conversion costs erase the GEMM win. Conversely, fused QKV and Triton
narrow-N GEMM were later overturned by better probes and the correct L2 roof.

**How to apply:** Read the rejection table before selecting. Name the new fact that changes a
rejected idea, set a kill condition, and append “Overturned in step N” only after promotion.
