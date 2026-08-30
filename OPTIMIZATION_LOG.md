# Transformer inference optimization log

Goal: make `UserOptimizedTransformer` faster than the stated bar (the
`torch.compile`d baseline) without breaking the harness accuracy contract
`abs(user - ref) <= atol OR abs(user - ref) <= rtol * abs(ref)`.

Hardware: RTX 5060 Ti (sm_120 Blackwell, 36 SMs, 16 GB), torch 2.8.0+cu129,
triton 3.4.0, Windows.

---

## 0. Environment fix (prerequisite)

`torch.compile` was **completely broken** on this machine. The installed
`triton-windows 3.6.0.post25` pairs with torch 2.9, and torch 2.8's inductor
failed with `ImportError: cannot import name 'triton_key'`. Downgraded to
`triton-windows==3.4.0.post21`. Reversible with
`pip install triton-windows==3.6.0.post25`.

## 1. Where the time actually goes

Profile of the compiled baseline, default shape (B=8, S=128, d_model=512,
heads=8, ffn=2048, layers=6), fp32/TF32:

| component | share | detail |
|---|---|---|
| GEMMs (projections + FFN) | **85%** (1.93 ms) | `cutlass_80_tensorop_s1688gemm` — Ampere-tuned TF32 kernels on an sm_120 part, ~20.8 TFLOPS |
| attention bmm + softmax | 5% | inductor did **not** pattern-match to SDPA |
| everything else | 10% | 146 kernel launches/iter; CPU time approximately equals GPU time |

GEMM ceiling, measured directly (TFLOPS):

| shape | TF32 | fp32 (no TF32) | fp16 | bf16 |
|---|---|---|---|---|
| `[1024,512]x[512,512]` qkv | 17.6 | 10.5 | 31.8 | 15.5 |
| `[1024,512]x[512,1536]` qkv fused | 20.7 | 12.7 | 38.0 | 34.4 |
| `[1024,512]x[512,2048]` ffn_in | 19.2 | 12.7 | 40.1 | 35.2 |
| `[1024,2048]x[2048,512]` ffn_out | 21.1 | 12.9 | 37.2 | 39.5 |

Launch-overhead breakdown of the optimized eager model:

| dtype | eager | kernel sum | launches | overhead | CUDA-graph replay |
|---|---|---|---|---|---|
| fp32 | 2.44 ms | 2.28 ms | 61 | 6.7% | 2.30 ms |
| fp16 | 2.39 ms | 1.64 ms | 157 | 31.6% | 1.71 ms |
| bf16 | 2.07 ms | 1.55 ms | 127 | 25.1% | 1.59 ms |

## 2. The constraint that shaped everything

**`torch.compile` changes the baseline's own low-precision numerics by more than
the tolerance.** Compiled baseline vs eager baseline, with none of our code
involved:

| dtype | max_abs | failing elements | verdict |
|---|---|---|---|
| fp32 | 0.00062 | 0 / 524288 | PASS |
| fp16 | 0.01172 | 21 | **FAIL** |
| bf16 | 0.09375 | 65102 (12%) | **FAIL** |

(`torch._inductor.config.emulate_precision_casts=True` only narrows bf16 to
51320 failures.)

So the eager baseline and the compiled baseline are **two mutually incompatible
references** — no implementation can be within tolerance of both. Because
accuracy is measured *against the baseline*, being **more** accurate than it
also counts as divergence. And low precision has no headroom at all: one bf16
ulp at magnitude 1.0 is 0.0078, roughly 4x the atol of 0.002.

This splits the problem in two:

* **fp32** has real headroom, so SDPA, fp16 GEMMs and aggressive fusion are all
  available.
* **fp16/bf16** admit only **arithmetic-preserving** changes: CUDA graphs and
  rounding-faithful fusion. `torch.compile` is off-limits *for our own model
  too*, not just for the baseline.

`run_bench.py` therefore separates the two things the harness conflates:
correctness is judged against the **eager** baseline (the harness default, and
the true semantic reference), while the speed bar is the **compiled** baseline's
latency, taken as the fastest of N runs so every reported speedup is a lower
bound. That bar has roughly 30% run-to-run variance across processes.

## 3. Changes, in order

Each step was developed on its own branch and merged only after measurement.

| step | change | geomean vs bar | accuracy |
|---|---|---|---|
| — | unmodified `UserOptimizedTransformer` | 0.80x | 5/5 |
| 1 | fused QKV + dtype-dependent attention path | 0.699x | 5/5 |
| 2 | CUDA graph capture/replay | 1.001x | 5/5 |
| 3 | fp16 GEMMs for fp32 models + calibration gate | 1.044x | 5/5 |
| 4 | bit-exact Triton scale+mask fusion | 1.038x* | 5/5 |
| 5 | fix long_fp16 accuracy bug; chunked FFN for extreme shapes | 1.060x | 5/5 |
| 6 | bit-exact fp16 GELU (no fp32 round-trip) | 1.072x | 5/5 |
| 7 | investigated autotuned Triton GEMM vs cuBLAS -- **no change merged** | 1.078x (unchanged) | 5/5 |
| 8 | fused QKV in fp16/bf16 where a probe proves it bit-exact | 1.096x | 5/5 |
| 9 | inductor-generated layout copies (bit-exact) | **1.189x** | 5/5 |
| 10 | investigated fp8 weight quantization -- **no change merged** | 1.189x (unchanged) | 5/5 |
| 11 | two shape-specialized Triton kernels for the graded shapes | 1.172x default suite; **3.764x** geomean on the graded matrix | 5/5 + 13/13 (+ 60/60 sweep) |

\* flat within the bar's noise; the optimized latency itself dropped 4-5% on the
causal and padded cases.

### Step 1 — fused QKV + dtype-dependent attention

One GEMM for q/k/v instead of three, `F.scaled_dot_product_attention`, a mask
fast-path, and a cached causal mask. Then split by dtype, because SDPA diverges
from the baseline by ~0.0015 per op in fp16 — *identically* for the MATH backend
and for fp32-upcast inputs, so it is not an accumulation-precision problem. The
baseline scales after the QK matmul and re-quantizes `probs` to the model dtype;
SDPA scales `q` first and keeps higher-precision probs.

* fp32 uses SDPA + fused QKV.
* fp16/bf16 reproduce the baseline arithmetic exactly, **with three separate
  projection GEMMs**, because fusing them changes cuBLAS kernel selection enough
  to break near-zero elements at atol=0.002.

Also found: **this Windows build has no flash attention at all**
(`Torch was not compiled with flash attention`); only the MATH, EFFICIENT and
CUDNN backends exist. The stated bar is really "torch.compile + inductor
bmm/softmax", not "torch.compile + flash SDPA".

### Step 2 — CUDA graphs

Graph replay runs identical kernels with identical arithmetic, which makes it
the one large lever that is legal in low precision. The cache is keyed on
`(shape, dtype, device, causal, mask_kind)`, and falls back to eager permanently
on capture failure, on non-CUDA devices, and under
`torch.compiler.is_compiling()`.

`mask.all()` is a device-to-host sync and is **illegal during capture** — it
poisons the entire CUDA context. It is hoisted out of the captured region and
memoized per mask tensor: at most one sync per distinct mask, zero when warm.

Largest wins on small shapes (fp32 b1/s64: **7.38x**), where launch overhead
dominates everything else.

### Step 3 — fp16 GEMMs for fp32 models, behind a calibration gate

fp32 models run their GEMMs and SDPA in fp16 with results cast straight back to
fp32, while LayerNorm, GELU, the residual adds and the final norm stay fp32.
Naive whole-model fp16 casting **fails** (max_abs 0.0082) and bf16 fails badly
(0.074); confining fp16 to the GEMMs passes everywhere at normal input scale
(max_abs 0.0012–0.0019).

The error scales as **1/std(residual)**, so small inputs break it:

| `--input-scale` | 4.0 | 2.0 | 1.0 | 0.5 | 0.25 | 0.1 |
|---|---|---|---|---|---|---|
| failures @ (2e-3, 2%) | 0 | 0 | 0 | 2 | 25 | 48 |
| max_abs | 0.00034 | 0.00064 | 0.00122 | 0.00218 | 0.00288 | 0.00287 |

So a **runtime calibration gate** compares the fp16 path against the fp32 path
on the *real* input at warmup, and enables fp16 only if nothing fails
`abs <= 0.9*atol OR rel <= 0.9*rtol`. The 0.9 margin was chosen empirically:
1.0 wrongly accepts input-scale 0.5, and <=0.8 wrongly rejects the
d_model=1024 / 12-layer config. Calibration runs before graph capture, since it
syncs. Tolerances are overridable via `TJ_ATOL` / `TJ_RTOL`.

Result: fp32 1.061x -> **1.588x** (2.28 ms -> 1.53 ms).

### Step 4 — Triton scale+mask fusion (bit-exact)

The attention epilogue streamed the score tensor about five times per layer
(~130 MB per forward). Fusing the scale and the combined mask into a single
Triton elementwise kernel, feeding `softmax(dtype=float32)` directly, removed
the explicit `.float()` pass: epilogue 56.09 -> 37.59 us/call (-33%).

**The softmax reduction itself could not be fused.** A Triton `tl.max`/`tl.sum`
reduction tree does not match ATen's softmax bit-for-bit, and the resulting
1e-5..1e-4 per-call difference compounds through the residual stream to
max_abs 0.0078 over 6 layers — real failures. The reduction stays on ATen.

### Step 5 — fix long_fp16 accuracy bug; chunked FFN for extreme shapes

A 10+ shape sweep (beyond the original 5-case suite) surfaced a real accuracy
failure at S=2048/4096 in fp16, and a separate hardware ceiling at extreme
`ffn_dim`. Root cause of the accuracy bug: `torch.softmax(scores, dim=-1,
dtype=torch.float32)` — the `dtype=` kwarg form — silently selects a
**different ATen reduction kernel** than an explicit `.float()` cast followed
by a plain softmax, and that kernel diverges by a few ulps specifically at
long rows (not seen at S<=512 fp16, or any bf16 shape). Fix: call
`torch.softmax()` natively on the already-model-dtype tensor — bit-identical
to the explicit-upcast form (verified `torch.equal` up to (8,4,4096,4096) in
both fp16/bf16), and cheaper too (no fp32 intermediate). fp16 at S=2048/4096
now PASSES bit-exact (previously failing 6-8/524288 elements at S=2048).

Separately, a grading-matrix stress case (`d_model=1024, ffn_dim=100000,
batch=32, seq=1024`) OOMs the **unmodified reference `BaselineTransformer`**
itself — the GELU intermediate is `[32,1024,100000]` fp32 ≈ 13.1GB, before
`UserOptimizedTransformer` even runs. Added `ChunkedBaselineTransformer`
(opt-in via `--chunk-baseline-ffn`, off by default, zero effect on
`BaselineTransformer`) and matching chunked-FFN paths inside
`UserOptimizedTransformer` (auto-triggered only when the estimated unchunked
FFN intermediate exceeds 3GB — never for any shape in this codebase's own
suites). Chunking is **not** automatically bit-exact even though the
underlying math is row-independent: cuBLAS can select a different GEMM
kernel depending on row count, the same class of bug as the softmax fix
above, and the relationship is non-monotonic (at one tested shape, chunk
sizes 128 and 8192+ were exact but 256-4096 were not) — so the chunk size
used (4096) was verified bit-exact at this codebase's actual large-`ffn_dim`
shape before use, not assumed safe by extrapolation. Result: the extreme
case now PASSES (0/67,108,864 failed, max_abs=0.00142), 15.77x vs its
(also chunked) eager baseline.

### Step 6 — bit-exact fp16 GELU (no fp32 round-trip)

The fp32 fp16-GEMM path (step 3) was upcasting the FFN's hidden activation to
fp32 just to run GELU, then downcasting back to fp16 for the next GEMM. GELU
follows the same pattern already found for softmax and LayerNorm: ATen's
kernel already accumulates internally in fp32 regardless of the input
tensor's dtype, so `F.gelu` on a fp16 tensor is bit-identical (`torch.equal`)
to the upcast/downcast round trip — removing it costs nothing numerically
while cutting 2 elementwise kernel launches and halving that op's memory
traffic. `default_fp32`: 1.586x -> **1.786x** vs the compiled bar.

### Step 7 — investigated: autotuned Triton GEMM vs cuBLAS (no change merged)

Re-opened the "Inductor max-autotune GEMM" rejection from §4 to check whether
its SM-count gate is actually well-calibrated for this 36-SM card, since our
workload amortizes autotuning at warmup (once) and replays via CUDA graph
indefinitely — the opposite of inductor's one-shot-compile assumption that
the gate is written for.

**Where the gate lives**: `torch/_inductor/utils.py::is_big_gpu()` —
`min_sms = 68  # 3080` (108/132 for A100/H100 aren't even it; 68 is an
RTX 3080). Our 36 SMs trip it. It gates `_use_template_for_gpu`, which is
the only caller.

**Bypassing it**: `torch._inductor.utils.is_big_gpu` is looked up as a
plain module global by its one caller, so monkeypatching the module
attribute (`inductor_utils.is_big_gpu = lambda *a, **k: True`) is enough —
no source patch, no private-API contortions. With
`config.max_autotune_gemm = True` this does make inductor autotune real
Triton GEMM templates for this GPU.

Two unrelated bugs surfaced immediately, both worth recording:
  * **Windows-only crash**: `torch._inductor.config.use_static_cuda_launcher`
    (on by default in torch 2.8) overflows a C `long` with the 64-bit CUDA
    stream handle inside `static_cuda_launcher.py`'s `_launch_kernel`
    (`OverflowError: Python int too large to convert to C long`) the moment
    a compiled Triton template actually runs. Unrelated to SM count or this
    project; setting `config.use_static_cuda_launcher = False` (falls back
    to the ordinary launcher) fixes it. Filed here so it isn't
    re-discovered as "autotune doesn't work" next time.
  * **Confirms the 99KiB shared-memory ceiling** stated in this project's
    hardware notes: several autotune candidates with `BLOCK_K=128` at
    128-wide M/N tiles failed with `OutOfMemoryError: out of resource:
    triton_mm Required: 131072/147456 Hardware limit:101376` (101376 B =
    99 KiB) during inductor's own candidate search — a real, measured
    number, not an assumption.

**Result (fp16 F.linear, matching what the fp32 fast path's GEMMs actually
run under; median of repeated `torch.cuda.Event`-timed trials, 20 warmup +
100-200 iters each)**:

| shape (M x K x N) | cuBLAS TFLOPS | inductor-autotuned Triton TFLOPS | ratio |
|---|---|---|---|
| qkv_fused, default (1024x512x1536) | 39.6 | 30.4 | 0.77x |
| out_proj, default (1024x512x512) | 21.7 | 12.2 | 0.56x |
| ffn_in, default (1024x512x2048) | 40.7 | 28.3 | 0.70x |
| ffn_out, default (1024x2048x512) | 39.9 | 30.3 | 0.76x |
| qkv_fused, big (16384x512x1536) | 46.2 | 41.5 | 0.90x |
| out_proj, big (16384x512x512) | 44.9 | 31.6 | 0.70x |
| ffn_in, big (16384x512x2048) | 46.4 | 43.1 | 0.93x |
| ffn_out, big (16384x2048x512) | 48.8 | 32.4 | 0.66x |

cuBLAS wins every shape, sometimes by a lot (out_proj default: 0.56x).
Inductor's stock Triton MM template (BLOCK_M/N capped at 128,
`num_stages<=5`, a handful of `num_warps`) never comes close on this card.

**Then hand-wrote a Triton GEMM** (`@triton.autotune` over
`BLOCK_M/N/K in {32,64,128}`, `num_stages in {2,3,4}`,
`num_warps in {2,4,8}`, filtered to configs whose estimated shared-memory
footprint stays under 99 KiB — 159 valid configs total; grouped-tile
ordering for L2 reuse) specifically to give the search room for tile sizes
that reach reasonable occupancy on 36 SMs at these shapes (e.g. `out_proj`
default at BLOCK_M=BLOCK_N=64 gives 128 tiles across 36 SMs, vs. inductor's
128x128 default giving only 32 — under-subscribed). This closed most of the
gap and even edged ahead at the bigger shapes, but the result is a wash, not
a win:

| shape | cuBLAS ms (best-of-3) | hand-written Triton ms (best-of-3) | ratio (triton/cublas) |
|---|---|---|---|
| out_proj, default | 0.0253 | 0.0340 | **1.34x slower** |
| qkv_fused, big | 0.5667 | 0.5443 | 0.96x (faster) |
| out_proj, big | 0.1899 | 0.1879 | 0.99x (~tied) |
| ffn_in, big | 0.7678 | 0.7171 | 0.93x (faster) |
| ffn_out, big | 0.7274 | 0.7417 | 1.02x (slower) |

At the shape that actually matters — `default` (batch=8, seq=128), the one
`run_bench.py`'s `default_fp32` case runs — Triton is at-best tied
(qkv_fused, ffn_in, ffn_out within 2-4% either way across repeated runs)
and clearly loses on `out_proj` (25-34% slower, reproducible across
repeated trials, not a one-off). Summed over one layer's 4 GEMMs at the
default shape, an all-Triton layer is **~7% slower** than the existing
all-cuBLAS path, so there's no per-shape substitution that helps here: even
cherry-picking cuBLAS for `out_proj` and Triton for the rest is still
slightly slower than what already ships, because none of the other three
GEMMs actually beat cuBLAS at this shape either — they just come close.

The bigger shape (batch=32, seq=512) shows a real but small edge on 3 of 4
GEMM types (qkv_fused, out_proj, ffn_in: 1-7%, reproducing in sign but not
magnitude across two independent runs) and no edge at all on `ffn_out`
(flipped sign between runs — noise). None of this shows up in
`run_bench.py`'s `default` suite, which only exercises the smaller shape,
and a few-percent edge is well inside the harness's own documented ~30%
run-to-run variance — not a "real, repeatable win" by this project's bar.

**Conclusion: no code change.** cuBLAS already wins or ties at every shape
this codebase actually runs. The SM-count gate's *literal* premise (`36 SMs
< 68`, calibrated off an RTX 3080) doesn't hold up as the reason Triton
loses here — bypassing it and autotuning for real does find configs that
almost close the gap, and even edges ahead at larger shapes — but the
practical reason autotuning doesn't pay off on this card is simpler:
**NVIDIA's Blackwell (sm_120) cuBLAS kernels are already very good for
these matrix sizes**, and a generic tile-based Triton MMA schedule (whether
inductor's own template or a hand-tuned one) doesn't have enough headroom
over them at this scale to justify the autotuning cost — which is real:
159 configs x compile-and-time per novel shape took on the order of tens of
seconds to minutes per shape in this investigation, far more than the
existing fp16-GEMM gate's "one extra forward pass" calibration cost, for a
payoff that isn't there at the shape that matters. `_resolve_fp16_gemm_enabled`'s
existing cuBLAS fp16-GEMM path (step 3) remains the fastest fp32 GEMM path
found on this hardware.

**Follow-up `ncu` observations (post-elevation access) — logged as
observations, not a settled root cause; the picture is still ambiguous:**

- Achieved occupancy (`sm__warps_active`) on the real cutlass GEMM/attention
  kernels is genuinely low: 8-16% for the two GEMM tile variants, 14.7-14.8%
  for `fmha_cutlassF`, versus 70-84% for LayerNorm/GELU/residual-add.
  `occupancy_limit_shared_mem` caps them at 1-2 resident blocks/SM (of 24
  possible). One narrow-N shape (`out_proj`/`ffn_out`, grid=32) has *fewer
  blocks than SMs* -- some SMs get zero work for that kernel's duration.
- A hand-written Triton GEMM re-tuned for `out_proj`'s exact shape (32x64x32
  tile, 256 blocks -- well above 36 SMs) reached 33.2% occupancy, more than
  2x cuBLAS's 15.1% at that shape -- and was still ~3.4% *slower* (20.29us
  vs 18.69us), with slightly lower SM throughput (32.45% vs 34.29%).
  Confirms occupancy alone isn't the limiter for these kernels; higher
  occupancy did not translate to a win.
- Initial diagnosis pinned this on shared-memory bank conflicts (broad
  `l1tex__data_bank_conflicts_pipe_lsu.sum`: 15,479 Triton vs 5,741 cuBLAS,
  ~2.7x). **That diagnosis does not hold up**: the shared-memory-*specific*
  counters (`..._mem_shared_op_ld/st.sum`) are actually comparable --
  4,651 (Triton) vs 5,116 (cuBLAS) -- so the broad metric was picking up
  something else (likely global-memory-related LSU-pipe replay activity,
  not shared-memory bank conflicts specifically). Retracting that claim.
- What *is* different: shared-memory load wavefronts are ~40% higher for
  Triton (404,361 vs 289,238) while store wavefronts are ~48% lower (9,890
  vs 18,982). Read as circumstantial evidence that cuBLAS's `s16816gemm`
  kernel (named for the `mma.sync.m16n8k16` instruction) moves tensor-core
  operands via wider/fewer shared-memory-to-register instructions
  (`ldmatrix`-style) than Triton's generic `tl.dot` lowering -- but this is
  an inference from wavefront counts, not a confirmed instruction-level
  cause (no SASS-level comparison was done). Global-memory coalescing is
  identical between the two (16 sectors/request, both ld and st) and is
  ruled out as a factor.
- Tested directly: shrinking the GEMM's M dimension (simulating "process a
  smaller chunk of the batch at a time") does **not** help and does not
  scale as a clean per-tile-constant story either -- M=256 (64 blocks)
  measured *worse* per-output-element time than M=1024 (256 blocks), with
  occupancy dropping from 33.4% to 13.9%. Smaller batches just reduce the
  already-thin margin of blocks-vs-36-SMs further; they don't touch
  whatever the real per-tile inefficiency is.
- Net: cuBLAS is beaten by only ~3.4% at the one shape with the largest
  gap after a real config search, and the exact instruction-level reason
  remains unconfirmed. Not worth further kernel-engineering effort given
  the small remaining margin and the cost already sunk here without a
  conclusive mechanism.

### Step 8 — fused QKV in fp16/bf16, gated on a bit-exactness probe

Step 1 banned fused QKV in low precision outright. That was too coarse, and
the stated reason was wrong.

Measured with `torch.equal` (bit-identical, not "within tolerance") across
10 shapes:

| dtype | verdict |
|---|---|
| bf16 | bit-exact at **all 10** shapes |
| fp16 | bit-exact at 8/10 — fails only at `(8,128,512)` and `(1,64,512)` |

Three candidate mechanisms were checked and ruled out:

* **Not kernel selection.** Profiling shows the separate (N=d_model) and
  fused (N=3·d_model) GEMMs dispatch to the *identical* cutlass kernel
  (`cutlass_80_tensorop_f16_s16816gemm_relu_f16_64x64_32x6_tn_align8` at the
  default shape) — same tile shape, same K-tiling.
* **Not split-K.** Forcing `CUBLAS_WORKSPACE_CONFIG=:0:0` (no workspace, so
  no split-K algorithms are available) leaves the exactness pattern
  completely unchanged, and only slows the GEMMs down.
* **Not expressible as a threshold.** Non-monotonic in M: at d_model=512
  fp16, M=256 ✗, M=512 ✓, M=1024 ✗, M≥2048 ✓. At d_model=128 every M tried
  was exact; at d_model=1024 the small-M end was not.

What actually varies is a cuBLASLt *algorithm* choice (threadblock swizzle /
CTA ordering) that shares one cutlass template name and is not observable or
controllable from Python. Any hand-written `(M,K,N,dtype)` predicate would be
curve-fitting a closed-source heuristic that a cuBLAS or driver update can
move — the same trap as the FFN chunk-size non-monotonicity in step 5.

So it is decided by measurement instead: `_probe_fused_qkv_exact` runs the
whole forward both ways on the real input and requires `torch.equal`, once
per configuration key, cached, before graph capture. Two extra forwards at
warmup, nothing at steady state.

**The result is narrower than the microbenchmark predicted.** Isolated, the
fused GEMM looks worth ~230 µs/iter (20.6 → 38.2 TFLOPS at the default
shape). In-model it is worth ~120 µs of GEMM time, because the three
separate GEMMs get L2 reuse that a cold microbenchmark does not show. And it
only helps at *small* M: at `big_fp16` (M=16384) the N=d_model form already
produces 2048 tiles, plenty for 36 SMs, so fusion changes nothing there. The
one configuration that gets both halves — M small enough for tile count to
matter, *and* bit-exact — is bf16 at the default shape:

| | main | step 8 |
|---|---|---|
| default_bf16 optimized median (3 reps) | 1.5284–1.5302 ms | **1.4815–1.4834 ms** (−3.1%) |

Everything else unchanged within noise; still 5/5 bit-exact.

### Step 9 — let inductor generate the layout copies

`torch.compile` is banned everywhere else in this project because it changes
numerics. It cannot change a rounding decision that is never made, though —
and the q/k/v head transpose and the post-attention merge are *pure data
movement*. Verified `torch.equal`, max_abs exactly 0, in fp16/bf16/fp32,
alongside the neighbouring ops that fail the same test and are therefore
deliberately left on ATen:

| op under inductor | bit-exact? | fp16 max_abs |
|---|---|---|
| split-heads copy / merge-heads copy | **yes** | 0 |
| residual add + masked_fill | **yes** | 0 |
| `layer_norm` | no | 9.8e-4 |
| fused `add + layer_norm` (what inductor prefers to emit) | no | 7.8e-3 |
| `gelu` | no | 3.8e-6 |

ATen's `direct_copy_kernel` is an elementwise kernel with 4-D index math and
no vectorization on the strided side; at these shapes on 36 SMs it is
launch/occupancy-bound, not bandwidth-bound. Timed **under CUDA-graph
replay** — the way it actually runs; eager wall-clock is swamped by dynamo's
per-call guard overhead, which graph replay removes entirely — for one
layer's three q/k/v copies:

| shape | ATen | inductor | hand-written Triton |
|---|---|---|---|
| B8 S128 D512 | 15.8 µs | **8.2 µs** | 9.3 µs |
| B32 S512 D512 | 293.9 µs | **269.9 µs** | 515.7 µs |
| B4 S2048 D512 | 154.4 µs | **130.3 µs** | 252.6 µs |
| B16 S256 D1024 | 150.1 µs | **132.4 µs** | 253.3 µs |

Inductor wins at every shape, and beats a Triton kernel hand-tuned for this
GPU everywhere except the smallest — step 7's lesson in the opposite
direction. Writing the kernel by hand was the wrong instinct twice now.

Two traps, both worth recording:

* **dynamo specializes on strides, not just shape/dtype.** With step 8's
  fused QKV enabled, q/k/v are non-contiguous column slices of one packed
  `[B,S,3D]` GEMM output, so a probe that passed *contiguous* tensors left a
  second variant to be compiled later — and "later" turned out to be inside
  `torch.cuda.graph()` capture, where that compile hit the Windows
  static-launcher `OverflowError`, failed the capture, and permanently
  demoted the model to the eager path. Measured: bf16 1.48 ms → **3.44 ms**,
  while still reporting bit-exact results the whole time. The probe now
  drives a real `_forward_core` both ways, so every variant compiles inside
  the workaround patch, before capture is ever attempted.
* **`use_static_cuda_launcher` is not limited to max-autotune GEMM
  templates**, as step 7 assumed — it fires on an ordinary pointwise clone
  too. It is patched only around the probe, never globally, so the harness's
  own `--compile-baseline` bar keeps its normal launcher and the comparison
  stays fair.

Optimized-latency A/B (median; repetitions are non-overlapping to four
significant figures):

| case | main | step 8 | step 9 |
|---|---|---|---|
| default_fp32 | 1.3715 | 1.3586 | 1.3572 |
| default_fp16 | 1.6706 | 1.6709 | **1.5395** |
| default_bf16 | 1.5312 | 1.4835 | **1.3580** |
| causal_fp16 | 1.6796 | 1.6803 | **1.5515** |
| padded_fp16 | 1.7566 | 1.7576 | **1.6368** |
| big_fp16 | 35.309 | 35.255 | **34.090** |
| long_fp16 | 37.528 | 37.627 | **36.952** |
| wide_fp16 | 41.102 | 40.785 | **39.635** |


### Step 10 — investigated: fp8 weight quantization (no change merged)

Question: can less-sensitive weights go to fp8 (e4m3) with fine-grained
group scales, staying close to the fp32 reference? Measured end to end.
Answer: **no, nowhere in this model** — and it fails on two independent
axes, either of which alone is disqualifying.

**Platform support first** (torch 2.8 / sm_120, `torch._scaled_mm`):

| scale granularity | supported? |
|---|---|
| per-tensor scalar | yes |
| per-row / per-channel | **no** — `Per-row scaling is not supported for this platform!` |
| MX block-scaled, e8m0, block=32 | **yes** (block=128 rejected; 32 is the only valid block) |
| `out_dtype` fp16 / bf16 / fp32 | all yes |
| fused bias | fp16/bf16 out only — *not* with fp32 out |

So the "fine-grained group scale" form is available, but only as MX
(`float8_e8m0fnu` scales, one per 32 elements along K). e8m0 scales are
powers of two, so the scaling itself contributes **no** rounding error;
all error is e4m3 mantissa truncation.

**Trap worth recording: the MX scale tensor must be in a 128x4 swizzled
layout, and passing a linear `[rows, K/32]` layout does not raise — it
silently returns wrong numbers.** A linear layout measured `err/std =
0.412` against an fp32 matmul; the same data in the swizzled layout gives
`0.0299`, which is exactly e4m3's expected round-trip error. The shape
check passes either way, so this reads as "fp8 is inaccurate" rather than
"the layout is wrong" unless it is checked against a reference.

**Measurement 1 — fp8 GEMM ceiling.** The raw hardware win is real and
large (best-of-3, `torch.cuda.Event`, 25 warmup + 100 iters, at the shapes
this codebase runs):

| shape (MxKxN) | fp16 ms | fp8/per-tensor ms | mxfp8 ms | mxfp8 TFLOPS | mx vs fp16 |
|---|---|---|---|---|---|
| qkv_fused default (1024x512x1536) | 0.0405 | 0.0232 | 0.0172 | 93.5 | 2.35x |
| out_proj default (1024x512x512) | 0.0248 | 0.0150 | 0.0149 | 36.0 | 1.66x |
| ffn_in default (1024x512x2048) | 0.0525 | 0.0306 | 0.0224 | 95.9 | 2.34x |
| ffn_out default (1024x2048x512) | 0.0537 | 0.0307 | 0.0191 | 112.7 | 2.81x |
| qkv_fused big (16384x512x1536) | 0.5588 | 0.2982 | 0.2104 | 122.5 | 2.66x |
| ffn_in big (16384x512x2048) | 0.7875 | 0.4022 | 0.2829 | 121.5 | 2.78x |
| ffn_out big (16384x2048x512) | 0.7446 | 0.3701 | 0.2422 | 141.8 | 3.07x |
| ffn_in wide (4096x1024x4096) | 0.7548 | 0.3769 | 0.2468 | 139.2 | 3.06x |
| ffn_out wide (4096x4096x1024) | 0.7774 | 0.3694 | 0.2339 | 146.9 | 3.32x |

mxfp8 peaks at ~147 TFLOPS against the measured fp16 peak of 48.8 (§5b),
the expected ~3x tensor-core generation ratio — and MX is *faster* than
per-tensor fp8, i.e. sm_120 has native block-scaled MMA rather than
emulating it. This is the largest single GEMM lever this project has
measured. Both other axes kill it anyway.

**Measurement 2 — weight distributions, and why group scaling is inert
here.** Default config, layer 0:

| matrix | numel | absmax | std | per-32-block amax min/med/max |
|---|---|---|---|---|
| q/k/v/out_proj | 262144 | 0.0442 | 0.0255 | 0.0332 / 0.0432 / 0.0442 |
| ffn_in | 1048576 | 0.0442 | 0.0255 | 0.0322 / 0.0433 / 0.0442 |
| ffn_out | 1048576 | 0.0221 | 0.0128 | 0.0158 / 0.0216 / 0.0221 |

The total spread of block maxima is ~1.37x. Group scaling exists to isolate
**outliers** — and there are none, because these are default `nn.Linear`
uniform-init weights, not trained ones. Weight round-trip relative RMS
error is therefore *identical to four decimal places* at every granularity:

| granularity | per-tensor | block=256 | block=128 | block=64 | block=32 |
|---|---|---|---|---|---|
| rel_rms | 0.02714 | 0.02714 | 0.02714 | 0.02714 | 0.02714 |

(fp16, for contrast: 0.000217 — 125x better.) **The finer granularity buys
literally nothing on this model.** The error is mantissa width, and no
scale layout fixes a 3-bit mantissa. This is a property of the benchmark's
untrained weights; on a real trained model with outlier channels, group
scaling would matter a great deal — but the wall below would still bind.

**Measurement 3 — the hard accuracy wall.** e4m3's own relative precision
(rel_rms **2.71%**) is *larger than the harness's rtol* (**2%**), before a
single layer of compounding. A format whose inherent error exceeds the
tolerance cannot be placed anywhere. End-to-end vs the fp32 eager baseline
at the default shape, harness criterion (`|err|<=0.002 OR rel<=0.02`, zero
failures required):

| weights quantized | mode | failed / 524288 | max_abs | verdict |
|---|---|---|---|---|
| all (current fast path) | **fp16** | **0** | **0.00129** | **PASS** |
| ffn_in+ffn_out | weight-only, fp32 GEMM | 239991 | 0.0803 | FAIL |
| ffn_in only | weight-only, fp32 GEMM | 185123 | 0.0587 | FAIL |
| ffn_out only | weight-only, fp32 GEMM | 182513 | 0.0566 | FAIL |
| out_proj only | weight-only, fp32 GEMM | 51678 | 0.0155 | FAIL |
| v_proj only | weight-only, fp32 GEMM | 51464 | 0.0161 | FAIL |
| all | mxfp8 (both operands) | 301518 | 0.1251 | FAIL |

Pushed to the most favorable case that exists — **one matrix, in one
layer, weight-only, with the GEMM itself still exact fp32** (strictly the
least damage an fp8 weight can do):

| layer | matrix | failed / 524288 | max_abs | verdict |
|---|---|---|---|---|
| L0 | attn.v_proj | 3186 | 0.00458 | FAIL |
| L0 | attn.out_proj | 4317 | 0.00515 | FAIL |
| L0 | ffn_in | 82198 | 0.0250 | FAIL |
| L0 | ffn_out | 81414 | 0.0266 | FAIL |
| L5 | attn.v_proj | 15065 | 0.00857 | FAIL |
| L5 | ffn_in | 74393 | 0.0238 | FAIL |

The gentlest possible application still lands at 2.3x the atol. Meanwhile
fp16 across **every** matrix in **every** layer uses only 64% of the atol
budget. There is no subset with headroom — the gap is a factor of ~3.5 at
the very best, not a few percent to be tuned away.

**Measurement 4 — even ignoring accuracy, it would be slower.** The GEMM
speedup is real but the operands have to get *into* fp8. Weights are
pre-quantized once and cached (the `_get_linear_fp16_weights` pattern), so
only activations are quantized per call — an amax reduction over each
32-element block, `log2`/`ceil`/`exp2`, a divide, a cast, and the 128x4
scale swizzle:

| shape | fp16 linear | fp8 GEMM alone | fp8 + activation quant | net |
|---|---|---|---|---|
| ffn_in default | 0.0526 | 0.0241 | 0.1991 | **0.26x** |
| ffn_out default | 0.0536 | 0.0190 | 0.1887 | **0.28x** |
| ffn_in big | 0.7470 | 0.2866 | 0.9067 | **0.82x** |
| ffn_out big | 0.7127 | 0.2490 | 2.8688 | **0.25x** |

Quantization costs 4-8x what the GEMM saves: fp8 is **3.5-4x slower than
the existing fp16 path** end to end at the default shape. (This quantizer
is straightforward PyTorch, so a fused Triton kernel would narrow the gap
— recorded as a caveat, not a defense, since the accuracy wall is
independent and unconditional.)

The same "the epilogue eats the win" pattern sinks the per-channel
workaround. Per-channel weight scaling is expressible without native
support, since `w[k,n] = wq[k,n]*s[n]` gives `out[m,n] = s[n] * sum_k
a[m,k]*wq[k,n]` — a post-GEMM broadcast multiply. Priced at ffn_in default:

| | ms | vs fp16 |
|---|---|---|
| fp16 `F.linear` | 0.0527 | 1.00x |
| `_scaled_mm` -> fp16 (fused bias) | 0.0306 | 1.72x |
| `_scaled_mm` -> fp32 + per-channel scale + bias | 0.0465 | 1.13x |
| ... + cast back to fp16 | 0.0539 | **0.98x** |

fp32 output forfeits the fused bias (unsupported), so the epilogue becomes
a separate full pass over `[M,N]` and gives the entire 1.72x back.

**Conclusion: no code change.** fp8 is disqualified twice over, and the
binding constraint is the one no amount of engineering moves: at rtol=2%,
the tolerance is *tighter than e4m3's own precision*. The three levers
that make fp8 work in production LLM inference — outlier-isolating group
scales, a loss-based sensitivity budget, and tolerance for ~1% output
drift — are all absent here. This project's accuracy contract is
bit-exactness-or-near-it against an fp32 reference, and fp16 already
spends 64% of that budget for a 1.8-2x GEMM win; fp8 asks for 20x the
error to buy at most another 1.7x that is then given back at the
quantization step. The existing `_resolve_fp16_gemm_enabled` path (step 3)
remains the correct precision floor for this workload.

### Step 11 - exploiting the fact that the test shapes are given

Everything through step 9 was tuned against the `default` suite, whose shape
(batch=8, seq=128, d_model=512, ffn=2048) is 85% GEMM and therefore
compute-bound. **The graded configurations are not that workload at all**, and
optimizing for them is a different problem.

#### The shapes, grouped

The 14-row grading matrix (`run_bench.py`'s `user_matrix` suite) splits into
four groups by what actually limits them:

| group | rows | shape | what limits it |
|---|---|---|---|
| **A. tiny model, batch/head/ffn sweep** | 01-06, 09-13 | d_model 128, ffn 32/128/1024, S=128, L=4 | memory + per-kernel overhead; dense matmul is only ~40% of the work |
| **B. degenerate small** | 07 | d_model 32, S=32, B=64 | almost entirely fixed per-kernel cost |
| **C. long/wide** | 08 | d_model 1024, S=1024, B=64, head_dim 256 | genuinely GEMM-bound, like the default suite |
| **D. extreme ffn** | 14 | ffn_dim 100000, B=32, S=1024 | capacity; already handled by step 5's chunked FFN |

The properties worth exploiting are in groups A and B, and they are structural,
not incidental:

* **d_model is 32 or 128 in 12 of 14 rows** (1024 in the other two), always a
  power of two, never above 1024. An entire residual row therefore fits in one
  Triton block.
* **seq_len is 32 or 128 in 12 of 14 rows.** An entire `[S, S]` attention score
  matrix for one (batch, head) therefore fits in one block.
* **ffn_dim is <= d_model in 11 of 14 rows** (32 or 128 against d_model=128).
* head_dim ranges over 8/32/64/128/256 - and 8 and 32 are small enough that
  cuBLAS and SDPA are both operating well outside the regime they are tuned
  for.

#### Where the time actually went (measured, not assumed)

Profiled under CUDA-graph replay at three representative group-A/B rows, on
the step-9 code:

| | 01_base | 07_seq32 | 13_ffn1024 |
|---|---|---|---|
| LayerNorm (9 launches) | **35.5%** | **42.5%** | 20.3% |
| GEMMs | 28.4% | 17.3% | 50.8% |
| attention (`fmha_cutlassF`) | 17.6% | 20.3% | 9.0% |
| residual adds + dtype casts | 9.6% | 11.4% | 6.1% |
| GELU | 2.0% | 3.2% | 8.4% |

LayerNorm being the single largest item is the whole story of this step. ATen's
`layer_norm_kernel` is written for wide rows; at d_model=32 it moved 512 KB in
9.4 us - about **55 GB/s** on a card that sustains several hundred. That gap
does not exist at d_model=512, which is why nine steps of tuning never
surfaced it.

#### Kernel 1: fused residual-add + LayerNorm (`_add_ln_kernel`)

One kernel replaces the four-to-five ATen kernels at each of the model's
residual sites:

```
(fp16->fp32 cast) -> masked_fill -> add -> layer_norm -> (fp32->fp16 cast)
```

Each of those streamed the whole `[B, S, d_model]` activation separately. The
fused version reads `x` and `delta` once and writes the updated residual and
the normalized output once. `_forward_core` was restructured around a single
`_ln_site` helper to make this a call-site decision rather than a second copy
of the layer body: the residual add that closes each sub-block is handed to the
*next* LayerNorm as a pending `delta` instead of being applied eagerly.

The given shapes are what make the kernel simple: `BLOCK_D = next_pow2(d_model)`
covers the entire feature axis, so the reduction is a single in-register pass
with no loop and no cross-block communication, and several rows fit in one
block (`BLOCK_M`), which is what fixes d_model=32 - one row per block would
have launched 65536 blocks of 32 elements.

Two things this surfaced that were not obvious:

* **The residual must be rounded to its own storage dtype before the norm
  reads it.** Keeping the fp32 accumulator alive into the reduction is *more*
  accurate than the eager chain, which materializes `x + delta` as a real
  tensor first - and this project counts being more accurate than the
  reference as divergence too. Measured 0.0156 (two bf16 ulps) off the ATen
  chain before the fix; a no-op for an fp32 residual.
* **No autotuning**, deliberately, unlike the other Triton kernels here. This
  one runs inside the CUDA-graph-captured region in several flag variants per
  layer, and autotuning benchmarks on first call, which syncs. A deterministic
  BLOCK_M/num_warps rule removes that hazard class outright, and this kernel is
  pure streaming plus one in-register reduction, where the spread between
  reasonable configs is small.

#### Follow-up: per-shape-group launch parameters for kernel 1

The first version picked `BLOCK_M`/`num_warps` from a guessed target of ~2048
elements per block. Sweeping `BLOCK_M x num_warps` over 1..64 x 1..8 at every
d_model the graded groups use -- timed under CUDA-graph replay, since an eager
sweep at 1-13 us per launch ranks the 5-10 us launch floor instead of the
kernel -- showed that guess was wrong in a consistent direction:

| shape | d_model | 2048-elem rule | measured best | gap |
|---|---|---|---|---|
| 07_seq32 | 32 | BM=8, w=1: 1.37 us | BM=16, w=4: 1.26 us | 1.08x |
| 02_batch1 | 128 | BM=8, w=4: 1.29 us | BM=1, w=1: 1.19 us | 1.08x |
| 04_batch16 | 128 | BM=8, w=4: 2.81 us | BM=4, w=1: 2.41 us | 1.16x |
| 01_base | 128 | BM=8, w=4: 7.33 us | BM=4, w=4: 6.60 us | 1.11x |
| default_fp32 | 512 | BM=4, w=8: 5.34 us | BM=1, w=4: 4.04 us | **1.32x** |
| 06_batch10000 | 128 | BM=8, w=4: 5263 us | BM=4, w=8: 5242 us | 1.00x |
| 08_seq1024 | 1024 | BM=2, w=8: 2158 us | BM=16, w=8: 2151 us | 1.00x |

The optimum tracks **~512 elements per block**, not 2048, across the whole
range: d_model=32 wants BLOCK_M=16, d_model=128 wants 4, d_model=512 wants 1 --
all the same tile. Separately, at batch=1 (M=128 rows) even the 512-element
rule leaves 32 blocks for 36 SMs, so a block-count floor (halve BLOCK_M until
the grid covers the device twice over) matters more there than tile size does.
`_add_ln_launch_params` is now those two rules and takes the row count and
device, and it reaches the measured optimum at 6 of 9 shapes with the worst
remaining gap 1.12x (was 1.32x).

**End to end this is worth nothing measurable: 1.000x geomean over the graded
matrix.** Repeating an identical configuration five times per case puts the
per-process noise floor at 1.006x-1.059x spread, and the whole A/B scatter
(0.958x-1.086x, no consistent sign) sits inside it. The kernel is 12-16% of the
forward, so a 10% kernel win is ~1.5% end to end -- below what this benchmark
can resolve at these shapes. Kept anyway: it is a strictly better-founded rule
at equal complexity, it is reproducibly faster at the kernel level at every
shape, and the two shapes where this kernel actually dominates the forward
(batch=10000, seq=1024) are bandwidth-bound and were already at their optimum
under either rule. Recorded here so it is not re-derived later as an
unexplored idea.

#### Kernel 2: single-block short-sequence attention (`_short_attn_kernel`)

With all `S` keys resident in one program, the softmax is a plain single-pass
row reduction - there is no need for the running-max/running-sum rescale that a
key-tiled flash-style kernel is forced into. `fmha_cutlassF` is tiled for long
sequences and at S=128 is doing that bookkeeping for a problem that needs none
of it: measured ~15 TFLOPS against a ~44 TFLOPS ceiling and 58% of the
bandwidth roof. That is a structural mismatch to the shape, not a tuning gap -
which is the distinction step 7 got wrong in the other direction.

The kernel also returns the context already merged to `[B, S, d_model]`, by
allocating the output in `[B, S, H, HD]` order and handing the kernel
transposed strides, so the merge stays a free view.

**Two measurements that changed the design:**

* **The query axis has to be tiled for occupancy, and BLOCK_Q/num_warps must be
  autotuned, not chosen by rule.** A fixed heuristic (`BLOCK_Q=128,
  num_warps=1` at head_dim=8) made `11_heads16` **4x slower than the ATen path
  it replaced**. Sweeping the config space at that shape:

  | config | time | vs best |
  |---|---|---|
  | BLOCK_Q=64, num_warps=4 | 75 us | 1.00x (SDPA: 121 us) |
  | BLOCK_Q=128, num_warps=4 | 85 us | 1.13x |
  | BLOCK_Q=32, num_warps=8 | 369 us | 4.9x |
  | BLOCK_Q=128, num_warps=1 | 803 us | **10.7x** |

  `num_warps=1` lost at every shape tried, by 2x to 10x - one warp has to hold
  the whole `[BLOCK_Q, BLOCK_S]` score tile and spills. The best config also
  genuinely moves with the shape (BLOCK_Q=128 wins at head_dim=32, BLOCK_Q=64
  at head_dim=8 and 128), so it is autotuned over 9 configs. That is legal
  inside the graph only because `_probe_short_attn` drives a whole forward with
  the kernel enabled before capture is ever attempted - the same pre-capture
  warmup contract `_fused_scale_mask_kernel` already relied on.
* A softmax denominator broadcast along the wrong axis (`probs / denom` instead
  of `probs / denom[:, None]`) produced max_abs 3.49. The gate caught it and
  disabled the kernel; the model stayed correct. Worth recording as the reason
  the probes compare *whole forwards* rather than spot-checking a tile.

#### What was investigated and NOT changed

**A Triton GEMM for the narrow-N shapes.** Profiles showed cuBLAS dispatching
these to `cutlass_80_wmma_tensorop` (~20-26 TFLOPS) rather than the `s16816`
mma.sync kernel it uses at d_model=512 (~39-44 TFLOPS), which looked like step
7's conclusion might not hold at these shapes. It does hold, for a different
reason than step 7's: at K=128 these GEMMs are **bandwidth-bound, not
compute-bound**. One layer's four GEMMs at `01_base` move 20 MB and cuBLAS runs
them at ~78% of this card's bandwidth roof, so the entire available win is
~1.28x even against a perfect kernel - and the fp16 TFLOPS figure is simply the
wrong yardstick for the shape.

A related trap worth recording: an eager microbenchmark says every GEMM with
K=32 or N=32 takes 12.8-13.9 us *regardless of size* (`2048x32x32` costs the
same as `8192x128x128`). That is the Python launch floor, not GPU time - under
CUDA-graph replay the same GEMMs measure ~2 us. Any conclusion about small
shapes drawn from an eager microbenchmark here is wrong.

**In-process A/B measurement.** Building many models in one process trips
dynamo's recompile limit (the step-9 compiled-layout probe silently turns
itself off after ~8 instances) and grows the CUDA graph pool; an in-process
three-arm A/B swung 30% between repetitions and reported a fictitious 0.65x
regression on `default_fp16`. All numbers below are one process per (case,
arm).

#### Results

Two things are being measured and they answer different questions. The A/B
below isolates *what these two kernels contributed*, against the step-9 code
on the same hardware in the same process-per-measurement protocol. The harness
table after it is the actual bar: `run_bench.py --suite user_matrix`, each row
scored against its own `torch.compile`d baseline.

Graded matrix, fp32, one process per measurement, median of 80
CUDA-event-timed replays (rows 06 and 14 excluded here - capacity cases whose
runtimes are dominated by data movement this A/B cannot resolve):

| case | step 9 | + fused LN | + short attn | LN | attn | total |
|---|---|---|---|---|---|---|
| 01_base | 0.8835 | 0.5254 | 0.4830 | 1.681x | 1.088x | **1.829x** |
| 02_batch1 | 0.1489 | 0.1129 | 0.0910 | 1.319x | 1.240x | **1.635x** |
| 03_batch4 | 0.1634 | 0.1109 | 0.1280 | 1.472x | 0.867x | 1.277x |
| 04_batch16 | 0.3029 | 0.1892 | 0.1948 | 1.601x | 0.972x | 1.556x |
| 05_batch128 | 1.7870 | 1.0185 | 0.9504 | 1.755x | 1.072x | **1.880x** |
| 07_seq32 | 0.2272 | 0.1228 | 0.1218 | 1.851x | 1.008x | **1.866x** |
| 08_seq1024 | 136.02 | 107.76 | 107.84 | 1.262x | 0.999x | 1.261x |
| 09_heads1 | 0.8866 | 0.5193 | 0.4710 | 1.707x | 1.103x | **1.882x** |
| 10_heads2 | 0.8441 | 0.4806 | 0.4680 | 1.756x | 1.027x | **1.804x** |
| 11_heads16 | 1.2231 | 0.8601 | 0.6755 | 1.422x | 1.273x | **1.811x** |
| 12_ffn32 | 0.8311 | 0.4660 | 0.4157 | 1.783x | 1.121x | **1.999x** |
| 13_ffn1024 | 1.4614 | 1.0954 | 1.0443 | 1.334x | 1.049x | 1.400x |
| **geomean** | | | | **1.566x** | **1.063x** | **1.664x** |

The attention kernel is worth +6.3% on top of the LayerNorm fusion and is a
clear win on the larger group-A rows; on the three smallest cases it sits
within run-to-run noise of 1.0x (those measurements are 0.1-0.2 ms and move
~20% between repetitions). `08_seq1024` is group C: head_dim=256 and S=1024 put
it outside both kernels' supported range, and it gets only the LayerNorm
fusion.

Kernel-level, at `01_base`: LayerNorm 314 us -> 61 us while also absorbing the
residual adds and all 16 dtype-cast kernels, and 59 -> 35 kernel launches per
iteration; attention 139 us -> 78 us.

Against the stated bar (`run_bench.py --suite user_matrix`, each row vs its own
`torch.compile`d baseline):

| case | optimized ms | vs compiled bar | MFU |
|---|---|---|---|
| 01_base | 0.4541 | 2.214x | 38.76% |
| 02_batch1 | 0.0711 | **12.191x** | 3.87% |
| 03_batch4 | 0.0854 | **10.006x** | 12.88% |
| 04_batch16 | 0.1674 | 5.153x | 26.29% |
| 05_batch128 | 0.9149 | 3.137x | 38.48% |
| 06_batch10000 | 108.6387 | 3.177x | 25.32% |
| 07_seq32 | 0.0772 | **11.040x** | 6.23% |
| 08_seq1024 | 107.2529 | 2.026x | 65.65% |
| 09_heads1 | 0.4459 | 1.744x | 39.48% |
| 10_heads2 | 0.4499 | 1.896x | 39.12% |
| 11_heads16 | 0.6643 | 5.255x | 26.50% |
| 12_ffn32 | 0.4008 | 2.355x | 35.68% |
| 13_ffn1024 | 1.0399 | 2.402x | 46.55% |
| **geomean (13/13 PASS)** | | **3.764x** | |

Row 14 (ffn_dim=100000) is a capacity case and has to be measured separately,
because the *unmodified* reference OOMs at that shape and needs step 5's
`--chunk-baseline-ffn`. It passes with the new kernels live - 0 of 33,554,432
elements failing, max_abs 0.0013 - at 688.3 ms against a 1453.9 ms chunked
eager baseline, 2.11x. (Not comparable to the 15.77x recorded for this shape in
step 5: that figure's warmup/repeat settings are not recorded, and this one was
taken with `--warmup 3 --repeats 10 --benchmark-rounds 1` to keep a 1.5 s/iter
baseline tractable. The point of re-measuring it here is that it still passes
and is still faster, not a like-for-like comparison.) Only the LayerNorm
fusion applies to it; S=1024 puts it outside the attention kernel's range. The very large ratios on rows 02/03/07 are the launch-overhead regime:
those forwards are 70-90 us, where CUDA-graph replay plus a 59 -> 35 launch
reduction dominates everything else, and MFU is correspondingly meaningless
there (3.9% at batch=1 - the GPU is essentially idle either way, we just stop
waiting on the CPU sooner).

The `default` suite is unaffected where it should be and better where it can
be - the fp16/bf16 cases stay **bit-exact** (both probes correctly refuse to
enable there), and `default_fp32` picks up the LayerNorm fusion and the
attention kernel:

| case | step 9 | step 11 | vs compiled bar | accuracy |
|---|---|---|---|---|
| default_fp32 | 1.3572 | **1.1804** | 2.051x | PASS |
| default_fp16 | 1.5395 | 1.5489 | 0.962x | PASS (max_abs 0) |
| default_bf16 | 1.3580 | 1.3573 | 1.318x | PASS (max_abs 0) |
| causal_fp16 | 1.5515 | 1.5518 | 0.950x | PASS (max_abs 0) |
| padded_fp16 | 1.6368 | 1.6307 | 0.896x | PASS (max_abs 0) |
| **geomean** | 1.189x | | **1.172x** | 5/5 |

A 60-configuration accuracy sweep (10 shapes x {fp32, fp16, bf16} x {no
padding, 40% padding}, 3 trials each) passes everywhere: fp32 max_abs
9.6e-4 - 1.6e-3 against an atol of 2e-3, and fp16/bf16 max_abs exactly 0.

#### Answering the question that started this step

Before it, **one** Triton kernel shipped (`_fused_scale_mask_kernel`, an
elementwise scale+mask on the fp16/bf16 attention epilogue), and it was not
shape-specialized. The two hand-written kernels of steps 7 and 9 - a GEMM and a
layout copy - were both measured and **rejected**. So: zero shape-specific
Triton kernels. This step adds two, and both win for the same reason the
earlier two lost: they are not attempts to out-tune a vendor kernel at its own
game, they exploit a structural property of the given shapes (a whole row, or a
whole score matrix, fits in one block) that the general-purpose kernel cannot
assume.


## 4. Rejected after measurement

| idea | why rejected |
|---|---|
| **KV caching** | Not applicable. The harness does one full-sequence forward per call, with no autoregressive decode and no cross-call state. There is no incremental step to cache K/V for, and caching across identical benchmark calls would just memoise the answer. |
| `torch.compile` on our own model (fp16/bf16) | Changes numerics past tolerance vs the eager reference — the same drift that makes the compiled baseline non-compliant. |
| Whole-model fp16 / bf16 cast | fp16 max_abs 0.0082, bf16 0.074 — both fail. |
| Error-compensated fp16 (hi+lo split GEMM) | Two fp16 GEMMs cost about as much as one TF32 GEMM. No win. |
| Folding the attention scale into `W_q`/`b_q` | 3.9e-3 divergence in fp16 (bit-exact in bf16 only). Unsafe. |
| ~~Fused QKV in fp16/bf16~~ | **Overturned in step 8.** Not kernel selection at all -- the fused and separate GEMMs dispatch to the *same* cutlass kernel. bf16 is bit-exact at every shape tried; fp16 at most. Now decided per-shape by a warmup `torch.equal` probe. |
| **fp8 weight quantization (e4m3, incl. MX block-32 group scales)** | Fails twice over (step 10). Accuracy: e4m3's own relative precision (2.71% rel_rms) **exceeds the harness's 2% rtol**, so no placement works -- even ONE matrix in ONE layer, weight-only with an exact fp32 GEMM, lands at 2.3x the atol, while fp16 across the whole model uses 64% of the budget. Group scaling is inert here (identical error at every granularity: these untrained weights have no outliers to isolate). Speed: activation quantization costs 4-8x what the faster GEMM saves -- 0.26x vs the fp16 path at the default shape -- despite mxfp8 hitting a genuine ~147 TFLOPS (3.3x fp16) in isolation. |
| Triton fused softmax reduction | Reduction tree differs from ATen; compounds to 0.0078 over 6 layers. Still true for fp16/bf16; step 11's short-sequence attention kernel does exactly this and is therefore fp32-only, gated on the calibration bar. |
| Triton GEMM for the graded shapes' narrow-N matmuls | Step 7's conclusion holds, but for a different reason at these shapes: at K=128 the GEMMs are bandwidth-bound and cuBLAS already runs at ~78% of the card's bandwidth roof, capping any possible win at ~1.28x. The low fp16 TFLOPS figure is the wrong yardstick. |
| Inductor max-autotune GEMM (or a hand-written Triton GEMM) | Disabled on this GPU by inductor's SM-count gate (36 < 68 SMs); bypassed it and also hand-tuned a Triton GEMM for this exact hardware in step 7 -- cuBLAS still wins or ties at every shape this codebase actually runs. |

## 5. Behaviour against a compiled baseline

If the harness is run with `--compile-baseline`, the reference itself changes:

| dtype | verdict | failing elements | speedup |
|---|---|---|---|
| fp32 | PASS | 0 / 2,621,440 | 1.589x |
| fp16 | FAIL | 108 / 2,621,440 (0.004%) | 1.030x |
| bf16 | FAIL | 325,859 (12.4%) | 1.128x |

Those fp16 failures closely match the count obtained by comparing the **eager
baseline against its own compiled self** (about 21 per trial over 5 trials). In
other words, this implementation is as close to the compiled baseline as the
reference implementation itself is; the failures are the compiled baseline's own
numerical drift, not an artifact of these optimizations.

## 5b. Standing measurement protocol: MFU alongside the latency ratio

Every `run_bench.py` run now reports each case's **MFU** (Model FLOPs
Utilization) alongside the existing latency-ratio speedup, and the
arithmetic-mean MFU across all shapes in the suite. Dense forward FLOPs
(QKV + attn QK^T + attn@V + out_proj + FFN GEMMs; LayerNorm/softmax/GELU
excluded, the standard convention; no discount for causal masking since the
full `[S,S]` score matrix is computed either way) divided by measured
latency, divided by this GPU's peak FLOPs/s at the precision *actually
executed*.

Peak FLOPs/s is **measured empirically** on this card (large square GEMMs),
not taken from a spec sheet — web sources for the RTX 5060 Ti's tensor-core
TFLOPS were inconsistent (47/120/200 TFLOPS all cited for the same SKU):
TF32 ~24.7, FP16 ~48.8, BF16 ~49.4 TFLOPS. FP16 measuring ~2x TF32 is the
expected, generation-independent tensor-core ratio and a sanity check these
are real.

Found and fixed a bug while wiring this up: an fp32 case where the
fp16-GEMM calibration gate (§3) silently enables its fast path was being
scored against the TF32 peak instead of the FP16 peak it actually runs on,
producing an impossible 120% MFU. Fixed by surfacing the gate's decision via
the existing `TJ_DEBUG_GATE` stderr marker and picking the matching peak.

**MFU is only a fair comparison within the same executed precision.**
Current `default` suite:

| case | speedup vs bar | MFU (ours) | MFU (compiled bar) |
|---|---|---|---|
| default_fp32 | 1.785x | 60.79%* | 67.31% |
| default_fp16 | 0.985x | 53.60% | 54.39% |
| default_bf16 | 1.319x | 60.02% | 45.50% |
| causal_fp16 | 0.966x | 53.18% | 55.04% |
| padded_fp16 | 1.059x | 50.41% | 47.58% |
| **average** | **1.189x geomean** | **55.60%** | **53.96%** |

As of step 9 the average MFU is above the compiled bar's for the first
time. The per-case speedup column still moves several percent between
runs -- the bar itself has ~30% cross-process variance -- so branch
decisions are made on the *optimized* latency column, which is stable to
four significant figures across repetitions.

\* against the FP16 peak (48.8 TFLOPS), since the gate enables fp16-GEMM
here; the bar's 67.19% is against TF32's peak (24.7 TFLOPS) since
`BaselineTransformer` never takes that path. Different ceilings — our lower
percentage of a higher ceiling (~29.8 TFLOPS achieved) still beats the
bar's higher percentage of a lower one (~16.6 TFLOPS achieved), which is
exactly why default_fp32 is faster despite the lower MFU%. For the fp16/bf16
rows both sides share the same peak, so those percentages ARE directly
comparable, and confirm the already-known story: the compiled baseline's
fusion (unavailable to us there without breaking bit-exactness) gives it a
genuine edge on `default_fp16`/`causal_fp16`/`padded_fp16`.

## 6. Reproducing

```
python run_bench.py --suite default      # 5 configs, correctness + speed
python run_bench.py --suite full         # 10 configs incl. long-seq and wide
python run_bench.py --suite user_matrix  # the 14-row graded matrix (step 11)
python profile_baseline.py float32       # kernel-level profile of the bar
```

Per-shape gate/probe decisions are printed to stderr with `TJ_DEBUG_GATE=1`:
which of `fp16-GEMM`, `fused-QKV`, `compiled-layout`, `short-attn` and
`fused-LN` were enabled for a given configuration, and the measured error
that decided it.

## 7. Observation: progressive residual-stream boundary push (fp32-target, not baseline-bf16-match)

Reframed question: not "match baseline's own low-precision arithmetic" but
"match baseline's fp32 output, using as much fp16 internally as passes
calibration" -- i.e., extend the existing fp16-GEMM gate's scope further into
the pipeline. Current merged design (`_forward_core`'s `use_fp16_gemm` path)
already runs every GEMM, attention, and GELU in fp16; only the residual
stream `x` itself and LayerNorm's protected input stay fp32.

Tested pushing the residual stream itself into fp16, progressively, at
default/causal/deep(L12,d1024) shapes, across 3 seeds x 6 input scales, at
the SAME 0.9-margin gate the production code uses:

- Level A (current): x fp32 throughout -> passes down to scale=1.0, fails at
  0.5 and below (matches the production gate's documented behavior).
- Level B (x fp16 through attention's residual add only, back to fp32 before
  FFN): fails starting at scale=1.0 (default/causal) or scale=2.0 (deep) --
  i.e., fails at the realistic default scale immediately.
- Level C (x fp16 through both residual adds): barely worse than B
  (max_abs 0.00853 vs 0.00782 at default/scale=1.0) -- the SECOND fp16 touch
  adds little on top of the first.

**Observation, not a closed conclusion**: there is no calibration-passing
middle ground between "residual stays fully fp32" (current) and "residual
touches fp16 at all" (fails immediately at realistic scale) for this specific
way of dropping precision (an immediate half-cast at the residual add). B and
C being nearly identical suggests the FIRST fp16 rounding of the accumulated
residual does most of the damage -- it re-rounds the entire accumulated value
at that point, which then propagates through every remaining layer -- rather
than damage accumulating gradually with more fp16 exposure. This has only
been tested for one specific perturbation shape (immediate cast, no
compensation); it does not rule out a differently-designed residual update
(e.g. compensated/error-feedback accumulation) surviving calibration --
though such a design would need its own scrutiny for whether it still
qualifies as a legitimate approximation of the true fp32 computation. Current
production boundary (Level A) is not merely "the one we picked" -- it appears
to sit right at the edge of what this shape of perturbation can survive.
