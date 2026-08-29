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

## 4. Rejected after measurement

| idea | why rejected |
|---|---|
| **KV caching** | Not applicable. The harness does one full-sequence forward per call, with no autoregressive decode and no cross-call state. There is no incremental step to cache K/V for, and caching across identical benchmark calls would just memoise the answer. |
| `torch.compile` on our own model (fp16/bf16) | Changes numerics past tolerance vs the eager reference — the same drift that makes the compiled baseline non-compliant. |
| Whole-model fp16 / bf16 cast | fp16 max_abs 0.0082, bf16 0.074 — both fail. |
| Error-compensated fp16 (hi+lo split GEMM) | Two fp16 GEMMs cost about as much as one TF32 GEMM. No win. |
| Folding the attention scale into `W_q`/`b_q` | 3.9e-3 divergence in fp16 (bit-exact in bf16 only). Unsafe. |
| Fused QKV in fp16/bf16 | Changes cuBLAS kernel selection; breaks near-zero elements. |
| Triton fused softmax reduction | Reduction tree differs from ATen; compounds to 0.0078 over 6 layers. |
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
| default_fp32 | 1.793x | 60.98%* | 67.19% |
| default_fp16 | 0.894x | 49.76% | 55.63% |
| default_bf16 | 1.170x | 53.33% | 45.58% |
| causal_fp16 | 0.878x | 49.24% | 56.09% |
| padded_fp16 | 0.855x | 47.06% | 55.07% |
| **average** | **1.071x geomean** | **52.07%** | **55.91%** |

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
python run_bench.py --suite default    # 5 configs, correctness + speed
python run_bench.py --suite full       # 10 configs incl. long-seq and wide
python profile_baseline.py float32     # kernel-level profile of the bar
```
