# Current SOTA implementation

## Target workload

The test shapes are causal transformer forwards with float32 model weights and
outputs. The optimized implementation uses fp16 tensor-core computation inside
that float32 model. There is no padding in the test matrix.

This distinction matters:

- the model, residual stream, LayerNorm accumulation, and final output remain
  float32;
- GEMMs and attention operate on fp16 tensors using cached fp16 weight copies;
- GELU preserves the same materialization and rounding boundary as the reference;
- correctness is checked against the naive float32 eager implementation.

The code still supports native fp16/bf16 models and padding as general fallbacks,
but their dtype-dependent manual-attention path is not part of the current target
workload or the benchmark summary below.

## Optimizations applied across the test shapes

### Static fp16 compute policy

Every test shape on CUDA uses the same static precision policy. It does not
inspect or calibrate on the input:

- LayerNorm writes normalized activations directly to fp16;
- packed QKV, attention, output projection, FFN input, and FFN output operate in
  fp16;
- projection weights and biases are converted once and cached;
- residual additions and the final normalization return to float32.

This is the largest broad compute optimization because the RTX 5060 Ti's fp16
tensor-core throughput is substantially higher than its float32/TF32 throughput.

### Packed QKV projection

Q, K, and V weights are packed into one `[3*d_model, d_model]` projection. One
wider GEMM replaces three narrow GEMMs. Besides removing two launches, the wider
output dimension creates more thread-block tiles and occupies the GPU's 36 SMs
more effectively.

The packed projection is used throughout the target float32 path. Depending on
the row count, it is executed by either the custom Triton projection kernel or
the cuBLAS-backed PyTorch linear fallback.

### Causal and mask fast paths

All test shapes are causal and unpadded, so no padding mask is built. SDPA receives
`is_causal=True` directly, and reusable causal metadata is resolved outside the
layer loop. This removes repeated mask construction and device-to-host checks
from every layer.

### CUDA graph replay

After all shape-specific choices have been resolved and warmed, the selected
kernel sequence is captured with `torch.cuda.CUDAGraph`. Replay removes Python
and launch overhead without changing kernel arithmetic.

Graph capture is skipped when its retained memory pool would be unsafe or when a
shape requires capacity streaming. Those shapes execute the same selected
kernels through ordinary launches.

### What `torch.compile` does here

Whole-model `torch.compile` is not part of the active SOTA route. The custom
Triton calls make whole-core Inductor compilation unreliable, and the shapes on
which it does compile produce more kernels rather than a faster graph. The active
route is the explicitly composed kernel sequence above, optionally replayed by a
CUDA graph.

Two small compiled layout helpers remain in the implementation. They contain
only `view`, `transpose`, `contiguous`, and `reshape`, so Inductor generates a
tiled copy without changing arithmetic. They are admitted only after a
bit-exact complete-forward check and fall back to ATen copies on failure.

These helpers materially improve native fp16/bf16 paths that must materialize
three split-head copies. They are not a demonstrated material contributor to the
current float32-plus-fp16 target: SDPA accepts strided Q/K/V views, and the custom
short-attention kernel already returns merged output. They should therefore be
treated as a minor layout fallback, not a headline SOTA feature.

## Shape-specific kernels and routing

The performance kernels below are handwritten in Triton unless the implementation
column says otherwise. Each approximate numerical kernel is tested by running a
complete forward with and without it before CUDA graph capture. The verdict is
cached by shape, dtype, device, causal mode, and the previously selected feature
flags. A compilation or launch failure permanently selects the fallback for that
model instance.

| Optimization | Implementation | Shape boundary | Exploited property | Fallback |
|---|---|---|---|---|
| Short causal attention | **Handwritten Triton** | `seq_len <= 128`, `head_dim <= 128` | All keys fit in one score tile, avoiding long-sequence online-softmax bookkeeping. Query rows are tiled only to create enough parallel blocks. The kernel fuses QK, causal masking, base-2 softmax, PV, and head merge. | PyTorch SDPA |
| Residual-add + LayerNorm | **Handwritten Triton** | Next-power-of-two feature width `<= 2048`, then complete-forward accuracy gate | One program owns a token row and can combine residual casting, residual add, masks, LayerNorm, output cast, and the optional final-store suppression. | ATen residual/add/LayerNorm chain |
| FFN input GEMM + GELU | **Handwritten Triton** | `batch*seq >= 256`, `ffn_dim >= 64`, and no capacity chunking | Applies GELU in the GEMM epilogue after rounding at the same point as the materialized PyTorch linear output. It deletes a launch and the write/read round trip of the FFN hidden tensor. | cuBLAS linear + ATen GELU |
| QKV/output/FFN-output projections | **Handwritten Triton**, using the GEMM kernel with GELU disabled | `batch*seq >= 1024`, then complete-forward accuracy gate | Narrow-N cuBLAS kernels underfill this GPU on several test shapes; the autotuned Triton tiling exposes more useful blocks. | cuBLAS-backed `F.linear` |
| Layout copies | **Inductor-generated Triton**, not handwritten | Only where a materialized split or merge copy is required, then bit-exact complete-forward gate | Replaces ATen's general strided copy with a tiled copy. This is secondary on the target path. | ATen view/copy/reshape |
| Capacity streaming | **PyTorch orchestration + SDPA**; tiled eager reference uses PyTorch GEMMs | Dense peak estimate above 65% of installed VRAM | Streams independent batch shards and tiles reference query rows so no full `[B,H,S,S]` score tensor or full resident input/output is required. | Ordinary dense execution below the gate |

The support boundaries are measured limits for this GPU and software stack, not
claims that the kernels are universally optimal outside those regions.

## How the fused layer is assembled

For a typical short-sequence test shape, one optimized transformer layer is:

1. fused pending-residual + LayerNorm, storing fp16 normalized activations;
2. one packed QKV GEMM, using Triton when the row-count gate admits it;
3. one fused short-attention Triton kernel, or SDPA outside its shape boundary;
4. output projection;
5. fused attention-residual + LayerNorm;
6. fused FFN-input GEMM + GELU;
7. FFN-output projection, leaving its result pending for the next LayerNorm.

Moving each residual addition to the following LayerNorm site is what makes the
add+LayerNorm fusion possible without duplicating the layer implementation. The
final normalization consumes the last pending residual and omits the otherwise
unused residual-stream store.

This gives three levels of optimization without relying on whole-model Inductor:

- operation fusion inside handwritten Triton kernels;
- schedule fusion through the pending-residual design;
- launch-overhead removal through CUDA graph replay.

## Capacity streaming for extreme sequences

Dense execution is replaced by streaming when:

```text
6 * (B*S*D*element_bytes)
+ 2 * (B*H*S*S*element_bytes)
+ 2 * (B*S*FFN*element_bytes)
    > 65% of installed VRAM
```

Installed capacity is used rather than current free memory, so routing is stable
and does not depend on other GPU processes.

The input remains on CPU and independent batch shards are transferred to CUDA.
The optimized side uses memory-efficient SDPA. The naive reference retains all
keys and values but evaluates fixed query tiles, so it preserves full causal
attention rather than substituting a local or sparse window. Its attention
workspaces are reused, unreachable future causal-key tiles are skipped, and the
query tile is limited to 10% of installed VRAM. Outputs are compared and released
one shard at a time.

For the extreme test shape `B=32, S=100000, D=1024, H=16, layers=2`, one float32
dense attention-score tensor alone would require 20.48 TB. The input and output
are each 13.1 GB. This shape therefore selects batch/query streaming. The path is
implemented and validated on feasible long-sequence surrogates, but a complete
accuracy-and-timing result for the full extreme shape has not yet been recorded.

## Latest test-shape benchmark

This is the latest complete operational run for the 13 test shapes that use the
dense path. All shapes are causal float32 models executing fp16 GEMMs, and all
13 passed against the naive float32 eager reference.

The run was collected with background GPU activity and only one sample per shape.
It is the latest result snapshot, not a promotion-grade repeated measurement.

| Shape | Optimized ms | vs compiled | vs naive | Optimized MFU |
|---|---:|---:|---:|---:|
| `01_base` | 0.3834 | 2.684x | 6.270x | 45.91% |
| `02_batch1` | 0.0629 | 26.413x | 44.017x | 4.37% |
| `03_batch4` | 0.1320 | 6.539x | 23.769x | 8.33% |
| `04_batch16` | 0.1305 | 6.516x | 18.696x | 33.72% |
| `05_batch128` | 0.7890 | 3.664x | 9.456x | 44.62% |
| `06_batch10000` | 101.9824 | 3.409x | 7.445x | 26.97% |
| `07_dmodel32` | 0.1305 | 6.654x | 18.262x | 14.75% |
| `08_dmodel1024` | 12.5621 | 2.109x | 2.498x | 70.06% |
| `09_heads1` | 0.3855 | 2.403x | 5.086x | 45.66% |
| `10_heads2` | 0.3665 | 2.609x | 6.494x | 48.03% |
| `11_heads16` | 0.4925 | 7.100x | 23.226x | 35.74% |
| `12_seq32` | 0.1223 | 7.939x | 19.845x | 29.24% |
| `13_seq1024` | 8.4777 | 6.712x | 21.839x | 45.68% |

Aggregate snapshot:

- **5.011x geomean versus the compiled baseline**;
- **12.188x geomean versus the naive float32 eager baseline**;
- **34.85% average optimized MFU**, versus 20.09% for the compiled baseline;
- **+14.76 percentage points / +73.49% relative MFU improvement**.

The complete benchmark artifact is
`agent/results/worktrees/step20/results/step-20-manual-full_user_matrix.json`.

Long-sequence surrogate validation at `B=1, S=8192, D=128, H=4, FFN=128,
layers=2` produced an exactly matching tiled-versus-dense naive reference and an
optimized-versus-dense maximum absolute error of `0.000745893`, with zero failing
elements. That artifact is
`agent/experiments/21-long-sequence-streaming/correctness-stable-b1-s8192.json`.
