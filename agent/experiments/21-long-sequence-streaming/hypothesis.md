# Step 21: capacity-derived long-sequence streaming protocol

- Status: SELECTED
- Branch: `opt/21-long-sequence-streaming`
- Pinned main SHA: `4e0c03b613049cde6c58a29a1be603adf2fa0247`
- Pinned experiment SHA: pending
- Started UTC: 2026-08-31T19:07:13Z

## Claim

When a static full-forward peak estimate containing both linear activation residency
(`B*S*D`) and quadratic dense-attention residency (`B*H*S^2`) exceeds a measured fraction
of installed VRAM, streaming independent batch shards and tiled query rows will execute the
same full causal-attention semantics without materializing the full input, output, FFN, or
attention score tensors on the GPU.

## Evidence and prior art

- Canonical row 14 needs 20.48 TB for one fp32 dense score tensor, while its input and
  output are each 13.1 GB. Flash/SDPA alone therefore does not solve the resident-tensor
  limit.
- `ChunkedBaselineTransformerBlock` already establishes row-wise FFN streaming and the
  warning that chunk size can alter floating-point reduction choices.
- The optimized fp32 path already uses the memory-efficient SDPA backend for long
  sequences, but the harness currently constructs the entire input on CUDA and requires a
  full CUDA output.
- The user explicitly approved changing the protocol from permanent row-14 preflight
  exclusion to a general capacity-derived streaming mechanism for any long sequence.

## Expected impact

- Cases expected to change mechanism: any shape whose estimated full-forward peak exceeds
  the empirically selected VRAM budget, including `14_extreme`.
- Cases expected to remain unchanged: all shapes comfortably below the gate, especially
  canonical rows 1-13.
- This is a capacity/protocol step. Latency improvement is not its promotion criterion;
  successful execution, zero accuracy failures on feasible cross-check shapes, and no
  behavior change below the gate are.

## Accuracy argument

Batch elements are independent, so batch streaming preserves mathematical semantics.
The reference attention keeps each complete key row for softmax while tiling only query
rows; this preserves full causal attention rather than substituting a local window. Since
GEMM algorithm selection can change with tile size, admission requires whole-model checks
against the unchanged dense eager baseline on feasible long-sequence surrogate shapes at
`atol=0.002`, `rtol=0.02`, with zero failing elements.

## Measurement plan

- Extend an agent-owned tool to report observed CUDA peak allocation and the static gate
  terms for arbitrary shapes in fresh child processes.
- Sweep feasible `(B,S)` pairs at fixed model widths across equivalent `B*S` values to show
  why the quadratic term is required and select a conservative total-VRAM fraction.
- Run dense-versus-streaming correctness on feasible long-sequence surrogates.
- Run the default suite and executable user-matrix rows to prove the below-gate path is
  unchanged. Row 14 receives a bounded smoke configuration first; a complete row-14 timing
  is reported separately because its quadratic compute cost makes the default repetition
  protocol inappropriate.

## Kill condition

Discard if any below-gate case changes path, any feasible dense-reference cross-check has a
failed element, the capacity decision depends on current free memory rather than installed
VRAM, or the streaming path changes full causal attention into windowed/sparse attention.

## Legitimacy check

The gate is a general function of shape, dtype, model dimensions, and installed device
capacity. It does not match benchmark names or fixed sequence lengths. Streaming computes
all causal query-key pairs and therefore changes storage/liveness, not model semantics.

## Protocol changes approved by the user

The streaming path may replace the unmodified eager `BaselineTransformer` only for shapes
that the static capacity gate proves cannot execute densely. Such cases must be labeled
`STREAMED`, must retain dense eager cross-check evidence from feasible surrogate shapes,
and must not contribute a compiled-baseline speedup unless an equivalent compiled streaming
bar exists. Long-case warmup/repetition counts must be explicit rather than silently using
the ordinary 20/100 defaults.
