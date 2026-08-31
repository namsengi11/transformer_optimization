# Step 20: admit an inductor-fused variant of the captured graph, behind a gate

- Status: SELECTED
- Branch: `opt/20-inductor-fused-graph`   (verify unused before `checkout -b`; a second
  session shares this tree and the `opt/NN` convention)
- Pinned main SHA: `03184a7e1955a0c2b1988a89a5dd4d8b7e2d8b9a`
- Pinned experiment SHA: `<pending>`
- Started UTC: `<pending>`

## Claim

Our CUDA-graph path replays the *eager* kernels, so it buys cheaper launches but no
fusion; the reference implementation reaches the same shapes through
`torch.compile(mode="reduce-overhead")`, which replays *inductor-fused* kernels and so
launches strictly fewer of them. On the launch-bound rows, admitting an inductor-fused
variant of the same captured region -- as a candidate gated per configuration, never as a
replacement -- should close the row-2-class gap without moving any row outside
`atol=0.002 / rtol=0.02`.

## Origin

Derived from the three-row mechanism comparison in the session of 2026-08-31:

| axis | ours today | theirs |
|---|---|---|
| mechanism | hand-rolled `torch.cuda.CUDAGraph` capture/replay | `torch.compile(reduce-overhead)` -> `cudagraph_trees` |
| kernels replayed | the eager kernels, unchanged | inductor-generated, fused |
| numerics | bit-identical to eager by construction | changed by fusion, absorbed by their budget |

Rows 4-5 of that table (graph breaks, capture-peak memory model) are deliberately OUT of
scope here; see Exclusions.

## Evidence and prior art

- `agent/OPTIMIZATION_LOG.md` step 2: graph replay is "the one large lever that is legal in
  low precision" *because* it preserves arithmetic. This step does not weaken that claim --
  it adds a second, non-bit-exact candidate behind an explicit gate.
- `agent/OPTIMIZATION_LOG.md` step 9: the precedent that `torch.compile` is admissible where
  its output is *proven* acceptable rather than assumed -- there by `torch.equal` on pure
  data movement. Step 20 extends the same discipline to a case that cannot be bit-exact, so
  the bar becomes the margin-scaled harness criterion, not `torch.equal`.
- `_probe_fp16_gemm_gate` (`torch_transformer_benchmark.py`) is the shape of the gate to
  reuse: run both variants on the configuration's real first-forward input, admit the
  candidate only if every element clears `_FP16_GEMM_GATE_MARGIN` x (atol, rtol).
- Measured gap motivating the work (rows whose shapes were unchanged by the matrix schema
  correction, RTX 5060 Ti, same harness, same session): row 2 ours 19.09x vs theirs 25.46x.
  Ours already won unchanged rows 1, 3, 4, 5.
  **Use the completed `ours_v2_logs/` + `theirs_v2_logs/` sweeps as the real before-baseline;
  the numbers above are from an interrupted run and are provisional. Do not use either
  sweep's full-matrix aggregate: rows 7, 8, 12, and 13 must be rerun using
  `agent/docs/USER_MATRIX.md`.**

## Expected impact

- Cases expected to improve: `02_batch1` primarily; possibly `03_batch4`, `07_dmodel32`,
  and `12_seq32` -- all expected to be launch-bound, where kernel *count* rather than
  kernel *cost* sets the floor. The corrected rows 7 and 12 require fresh profiling.
- Cases expected to decline the gate: `08_dmodel1024` and `13_seq1024` (expected compute-
  and attention-bound, respectively; verify with the corrected shapes before relying on it).
- Expected latency/MFU range: row 2 optimized 0.127 ms -> 0.09-0.11 ms if the mechanism is
  real. No change expected on rows 6, 8, 13. A null result on every row is a legitimate
  and publishable outcome.

## Accuracy argument

The fused variant is NOT bit-exact and must not pretend to be: inductor reorders and fuses
elementwise work around the GEMMs. Three containments:

1. It is never the only path. The existing hand-rolled bit-exact graph stays as the
   fallback and remains the default for any configuration the gate rejects.
2. Admission is per `(shape, dtype, device, causal, mask_kind)` -- the existing graph-cache
   key -- decided on that configuration's real input, not on a synthetic probe.
3. The gate margin is the already-calibrated `_FP16_GEMM_GATE_MARGIN` (0.9), so an admitted
   variant holds 10% headroom against the harness threshold rather than sitting on it.

Fusion stacks on top of the fp16-GEMM error already spent (max_abs 0.0012-0.0019 at normal
input scale). If the combined error clears the gate at margin 0.9 it is spending headroom
that was already measured; if it does not, the gate rejects and nothing ships.

## Implementation sketch

1. Add `_compiled_core` alongside `_forward_core`: `torch.compile(self._forward_core,
   mode="reduce-overhead")`, built once per configuration, stored beside the graph cache.
2. Extend the existing candidate/probe machinery rather than inventing a new one -- follow
   `_probe_fp16_gemm_gate`'s structure exactly: reference = current graph path, candidate =
   compiled path, same input, margin-scaled criterion, verdict cached on the graph-cache key.
3. Two traps that WILL fire if ignored, both already documented in this repo:
   - Hold the `_no_static_cuda_launcher()` patch across BOTH warmup and first compile.
     Windows/torch-2.8 overflows a C long with the 64-bit stream handle otherwise.
   - Warm the compiled variant under the SAME dispatch key set the caller uses. dynamo
     specializes on the dispatch key set, so a helper compiled under `inference_mode()` is
     a cache miss under `no_grad()`, and the recompile lands inside capture. Step 2's note
     measures this failure at 2.3x slower, silently, while still correct.
4. Compilation is one-time and must land in warmup, never inside a timed region. Assert it,
   do not assume it: the harness's `--warmup 20` is the budget.
5. No `mem_get_info`, no `.item()`, no `mask.all()` inside any captured or compiled region.

## Measurement plan

- **A/B by flag on ONE fixed checkout, never by branch.** A second session shares this GPU
  and this working tree; branch-to-branch timing comparisons on a shared card are invalid.
  Add an env flag (e.g. `TJ_FUSED_GRAPH=0|1`) and alternate within a single process run.
- Before/after: full 13-row executable causal matrix, one row per process, using the exact
  `run_bench.py` cases checked against `agent/docs/USER_MATRIX.md`. Do not copy row arguments
  from the pre-correction `ours_v2_logs` driver.
- Report medians AND the run-to-run band; a row moves only if the bands do not overlap.
- MFU with the same FLOP model used in-session: `12*M*d^2*L` + causal attention
  `2*B*S^2*d*L`, denominator 47.4 TFLOP/s on this card (36 SM x 2572 MHz x 512 FLOP/clk).
- GPU hygiene: confirm the card is idle (<4 GB used) before starting; sysmem fallback is now
  DISABLED at the driver, so an over-budget row errors instead of silently spilling. Treat
  any OOM as a real result, not an infrastructure failure to work around.

## Kill condition

Discard if ANY of:
- No row improves with non-overlapping bands (a pure-null result).
- Any row fails the harness gate (`failed != 0`) with the fused variant admitted.
- Any admitted row regresses more than 2% against the pinned baseline.
- The gate admits a configuration whose max_abs exceeds 0.9 x atol -- indicates the gate
  is being consulted after the fact rather than before admission.
- Compilation cost cannot be kept out of the timed region.

## Legitimacy check

This optimizes the computation, not the benchmark: fusion reduces the number of kernels
launched to produce the same mathematical result, and the admission gate is evaluated on the
configuration's real input against the model's own eager reference -- it cannot memorize an
answer, cannot depend on `data_ptr`, and cannot narrow to the official matrix, because the
same gate runs for any shape the model is handed. The fallback path is the current shipped
one, so a rejected configuration is exactly as correct as it is today.

## Exclusions

- Graph-break engineering (their `torch.library` custom-op wrapper for cuBLASLt) is out of
  scope: we have no extension call inside the region to break on.
- The capture-peak VRAM model (`_would_oom_causal`) is out of scope; executable rows 1-13 do
  not engage chunking. Official row 14 is retained only as `PREFLIGHT_BLOCKED`.
- Row 14 is not in scope.

## Handoff state

- `torch_transformer_benchmark.py` carries an UNCOMMITTED capacity-aware chunk-sizer change
  (`_FFN_CHUNK_BUDGET_FRAC`, `_ffn_chunk_budget_bytes`, `_ffn_chunk_size(device=...)`),
  verified inert on all 13 official rows on a 16 GB card. Decide deliberately whether step 20
  branches from it or from clean `03184a7`; do not silently absorb it into this experiment.
- A copy of that patched file is preserved outside the repo at the session scratchpad path
  `ours_v2/torch_transformer_benchmark.py` if it needs recovering.
