# Step 22: make fused FFN composable with Inductor

- Status: DISCARDED (superseded before implementation)
- Branch: `opt/22-inductor-triton-erf`
- Pinned main SHA: `42732832f909ce0e3d54a82c8614df3a5a28e524`
- Pinned experiment SHA: pending
- Started UTC: 2026-08-31T20:16:34Z

## Claim

The whole-core Inductor candidate declines ordinary FFN shapes because the raw Triton GELU
kernel closes over the private Python global `_tl_libdevice`; replacing that call with the
public `tl.erf` intrinsic will let Inductor materialize the `APPLY_GELU=True` specialization,
admit numerically safe configurations, and reduce latency versus the current manual captured
graph on at least one suite without regressing either suite by more than 2%.

## Evidence and prior art

- The supplied failure trace reaches `_gemm_gelu_kernel` only for the fused FFN GELU path
  and fails with `NameError: _tl_libdevice is not defined` in Inductor's generated module.
- Rows 2 and 7 avoid that specialization through `_FUSED_FFN_MIN_ROWS` or
  `_FUSED_FFN_MIN_FFN`, matching their successful real Inductor CUDA graphs.
- Step 20 was superseded without measurement, so its mechanism remains eligible for a fresh
  numbered revisit.
- `agent/LESSONS.md` requires full-forward eager-reference accuracy, clean fixed-checkout
  measurements, and CUDA-graph timing.

## Expected impact

- Cases expected to improve: ordinary FFN shapes admitted by the full-forward gate,
  particularly `03_batch4`; possibly other launch-bound default and matrix rows.
- Cases expected to decline the gate or retain the manual graph: numerical failures and the
  capacity-gated row-6 dynamic compile path, which explicitly disables Inductor CUDA graphs.
- Expected latency/MFU range: a measurable latency reduction on at least one suite; no suite
  may regress more than 2%.

## Accuracy argument

`tl.erf` is Triton's public spelling of the same GELU ingredient, but numerical equivalence
is not assumed. Every admitted configuration must pass the existing 0.9-margin full-core
probe, and suite correctness is judged against full eager `BaselineTransformer` output at
`atol=0.002`, `rtol=0.02` with zero failing elements.

## Measurement plan

- Before: `agent/tools/bench.py` on default and user-matrix suites for three runs with
  `TJ_FUSED_GRAPH=0`, producing `default-before.json` and `matrix-before.json`.
- Mechanism: fixed-checkout `agent/tools/probe_ab.py` with `TJ_FUSED_GRAPH=0|1`, plus an
  explicit B4 debug-gate run proving an `ENABLE` verdict and no `_tl_libdevice` failure.
- After: the same two three-run suites with `TJ_FUSED_GRAPH=1`, compared to the before
  artifacts and stored as `default-after.json` and `bench.json`.
- Graph evidence: `agent/tools/nsys_trace.py` on `user_matrix:03_batch4`, followed by
  generated-kernel inspection. Tier 3 is unavailable per `env_check.py` and is not required.

## Kill condition

Discard on any compilation failure, eager-reference accuracy failure, untrusted run, row-6
OOM, missing CUDA-graph replay on an admitted normal shape, overlapping before/after ranges
for every possible win, or a regression above 2% on either suite. Also discard if the
candidate does not beat the current manual captured graph under repeatable measurements.

## Legitimacy check

The change replaces a private Python reference with a public Triton intrinsic for every
shape. It neither recognizes benchmark cases nor changes the model computation, tolerances,
timing controls, parameters, or fallback path.
