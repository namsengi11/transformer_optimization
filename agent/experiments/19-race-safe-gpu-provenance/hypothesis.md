# Step 19: make GPU provenance sampling race-safe

- Status: SELECTED
- Branch: `opt/19-race-safe-gpu-provenance`
- Pinned main SHA: `03184a7e1955a0c2b1988a89a5dd4d8b7e2d8b9a`
- Pinned experiment SHA: `<pending>`
- Started UTC: `2026-08-31T14:36:00Z`

## Claim

The foreign-load sampler mislabels an own GPU child when that child exits after `pmon` but
before descendant discovery; unioning descendant snapshots taken immediately before and
after `pmon`, together with step 18's replacement-tolerant decoding, will classify the
benchmark subprocesses as own and produce a trusted predefined quick-suite artifact when no
actual foreign GPU work occurs.

## Evidence and prior art

- `agent/experiments/18-windows-text-decoding/quick.json` completed after the decoder fix
  but marked all three runs untrusted. Each run reported three new 96-98% SM PIDs, matching
  `run_bench.py`'s accuracy, optimized, and compiled-bar subprocesses.
- In `ForeignLoadSampler._run`, `process_sm_utilization()` blocks in `nvidia-smi pmon`
  before `_descendants(os.getpid())` is called. A just-finished child is therefore present
  in the utilization sample and absent from the later live process tree.
- `agent/LESSONS.md` requires foreign work to invalidate a run, so the correction must add
  own-process evidence rather than weaken the 5% foreign-load threshold.

## Expected impact

- Cases expected to improve: none; this is provenance tooling.
- Cases expected to decline the gate: none.
- Expected latency/MFU range: unchanged. The required result is a trusted 1/1 quick suite
  with own high-utilization children excluded and any genuinely foreign PID still reported.

## Accuracy argument

The change is confined to host-side process attribution and decoding in `agent/tools/`.
Model code, eager-reference accuracy, benchmark controls, timings, and GPU work are unchanged.

## Measurement plan

- `python -m compileall -q agent/tools`.
- Deterministic mocked attribution test stored in this experiment directory: verify an own
  child visible only in the pre-`pmon` tree is excluded, an own child visible only afterward
  is excluded, and an unrelated high-SM PID is retained.
- `python agent/tools/bench.py --suite quick --tag step-19-tooling --runs 3 --output agent/experiments/19-race-safe-gpu-provenance/quick.json`.
- Inspect clean SHA, 1/1 eager-reference accuracy, sampler fields, and `trusted:true`.

## Kill condition

Discard if the deterministic test loses a real foreign PID, any decoder/sampler exception
occurs, the quick suite is not 1/1 PASS, the artifact is untrusted in an otherwise idle
window, or any product/harness behavior changes.

## Legitimacy check

The fix uses parent/child process identity around the existing utilization sample. It does
not whitelist names, PIDs, suites, or utilization levels and does not suppress real foreign
GPU activity.
