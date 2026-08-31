# Step 18: make Windows measurement decoding loss-tolerant

- Status: SELECTED
- Branch: `opt/18-windows-text-decoding`
- Pinned main SHA: `7e30a38e12c5d67fdc600f02c8cd99f9b3dc6cd4`
- Pinned experiment SHA: `c3849ee5396dd7592bfa7d62419aea2ba7a537c8`
- Started UTC: `2026-08-31T14:28:00Z`

## Claim

Windows measurement subprocesses inherit a strict `cp1252` decoder that crashes on bytes
emitted by `nvidia-smi` and can silently kill the foreign-load sampler; making captured-text
decoding replacement-tolerant will keep provenance monitoring alive without changing any
measured workload or parsed numeric field.

## Evidence and prior art

- The first attempted current-main run of
  `agent/tools/bench.py --suite default --tag step-18-main --runs 3` raised
  `UnicodeDecodeError: 'charmap' codec can't decode byte 0x90` inside
  `process_sm_utilization()` and then `AttributeError` when its sampler thread received
  `stdout=None`.
- No suite artifact was emitted, so no partial number is retained or trusted.
- `agent/ORCHESTRATOR.md` section 1 says a missing measurement capability is a numbered
  tooling step of its own; product code and harness controls remain out of scope.

## Expected impact

- Cases expected to improve: none; this is provenance tooling.
- Cases expected to decline the gate: none.
- Expected latency/MFU range: unchanged. The expected effect is a completed sampler thread,
  a clean trusted artifact, and no decoding exception.

## Accuracy argument

Only host-side capture/decoding in `agent/tools/` changes. The eager baseline, optimized
model, benchmark arguments, accuracy thresholds, warmup, and repeats are unchanged.

## Measurement plan

- Static check: `python -m compileall -q agent/tools`.
- Exercise both GPU query paths repeatedly from Python imports, recording console output in
  `agent/experiments/18-windows-text-decoding/tool-smoke.txt`.
- Trusted end-to-end check:
  `python agent/tools/bench.py --suite quick --tag step-18-tooling --runs 3 --output agent/experiments/18-windows-text-decoding/quick.json`.
- Inspect the artifact's provenance, per-run `gpu_load`, accuracy, and trust fields.

## Kill condition

Discard if any capture still raises a decoding exception, the sampler thread terminates,
the quick artifact is missing or untrusted, accuracy is not 1/1 PASS, or the change alters
benchmark/model code or numeric parsing.

## Legitimacy check

The change only preserves ASCII numeric/control data while replacing undecodable diagnostic
bytes. It neither changes the workload nor suppresses foreign-process detection.
