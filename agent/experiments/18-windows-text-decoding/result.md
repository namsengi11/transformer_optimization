# Step 18 result: make Windows measurement decoding loss-tolerant

- Outcome: DISCARDED
- Branch: `opt/18-windows-text-decoding`
- Base SHA: `7e30a38e12c5d67fdc600f02c8cd99f9b3dc6cd4`
- Measured SHA: `c3849ee5396dd7592bfa7d62419aea2ba7a537c8`
- Completed UTC: `2026-08-31T14:34:00Z`
- GPU time: approximately 3 minutes

## Verdict

Discard under the stated kill condition: replacement-tolerant decoding eliminated the
exception, but `agent/tools/bench.py` still produced an untrusted quick artifact because its
foreign-load sampler misclassified short-lived benchmark descendants.

## Accuracy

| suite | passing | max abs | max rel | tool artifact |
|---|---:|---:|---:|---|
| quick | 1/1 | 0.00116372 | 560.776 (near-zero reference elements; OR tolerance passed) | `agent/experiments/18-windows-text-decoding/quick.json` |

## Latency and MFU

| suite/case | before ms range | after ms range | delta | MFU before/after | precision | trusted |
|---|---:|---:|---:|---:|---|---|
| quick/default_fp32 | n/a | 1.0917-1.0958 | tooling step; not judged | n/a / 0.7537 | fp16 GEMMs enabled | no |

## Mechanism evidence

`agent/experiments/18-windows-text-decoding/tool-smoke.txt` records five successful
`process_sm_utilization()` samples after the change, with no `UnicodeDecodeError` or sampler
thread exception. The clean fixed-checkout run in
`agent/experiments/18-windows-text-decoding/quick.json` completed all three child runs, but
each recorded three transient 96-98% SM PIDs as foreign. The sampler queries `pmon` before
it snapshots descendants, so an own child that exits between those operations is absent
from the later tree and is mislabeled.

## Exclusions

No cases were excluded. This tooling investigation ran only the predefined quick suite and
made no optimization-promotion claim.

## Verifier

`REFUTED`; see `agent/experiments/18-windows-text-decoding/verifier.md`. The decoder mechanism
worked, but the hypothesis's trusted-artifact kill condition failed. Product and harness
arithmetic were untouched.

## Code disposition

The implementation remains unmerged on `opt/18-windows-text-decoding`. Main receives only
the documentation record; step 19 will retest replacement-tolerant decoding together with
race-safe descendant attribution.

## Lessons delta

Added the requirement that Windows GPU provenance must both tolerate undecodable diagnostic
bytes and snapshot own descendants on both sides of `pmon` sampling.

## Next candidate

Step 19 is a tooling prerequisite that combines the valid decoding change with race-safe
child attribution. The streaming/flash-FFN candidate remains first once trusted baseline
measurement is possible.
