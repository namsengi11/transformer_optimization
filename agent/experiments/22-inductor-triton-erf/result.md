# Step 22 result: Inductor Triton compatibility

- Outcome: DISCARDED
- Branch: `opt/22-inductor-triton-erf`
- Base SHA: `42732832f909ce0e3d54a82c8614df3a5a28e524`
- Measured SHA: not implemented or measured
- Completed UTC: 2026-08-31T20:27:11Z
- GPU time: 0

## Verdict

Discarded before implementation because the user explicitly reprioritized general
long-sequence streaming, an 85% installed-VRAM target, and shape-independent autotuning.

## Accuracy

Not measured; no implementation was made.

## Latency and MFU

Not measured.

## Mechanism evidence

None. This priority-driven termination establishes neither a win nor a rejection for the
public `tl.erf` hypothesis.

## Exclusions

All suites were excluded because the experiment stopped in HYPOTHESIZE.

## Verifier

Not applicable: no product change or measurement claim is promoted.

## Code disposition

The branch contains documentation only. The Inductor hypothesis remains eligible for a
later numbered revisit.

## Lessons delta

An explicit user priority change closes the active serial experiment without treating the
unmeasured mechanism as rejected evidence.

## Next candidate

General streaming autotuning keyed by shape, dtype, installed VRAM, and measured candidate
memory/latency, with an 85% VRAM target and no benchmark-name or exact-shape predicates.
