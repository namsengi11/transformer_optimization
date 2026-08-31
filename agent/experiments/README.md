# Experiment records

Every numbered optimization or investigation owns one directory:

```text
agent/experiments/
  NN-short-slug/
    hypothesis.md
    result.md
    bench.json            # generated, machine-local, ignored by git
    default-before.json   # generated, machine-local, ignored by git
    default-after.json    # generated, machine-local, ignored by git
    probe.json            # generated, machine-local, ignored by git
```

The Markdown files are committed. Measurement JSON is retained locally and cited by path,
but ignored because it contains machine-local profiler and environment detail.

## `hypothesis.md` schema

```markdown
# Step NN: short title

- Status: SELECTED
- Branch: `opt/NN-slug`
- Pinned main SHA: `<40 hex>`
- Pinned experiment SHA: `<40 hex after implementation commit>`
- Started UTC: `<timestamp>`

## Claim

One falsifiable sentence: workload fact -> mechanism -> expected measured effect.

## Evidence and prior art

- Relevant `agent/OPTIMIZATION_LOG.md` steps/lines only.
- Relevant `agent/LESSONS.md` entries.
- Existing profiler artifacts.

## Expected impact

- Cases expected to improve:
- Cases expected to decline the gate:
- Expected latency/MFU range:

## Accuracy argument

Why the eager reference should remain within `atol=0.002`, `rtol=0.02` with zero failures.

## Measurement plan

- Mechanism command and artifact:
- Default suite before/after commands and artifacts:
- User-matrix before/after commands and artifacts:
- Profiler commands, if any:

## Kill condition

An exact result that causes discard, including overlapping ranges, accuracy, trust, and
cross-suite regression limits.

## Legitimacy check

Why this optimizes the computation rather than memorizing or narrowing the benchmark.
```

## `result.md` schema

```markdown
# Step NN result: short title

- Outcome: MERGED | DISCARDED | BLOCKED
- Branch: `opt/NN-slug`
- Base SHA:
- Measured SHA:
- Completed UTC:
- GPU time:

## Verdict

One sentence applying the `agent/ORCHESTRATOR.md` promotion rule.

## Accuracy

| suite | passing | max abs | max rel | tool artifact |
|---|---:|---:|---:|---|

## Latency and MFU

| suite/case | before ms range | after ms range | delta | MFU before/after | precision | trusted |
|---|---:|---:|---:|---:|---|---|

## Mechanism evidence

Probe/microbenchmark/profile result, including non-overlap judgment and artifact.

## Exclusions

Cases excluded, the pinned-main evidence allowing exclusion, and why.

## Verifier

`CONFIRMED` or `REFUTED`, adversarial checks performed, and artifact references.

## Code disposition

Merge commit or reason the implementation branch remains unmerged.

## Lessons delta

Exact `agent/LESSONS.md` entry added or changed.

## Next candidate

Backlog update justified by this result.
```

Do not edit generated JSON to improve a result. Rerun the named tool with a new tag and cite
both artifacts when invalidating an earlier measurement.
