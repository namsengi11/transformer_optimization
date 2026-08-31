# Continuous-agent optimization log

This is the fresh durable log for experiments governed by `agent/ORCHESTRATOR.md`. It starts
after the legacy optimization workflow; it does not replace or rewrite that history.

## 1. Legacy boundary

- Historical log: `OPTIMIZATION_LOG.md` at repository root.
- Historical research: `KERNEL_METHODS_ANALYSIS.md` and `SOTA_AGENT_KERNEL_METHODS.md` at
  repository root.
- Historical raw benchmark output: `results/` and the root `*_results.txt` / `14_extreme*.txt`
  files.
- Last legacy numbered step: 17. The first continuous-agent experiment is step 18.

Treat legacy measurements as prior evidence, not as fresh promotion evidence. Reproduce any
number that affects a new decision with `agent/tools/` and store the resulting artifact under
`agent/experiments/` or `agent/results/`.

## 2. Inherited standing

The legacy log reported, at its recorded checkout and environment, 1.207x geomean versus the
compiled bar on the default suite, 3.764x on the graded matrix, and 55.6% average MFU. These
figures are a starting hypothesis only until the continuous agent records a trusted current-
main baseline.

## 3. Agent experiment index

| Step | Outcome | Branch | Claim | Default delta | Matrix delta | Evidence |
|---:|---|---|---|---:|---:|---|

## 4. Agent rejection index

| Mechanism | Step | Reason | Revisit condition |
|---|---:|---|---|

## 5. Agent measurement notes

Add only durable protocol or environment observations that affect interpretation of more than
one experiment. Put step-specific detail in `agent/experiments/NN-slug/result.md`.

## Step records

Append one `### Step NN: title` section per completed experiment, in numeric order. Every
number must name its `agent/tools/` producer and an artifact under `agent/`.
