---
description: Boot or resume the serial recursive transformer optimization loop.
argument-hint: "[number-of-steps]"
allowed-tools: Read, Grep, Glob, Bash, Edit, Write, Task
---

Run the workflow in `agent/ORCHESTRATOR.md` exactly. `$ARGUMENTS` is an optional maximum number of
new numbered steps for this invocation; omitted means continue until a documented stop or
blocking condition.

Boot by reading `agent/README.md` and `agent/ORCHESTRATOR.md`, running
`python agent/tools/env_check.py`, reading `agent/LESSONS.md`, reading only the indexed
`agent/OPTIMIZATION_LOG.md` tables/measurement notes, and inspecting the
last 20 commits and branches. Report the current standing and next step before proposing
anything. Do not edit code during boot.

Resume an existing `agent/experiments/NN-slug` record at its first incomplete state. Otherwise
select one seeded-backlog item and create the hypothesis record. Remain fully serial and use
at most one subagent at a time under the roster rules. All measurements must come from
`agent/tools/`; no inline timing, profiling, or A/B code is allowed. Store every document,
tool, scratch file, experiment artifact, measurement, profiler capture, digest, and dashboard
created by this workflow under `agent/`; product code changes are the sole experiment-scoped
exception.

At every step boundary, apply the promotion rule mechanically, use the verifier, record both
successes and failures, print the required summary block, and observe the digest/plateau
rules. Ask the user only for the four blocking classes in `agent/ORCHESTRATOR.md` section 6.
