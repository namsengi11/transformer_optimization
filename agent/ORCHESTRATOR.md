# Recursive optimization orchestrator

This is the master instruction document for optimization work on
`torch_transformer_benchmark.py`. Read it first in every optimization session. The goal is
lower optimized latency and higher MFU without leaving the harness accuracy contract.

## 0. Boot sequence

Run this sequence before proposing or editing anything:

1. Read `agent/README.md`, especially its storage invariant, and then this file.
2. Run `python agent/tools/env_check.py`. Stop if its pinned environment checks fail. Tier 3 being
   unavailable is a supported degraded mode; report it in the next digest.
3. Read `agent/LESSONS.md`.
4. Read sections 2-5 of the fresh `agent/OPTIMIZATION_LOG.md`.
5. Treat root `OPTIMIZATION_LOG.md` as read-only legacy evidence. Read only its step/rejection
   index during boot, and open a legacy step body only when the new hypothesis touches that
   mechanism.
6. Run `git log --oneline -20` and `git branch -a`. Derive the next step number from the
   agent experiment index. The first continuous-agent step is 18.

Do not read the entire legacy optimization log. Its recorded default-suite standing was
1.207x geomean versus the compiled bar. The reported 3.764x graded-matrix geomean and 55.6%
average MFU used the pre-2026-08-31 misdecoded shapes and are invalid for the canonical
matrix; do not use them as inherited standing or comparison evidence. Read
`agent/docs/USER_MATRIX.md` and run `python agent/tools/check_user_matrix.py` before any
matrix measurement.

## 1. Invariants

These rules are not optimization candidates.

1. Accuracy is judged against eager `BaselineTransformer` where its calibrated dense peak
   fits installed VRAM. Above that boundary, use the step-21 query-tiled eager reference,
   which preserves full attention and has been cross-checked against the dense reference on
   feasible long-sequence surrogates. Both require zero failing elements under
   `abs(user-reference) <= 0.002 OR abs(user-reference) <= 0.02*abs(reference)`. Never use
   the compiled baseline as the numerical reference.
2. Measurement goes through `agent/tools/` or it does not count. Every number in a hypothesis,
   result, commit, digest, or dashboard names the tool and artifact that produced it. Do not
   write inline timing loops, one-off profiler snippets, or ad-hoc A/B scripts. A missing
   capability is a numbered tooling step of its own.
3. Do not edit or route around these harness controls:
   - `compare_outputs` (currently near line 3583);
   - `run_accuracy_tests` (near line 3653);
   - `benchmark_once` or `benchmark_models` (near lines 3771 and 3805);
   - any `BaselineTransformer` class;
   - `PEAK_TFLOPS` or `count_dense_flops` in `run_bench.py`;
   - default `--atol`, `--rtol`, `--warmup`, or `--repeats`.
   A protocol change is its own step and requires user approval before editing.
4. Do not add parameters or buffers to `UserOptimizedTransformer`.
   `load_state_dict(strict=True)` must continue to work.
5. New kernels use the established house pattern:
   `_probe_X` -> `_resolve_X_enabled` with a shape/dtype/flag cache -> permanent
   `_X_disabled` kill switch -> resolve in `forward()` before graph capture -> pass a
   `use_X` boolean into `_forward_core` -> include it in the CUDA graph cache key -> print
   the verdict under `TJ_DEBUG_GATE`. Probes compare a complete `_forward_core` pass, never
   a tile or isolated layer.
6. Constants carry the artifact and measurement that selected them. An unmeasured constant
   does not ship.
7. `nvprof` is banned: it is deprecated and does not support sm_120.
8. Store every agent-owned document, script, experiment record, measurement, profiler
   capture, digest, dashboard, and scratch artifact under `agent/`, following
   `agent/README.md`. Product changes made by a numbered experiment and the fixed discovery
   shims are the only workflow files permitted outside that directory. The tools reject
   `--output` paths outside `agent/`.
9. The root legacy log, research documents, diagnostic, and historical results named in
   `agent/README.md` are read-only. Never append new agent work to them or move them into
   `agent/`.

## 2. Objective and promotion rule

Promote step N only when all conditions hold:

1. `agent/tools/bench.py --suite default` reports 5/5 PASS.
2. `agent/tools/bench.py --suite user_matrix` reports 14/14 PASS. Capacity-safe rows use the
   dense eager reference; over-budget rows such as `14_extreme` report `STREAMED` and use
   batch/query streaming. A streamed row has no compiled-baseline ratio and is excluded from
   compiled-bar geomeans. The gate, long-case counts, and schema are in
   `agent/docs/USER_MATRIX.md`.
3. At least one of these is true:
   - median `optimized_ms` improves on at least one suite with non-overlapping min/max ranges
     over at least three independent runs, and no suite regresses more than 2%; or
   - average MFU improves at the same executed precision while every case's `optimized_ms`
     remains within +/-1%.
4. Every contributing case has `trusted: true` and no foreign GPU process was observed.

Discard otherwise. A no-code investigation still consumes a step number and gets recorded.
Judge `optimized_ms`, never `speedup_vs_compiled`; the compiled bar varies substantially
across processes. Compare MFU only at the same executed precision, using
   `fp16_gemm_enabled` for fp32 rows that actually ran fp16 tensor-core GEMMs. Judge streamed
   rows by accuracy, optimized latency, and same-precision MFU, not by a nonexistent compiled
   streaming bar.

## 3. Serial state machine

Only one experiment may be active. Use the following states in order.

### SELECT

Entry: boot complete and no unfinished experiment directory. Select the highest-priority
backlog item whose prerequisites are satisfied. After two consecutive discards, select a
profiling step instead of an implementation.

Exit artifact: `agent/experiments/NN-slug/hypothesis.md` created from the schema in
`agent/experiments/README.md`, including a falsifiable mechanism, expected affected cases, kill
condition, and required tool invocations.

### HYPOTHESIZE

Pin the base:

```powershell
git rev-parse main
git rev-parse HEAD
```

Record both SHAs. If no trusted current-main comparison exists, collect it before editing:

```powershell
python agent/tools/bench.py --suite default --tag step-NN-main --runs 3 --output agent/experiments/NN-slug/default-before.json
python agent/tools/bench.py --suite user_matrix --tag step-NN-main --runs 3 --output agent/experiments/NN-slug/matrix-before.json
```

Failure exit: if the claim cannot name a measurable mechanism and kill condition, discard it
before branching and record why.

### BRANCH

Create exactly one branch from the recorded main SHA:

```powershell
git switch main
git switch -c opt/NN-slug <PINNED_MAIN_SHA>
```

Never delete numbered branches. If the working tree contains unrelated changes, preserve
them and do not absorb them into the experiment.

### IMPLEMENT

Make the smallest change that tests the hypothesis. Do not change measurement tooling inside
an optimization step. Run non-performance correctness checks, inspect the diff, and commit
the implementation so fixed-checkout A/B can reject a dirty tree.

Failure exit: compilation failure, strict state-dict incompatibility, or a probe that cannot
be made whole-forward correct means discard or create a separate tooling step.

### MEASURE

Set `TJ_DEBUG_GATE=1`; the tools do this by default. Use only predefined commands.

Mechanism claim, when one gate can expose both arms:

```powershell
python agent/tools/probe_ab.py --suite quick --tag step-NN-probe --flag TJ_EXPERIMENT_GATE --a-value 0 --b-value 1 --pinned-sha <HEAD_SHA> --base-sha <PINNED_MAIN_SHA> --output agent/experiments/NN-slug/probe.json
```

Single-kernel evidence, when required:

```powershell
python agent/tools/microbench.py --target short_attn --output agent/experiments/NN-slug/microbench.json
```

Suite judgment:

```powershell
python agent/tools/bench.py --suite default --tag step-NN-after --runs 3 --compare agent/experiments/NN-slug/default-before.json --output agent/experiments/NN-slug/default-after.json
python agent/tools/bench.py --suite user_matrix --tag step-NN-after --runs 3 --compare agent/experiments/NN-slug/matrix-before.json --output agent/experiments/NN-slug/bench.json
```

For an unprofiled region, use the tiers in order:

```powershell
python agent/tools/profile.py --arm optimized --case user_matrix:01_base --output agent/experiments/NN-slug/profile.json
python agent/tools/nsys_trace.py --tag step-NN --arm optimized --case user_matrix:01_base --output agent/experiments/NN-slug/nsys.json
python agent/tools/ncu_profile.py --tag step-NN --arm optimized --case user_matrix:01_base --kernel-regex ".*target.*" --output agent/experiments/NN-slug/ncu.json
```

Tier 3 returning `available:false` is not an experiment failure. Use Tier 1+2, report the
counter restriction, and do not invent missing occupancy or bandwidth numbers.

Failure exit: any dirty provenance mismatch, moved checkout, moved main, overlapping ranges,
foreign GPU PID, environment mismatch, or failed accuracy invalidates the result.

### JUDGE

Apply section 2 mechanically. Ask the verifier role to look for harness gaming, gate
declines masking precision errors, missing shapes, and a mismatch between the claimed
mechanism and the measured arm. The verifier returns only `CONFIRMED` or `REFUTED` with
artifact-backed evidence.

### MERGE or DISCARD

Before either outcome:

```powershell
git rev-list --left-right --count main...HEAD
```

If main moved, rebase/port onto the new main and repeat all judged measurements. On promotion:

```powershell
git switch main
git merge --no-ff opt/NN-slug -m "Merge opt/NN: <summary>"
```

The merge body records per-case optimized-ms ranges, geomean delta, MFU delta, eager-baseline
accuracy, tool artifact paths, exclusions, and reasons. Do not delete the branch. On discard,
leave the branch and its commits unmerged.

### RECORD

For both outcomes:

1. Complete `agent/experiments/NN-slug/result.md`.
2. Add the section 3 table row and `### Step NN` section to `agent/OPTIMIZATION_LOG.md`.
3. Append or update the relevant `agent/LESSONS.md` entry.
4. Commit records on the experiment branch; for discarded work, merge a documentation-only
   record commit if and only if it does not carry implementation changes.
5. Print the step boundary block from section 6.

### THROTTLE

After two consecutive discards, profile rather than implement. After three discards among
the last five steps, use the researcher role. After four consecutive discards, stop and ask
the user. Then return to SELECT.

## 4. Measurement protocol and enforced traps

| Trap | Structural guard |
|---|---|
| Eager timing measures Python/Triton launch overhead | `microbench.py` captures CUDA graphs and subtracts empty replay |
| An L2-resident isolated op is scored against DRAM | `roofline.py` switches at the measured 32 MiB working-set boundary |
| Sync-per-call tiny-shape timing and autotune noise | `microbench.py` uses back-to-back replay; `bench.py` requires non-overlapping ranges |
| Branch checkout A/B while main moves | `probe_ab.py` toggles one flag on one SHA and rechecks HEAD/main |
| Benchmark starts on a busy GPU | `bench.py` requires five idle seconds below 10% utilization |
| Foreign CUDA work begins mid-run | `bench.py` samples CUDA PIDs and marks the case untrusted |
| In-process A/B carries caches and autotune state | `probe_ab.py` starts a separate process for every arm/case |
| Compiled baseline is used as a correctness reference | `bench.py` delegates to `run_bench.py`, which separates eager accuracy and compiled speed |

## 5. Role roster

Default to inline work. Use at most one role at a time, only for bounded work whose raw output
would flood the orchestrator context.

| Role | Use when | Required return |
|---|---|---|
| profiler | target is an unprofiled region | ranked Tier 1 table, Tier 2 gaps/graph evidence, Tier 3 or explicit unavailable result, bound classification |
| kernel-author | change touches more than one site in the benchmark file | branch diff with probe/gate/cache wiring; no measurement claims |
| verifier | before every merge | `CONFIRMED` or `REFUTED`, with artifact paths and adversarial checks |
| librarian | after every merge or discard | result record, log row/section, and lessons delta |
| researcher | backlog exhausted or 3 of 5 steps failed | hardware-mapped candidates with mechanism, expected gain, risk, and kill condition |

The canonical prompt contracts are in `agent/roles/`; `.claude/agents/` contains discovery
shims only. Hand each role the pinned SHA, experiment
directory, hypothesis, exact allowed tools, and relevant prior step excerpts. Do not ask a
role to rediscover those inputs.

## 6. Human interaction and digests

Routine promotion and discard decisions are autonomous. Print this at every boundary:

```text
=========================================================
STEP NN | MERGED|DISCARDED | short description   [GPU time]
---------------------------------------------------------
CLAIM    falsifiable mechanism
RESULT   default before -> after | matrix before -> after
         MFU before -> after     | accuracy 5/5 + 14/14 (streamed rows labeled)
DETAIL   strongest mechanism measurement [tool + artifact]
EXCLUDED cases and evidence, or none
=========================================================
```

Every four steps or two elapsed hours, whichever comes first, produce a digest containing
step history, per-suite optimized-latency and MFU trends, standing versus the compiled bar,
backlog, open questions, and profiler availability. At the first digest, publish an Artifact
dashboard if the session exposes an Artifact capability; update the same artifact afterward
and print its link. If no Artifact capability exists, state that explicitly and write the
same dashboard data to `agent/results/digests/` without fabricating a URL. Store every
dashboard or digest artifact under `agent/results/`.

Block and ask the user only for:

1. a plateau (three of the last five discarded, or under 0.5% cumulative gain over three);
2. a non-dominant tradeoff between suites;
3. questionable benchmark legitimacy, including new grading-shape specialization,
   memoization, or narrowed generality;
4. protocol/invariant changes, broken pinned environment, or a request to enable Tier 3 GPU
   counters.

## 7. Seeded backlog

Priority can change only when a measurement or dependency justifies it.

| Candidate | Mechanism | Risk / kill condition |
|---|---|---|
| Streaming/flash FFN | avoid materializing `[M, ffn]`; inspect the `minseok` prototype | high complexity; kill if traffic reduction does not offset recompute or accuracy cannot match |
| Online-softmax attention for `S>128` fp32 | replace the current SDPA fallback with a tiled one-pass algorithm | reduction-order accuracy; kill on any eager-reference failure |
| Pin autotuner configs per shape key | remove the documented tiny-shape configuration variance | kill unless ranges narrow or optimized latency improves |
| Split-K/stream-K narrow-N GEMMs | occupy 36 SMs despite tile quantization | reduction overhead; kill on overlapping or slower ranges |
| Extend Triton GEMM to attention BMMs | apply step 17's L2-resident result to remaining matmuls | stride/layout and numerical risk |
| bf16 headroom | locate the remaining gap from 60% MFU | profile before implementation |
| Honest-bar reconciliation | finish `agent/tools/bench_equal_bar.py` with reduce-overhead graph bar | may reduce headline ratio; still record honestly |
| Revisit GEMM+residual+LayerNorm | step 17 changed the epilogue economics | kill if low grid occupancy persists |

Strike through rejected rows and link the numbered step. Use the log's “Overturned in step N”
convention when new evidence reverses an earlier result.

## 8. Stop conditions

Stop when the accuracy contract cannot be met without relaxing an invariant, after four
consecutive discards, when a required protocol decision needs user approval, or when the user
says stop. Difficulty, slow profiling, or Tier 3 unavailability alone are not stop conditions.
