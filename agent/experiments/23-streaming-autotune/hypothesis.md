# Step 23: general 85%-VRAM streaming autotuner

- Status: PROMOTED
- Branch: `opt/23-streaming-autotune`
- Pinned main SHA: `ba933aa7a8f1b0db398898ecd9416510c936b5f5`
- Pinned experiment SHA: `d115f513fb999cc4ba2ff5a3b06cd59a86b72a2b`
- Started UTC: 2026-08-31T20:29:27Z

## Claim

For any capacity-streamed shape, independently selecting the optimized batch shard and the
reference query tile from `(B,S,D,H,FFN,dtype,installed VRAM)` under an 85% measured-memory
target will reduce long-case latency and raise optimized MFU versus the conservative shared
65%/10% plan, without benchmark-name or exact-shape specialization.

The capacity target is a ceiling, not a requirement to fill every byte. Candidate selection
maximizes useful work under that ceiling and may deliberately use less memory when a larger
candidate is slower or crosses the allocator-reserved limit.

## Evidence and prior art

- Step 21 made canonical row 14 executable and passed 3,276,800,000 output elements with
  zero failures (`max_abs=0.000951409`) using batch shard 1 and query tile 128.
- That safe reference used about 8.3 GiB of 16 GiB and took about 20 minutes for one
  reference-plus-optimized accuracy trial, leaving substantial unused capacity.
- Equal-`B*S` capacity probes proved that linear token count alone is insufficient; the
  planner must retain both linear residency and quadratic attention terms.
- Experimental variable-key and extra softmax-workspace variants OOMed despite smaller
  arithmetic, so candidates must keep fixed reusable workspaces and preserve headroom for
  vendor workspaces and allocator behavior.

## Expected impact

- Capacity-streamed shapes: larger optimized batch shards where the memory-efficient SDPA
  path fits, fewer host/shard iterations, and a larger reference query tile.
- Dense shapes: no behavior or latency change because the streaming gate is not entered.
- Expected row-14 effect: optimized batch shard 1 -> at least 2 if measured safe; reference
  tile 128 -> the largest aligned candidate below the 85% target. These are expectations,
  not predicates.

## Accuracy argument

Batch elements are independent. Changing batch shard size changes scheduling only, while
the query-tiled reference still retains every key and computes full attention. Whole-output
accuracy remains judged at `atol=0.002`, `rtol=0.02` with zero failed elements. Candidate
selection depends only on shape/dtype/device capacity and measured execution, never values.

## Measurement plan

- Use `agent/tools/streaming_tile_probe.py` in isolated processes to sweep aligned query
  tiles and record latency plus peak allocated/reserved memory.
- Add an equivalent isolated optimized-shard probe and sweep feasible shard sizes.
- Select the fastest candidate whose observed peak is at most 85% of installed VRAM.
- Validate on at least two feasible long-sequence surrogates and canonical row 14; confirm
  ordinary quick/default cases stay on the dense path.

## Kill condition

Discard if the selected candidate exceeds 85% observed peak, OOMs, changes any below-gate
path, fails any output element, or cannot beat the conservative streamed plan. A candidate
that merely uses more VRAM without reducing latency is rejected.

## Legitimacy check

Candidate generation is a general function of shape, dtype, and installed device capacity.
The cache key contains those properties, not suite/case names or exact official values.
The tuner changes storage and batching only; it does not use local/windowed attention,
memoize outputs, relax tolerances, or omit causal relationships.
