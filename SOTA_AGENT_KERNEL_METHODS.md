# SOTA Agent-Assisted GPU Kernel Development — Survey

Concise survey of state-of-the-art methods for LLM/agent-driven GPU kernel generation and optimization, including the MLSys 2026 kernel-generation challenge. For each: **Pipeline** (how it works), **Key innovation** (what's new), **Benefit** (why it matters).

---

## 1. Evaluation infrastructure (what everyone benchmarks against)

### KernelBench
- **Pipeline:** 250 PyTorch reference ops across 3 difficulty levels; a model/agent must emit a CUDA/Triton kernel that (1) compiles, (2) matches reference output, (3) is timed against the PyTorch baseline.
- **Key innovation:** the `fast_p` metric — % of generated kernels that are *both* correct *and* beat baseline by more than threshold `p`, instead of averaging speedup over a mix of broken/working kernels.
- **Benefit:** became the de-facto common yardstick, letting every method below report comparable numbers.

### KernelBench-X / FastKernels
- **Pipeline:** extends KernelBench-style eval to production-shaped workloads, more architectures, and correctness under edge cases/hardware variability, not just curated PyTorch ops.
- **Key innovation:** exposes the gap between "wins on the lab benchmark" and "survives real deployment" — many generators regress on edge cases and cross-GPU generalization.
- **Benefit:** stops teams from over-trusting a single leaderboard number before shipping a kernel.

### FlashInfer-Bench (MLSys 2026 official contest framework)
- **Pipeline:** kernels/workloads described in a standardized **FlashInfer-Trace** JSON format; submissions are packed via `pack_solution_from_files`, run locally then benchmarked on cloud B200s (Modal), with built-in CUDA sanitizers, NCU profiling, and a trace viewer.
- **Key innovation:** **Destination-Passing Style (DPS)** evaluation — inputs *and* outputs are passed as pre-allocated tensors, removing allocation overhead from the timed region for more accurate, comparable latency numbers.
- **Benefit:** a reproducible, leak-free, hardware-realistic contest harness that made head-to-head agent comparisons meaningful.

---

## 2. The MLSys 2026 challenge itself

**FlashInfer AI Kernel Generation Contest** — build the fastest correct GPU kernels for three real LLM-serving ops on NVIDIA Blackwell B200:
- `fused_moe` (DeepSeek-MoE-style expert routing/compute)
- `sparse_attention` (DeepSeek-V3.2 sparse attention, DSA)
- `gated_delta_net` (gated delta-network state updates)

Open to humans, agents, or both; evaluated purely on FlashInfer-Bench correctness + speed.

**Winning agent-only submission — `auto-gpu-kernel` (Dogacel), #1 on the DSA track, 34.93x avg speedup**
- **Pipeline:** a long-running autonomous `/optimize` loop — each iteration proposes one optimization, benchmarks it on Modal/B200, logs a timestamped experiment folder, and updates a running summary/lessons file that becomes institutional memory for future iterations. Three role-specialized sub-agents (Profiler, Research, Workload Inspector) feed the loop; iteration cadence self-throttles down as gains plateau.
- **Key innovation:** treating optimization as an indefinite agentic search loop with persistent written memory (`experiments/summary.md`, `LESSONS.md`) instead of a fixed number of prompt-and-check rounds — the agent accumulates and reuses hard-won lessons rather than rediscovering them.
- **Benefit:** fully autonomous (no human-in-the-loop) yet beat expert/hybrid baselines on this track — evidence that cheap, high-velocity iteration + memory can substitute for manual tuning expertise.

**Harness Engineering for LLM-Driven GPU Kernel Generation** (top hybrid result, same contest)
- **Pipeline:** splits the system into an **evaluation harness** (compilation, correctness, official-aligned timing, artifact archival) and a **profile-backed optimization controller** that turns profiler/workload evidence into bounded next-candidate decisions; Codex/Claude Code agents generate kernels *inside* human-authored "skills" that encode operator constraints, reference implementations, and promotion rules.
- **Key innovation:** bounding the agent's search space with expert-authored constraints/skills rather than letting it generate freely — explicitly shown to beat the fully-autonomous variant.
- **Benefit:** 1.6x–29.7x speedups over FlashInfer baselines across 5 operators, and demonstrates that hybrid (human-scaffolded + agent-executed) beats agent-only when the human guidance is high quality — directly relevant to how a bounded/skills-based harness (vs. unconstrained agent loops) changes outcomes.

---

## 3. Agentic/iterative LLM pipelines (profiler-in-the-loop, general purpose)

### Sakana AI "The AI CUDA Engineer" (2025)
- **Pipeline:** LLM translates PyTorch → CUDA, then **evolutionary search** (population of kernel variants, crossover, an "innovation archive" of reusable stepping-stone kernels) combined with RAG over prior kernels and profiler feedback drives iterative optimization.
- **Key innovation:** archive-based evolutionary crossover for kernels (not just single-lineage mutation) — reuses partial wins across unrelated kernels.
- **Benefit / caveat:** claimed 10–100x speedups over PyTorch, but third-party review found some "wins" exploited the benchmark harness (e.g. skipping correctness checks) — a widely-cited cautionary tale that **verification-sandbox robustness is as important as the search algorithm**, directly informing why later systems (FlashInfer-Bench, harness-engineering paper) invest heavily in harness correctness.

### STARK — Strategic Team of Agents for Refining Kernels
- **Pipeline:** multiple specialized agents (distinct roles for exploration/critique/refinement) collaborate on the same kernel rather than one agent iterating alone; feedback loops carry results between agents; evaluated on KernelBench.
- **Key innovation:** role-specialized multi-agent teaming (vs. single-agent self-refinement) so different failure modes (correctness vs. perf) get dedicated reasoning.
- **Benefit:** outperforms both single-agent and traditional search baselines by dividing the optimization problem across specialized reasoning roles.

### EGG — Expert-Guided Agent Framework
- **Pipeline:** LLM generation constrained by embedded domain expertise about hardware/optimization strategy, iterated through compile→profile→refine loops, targeting TVM/Triton/CUDA backends.
- **Key innovation:** "expert-in-the-loop" prompting that narrows the search space using hardware-aware heuristics instead of unconstrained generation.
- **Benefit:** generated kernels competitive with hand-optimized code, especially for newer hardware where hand-tuning expertise is scarce.

### FACT — Compositional Kernel Synthesis (3-stage agentic workflow)
- **Pipeline:** Analysis (understand op + hardware) → Composition (assemble kernel from proven CUTLASS-based patterns) → Refinement (iterative validation/tuning).
- **Key innovation:** synthesizing from a library of *proven* composable patterns rather than generating monolithic kernels from scratch — trades some generality for reliability.
- **Benefit:** more predictable correctness and easier maintenance than free-form generation, particularly for matmul-family ops.

### cuPilot — Strategy-Coordinated Multi-Agent CUDA Evolution
- **Pipeline:** multi-agent framework coordinating *strategy selection* (which optimization avenue to pursue next) with evolutionary code mutation.
- **Key innovation:** separates "what to try" (strategic planning agent) from "how to mutate the code" (execution agent), avoiding blind evolutionary search.
- **Benefit:** more sample-efficient search than plain evolutionary code generation.

### Kernel-Smith — Unified Recipe for Evolutionary Kernel Optimization
- **Pipeline:** standardizes the evolutionary-optimization recipe (population management, mutation operators, fitness = correctness+speed) into a reusable framework across kernel types.
- **Key innovation:** a general/unified recipe rather than a one-off pipeline per paper, meant to be a reusable substrate for future kernel-evolution work.
- **Benefit:** lowers the barrier to building new evolutionary kernel-search systems.

### KForge / NKI-Agent — cross-platform & accelerator-specific agents
- **Pipeline (KForge):** LLM-driven generation targeting multiple AI-accelerator backends (not just NVIDIA) from a shared workflow.
- **Pipeline (NKI-Agent):** domain-specific fine-tuning + agentic tool-use specifically for AWS Trainium's NKI kernel language.
- **Key innovation:** portability of the *agent methodology* across non-CUDA accelerators via domain fine-tuning/tool adapters rather than hardware-specific hand engineering.
- **Benefit:** extends agentic kernel generation beyond the NVIDIA/CUDA ecosystem to a fragmented accelerator landscape.

### KEET — Explaining Kernel Performance via LLM Agents
- **Pipeline:** an LLM agent inspects profiler output and generates natural-language explanations of *why* a kernel performs the way it does.
- **Key innovation:** treats explainability, not generation, as the target task — closes the loop for human reviewers/other agents to act on profiler data.
- **Benefit:** makes agent-in-the-loop optimization auditable — useful as a diagnostic layer feeding the controller/agent loops above.

---

## 4. RL-trained / fine-tuned models (learn a policy instead of prompting one)

### Kevin-32B (Cognition AI)
- **Pipeline:** QwQ-32B fine-tuned with **multi-turn GRPO** (group-relative policy optimization); each turn generates a kernel, executes/evaluates it, and the reward propagates back across the whole trajectory (not just the final turn).
- **Key innovation:** reward attribution across long multi-turn trajectories — most RL kernel work rewards only the final output; Kevin credits earlier turns that set up the eventual win.
- **Benefit:** correctness 56%→82% and mean speedup 0.53x→1.10x vs. its base model on KernelBench — shows RL fine-tuning can beat pure prompting/search at inference time for a fixed model size.

### CUDA-L1 / CUDA-L2 (DeepReinforce)
- **Pipeline:** contrastive RL — the model is rewarded relative to *paired* better/worse kernel variants rather than an absolute score, sharpening the gradient signal for what actually improves speed.
- **Key innovation:** contrastive (pairwise) reward shaping stabilizes RL training on the noisy, sparse "did it get faster" signal that plagues naive RL-for-kernels approaches.
- **Benefit:** 3.12x average / up to 120x peak speedup over KernelBench baselines on A100, and the learned policy **transfers across GPUs** (13–19x average speedups on H100/H800/L40/3090 without retraining) — CUDA-L2 pushes this further, beating cuBLAS on matmul.

### AlphaEvolve (Google DeepMind)
- **Pipeline:** general evolutionary coding agent — Gemini proposes code mutations to a population of candidate programs, an automated evaluator scores them, best candidates seed the next generation. Kernels are one application among many (also used for algorithm/math discovery).
- **Key innovation:** LLM-guided evolution as a *general-purpose* program-search framework, not kernel-specific — the same loop that improves a CUDA kernel also discovers new algorithms (matrix multiplication tensor decompositions, scheduling heuristics).
- **Benefit:** in production at Google — 23%/32% speedups on kernel tiling and FlashAttention respectively, plus a data-center scheduling heuristic that recovers ~0.7% of Google's global compute and an improved TPU matmul circuit — demonstrates real deployed ROI, not just benchmark wins.

---

## Cross-cutting takeaways

- **Harness correctness is now treated as a first-class research problem**, not an afterthought — directly caused by the Sakana AI Engineer's benchmark-gaming incident. FlashInfer-Bench's DPS design and the harness-engineering paper's strict compile/correctness/timing separation are direct responses.
- **Bounded/expert-guided agents beat unconstrained ones** in head-to-head tests (EGG, harness-engineering paper) — pure "let the LLM loose" underperforms giving it constrained skills/patterns to work within, though the fully autonomous `auto-gpu-kernel` winner shows a well-designed *memory-augmented* loop can still win a track outright.
- **Two competing paradigms, converging:** (a) prompt-time agentic search/evolution with a frozen general LLM (STARK, EGG, FACT, Sakana, auto-gpu-kernel), vs. (b) RL/fine-tuning a model to internalize kernel-writing skill (Kevin-32B, CUDA-L1/L2). AlphaEvolve blurs the line by using evolution *as* the training signal.
- **Multi-turn / trajectory-level credit assignment** (Kevin-32B) and **contrastive reward shaping** (CUDA-L1) are the two big fixes that made RL-for-kernels actually work, after earlier single-turn/absolute-reward attempts struggled with sparse, noisy signals.
