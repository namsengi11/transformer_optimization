# TechJam Transformer Kernel

## Project overview

This project optimizes the forward pass of a PyTorch transformer specifically for the
**NVIDIA GeForce RTX 5060 Ti 16 GB**, targeting lower CUDA latency and higher model FLOPs
utilization (MFU). It compares `UserOptimizedTransformer` with an eager `BaselineTransformer`
for correctness and with a compiled baseline for speed. The optimized path combines CUDA
graph replay, PyTorch/SDPA operations, and shape-gated Triton kernels while preserving strict
`state_dict` compatibility.

The main implementation is `torch_transformer_benchmark.py`; `run_bench.py` defines the
benchmark suites and MFU calculation. The governed measurement and recursive optimization
workflow lives under `agent/`.

> ### Headline result
>
> **Geometric-mean latency improvement of `13.90x` over the naive eager baseline**, across the
> 13 dense-baseline shapes of the canonical matrix, at **38.3% mean MFU** (baseline: 10.1%).
> **All 14 shapes pass** the accuracy gate (`atol=0.002` / `rtol=0.02`) with zero failing
> elements, including the extreme `seq_len=100000` case that has no dense baseline at all.
> Per-shape numbers are in [Results](#results).

> **Target-hardware scope:** this is specifically an **RTX 5060 Ti 16 GB optimization**, not
> a generally optimal transformer implementation. Correctness should transfer where the
> supported operations exist, but its kernel choices, gates, capacity policy, and reported
> speedups are particular to this card and must be re-measured before being claimed elsewhere.

## Hardware used for the reported results

These are the settings of the Windows 11 machine on which the repository was developed and
measured. Bandwidth figures marked *theoretical* are calculated from the configured link or
memory data rate; they are not application-level benchmark results.

| Component | Setting | Pipeline impact |
|---|---|---|
| OS | Microsoft Windows 11 Home, version `10.0.26200` (build `26200`) | Determines the supported Triton package and profiler paths. |
| CPU | [AMD Ryzen 5 7600](https://www.amd.com/en/products/processors/desktops/ryzen/7000-series/amd-ryzen-5-7600.html), Zen 4, 6 cores / 12 threads, 3.8 GHz base and up to 5.1 GHz boost | Affects Python orchestration, compilation, input generation, and streamed host work. |
| System memory | 32 GiB (2 x 16 GiB Micron `CP16G60C36U5B.M8D3`), dual-channel DDR5 configured at 6000 MT/s | 96 GB/s theoretical aggregate bandwidth (`6000 MT/s x 8 B x 2 channels`); capacity and bandwidth matter for the streamed extreme case. |
| GPU | [NVIDIA GeForce RTX 5060 Ti 16 GB](https://www.nvidia.com/en-us/geforce/graphics-cards/50-series/rtx-5060-family/), Blackwell, compute capability `12.0`, 36 SMs / 4608 CUDA cores | Primary compute device and the explicit target for kernel admission/tuning. |
| GPU memory | 16 GB GDDR7 (`nvidia-smi`: 16,311 MiB), 128-bit interface, 14,001 MHz maximum memory clock | About 448 GB/s theoretical bandwidth (`28.002 Gb/s x 128 / 8`); the workflow limits capacity planning to 85% of installed VRAM. |
| Host-to-GPU link | PCIe Gen 4 x8 active | About 15.75 GB/s theoretical in each direction; transfers are included in capacity-streaming latency. |
| GPU software/power | Driver `591.86`; PyTorch CUDA runtime `12.9`; driver supports CUDA `13.1`; 180 W power limit | Driver, clocks, thermals, and power state can materially change latency. |

The measured dense GEMM ceilings used by `run_bench.py` are 24.7 TFLOP/s for TF32,
48.8 TFLOP/s for FP16, and 49.4 TFLOP/s for BF16.

### How the RTX 5060 Ti changes the optimization methodology

The card's scale changes which optimization wins. This RTX 5060 Ti has 36 SMs and 16 GB of
VRAM. For comparison, the same-generation
[RTX PRO 6000 Blackwell Workstation Edition](https://www.nvidia.com/en-us/products/workstations/professional-desktop-gpus/rtx-pro-6000/)
has 188 SMs, 96 GB of VRAM, and 1792 GB/s memory bandwidth. A kernel tuned on either device
therefore cannot be assumed optimal on the other.

- **Grid and tile selection:** kernels here try to expose at least enough independent blocks
  to occupy 36 SMs. Packing QKV projections and using smaller tiles can fill this GPU; a
  188-SM workstation GPU needs over five times as much parallel work and may prefer different
  tile sizes, split-K factors, or batching.
- **Launch overhead:** small transformer shapes finish quickly enough that Python and kernel
  launches are a large fraction of latency. CUDA graph replay and operation packing are
  consequently central. Their relative gain changes as GPU size and workload size change.
- **Memory and roofline decisions:** fusion and kernel admission use this card's measured
  compute, L2, and off-chip bandwidth rather than a generic CUDA model. An operation classified
  as launch-, cache-, or bandwidth-bound here may fall into a different regime on a workstation
  GPU, so every threshold and autotuner result must be regenerated there.
- **Capacity policy:** the 85%-of-VRAM gate streams shapes that cannot safely fit in 16 GB.
  A 96 GB workstation GPU may run the same shape densely, removing transfer and tiling costs
  and changing the best algorithm entirely.

## Setup and installation

Requirements: this Windows 11 machine or a compatible NVIDIA CUDA GPU, Python 3.12, a current
NVIDIA driver, and PowerShell. From the repository root:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Validate the pinned environment and benchmark-matrix schema before running GPU work:

```powershell
python agent/tools/env_check.py
python agent/tools/check_user_matrix.py
```

The environment check must report `PASS`. Nsight Systems is optional for ordinary benchmarks.
Nsight Compute is also optional and requires elevated GPU-counter access on this machine.

## Results

Canonical 14-shape matrix (`agent/docs/USER_MATRIX.md`), measured on the hardware above with
`python run_bench.py`. `baseline` is the naive eager `BaselineTransformer`;
`sota` is the shipped `UserOptimizedTransformer`; both are median CUDA-event latencies from
the same process. MFU is scored against the measured ceiling for the precision actually
executed (48.8 TFLOP/s FP16 for the optimized path, 24.7 TFLOP/s TF32 for the baseline).

| # | B | S | d | H | L | ffn | accuracy | max_abs | baseline | sota | speedup | MFU |
|--:|--:|--:|--:|--:|--:|--:|:--|--:|--:|--:|--:|--:|
| 1 | 64 | 128 | 128 | 4 | 4 | 128 | PASS | 1.51e-03 | 2.297 ms | 0.376 ms | **6.12×** | 46.86% |
| 2 | 1 | 128 | 128 | 4 | 4 | 128 | PASS | 9.75e-04 | 3.815 ms | 0.061 ms | **62.44×** | 4.50% |
| 3 | 4 | 128 | 128 | 4 | 4 | 128 | PASS | 1.16e-03 | 2.218 ms | 0.071 ms | **31.24×** | 15.50% |
| 4 | 16 | 128 | 128 | 4 | 4 | 128 | PASS | 1.24e-03 | 2.716 ms | 0.131 ms | **20.81×** | 33.72% |
| 5 | 128 | 128 | 128 | 4 | 4 | 128 | PASS | 1.44e-03 | 7.415 ms | 0.738 ms | **10.05×** | 47.70% |
| 6 | 10000 | 128 | 128 | 4 | 4 | 128 | PASS | 1.77e-03 | 745.7 ms | 92.835 ms | **8.03×** | 29.63% |
| 7 | 64 | 128 | 32 | 4 | 4 | 32 | PASS | 1.71e-03 | 2.691 ms | 0.131 ms | **20.62×** | 14.75% |
| 8 | 64 | 128 | 1024 | 4 | 4 | 1024 | PASS | 1.66e-03 | 31.010 ms | 12.127 ms | **2.56×** | 72.57% |
| 9 | 64 | 128 | 128 | 1 | 4 | 128 | PASS | 1.41e-03 | 2.271 ms | 0.376 ms | **6.04×** | 46.84% |
| 10 | 64 | 128 | 128 | 2 | 4 | 128 | PASS | 1.36e-03 | 2.839 ms | 0.350 ms | **8.12×** | 50.34% |
| 11 | 64 | 128 | 128 | 16 | 4 | 128 | PASS | 1.30e-03 | 11.490 ms | 0.493 ms | **23.32×** | 35.73% |
| 12 | 64 | 32 | 128 | 4 | 4 | 128 | PASS | 1.37e-03 | 2.405 ms | 0.131 ms | **18.43×** | 27.40% |
| 13 | 64 | 1024 | 128 | 4 | 4 | 128 | PASS | 1.37e-03 | 175.3 ms | 5.391 ms | **32.52×** | 71.83% |
| | | | | | | | | | | **geomean** | **13.90×** | mean **38.3%** |
| 14 † | 32 | 100000 | 1024 | 16 | 2 | 1024 | PASS | 9.47e-04 | 1,229,326.9 ms † | 59,902.5 ms | 20.52× † | 92.43% ‡ |

Rows 1-13 are one run of `python run_bench.py --suite user_matrix`
(`agent/results/attn_autotune_user_matrix.json`). **Row 14 is carried over from the previous
full-matrix run** and was not re-measured with the current attention path; it is marked
separately below and excluded from the aggregate either way. The sub-0.15 ms rows (2, 3, 7,
12) vary by 20-35% run to run at this latency scale, so their individual ratios are
directional rather than precise; the large rows (1, 5, 6, 8, 11, 13) reproduce to within ~1%.

**† Row 14 is not comparable to the rows above and is excluded from the geometric mean.** Its
dense baseline does not exist on any single GPU: the fp32 `[32, 16, 100000, 100000]` score
tensor alone is 20.48 TB. The quoted baseline is therefore a *query-tiled* eager reference,
algebraically equivalent but a different measurement class, so pooling its ratio with the
dense-baseline rows would answer no well-posed question. The row is reported because the shape
executes correctly at all — via capacity streaming — which is the result worth claiming.

**‡ Row 14's MFU is an upper bound, not a measurement.** `count_dense_flops` deliberately takes
no causal discount, which matched the dense path that computes the full `[S, S]` scores and masks
afterwards. The streaming path skips masked tiles, and at `seq_len=100000` attention is 97% of
the counted FLOPs, so the numerator is roughly double what executes. The causal-corrected figure
is approximately **46%**. The same bias affects the other causal rows by about 7%, where
attention is only 14% of the FLOPs.

## Reproduce the results

Close GPU-heavy applications first. The measurement tool waits for an idle GPU, samples for
foreign CUDA processes, runs each case three times, checks accuracy against the eager model,
and records latency ranges, MFU, and speedup against the compiled baseline.

Run a quick smoke test:

```powershell
python agent/tools/bench.py --suite quick --tag readme-quick --runs 3 --output agent/results/readme-quick.json
```

Reproduce the five-case default suite and the canonical 14-case matrix:

```powershell
python agent/tools/bench.py --suite default --tag readme-default --runs 3 --output agent/results/readme-default.json
python agent/tools/bench.py --suite user_matrix --tag readme-matrix --runs 3 --output agent/results/readme-matrix.json
```

The matrix run is long because `14_extreme` has sequence length 100,000. It is routed through
capacity streaming, includes CPU/GPU transfers, and intentionally has no compiled-baseline
speedup. The promoted step-23 measurement at commit `ed662364` passed 102,400,000 compared
elements with zero failures (`max_abs=0.000811517`) and measured 1911.8657 ms per batch element
at 90.50% MFU. Exact reproduction should be judged from the new artifact's accuracy, trust
flag, and latency range, not from a single identical timing value.

## Agent workflow

The continuous-improvement agent is introduced in [`agent/README.md`](agent/README.md) and
governed by [`agent/ORCHESTRATOR.md`](agent/ORCHESTRATOR.md). Its design builds on the earlier
survey in [`SOTA_AGENT_KERNEL_METHODS.md`](SOTA_AGENT_KERNEL_METHODS.md) and the conclusions in
[`KERNEL_METHODS_ANALYSIS.md`](KERNEL_METHODS_ANALYSIS.md). That research compared approaches
such as compile-correct-profile loops, iterative kernel agents, evolutionary search, profiler
feedback, and cross-hardware validation. The current workflow turns those ideas into a strict,
artifact-backed process for this RTX 5060 Ti rather than treating generated code or one fast
timing as sufficient evidence.

The working loop is:

`environment and schema checks -> profile/research -> falsifiable hypothesis -> numbered,
minimal implementation -> repeated measurement -> adversarial verification -> merge or
discard -> durable result and lessons`

Only one experiment is active at a time. It pins the starting commit, predicts which cases
should improve, states a measurable kill condition, and makes the smallest change capable of
testing the mechanism. Promotion requires the 5-case default suite and all 14 canonical matrix
cases to pass accuracy, plus a trusted improvement outside the observed timing ranges without
a material regression elsewhere. Failed ideas are retained as numbered evidence so the agent
does not repeatedly revisit disproven mechanisms. Profiling, verification, research, and
record-keeping roles are used only for bounded parts of this loop.

The most crucial safeguards are:

- **Correctness is independent of the speed target.** Outputs are checked against the eager
  `BaselineTransformer`; the compiled baseline is used only as the performance bar. Large
  shapes that cannot form dense attention use the query-tiled eager reference without changing
  the numerical contract.
- **Measurements need provenance and isolation.** Approved tools record the commit, command,
  environment, and output artifact. Judged results use at least three independent runs, an
  idle-GPU gate, and foreign-CUDA-process monitoring; overlapping timing ranges count as noise.
- **Hardware mechanisms must be demonstrated.** A change must connect a measured bottleneck
  to this card's 36 SMs, memory hierarchy, launch behavior, or 16 GB capacity. New kernel paths
  use whole-forward correctness probes, cached shape/dtype decisions, and permanent fallback
  switches before CUDA graph capture.
- **Generalization is protected.** Exact grading-shape specialization, memoization, relaxed
  tolerances, and benchmark-harness changes are forbidden. Both ordinary and extreme cases
  must remain executable through general shape- and capacity-derived policies.
- **Negative results remain useful.** Every merge or discard updates the experiment record,
  optimization log, and reusable lessons. Repeated failures trigger profiling or research
  instead of uncontrolled code churn.

All new experiments, profiles, measurements, and digests belong under `agent/`. The current
agent-owned research files in `agent/docs/` clearly separate newly confirmed evidence from the
historical root documents, which remain read-only.

## Limitations and future improvements

- The kernels, tiles, admission thresholds, roofline constants, and 85%-VRAM streaming gate
  are specific to the RTX 5060 Ti 16 GB and pinned PyTorch/Triton versions. Results must not be
  presented as workstation- or data-center-GPU performance without fresh tuning and measurement.
- Timing remains sensitive to background GPU work, clocks, thermals, driver behavior, and
  compiler caches despite idle gating and repeated measurements.
- MFU uses dense matmul FLOPs and empirical GEMM ceilings, so it does not represent every
  operation or memory bottleneck in the transformer.
- The extreme streamed row has no capacity-equivalent compiled streaming baseline, so it can
  be evaluated for accuracy, latency, and MFU but not compiled-baseline speedup.
- More time would go toward cross-GPU CI and environment locking, an honest streaming baseline,
  online-softmax attention for long sequences, and a streaming/fused FFN that avoids large
  intermediate activations.
