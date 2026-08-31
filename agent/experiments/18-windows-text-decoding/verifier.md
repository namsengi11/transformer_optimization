REFUTED

- `agent/experiments/18-windows-text-decoding/tool-smoke.txt` confirms the decoding
  exception is gone, but the hypothesis kill condition also requires a trusted end-to-end
  artifact.
- `agent/experiments/18-windows-text-decoding/quick.json` has clean fixed-checkout
  provenance at `c3849ee5396dd7592bfa7d62419aea2ba7a537c8` and 1/1 eager-reference
  accuracy, but `trusted:false` in all three runs.
- Each run attributes three transient PIDs at 96-98% SM to foreign work, matching the
  benchmark's short-lived child-process pattern; the process-tree snapshot occurs after
  `nvidia-smi pmon`, so exited descendants can be misclassified.
- No product, harness-control, baseline, shape, dtype, or precision behavior changed.
