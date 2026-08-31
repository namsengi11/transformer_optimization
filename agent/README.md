# Continuously improving agent

This directory is the canonical home of the recursive optimization agent. Start with
`agent/ORCHESTRATOR.md`.

## Storage invariant

The improving agent must store every file it creates for its own operation under `agent/`:

- durable instructions and memory: `agent/*.md`;
- research and supporting documentation: `agent/docs/`;
- role and command contracts: `agent/roles/` and `agent/commands/`;
- measurement tooling: `agent/tools/`;
- numbered experiment records: `agent/experiments/`;
- raw measurements, profiler captures, summaries, digests, and dashboards: `agent/results/`.

Do not create agent documentation, scratch scripts, timing snippets, benchmark output,
profiler output, experiment records, or digests at repository root or elsewhere. If an
external program can only emit to another location, move its output into `agent/results/`
before continuing and cite the final path.

The only permitted files outside `agent/` are:

1. product/source changes that are the subject of a numbered optimization experiment;
2. fixed discovery shims (`AGENTS.md` and `.claude/`) that point back here;
3. ordinary repository metadata such as `.gitignore`.

Raw JSON, SQLite, and Nsight reports are machine-local and ignored by Git. Markdown records
and concise text result summaries remain trackable.

## Legacy boundary

Files that belonged to the optimization workflow before this directory was introduced remain
at repository root. In particular, `OPTIMIZATION_LOG.md`, `KERNEL_METHODS_ANALYSIS.md`,
`SOTA_AGENT_KERNEL_METHODS.md`, `profile_baseline.py`, `bench_equal_bar.py`, `results/`, and
the root result text files are historical inputs, not agent-owned outputs. Do not move,
rewrite, or append to them. Use the fresh counterparts under `agent/` for all continuous-
agent work.

## Layout

```text
agent/
  README.md
  ORCHESTRATOR.md
  LESSONS.md
  OPTIMIZATION_LOG.md
  docs/
  roles/
  commands/
  tools/
  experiments/
  results/
```

All commands are run from the repository root and therefore use paths beginning with
`agent/`.
