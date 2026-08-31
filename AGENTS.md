# Continuous-improvement agent entrypoint

When running or modifying the recursive optimization workflow, first read
`agent/README.md` and `agent/ORCHESTRATOR.md`. Treat `agent/` as the exclusive storage root
for the improving agent's documentation, tools, experiments, measurements, profiler output,
digests, dashboards, and other generated files.

Product code may be changed outside `agent/` only as part of a numbered experiment governed
by `agent/ORCHESTRATOR.md`. Keep the fixed `.claude/` files as discovery shims only.
