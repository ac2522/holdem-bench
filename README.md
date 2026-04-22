# HoldEmBench

A No-Limit Texas Hold'em benchmark for large language models, featuring natural-language table-talk.

**Status:** Phase 0 — engine + harness + baseline agents. No live tournaments yet.

## Quick start

```bash
uv sync
uv run holdembench run --config evals/stub-phase0-smoke.yaml --seed 42
```

## Architecture

See [docs/design/2026-04-22-holdembench-design.md](docs/design/2026-04-22-holdembench-design.md).

## License

Code: Apache-2.0. Data & prompts: CC-BY-4.0.

## Citation

See `CITATION.cff`.
