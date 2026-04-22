# HoldEmBench

[![test](https://github.com/ac2522/holdem-bench/actions/workflows/test.yml/badge.svg)](https://github.com/ac2522/holdem-bench/actions/workflows/test.yml)
[![lint](https://github.com/ac2522/holdem-bench/actions/workflows/lint.yml/badge.svg)](https://github.com/ac2522/holdem-bench/actions/workflows/lint.yml)
[![types](https://github.com/ac2522/holdem-bench/actions/workflows/types.yml/badge.svg)](https://github.com/ac2522/holdem-bench/actions/workflows/types.yml)

A No-Limit Texas Hold'em benchmark for large language models, with natural-language **table-talk**.

## What it measures

Three orthogonal axes (never collapsed into a single scalar):

1. **Chip EV** — mbb/100 with bootstrap 95% CI
2. **Persuasion / deception** — bluff induction, bluff resistance, claim calibration, identity-leak rate
3. **Strategic soundness** — GTO-match rate + EV-loss vs GTO reference (mbb)

## Phase 0 status

Phase 0 ships an engine + harness against deterministic baseline agents (`RandomAgent`,
`TightPassiveAgent`, `GTOApproxAgent`, `CannedTalkAgent`). Zero API cost. Byte-identical
re-run under fixed seeds. No real-model tournaments yet — Phase 1.

## Quick start

```bash
git clone https://github.com/ac2522/holdem-bench && cd holdem-bench
uv sync
uv run holdembench run --config evals/stub-phase0-smoke.yaml --seed 42
```

Artifacts land in `results/stub-phase0-smoke/`:
- `events.jsonl` — append-only JSONL event log (single source of truth)
- `manifest.json` — SHA-256 of events + tournament metadata

## Architecture

See [`docs/design/2026-04-22-holdembench-design.md`](docs/design/2026-04-22-holdembench-design.md).

## Development

```bash
uv sync --all-extras
uv run pytest                          # unit + property + golden
uv run pytest --runslow                # + integration
uv run ruff check .
uv run pyright src
uv run pre-commit install              # optional
```

## License

Code: [Apache-2.0](LICENSE). Data, prompts, scenarios: [CC-BY-4.0](LICENSES/DATA-CC-BY-4.0.txt).

## Citation

See [`CITATION.cff`](CITATION.cff). Zenodo DOI on first tagged release.
