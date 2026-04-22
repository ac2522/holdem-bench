# Contributing to HoldEmBench

## Development setup

```bash
git clone https://github.com/ac2522/holdem-bench && cd holdem-bench
uv sync --all-extras
uv run pre-commit install
```

## TDD required

Every PR that modifies `src/holdembench/engine/` or `src/holdembench/chat/` must include tests written **before** the implementation change. Reviewers check the commit order.

## Running checks locally

```bash
uv run ruff check .
uv run ruff format --check .
uv run pyright src
uv run pytest                         # fast suites
uv run pytest --runslow               # + integration
```

## Submitting a new model to the leaderboard

1. Add a model card under `MODEL_CARDS/<provider>-<model>.md` (template forthcoming in Phase 1).
2. Open a PR with the card and a stub `evals/` config referencing the new model.
3. Maintainer will run the tournament on the author's hardware (tier-2 verification).
