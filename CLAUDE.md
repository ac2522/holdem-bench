# CLAUDE.md — Collaboration rules for HoldEmBench

These instructions override the default Claude Code style guide where they
conflict.

## Communication rules

### 1. Put the answer at the TOP, explanations at the BOTTOM

The user only sees the most recent assistant turn. Anything explained
two turns ago is invisible. Structure every response as:

1. **Top:** the result, the change, the bottom-line answer.
2. **Bottom:** glossary — re-define every acronym, code, project-noun,
   and phrase used above.

Even if you defined a term last turn, define it again. Don't assume
the user remembers.

### 2. Re-explain every project-specific term

Always re-define inline. Examples that need glossing every single time:

- **Phase 1.5 / 1.6** — the smoke test phases (small-scale / full dry run)
- **P1.1-A / P1.1-B / etc.** — follow-up issue IDs we tracked
- **TDAValidator** — our internal poker-rule validator
- **CHIPS_PUSHING / CHIPS_PULLING** — pokerkit automations
- **DecisionContext / ctx.legal** — what we pass to LLM agents
- **smoke** — a low-cost end-to-end real-LLM run

### 3. No "as before" / "from the last run"

The user can't see prior outputs. Always re-state what you're comparing
to (e.g. "vs the smoke I ran 5 min ago which had 3 rejections, this run
had 0").

### 4. Plain English first

Never write "P1.1-D blocks 1.6 because state.stacks is read pre-pull."
Write the same idea as "There's a chip-accounting bug — chips disappear
from the table — that we have to fix before the bigger run."

### 5. Confirm before invasive refactors

A change that touches more than ~5 files or ~100 lines: send a short
plan first, wait for approval. Small fixes don't need this.

### 6. Don't bury status in the middle

If a smoke run failed / passed / partially passed, that goes in the
first line of the response. Not paragraph 3.

## Project guardrails

- The user's only API key is OpenRouter (in `.env`, gitignored).
  All real-LLM runs use `evals/smoke-openrouter-only.yaml` until
  per-provider keys are added.
- Each smoke run costs real money — typically <$0.01 but failed
  startups still cost a few cents in retries.
- Tests + ruff + pyright must stay green before any commit.
