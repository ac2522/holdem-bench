# HoldEmBench — Phase 1 Handoff

**For the next agent picking up this project.** Read this document first. Everything else linked from here.

---

## TL;DR

**HoldEmBench** is a public open-source benchmark where LLMs play 9-seat No-Limit Texas Hold'em against each other with natural-language table-talk. Measures three orthogonal axes: chip EV (money), persuasion/deception sub-scores (chat), and GTO-match rate (strategic soundness).

**Phase 0 is COMPLETE** — engine, harness, chat protocol, 4 baseline stub agents, scoring, CLI, full CI. 127 tests passing, deterministic stub-vs-stub tournaments with byte-identical re-runs. Released as v0.1.0, public at https://github.com/ac2522/holdem-bench. Zero API cost so far.

**Phase 1** is what you are here for: real model adapters + cheap-tier validation + Django leaderboard + first public tournament.

---

## Repo facts

- **URL**: https://github.com/ac2522/holdem-bench
- **Local path**: `/home/zaia/Development/holdem-bench/`
- **Release**: `v0.1.0` → https://github.com/ac2522/holdem-bench/releases/tag/v0.1.0
- **License**: Apache-2.0 (code), CC-BY-4.0 (data/prompts/scenarios)
- **Owner / maintainer**: Alex Carter (`alex@alexander-carter.co.uk`), GitHub `@ac2522`
- **CI**: 5 workflows all green (lint, types, test, integration, scorecard) + dependabot
- **Default branch**: `main`
- **uv 0.11.7**; `uv.lock` committed for reproducibility; Python 3.12 pinned
- **Deps pinned**: `pokerkit>=0.7,<0.8`, `pydantic>=2.7,<3`, `tiktoken>=0.7,<1`, `numpy>=1.26,<3`, `scipy>=1.13,<2`, `click>=8.1,<9`, `pyyaml>=6.0,<7`, `ruff>=0.15,<1`

---

## Architecture (locked, don't re-decide)

### Core principle

**The JSON event log is the only source of truth.** Every downstream surface — leaderboard, hand replay viewer, YouTube UI, paper figures, audit tooling — is a read-only consumer of `events.jsonl`. Nothing ever bypasses the log for persistent state.

### Two-repo split

- **`holdem-bench`** (this repo) — fully portable benchmark. Anyone can `git clone && uv sync && uv run holdembench run`.
- **`alexander-carter.co.uk/poker_bench`** (Django app inside existing `djangoproj` at `/home/zaia/Development/djangoproj/`) — thin integration layer that ingests event logs and renders the leaderboard. **Not yet built.** Phase 1 scaffolds it.

### Key design decisions (approved during Phase 0 brainstorming, don't re-open without reason)

| Decision | Resolution |
|---|---|
| Table format | 9-seat full-ring, one model per provider |
| Tournament unit (v2) | 10 cash-game sessions with escalating blinds (monthly) |
| Tournament unit (v1 pilot) | 1 cash-game session per week (flat blinds, ~150 hands) |
| Score | Cumulative chip winnings + mbb/100 with bootstrap 95% BCa CIs |
| Tiebreaker | Single extra full cash-session at max blinds, tied players only |
| Chat attached message | ≤80 tokens optional per action |
| Chat probe | Player speaks instead of acting; opponents forced to reply ≤80 tokens each |
| Chat budget | 400 tokens per player per orbit (shared across all chat types) |
| Context between games | Cleared; summarizer produces ~2k-token session notes |
| Identity | Anonymized Seat 1-9, salted per tournament, revealed at tournament_end |
| Tokenizer | `tiktoken` cl100k_base canonical (fairness across models) |
| Monthly roster | Top-4 advance + 5 new models |
| Human seat option | Supported as "Seat9" with 120s timeout (vs 60s for models); doubles as human baseline |
| Budget ceiling (pilot) | ~$100/week target via aggressive prompt caching |

Full rationale in `docs/design/2026-04-22-holdembench-design.md`.

### Directory structure (Phase 0 as shipped)

```
src/holdembench/
├── cli.py                            `holdembench run --config ...`
├── events/
│   ├── schema.py                     15 pydantic event models (discriminated union on "type")
│   └── log.py                        Append-only JSONL writer + reader + sha256_hex
├── engine/
│   ├── config.py                     TableConfig (frozen dataclass)
│   ├── deck.py                       Deterministic seeded shuffle
│   ├── table.py                      pokerkit NoLimitTexasHoldem wrapper (public API stable)
│   ├── validator.py                  TDAValidator + RawDecision (uses pokerkit min_raise_to)
│   └── ev_adjustment.py              Monte Carlo all-in equity (not yet wired into runner)
├── chat/
│   ├── tokenizer.py                  cl100k_base wrapper
│   ├── protocol.py                   ChatProtocol (budgets, probes, folded-silencing)
│   └── content.py                    card-claim & identity-leak detect + injection reject
├── agents/
│   └── base.py                       Agent Protocol + DecisionContext + Pricing + RawDecision re-export
├── baselines/
│   ├── random_agent.py               stub:random
│   ├── tight_passive.py              stub:tight_passive
│   ├── gto_approx.py                 stub:gto_approx (9-max push-fold chart)
│   └── canned_talk.py                stub:canned_talk (GTOApprox + rotating canned messages)
├── harness/
│   ├── runner.py                     Tournament game loop + TournamentConfig
│   └── manifest.py                   manifest.json writer + SHA verifier
├── scoring/
│   ├── chip_ev.py                    compute_chip_ev + mbb_per_100
│   ├── multi_way_elo.py              Plackett-Luce (MM algorithm)
│   └── bootstrap_ci.py               scipy BCa mean CI
└── replay/
    └── (empty — replay parser lives in events.log.EventLog.replay)
```

### Public API contracts (callers depend on these; be careful)

- `Table.next_actor() -> int | None`
- `Table.current_bet() -> int`
- `Table.big_blind -> int`  (property)
- `Table.min_raise_to() -> int`  (pokerkit `state.min_completion_betting_or_raising_to_amount`)
- `Table.hand_is_over() -> bool`
- `Table.apply_fold(seat: int) -> None`
- `Table.apply_check_or_call(seat: int) -> None`
- `Table.apply_raise(seat: int, to: int) -> None`
- `Agent` Protocol: `model_id: str`, `pricing: Pricing`, `async decide(ctx: DecisionContext) -> RawDecision`
- `EventLog(path)` context manager with `.emit(event)` and static `.replay(path)` + `.sha256_hex(path)`
- `TournamentConfig` dataclass; `run_tournament(cfg, agents) -> TournamentResult` async
- `parse_event(dict) -> Event` for reading logs

---

## Phase 0 — what's DONE

- Repo bootstrap (pyproject, LICENSE, Dockerfile, pre-commit, conftest)
- 15-type pydantic event schema with discriminated union + parse_event
- Append-only JSONL event log with SHA-256 manifest integrity
- Canonical cl100k_base tokenizer wrapper
- Deterministic 52-card deck shuffler
- pokerkit-backed `Table` wrapper (all API adaptations documented in `src/holdembench/engine/table.py` module docstring)
- TDA action-protocol validator (uses pokerkit's `min_completion_betting_or_raising_to_amount` for correct Rule 40 min-raise)
- Monte Carlo all-in EV adjustment (implemented but **not yet wired** into runner — deferred to Phase 1)
- Full ChatProtocol: 400 tokens/orbit, ≤80 per message, max 2 probes/hand, folded-silencing, retry-once policy
- Content validation: card-claim & identity-leak tagging, HTML/tool-call injection rejection
- Agent Protocol + DecisionContext + Pricing
- 4 stub baselines (Random / TightPassive / GTOApprox / CannedTalk)
- Tournament runner with: event-log emission, `action_request` events, chat.spend() wiring, mark_folded(), retry-once on invalid actions, deterministic time mode, manifest SHA verification
- CLI (`holdembench run --config evals/stub-phase0-smoke.yaml --seed 42 --deterministic-time`)
- Scoring: chip EV / mbb/100 / Plackett-Luce multi-way Elo / bootstrap BCa CIs
- Tests: 127 passing (unit / property / golden / integration); 8 golden hands scaffolded TODO
- CI: lint (ruff 0.15), types (pyright strict), test matrix (ubuntu/macos, py3.12), integration with `--runslow`, OpenSSF Scorecard, dependabot weekly
- Byte-identical deterministic tournament re-runs under fixed seed + `--deterministic-time` flag

**Commits**: 33 on main, all authored `Alex Carter <alex@alexander-carter.co.uk>`, tag `v0.1.0`.

---

## Phase 0 — what's DEFERRED (follow-ups)

See `docs/reviews/follow-ups.md` for the full list (8 P1s, 22 P2s, 10 P3s) with file:line refs and fix sketches. The most important ones for Phase 1 to be aware of:

- **P1-A**: narrow `_apply_raw_to_table` `ValueError` catch — currently too broad, could mask seat-index desync
- **P1-B**: wire Showdown events + EV adjustment into runner. Important for Phase 2 (real tournaments reach showdown frequently).
- **P1-C**: consolidate duplicate `ActionName`/`Street`/`ActionKind`/`ChatKind` aliases from 4 modules into one `src/holdembench/types.py`
- **P1-D**: `_Base` leaks from `events/schema.py.__all__`; `parse_event` returns `_Base` not `Event`
- **P1-E**: merge `RawDecision` (engine/validator.py) and `ActionResponse` (events/schema.py) invariant validation so they can't drift
- **P1-F/G/H**: missing property-test invariants (side-pot sum, showdown non-empty, canonical-log token count) + 5 event types never instantiated + no event-shape assertion

**Recommendation**: address P1-C (type consolidation) BEFORE Phase 1.0 adapter work — new adapters will inherit cleaner types. P1-B (Showdown emission) should land BEFORE Phase 1.6 (full-format dry run) since real models reach showdown often.

---

## Phase 1 — the plan

### Phase 1.0 — Adapter scaffolding (no network yet, ~1-2 weeks)

**Goal**: make it possible to plug in any of Anthropic / OpenAI / Google / xAI / Moonshot / OpenRouter / human, without actually calling any of them yet.

**Do these tasks in this order**, each TDD, each its own commit:

1. **Type consolidation (P1-C + P1-D + P1-E)** — create `src/holdembench/types.py` with canonical `ActionName`, `Street`, `ActionKind`, `ChatKind`. Update schema.py, validator.py, agents/base.py, chat/protocol.py to import from there. Fix `_Base` → `Event` in `parse_event`. Merge RawDecision/ActionResponse invariants.
2. **Canonical prompt template** (`src/holdembench/agents/prompt.py`). Renders system / session / hand / current-decision blocks per spec §8.3. Cache breakpoints at block boundaries. Uses canonical tokenizer for any length measurement. Returns a provider-agnostic structured prompt object.
3. **Output schema + pydantic validator** (`src/holdembench/agents/output_schema.py`). `AgentOutput` pydantic model with `kind`, `action`, `amount`, `message`, `thinking` optional. Parse failures → retry-once-then-autofold (piggyback on existing runner logic).
4. **Shared contract test suite** (`tests/contract/test_agent_contract.py`). Parametrized over all adapter classes. Asserts: output shape, timeout handling, retry behavior, cost logging, cache-control injection. Uses a shared `@pytest.fixture recorded_agent_session` pattern (e.g. vcrpy-style cassettes) so no network at test time.
5. **AnthropicAgent** (`src/holdembench/agents/anthropic.py`). Uses `anthropic>=0.50` SDK. `cache_control={"type": "ephemeral"}` on system blocks. Tool-use mode for JSON output reliability. Pricing: pull from provider pricing sheet as of build date, commit as dataclass literal. Supports `extended_thinking` with configured token budget.
6. **OpenAIAgent** (`src/holdembench/agents/openai.py`). Uses `openai>=2.0` SDK. Automatic caching (OpenAI doesn't need explicit breakpoints). Structured output via `response_format={"type": "json_schema", ...}`. Supports `reasoning_effort` for gpt-5 models.
7. **GoogleAgent** (`src/holdembench/agents/google.py`). Uses `google-genai>=1.0`. Implicit caching for Gemini 1.5+ Pro. Structured output via `response_mime_type="application/json"` + schema.
8. **XAIAgent** (`src/holdembench/agents/xai.py`). Via OpenAI-compatible endpoint (xAI's API is OpenAI-schema).
9. **MoonshotAgent** (`src/holdembench/agents/moonshot.py`). Explicit cache_control similar to Anthropic.
10. **OpenRouterAgent** (`src/holdembench/agents/openrouter.py`). Covers Llama/DeepSeek/Qwen/Yi/GLM/etc. uniformly. OpenAI-compatible. Read pricing from OpenRouter's `/models` endpoint; cache locally.
11. **HumanAgent** (`src/holdembench/agents/human.py`). Blocking asyncio call awaiting input via a `Queue`. Django view will write to the queue. 120s timeout. For now, a CLI prompt fallback in the absence of Django (`input()`).
12. **Budget circuit breaker in runner**. If any model exceeds 2× declared per-game ceiling, auto-fold for rest and emit `budget_circuit_break` event.
13. **Cost-aware summary in `TournamentResult`**. Per-model breakdown: input tokens, output tokens, cache_read tokens, cache_creation tokens, thinking tokens, USD total, retries, rate-limited time.

Landmark: at end of Phase 1.0, `uv run pytest` still passes in CI without any API keys set. Recorded-cassette contract tests prove adapters can be loaded and parse canned responses correctly.

### Phase 1.5 — Cheap-tier smoke run (~$1-3 target)

**Goal**: prove the whole thing actually works end-to-end with real LLMs, before burning money on frontier models.

- Create `evals/smoke-cheap-tier.yaml` — 50 hands, 6 seats: `anthropic:claude-haiku-4-5`, `openai:gpt-5-mini-2025-XX-XX`, `google:gemini-3-flash-preview`, `openrouter:deepseek/deepseek-chat-v3`, `openrouter:qwen/qwen3-32b`, `stub:gto_approx` (anchor).
- Pin the exact model IDs at run time by hitting Chatbot Arena / Artificial Analysis / LiveBench to see what's actually available.
- Store API keys in `~/.holdembench/credentials.toml` (NOT in repo; `.gitignore` already covers `*.env` — extend to cover `credentials.*` just in case).
- Run: `uv run holdembench run --config evals/smoke-cheap-tier.yaml --seed 42 --deterministic-time=false`.
- Verify:
  - cache hit rate ≥80% on each model (inspect per-call `cache_read_tokens` / `input_tokens`)
  - token accounting matches provider-returned usage (every `cost_usd` in the log is within 1% of what you'd compute from returned usage × pricing)
  - no model catastrophically breaks the JSON schema (retry rate <5%)
  - chat messages actually appear in the log with non-zero `tokens`
  - orbit budgets are correctly debited and truncation/rejection events appear when models try to overflow
  - total cost ≤$3 (from the `TournamentResult.total_cost_usd`)
- If any of those fail, file issues and fix before Phase 1.6.

### Phase 1.6 — Full-format dry run (~$10-20 target)

- Same cheap-tier roster, scaled to 9 seats × 150 hands (the v1 pilot format).
- If cost is too high, halve to 75 hands and note it.
- Verify the variance-reduction pieces work:
  - All-in EV adjustment runs on appropriate hands (now that P1-B is done)
  - Showdown events emitted correctly
  - Bootstrap 95% CIs refuse to rank when N<3 tournaments
- Publish the result as `results/smoke-cheap-tier-2026-XX-XX/events.jsonl` to a separate "smoke" tag (not v0.2.0) for audit.

### Phase 1.7 — Django leaderboard scaffold

- In `/home/zaia/Development/djangoproj/portfolio/poker_bench/` create the Django app per spec §11.
- Vanilla JS frontend only (CLAUDE.md in djangoproj forbids React/Next.js).
- Models: `Tournament`, `ModelAppearance`, `SeasonRanking` (thin index; data stays in JSONL).
- Views: Leaderboard / TournamentDetail / HandReplay / Play / RawData / Methodology.
- Hand-replay viewer: vanilla JS + canvas, reads `events.jsonl` via a single GET, iterates events forward, renders table with seats / cards / pot / chat column.
- `management/commands/sync_holdembench_results.py` — fetches latest release assets from GitHub.
- Deploy via existing `deploy.sh` and GitHub Actions on the djangoproj side.
- Host at `alexander-carter.co.uk/poker_bench/`.

### Phase 2 — First public tournament

- Roster: frontier (Opus 4.7, GPT-5.4 Pro, Gemini 3.1 Pro, Llama 4, Grok-latest, Kimi K2, 2 other Chinese frontier, 1 open wildcard).
- PREREGISTRATION.md filed, Zenodo DOI issued via CITATION.cff auto-integration.
- Canary UUID per tournament (already scaffolded in runner).
- Event log published as GitHub Release asset.
- Django leaderboard live.
- Announce on Twitter/HN/LessWrong.

---

## Testing budget strategy (do NOT skip this)

**The single most-cited failure mode in deception-eval papers (see `docs/design/2026-04-22-holdembench-design.md` §18) is tiny sample sizes in high-variance games.** Poker variance is brutal. Guard rails:

1. **Never skip Phase 1.5** before Phase 1.6. Cheap-tier validation is ~$3; it catches adapter bugs that would cost hundreds to surface at frontier tier.
2. **Leaderboard must refuse to rank** when bootstrap 95% CIs overlap. This is already scaffolded in the `scoring/bootstrap_ci.py` module — just needs to be wired into the leaderboard view (Phase 1.7).
3. **Minimum sample sizes** to claim a ranking: 500 hands for coarse ordering, 5000 hands for publication-grade separation between similar models. See spec §9.3.
4. **Duplicate-poker** (mirror seats, 2× cost) is a v2 feature — add once single-table pipeline is proven and budget supports.
5. **Contamination canaries**: runner already emits a `canary_uuid` per tournament in `tournament_start`. Rotate each run. Log the canary in a separate file (`docs/canaries/`) so we can check for training-data leakage post-hoc.

---

## First actions for the new agent

When you pick up this project:

1. `cd /home/zaia/Development/holdem-bench && git pull && uv sync --all-extras`
2. Read `docs/design/2026-04-22-holdembench-design.md` (the spec — §8 is most relevant to Phase 1)
3. Read `docs/reviews/follow-ups.md` (deferred items)
4. Read `docs/superpowers/plans/2026-04-22-holdembench-phase-0.md` in the `ai_games` brainstorming dir at `/home/zaia/Development/ai_games/docs/superpowers/plans/` (the Phase 0 plan — see how task structure worked; replicate for Phase 1)
5. Run `uv run pytest --runslow` to confirm 127 pass before touching anything
6. Invoke `superpowers:brainstorming` OR (if user wants to jump straight in) `superpowers:writing-plans` with this handoff doc as input to produce `docs/superpowers/plans/YYYY-MM-DD-holdembench-phase-1.md`
7. Execute with `superpowers:subagent-driven-development`, TDD per task, commit per task, mid-plan `code-reviewer` audit after adapter layer complete

---

## Reference file paths

| What | Where |
|---|---|
| Design spec (canonical) | `docs/design/2026-04-22-holdembench-design.md` |
| Phase 0 implementation plan | `/home/zaia/Development/ai_games/docs/superpowers/plans/2026-04-22-holdembench-phase-0.md` |
| Phase 0 review findings | `docs/reviews/2026-04-22-phase-0-review.md` |
| Deferred follow-up backlog | `docs/reviews/follow-ups.md` |
| This handoff | `docs/handoff/2026-04-23-phase-1-handoff.md` |
| Spec location original copy | `/home/zaia/Development/ai_games/docs/superpowers/specs/2026-04-22-holdembench-design.md` (identical to repo copy) |
| Reproducibility instructions | `docs/reproducibility.md` |
| Methodology (short) | `docs/methodology.md` |
| Repo root | `/home/zaia/Development/holdem-bench/` |
| Existing Django site (where leaderboard will live) | `/home/zaia/Development/djangoproj/` |
| Django site conventions (vanilla JS preferred, etc.) | `/home/zaia/Development/djangoproj/CLAUDE.md` |

---

## Things NOT to do

- Do not re-open the locked design decisions in the table above without explicit user confirmation. We brainstormed these over multiple rounds.
- Do not introduce React / Next.js / shadcn into the Django app — the existing `djangoproj/CLAUDE.md` rules require vanilla JS for consistency.
- Do not skip TDD on any `src/holdembench/engine/` or `src/holdembench/chat/` change — these are rules-critical code with property-test invariants.
- Do not commit API keys, even for "just a test". `.gitignore` covers `.env*`, but `credentials.toml` or `config.local.*` should also be avoided.
- Do not skip the cheap-tier smoke run (Phase 1.5) before frontier-tier (Phase 2). Budget discipline is non-negotiable.
- Do not silently drop Phase 0 follow-ups when touching related code. If you're in `src/holdembench/events/schema.py`, take the chance to also fix P1-D (`_Base` leak). Link the fix commit to the follow-up item in the commit message.
