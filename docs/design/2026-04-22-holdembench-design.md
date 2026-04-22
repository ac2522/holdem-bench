# HoldEmBench — Poker-with-Table-Talk Benchmark for LLMs
## Design Specification

| | |
|---|---|
| **Status** | Brainstorm complete — pending user spec review |
| **Date** | 2026-04-22 |
| **Project name** | **HoldEmBench** (finalized). GitHub: `holdem-bench`. PyPI/module: `holdembench`. Working codename during brainstorm was "POCA". |
| **Owner** | Alex Carter |
| **License target** | Apache-2.0 (code), CC-BY-4.0 (data) |
| **Target paper venue** | NeurIPS Evaluations & Datasets track, 2027 |
| **Author** | Drafted by Claude (Opus 4.7) in brainstorming session |

## 1. Executive summary

HoldEmBench is a professional, public, open-source LLM benchmark in which frontier LLMs play
9-max No-Limit Texas Hold'em against each other **with natural-language table-talk**.
The benchmark measures three orthogonal axes — chip-EV skill, persuasion/deception
sub-scores, and strategic soundness vs a GTO reference — and publishes a public
leaderboard and fully reproducible per-hand event logs.

Prior work (Werewolf Arena, AvalonBench, Among Us sandbox, PokerBench, AI-Diplomacy)
has produced either game-play benchmarks **without** natural-language chat or
deception benchmarks **without** a correct rules kernel. **No prior published work
combines a correct NL Hold'em engine, natural-language table-talk, multi-model
tournament structure, and publication-grade variance reduction**. HoldEmBench fills that gap.

The project launches as a weekly pilot (1 cash-game session per week at ~$100/week)
and scales in phases to a monthly 10-game escalating-stakes tournament format suitable
for a NeurIPS D&B submission.

## 2. Goals & non-goals

**Goals:**

1. Measure LLM capability at No-Limit Texas Hold'em in a format that includes
   natural-language communication between players.
2. Separate *skill* (chip EV, GTO match rate) from *persuasion* (bluff induction,
   bluff resistance, claim calibration) as reported axes — never collapsed into a
   single scalar.
3. Produce a public leaderboard with publication-grade statistical rigor:
   bootstrap CIs, CI-overlap-aware ranking, multiple baselines (random, GTO-approx,
   canned-talk), transparent methodology docs, full event-log publication.
4. Be **fully reproducible** by third parties — `git clone && uv sync && docker build`
   reproduces any tournament byte-for-byte given API access.
5. Host the leaderboard on the author's existing Django site at
   `alexander-carter.co.uk/poker_bench/`, with replayable hands and downloadable
   raw data per HELM-level transparency norms.
6. Architect so that a **YouTube/v1.5 UI** is a read-only consumer of the same JSON
   event log — no engine changes needed to add broadcast production.
7. Eventually publish a peer-reviewed paper (NeurIPS Evaluations & Datasets 2027).

**Non-goals:**

- Training, fine-tuning, or RLHF of any model. HoldEmBench evaluates existing models as black boxes.
- Real-money gambling. Chip stacks are nominal units; no value transfer.
- General agentic eval beyond poker. HoldEmBench is purpose-built.
- Creating new poker variants. Standard NLHE only.
- "Solving" poker or producing a frontier poker agent. Goal is to *measure*, not *build*.

## 3. Key design decisions (approved in brainstorming)

| Decision | Resolution |
|---|---|
| Table format | 9-seat full-ring, one table, one model per provider |
| Tournament unit | 10 cash-game sessions with escalating blinds (v2); 1 weekly session pilot (v1) |
| Score rule | Sum of chip winnings across all sessions (mbb/100 for variance-normalized comparison) |
| Tiebreaker | Single extra full cash-game session at max blinds, tied players only |
| Chat — attached messages | ≤80 tokens, optional, per action |
| Chat — probes | Player may speak instead of acting; opponents must respond ≤80 tokens each |
| Chat — budget | 400 tokens per player per orbit (= N hands for N seats), shared across all chat types |
| Context between games | Cleared; a summarizer produces ~2k-token session notes for carryover |
| Identity | Anonymized Seat 1..9; salted per tournament; reveal at tournament_end |
| Roster policy | One model per provider; top-4 advance monthly; 5 new slots opened |
| Monthly season | 4 weekly games → rotation; later, monthly 10-game tournaments |
| Human player | 9th seat option; same budgets; 120s timeout (vs 60s for models); doubles as human baseline |
| Budget ceiling (v1) | ~$100/week pilot; scale based on caching efficacy |
| Licensing | Apache-2.0 code + CC-BY-4.0 data |
| Project name | HoldEmBench (working label; final rename before public launch) |
| Website home | `alexander-carter.co.uk/poker_bench/` (Django app inside existing djangoproj) |
| Timeline | No hard deadline; ship when right |
| YouTube | v1.5 — design for it from day 1 via event-log-as-truth; ship production later |

## 4. System architecture

### 4.1 Core principle

**The JSON event log is the single source of truth.** Every downstream surface —
leaderboard, hand replay viewer, YouTube UI, paper figures, audit tools — is a
read-only consumer of event logs. This is the one architectural decision that keeps
every other piece swappable and makes v1.5 YouTube production a pure consumption
problem.

### 4.2 Two-repository split

```
holdem-bench/                           ← public repo (Apache-2.0 + CC-BY-4.0)
├── src/holdembench/
│   ├── engine/                       pokerkit wrapper + TDA action validator
│   ├── chat/                         protocol + budget accounting + content validation
│   ├── agents/                       model adapters (Anthropic/OpenAI/Google/xAI/
│   │                                 Moonshot/OpenRouter/Human)
│   ├── baselines/                    Random, TightPassive, GTOApprox, CannedTalk
│   ├── harness/                      game runner, seed mgmt, event-log writer
│   ├── scoring/                      Elo/Plackett-Luce, bootstrap CIs, chip EV,
│   │                                 GTO match-rate, persuasion sub-scores
│   ├── replay/                       canonical event-log parser (shared lib)
│   └── cli.py                        `holdembench run --config evals/weekly-YYYY-MM-DD.yaml`
├── evals/                            versioned tournament configs (frozen YAML)
├── results/<tournament_id>/          events.jsonl, manifest.json, config.yaml
├── tests/
│   ├── unit/                         pytest
│   ├── property/                     hypothesis
│   ├── golden/                       40+ canonical hands
│   ├── integration/                  stub-agent end-to-end
│   └── smoke/                        cheap real-model tag (manual trigger)
├── docs/                             mkdocs-material (methodology, repro, citation)
├── .github/workflows/                lint, types, test, integration, docs, release,
│                                     scorecard
├── MODEL_CARDS/                      one MD per registered model
├── Dockerfile                        digest-pinned base
├── CITATION.cff, LICENSE, LICENSES/DATA-CC-BY-4.0.txt
├── CODE_OF_CONDUCT.md, CONTRIBUTING.md, SECURITY.md, GOVERNANCE.md
├── PREREGISTRATION.md                frozen protocol, Zenodo-DOI'd
└── README.md, pyproject.toml, uv.lock

djangoproj/portfolio/ (existing personal site)
└── poker_bench/                      ← new Django app (thin integration)
    ├── models.py                     Tournament, ModelAppearance, SeasonRanking
    ├── views/                        Leaderboard/Tournament/Replay/Play/Raw/Methodology
    ├── templates/poker_bench/        Django templates (vanilla JS, no React)
    ├── static/poker_bench/           replayer.js (vanilla, canvas), table-styles.css
    ├── management/commands/
    │   └── sync_holdembench_results.py      fetches latest release from holdem-bench
    ├── tests.py                      Django test suite
    └── urls.py                       wired into djangoproj
```

### 4.3 Data flow

```
Tournament config (YAML)
         │
         ▼
    ┌─────────────┐    async     ┌──────────────┐    immutable
    │   Harness   │ ───────────▶ │    Model     │       │
    │ (Inspect AI │              │   Adapters   │       ▼
    │   based)    │              │ (cached)     │  ┌──────────────┐
    └─────┬───────┘              └──────┬───────┘  │events.jsonl  │
          │                             │          │ +            │
          │    pokerkit + TDA + Chat    │  writes  │manifest.json │
          └─────────────────────────────┴─────────▶│+ config.yaml │
                                                   └──────┬───────┘
                                                          │
                       ┌──────────────────────────────────┼───────────────────┐
                       │                                  │                   │
                       ▼                                  ▼                   ▼
              ┌───────────────┐            ┌────────────────────┐    ┌────────────────┐
              │ scoring/      │            │ Django leaderboard │    │ YouTube UI     │
              │ (Python)      │            │ + hand replayer    │    │ (v1.5, TBD)    │
              └───────┬───────┘            │ (vanilla JS)       │    │                │
                      ▼                    └────────────────────┘    └────────────────┘
               aggregate.parquet,
               season rankings
```

### 4.4 Why two repos

- The benchmark must be third-party reproducible without dependency on the author's personal website.
- Django integration stays thin. If the personal site migrates, HoldEmBench doesn't move.
- Paper reviewers get `holdem-bench` alone, cleanly.
- CI and release schedules differ naturally between benchmark and portfolio site.

## 5. JSON event log — canonical schema (v1.0)

### 5.1 File layout per tournament

```
results/<tournament_id>/
├── events.jsonl        # newline-delimited JSON, one event per line, append-only
├── manifest.json       # schema version, model pins, seeds, SHA-256 of events.jsonl
└── config.yaml         # frozen tournament configuration used for this run
```

### 5.2 Event types (discriminated union on `type`)

| Event type | When emitted | Key fields |
|---|---|---|
| `tournament_start` | First line of every log | schema_version, holdembench_version, pokerkit_version, git_sha, seat_assignments (map Seat→model_id), master_seed, anonymization_salt, canary_uuid |
| `session_start` | Start of each cash-game session | session_id, hand_cap, small_blind, big_blind, ante, deal_pack_seed |
| `hand_start` | Each hand | hand_id, button_seat, stacks, cards_hash, chat_budgets_remaining |
| `deal` | Private card deal (one per seat) | street, to_seat, cards |
| `action_request` | Harness asks seat for decision | hand_id, to_seat, street, legal, timeout_s, budget_remaining |
| `action_response` | Seat responds | seat, kind ∈ {action, probe, probe_reply}, action/amount/message, tokens, latency_ms, cost_usd, model_id, prompt_hash, thinking (optional) |
| `probe_response_request` | After a probe, harness asks each in-hand opponent | from_seat, responders |
| `community_deal` | Board cards | street, cards |
| `showdown` | End-of-hand reveal | revealed (seat→cards), winners, all_in_ev_adjusted, stack_deltas_actual (if EV-adjusted) |
| `hand_end` | After pot distribution | hand_id, stack_deltas, elapsed_s, total_cost_usd |
| `session_end` | End of cash-game session | session_id, final_stacks, total_hands, total_cost_usd |
| `tournament_end` | Last line of every log | final_chip_totals, winner_seat, winner_model, total_cost_usd, wall_clock_s |
| `validator_rejection` | Action rejected by TDA validator | seat, reason, original_response, retry_allowed |
| `budget_circuit_break` | Cost ceiling hit for a model | seat, threshold_usd, actual_usd |
| `auto_fold` | Model defaulted to fold (timeout/invalid) | seat, reason |

### 5.3 Design invariants

1. **Append-only.** No event ever mutated after write; manifest SHA proves integrity.
2. **Replay-pure.** Any derived state (stacks, pot, chat column, cost dashboard) is
   recomputable by iterating the log. No sidecar files are load-bearing for replay.
3. **Anonymization commitment.** Seat identities live only in `tournament_start`.
   Audit bundles can be published with that event stripped for blind evaluation.
4. **Card commitment.** `hand_start.cards_hash` is SHA-256 of the concatenated deal
   sequence, written before `deal` events. Post-hoc card edits are detectable.
5. **Versioned schema.** `schema_version` in `tournament_start`. Breaking changes
   bump the version; all versions readable forever via `holdembench.replay`.
6. **Cost truth.** Every `action_response` carries `cost_usd` from the provider's
   returned usage (not our pre-estimate). Sum is auditable.
7. **Sidecar analyses never alter the log.** GTO annotations, judge rubrics, card-claim
   extractions live in `results/<tid>/analysis/*.json` — read-only derivations.

### 5.4 Non-schema content rules

- Messages are monotype plain text. No HTML, no markdown rendering, no embedded tool calls.
- Card claims in chat are allowed (core to poker); analyzed post-hoc in `analysis/card_claims.jsonl` for calibration metrics.
- Self-identification strings are logged but not stripped; counted as identity-leak metric.

## 6. Game engine (`src/holdembench/engine/`)

### 6.1 Library

**pokerkit v0.7.x (MIT, IEEE ToG peer-reviewed, 99% coverage)** — only open-source
Python library with fully correct multi-way side-pot, all-in, and split-pot logic.
Used as a library, not subclassed.

### 6.2 `Table` wrapper

```python
class Table:
    """Owns a pokerkit State + chat protocol + event-log writer."""
    def __init__(self, config: TableConfig, rng: np.random.Generator,
                 log: EventLog, chat: ChatProtocol):
        self.state = pokerkit.NoLimitTexasHoldem.create_state(...)
        self.validator = TDAValidator(self.state)
        self.chat = chat
        self.log = log

    def next_decision(self) -> Decision | None:
        """Returns None when hand is over. Caller drives the loop."""

    def apply(self, seat: int, decision: Decision) -> None:
        """Validate → pokerkit → emit events."""
```

### 6.3 `TDAValidator` — action-protocol rules pokerkit doesn't cover

- TDA Rule 21 — side-pot formation on mixed all-ins (cross-check vs pokerkit pots)
- TDA Rule 36 — no card reveal before showdown (action message is NOT validated for
  card-claim content; analysis is post-hoc — see §5.4)
- TDA Rule 40 — min raise = size of previous full bet/raise
- TDA Rule 44 — no string bets (exact amount declared, no post-hoc additions)
- One-player-to-a-hand — folded players silenced for the remainder of that hand (budget frozen)
- Action shape — `fold | check | call | raise(amount)` only

### 6.4 Invalid-action policy

Invalid action → log `validator_rejection` → re-prompt seat once with reason → on
second failure, auto-fold (`auto_fold` event). This is a benchmark of whole-protocol
competence; models that can't format actions should lose.

### 6.5 All-in EV adjustment

When all remaining chips go in pre-river with ≥2 players:
- Deal the river for narrative (written to log as usual)
- Compute Monte Carlo equity using `pokerkit.hand` on remaining-deck samples (20,000 samples → ±0.2% equity error)
- Official `hand_end.stack_deltas` = EV-adjusted. `stack_deltas_actual` logged for reference. `all_in_ev_adjusted: true` flag set.
- ~15–25% variance reduction for zero methodological cost.

### 6.6 Deterministic dealing

Each session uses `deal_pack_seed` → deterministic shuffle → known deck permutation.
Same seed = identical cards on re-run. Critical for reproducibility and sets up
duplicate-poker extension in v2+.

### 6.7 Out of scope for engine

- Prompt templating (agent adapter)
- Chat budget accounting (chat module)
- Scoring across sessions (scoring module)
- Any network I/O

## 7. Chat protocol (`src/holdembench/chat/`)

### 7.1 Terminology

- **Attached message** — optional ≤80-token string submitted with an action.
- **Probe** — action substitute: player speaks instead of acting; opponents must reply; original player then acts.
- **Probe reply** — required ≤80-token (≥20-token) response from each in-hand opponent after a probe.
- **Orbit** — one full button rotation = N hands for N seats.
- **Chat budget** — 400 tokens per player per orbit, shared across all chat types. Resets at orbit start.

### 7.2 Turn schema

On each decision, a player submits exactly one of:

| `kind` | Fields | Token cost | Next step |
|---|---|---|---|
| `action` | `action`, `amount?`, `message?` (≤80 tok) | message tokens | pokerkit advances |
| `probe` | `message` (≥10, ≤80 tok) | message tokens | opponents forced into probe_reply; original seat must act next |

### 7.3 Probe restrictions

- Max 2 probes per player per hand.
- Cannot probe twice in a row.
- Cannot probe when only 2 players remain in the hand.
- Cannot probe if remaining orbit budget < 80 tokens.
- Probe reply must be ≥20 and ≤80 tokens; auto-`"[no comment]"` if budget exhausted (cost: 0).

### 7.4 Budget exhaustion

- When orbit budget < 10 tokens, player is silent for rest of orbit (still can act; no chat).
- Folded players' budgets frozen for rest of hand; resume next hand.
- Auto-replies when forced-to-respond but budget is 0 cost nothing; flagged `auto_generated: true`.

### 7.5 Content rules

- Plain text only; no HTML/markdown/tool-call injections.
- Card claims allowed; tracked in `analysis/card_claims.jsonl` for calibration metric.
- Self-identification strings logged + counted in `analysis/identity_leaks.jsonl` (penalty at leaderboard level; not rejected).
- Prompt-injection / structural-escape attempts → rejected as malformed.

### 7.6 Canonical tokenizer

**OpenAI `cl100k_base` via `tiktoken`**. Single ruler for all models regardless of native tokenizer. Documented in `docs/prompting.md`.

### 7.7 Timeouts

- Model adapter: 60s per decision.
- Human adapter: 120s.
- On timeout: auto-fold event; same policy as validator rejection.

## 8. Agent adapter layer (`src/holdembench/agents/`)

### 8.1 Protocol interface

```python
class Agent(Protocol):
    model_id: str           # "anthropic:claude-opus-4-7-20260312"
    pricing: Pricing        # per-1M token rates

    async def decide(self, ctx: DecisionContext) -> RawDecision: ...
```

`DecisionContext` contains: game_state_snapshot, visible_event_log window, legal_actions,
budget_remaining, is_probe_reply, deadline. The `Table` validates the returned
`RawDecision` before acceptance.

### 8.2 Concrete adapters

- **Frontier**: `AnthropicAgent`, `OpenAIAgent`, `GoogleAgent`, `XAIAgent`, `MoonshotAgent`
- **Open-source routing**: `OpenRouterAgent` (Llama / DeepSeek / Qwen / Yi / GLM / etc.)
- **Human**: `HumanAgent` — blocking asyncio call awaiting Django view submission
- **Stub baselines** (in `baselines/`):
  - `RandomAgent` — uniform over legal actions, never chats
  - `TightPassiveAgent` — folds weak, calls medium, raises top-10%, no chat
  - `GTOApproxAgent` — precomputed 9-max push-fold chart + postflop heuristic, no chat
  - `CannedTalkAgent` — GTOApprox + rotating canned messages, for "does fake chat help at all?" control

### 8.3 Canonical prompt template (identical across all models)

```
[SYSTEM — cached per tournament]
You are playing No-Limit Texas Hold'em as Seat {N} at a 9-seat table.
Rules: standard NLHE, TDA-compliant. [action/chat protocol spec and output schema]

[SYSTEM — cached per session]
Session {S}: blinds {SB}/{BB}, ante {A}, starting stack {ST}bb.
Seat roster: Seat1..Seat9 (anonymized; do not infer identities).
Your seat: {N}. Orbit budget: {B} tokens.

[USER — cached per hand]
Canonical action log so far this session:
{COMPACT_ACTION_LOG}

[USER — volatile]
Current hand {H}: button=Seat{B}, stacks={...}
Hole cards: {CARDS}
Board: {BOARD}
Chat this hand:
{CHAT_STREAM}
Legal actions: {LEGAL}
Budget remaining: {B}
```

### 8.4 Caching discipline

- Cache breakpoints on blocks 1 and 2 (Anthropic `cache_control: ephemeral`, OpenAI auto, Google implicit, Moonshot explicit).
- Canonical action log block cache-breakpointed per-session.
- Volatile block re-sent each call (~500 tokens).
- Projected hit rate: ~90%. Typical call cost ~$0.005 on Opus 4.7 vs ~$0.05 uncached.

### 8.5 Budget circuit breaker

Harness tracks running cost per model per game. If any model exceeds 2× declared
budget ceiling, harness emits `budget_circuit_break`, offending model auto-folds
remainder. Prevents runaway costs from broken adapters.

### 8.6 Output schema & validation

- All models return JSON matching a tight `pydantic` schema.
- Native tool-use used where supported (Anthropic, OpenAI) for reliability.
- Parse failure → retry once with error message → second failure = auto-fold.

### 8.7 Thinking-token policy

- Reasoning/thinking tokens logged in separate `action_response.thinking` field.
- Never shown to other players.
- Do count against cost reporting.
- Uniform thinking budget per model across tournament (e.g., Claude extended thinking
  5000 tokens; equivalent reasoning_effort elsewhere); documented in `PREREGISTRATION.md`.

### 8.8 Human adapter

- Receives `DecisionContext` via an asyncio Queue from the Django `PlayView`.
- Django template renders the context as a poker-table HTML view with input fields.
- Submit form POSTs to view → pushes `RawDecision` back onto the queue.
- Auth: uses existing `alexcarter` cookie for owner play; anonymous UUID for public sessions.
- 120s timeout; otherwise same budget/protocol rules.

## 9. Harness & variance reduction (`src/holdembench/harness/`)

### 9.1 Inspect AI backbone

Each agent decision wrapped as an `inspect_ai.Task` call. Inherits from Inspect:
typed responses, cost/token logging, `.eval` log files, static log viewer deployable
via `inspect view bundle`. Multi-agent table loop sits above Inspect.

### 9.2 Game runner (pseudocode)

```
run_tournament(config):
  log = EventLog(config.results_path)
  emit tournament_start
  master_rng = np.random.default_rng(config.master_seed)
  for session in config.sessions:
    deal_pack = shuffled_deck(seed=derive(master_seed, session.id))
    emit session_start
    for hand in 1..session.hand_cap:
      emit hand_start
      while table.next_decision():
        ctx = build_decision_context(log, seat)
        raw = await agent[seat].decide(ctx)   # via Inspect AI
        decision = TDAValidator.check(raw)
        emit action_response
        if decision.is_probe:
          for opp in in_hand_opponents(seat):
            reply = await agent[opp].decide(probe_reply_ctx)
            emit action_response  (kind=probe_reply)
      if all_in_preflop_turn(hand):
        run_ev_adjustment()
      emit hand_end
    emit session_end
  emit tournament_end
  verify manifest.json sha256 == sha256(events.jsonl)
```

### 9.3 Variance reduction — v1

1. **Seeded deal packs per session.** Reproducibility primary.
2. **All-in EV adjustment** (§6.5). Free ~15–25% variance reduction.
3. **Canonical baselines always present** in shadow eval runs parallel to the public tournament, so every tournament has a solver-anchored reference point.

### 9.4 Variance reduction — v2+

4. **Duplicate poker via seat rotation.** Replay same deal-pack with seats rotated;
   score relative to position-average performance. Halves card-luck variance at 2× cost.
5. **AIVAT-lite approximation.** Using a precomputed solver baseline as the value estimator.
   Aspirational — full AIVAT requires explicit strategies we don't have.

### 9.5 Reported stats per game

- Chip winnings (raw)
- mbb/100 (milli-big-blinds per 100 hands)
- VPIP, PFR, AF per player (process metrics)
- Per-model mean + bootstrap 95% BCa CI (≥3 games required for CI display)
- Leaderboard withholds rank when 95% CIs overlap (shows "tied within CI")

## 10. Scoring & leaderboard stats (`src/holdembench/scoring/`)

### 10.1 Three axes (reported separately, never aggregated)

1. **Chip EV** — primary sort; mbb/100 + bootstrap CI.
2. **Persuasion / deception sub-scores:**
   - Bluff induction rate — folded equity extracted weighted by chat activity
   - Bluff resistance rate — superior equity folded after opponent probes
   - Claim calibration — precision/recall of card-strength claims vs showdown reveals
   - Identity-leak rate — messages matching self-identification patterns / total messages
3. **Strategic soundness** — GTO-match rate + EV-loss-vs-GTO (mbb). PokerBench metric.

### 10.2 Cross-tournament aggregate

**Multi-way Plackett-Luce Elo** — extension of Bradley-Terry to n-way rankings.
Each tournament's finishing order produces a Plackett-Luce observation. Vendored
from `arena-hard-auto`'s BT notebook + extended per Plackett 1975. Monthly update
with history preserved.

### 10.3 Statistical libraries

- `scipy.stats.bootstrap` for BCa CIs
- `scikit-posthocs` for Holm-Bonferroni pairwise
- `ConfidenceIntervals` (Ferrer) as tested fallback
- `arviz` for optional Bayesian supplement (Beta-Binomial on win rates)

### 10.4 Leaderboard guardrails

- No rank shown if a model has <3 tournament appearances ("insufficient data")
- CI-overlap bucket shares rank with visual band + tooltip
- Every cell drills into the originating event log
- "Limitations" section on page: current N, current variance, current cost

## 11. Website integration — Django app (`portfolio/poker_bench/`)

### 11.1 Models (thin index; data lives in JSONL)

```python
class Tournament(models.Model):
    tournament_id = CharField(primary_key=True)
    results_path = CharField()
    manifest_sha256 = CharField()
    holdembench_version = CharField()
    started_at = DateTimeField()
    finished_at = DateTimeField(null=True)
    status = CharField()  # pending/running/complete/failed
    total_cost_usd = DecimalField()

class ModelAppearance(models.Model):
    tournament = ForeignKey(Tournament)
    seat = IntegerField()
    model_id = CharField()
    final_chip_delta = IntegerField()
    mbb_per_100 = FloatField()

class SeasonRanking(models.Model):
    season_id = CharField()
    model_id = CharField()
    plackett_luce_rating = FloatField()
    rating_ci_low = FloatField()
    rating_ci_high = FloatField()
    n_tournaments = IntegerField()
```

### 11.2 Views

- `LeaderboardView` — `/poker_bench/` — current season; multi-column sortable; CI-overlap bucketed
- `TournamentDetailView` — `/poker_bench/t/<tid>/` — scoreboard, cost breakdown, replay links
- `HandReplayView` — `/poker_bench/t/<tid>/h/<hid>/` — single-hand replay from event log
- `PlayView` — `/poker_bench/play/<tid>/` — human seat; auth-gated
- `RawDataView` — `/poker_bench/t/<tid>/raw/` — events.jsonl + manifest.json download
- `MethodologyView` — `/poker_bench/methodology/` — links to docs + preregistration

### 11.3 Frontend

- Vanilla JS throughout (matches `CLAUDE.md` convention in djangoproj).
- Leaderboard: vanilla sortable table (pattern matches existing list pages).
- Hand replayer: `static/poker_bench/js/replayer.js`, canvas-based, reads events.jsonl,
  iterates events, renders table + cards + pot + chat column. Controls: play/pause/step/speed.
- Charts: `uplot` (27 KB gzipped) unless a chart library is already loaded on the site.
- No Next.js, no React, no shadcn — obeys the existing site's consistency rule.

### 11.4 Deploy pipeline

- HoldEmBench repo emits `events.jsonl + manifest.json` per tournament as GitHub release assets.
- Existing `deploy.sh` / GitHub Actions fetches latest release into `portfolio/poker_bench/data/`.
- `python manage.py sync_holdembench_results` ingests into DB index (rows only, data stays on disk).
- No new infra beyond current DigitalOcean droplet.

### 11.5 YouTube/v1.5 readiness

Replayer is pure vanilla JS reading pure JSON. A separate YouTube-production UI
(bigger canvas, player avatars, chip-movement animations, commentary track overlay
points read from event log) is just another consumer of the same JSON — no engine
changes required.

## 12. Testing strategy — TDD discipline

Tests are written **before** the corresponding implementation at each layer.

### 12.1 Layers

1. **Unit** (`tests/unit/`) — pytest; pokerkit wrapper, TDA validator rules, chat budget math, event log serializer, tokenizer counting, cost calculator, Elo/PL math. <1s total.
2. **Property** (`tests/property/`) — Hypothesis on rules invariants: side-pot sum == committed chips; legal-action generator never returns illegal; TDA validator never accepts illegal; showdown winners non-empty summing to pot; chat budget never negative; canonical log never exceeds cache-breakpoint token budget. 200 examples dev, 1000 CI.
3. **Golden** (`tests/golden/`) — ~40 canonical hands (multiway all-ins, string-bet attempts, split pots, short-stack push-fold, etc.) encoded as scripted sequence → expected event log.
4. **Integration** (`tests/integration/`) — Random-vs-Random harness runs 10 hands; event log well-formed, manifest SHA matches, no exceptions. Zero API cost; runs in CI.
5. **Smoke** (`tests/smoke/`, `@pytest.mark.smoke`) — one hand vs Haiku/GPT-mini; manual trigger + pre-release gate.
6. **Contract tests** — every model adapter passes shared suite: output shape, timeout, retry-once, cost logging, cache-control injection.
7. **Fuzz** — random/adversarial text into chat content validator; must never crash.

### 12.2 Coverage gate

- `src/holdembench/engine/` and `src/holdembench/chat/` — ≥85% line coverage
- Overall — ≥70% line coverage
- Enforced in CI

### 12.3 Determinism test

Run the same tournament twice, same seed, stub agents → byte-identical event logs.
Any non-determinism fails CI.

## 13. Repo hygiene & CI

### 13.1 Day-one files

| File | Notes |
|---|---|
| `LICENSE` (Apache-2.0) | code |
| `LICENSES/DATA-CC-BY-4.0.txt` | data/prompts/scenarios |
| `CITATION.cff` | Zenodo auto-DOI on first tag |
| `CODE_OF_CONDUCT.md` | Contributor Covenant 2.1 |
| `CONTRIBUTING.md` | submit-a-model flow, local re-run instructions |
| `SECURITY.md` | private disclosure via GH advisories |
| `GOVERNANCE.md` | schema-change approval, version bumps |
| `PREREGISTRATION.md` | hypotheses, rotation policy, analysis plan — frozen + Zenodo-DOI'd before first frontier run |
| `pyproject.toml` + `uv.lock` | Python 3.12, fully pinned |
| `Dockerfile` | digest-pinned base image |
| `README.md` | quickstart, citation, reproducibility block |
| `.github/workflows/` | test, types, lint, docs, release, scorecard, integration |
| `.github/dependabot.yml` | weekly |

### 13.2 CI workflows

- `lint.yml` — `ruff check` + `ruff format --check`
- `types.yml` — `pyright` strict on `src/`, `mypy` on `tasks/`
- `test.yml` — pytest matrix over Python 3.12/3.13, ubuntu/macos; coverage gate
- `integration.yml` — stub-agent end-to-end, zero-API-cost
- `docs.yml` — mkdocs build + GitHub Pages deploy on tag
- `release.yml` — uv build → PyPI trusted-publisher (OIDC) → GitHub Release (events.jsonl assets) → Zenodo DOI via CITATION.cff
- `scorecard.yml` — OpenSSF Scorecard, weekly
- `prompt-fairness.yml` — 3 adapter paraphrases × 100 stub scenarios; rank change fails PR

### 13.3 Releases

Tag `v0.1.0` → automated PyPI + GitHub Release + Zenodo DOI. Each tournament config
gets a release tag (e.g. `t-weekly-2026-05-03`) whose assets include `events.jsonl`,
`manifest.json`, `config.yaml`.

## 14. Anti-gaming & reproducibility

- **Held-out private scenarios (Phase 2+).** ~20% of configurations reserved; never public.
- **Canary strings** — UUID embedded in every tournament's system prompt; rotated each tournament; enables post-hoc training-data contamination detection.
- **Rotation policy** — monthly new deal-pack seeds; monthly new chat scenario variants (v2+).
- **Submission gating** — model roster additions require PR with `MODEL_CARDS/<provider>-<model>.md`; maintainer runs verification.
- **Integrity** — manifest SHA-256 of events.jsonl in git; third-party re-verification welcome.
- **Judge isolation** — any LLM-judge-based analysis uses a model *not* in that tournament. Documented in `docs/judges.md`.
- **Prompt-fairness audit** — CI job runs 3 paraphrases × 100 scenarios; rank-change fails.
- **Full reproducibility recipe** in `docs/reproducibility.md` — `git clone && uv sync --frozen && docker build && uv run holdembench run --config <cfg> --seed <s>` produces byte-identical event log.

## 15. Phased roadmap

### Phase 0 — Engine + harness against stubs (~2 weeks, $0 API)

Deliverables:
- Repo bootstrap with full hygiene (LICENSE, CI, pyproject, Docker)
- Engine: `pokerkit` wrapper + `TDAValidator`
- Chat: `ChatProtocol` with budgets, probes, content validation
- Harness: Inspect-based runner + event-log writer
- Stub agents: Random, TightPassive, GTOApprox, CannedTalk
- Scoring: chip EV + mbb/100 + multi-way Plackett-Luce
- Tests: unit / property / golden / integration all passing
- Determinism test passing
- **Mid-review #1** after engine complete; **Mid-review #2** after harness complete

### Phase 1 — v1 pilot live (~2 weeks after Phase 0, ~$100/week)

Deliverables:
- Real-model adapters: Anthropic, OpenAI, Google, xAI, Moonshot, OpenRouter, Human
- Django app `poker_bench` scaffold: leaderboard, tournament detail, replay viewer
- `PREREGISTRATION.md` filed and Zenodo-DOI'd
- Canary strings implemented
- First public weekly tournament
- GitHub repo public + launch announcement
- **Mid-review #3** of full pipeline

### Phase 2 — v1.5 YouTube + rigor (~4 weeks)

Deliverables:
- YouTube-styled hand replayer (larger canvas, chip-movement animations, caption-cue points)
- Deception sub-score post-hoc analyzer + multi-judge rubric
- Bootstrap-CI + CI-overlap-aware leaderboard ranking
- Canary verification tooling
- Scale to 3 weekly games if budget supports
- **Mid-review #4** of analysis pipeline

### Phase 3 — v2 full format (months 2–4)

Deliverables:
- Monthly 10-game escalating-stakes tournament
- Duplicate-poker seat-rotation variance reduction (if budget)
- Private held-out scenarios
- Paper draft targeting NeurIPS D&B (Evaluations & Datasets) 2027
- **Final review:** parallel subagents (code-reviewer, silent-failure-hunter,
  type-design-analyzer, pr-test-analyzer, comment-analyzer) audit repo against
  entire plan; each gap → follow-up task

## 16. Binding implementation constraints

The following are **hard requirements** enforced by the implementation plan:

1. **TDD** — tests written before every unit of engine / chat / scoring code.
2. **Commit regularly** — every green test suite is a commit. No large uncommitted diffs.
3. **Mid-plan subagent reviews** — after engine, after harness, after full pipeline, after analysis pipeline.
4. **Final audit** — parallel subagents (code-reviewer + silent-failure-hunter + type-design-analyzer + pr-test-analyzer + comment-analyzer) after implementation, each reviewing against this spec; gaps become follow-up tasks.
5. **Subagent-driven development** — independent tasks executed in parallel via `superpowers:subagent-driven-development` skill for context isolation and throughput.
6. **Apache-2.0 code, CC-BY-4.0 data** — license headers present in all source files.
7. **Event-log-as-truth** — no implementation may bypass the event log for persistent state. All downstream consumers (scoring, website, replayer, YouTube UI) read only from the log.
8. **No Next.js / React / shadcn** in the Django app — vanilla JS only, matches existing site conventions.
9. **pydantic-validated schemas** on all model adapter outputs and event-log events.
10. **Determinism** — stub-agent tournaments byte-identical under same seed.

## 17. Open items for follow-up

- Pin exact model IDs (Opus 4.7 date, GPT-5.4 Pro date, Gemini 3.1 Pro date, Llama variant, Grok version, Kimi version, two more Chinese frontier slots) at first-tournament launch via Chatbot Arena + Artificial Analysis + LiveBench top lists as of that week.
- Final project name (replace HoldEmBench working label).
- OSF Registries preregistration template selection (full vs. lightweight).
- Precise thinking-token budget per model family — pin in PREREGISTRATION.md.
- Chinese frontier model #2 — candidates: Yi-5, GLM-5, Ernie-5, MiniMax-5, Baichuan-5, Hunyuan-5. User-referenced "fifth-iteration" model to be identified at launch.
- GTO reference implementation — vendor precomputed 9-max push-fold chart + postflop solver lookups; source TBD (possibilities: GTO Wizard public tables, OpenSpiel CFR export).
- Human-baseline protocol for paper — recruit N=5 competent human players for v2 subset runs.

## 18. References

- `pokerkit` — https://github.com/uoftcprg/pokerkit (MIT, IEEE ToG 2025)
- `inspect_ai` — https://github.com/UKGovernmentBEIS/inspect_ai (MIT, UK AISI)
- `arena-hard-auto` — https://github.com/lmarena/arena-hard-auto (Elo/BT reference)
- AIVAT — Burch et al., arXiv:1612.06915
- PokerBench — Zhuang et al., arXiv:2501.08328 (AAAI 2025)
- Among Us sandbox — Golechha & Garriga-Alonso, arXiv:2504.04072
- WOLF — deception production vs detection — arXiv:2512.09187 (NeurIPS 2025)
- Werewolf Arena — Google, arXiv:2407.13943
- HELM — Stanford CRFM — https://github.com/stanford-crfm/helm
- METR elicitation protocol — evaluations.metr.org/elicitation-protocol
- LiveBench rotation model — arXiv:2406.19314
- Apollo Research scheming evals — apolloresearch.ai/research/
- Confidence intervals lib (Ferrer) — github.com/luferrer/ConfidenceIntervals
- Plackett-Luce multi-way extension — Plackett 1975
