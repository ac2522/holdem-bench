# Phase 0 Review Findings

**Date:** 2026-04-22
**HEAD at review:** `06ad8a5`
**Test status at review:** 107 pass + 8 TODO-skipped, ruff + pyright clean.

Five parallel subagents reviewed Phase 0 against the design spec and implementation plan.
Findings are consolidated below with [P1] must-fix, [P2] should-fix, [P3] nice-to-have tags.

## [P1] Must-fix — behavioral / correctness gaps

| # | Area | Finding | File(s) |
|---|---|---|---|
| 1 | Runner | `chat.spend()` / `validate_content()` are never called from the harness. Chat budget is declared in `HandStart` but never debited. Message token caps and injection filtering are inoperative during a live run. | `harness/runner.py` |
| 2 | Runner | `mark_folded()` is never called when a seat folds. Folded seats can keep chatting for the rest of the hand (spec §6.3, §7.4 violated). | `harness/runner.py` |
| 3 | Runner | `action_request` events are never emitted. Log consumers cannot reconstruct who was asked and what legal actions were. Breaks replay-purity invariant (spec §5.3 #2). | `harness/runner.py` |
| 4 | Runner | Retry-once policy (spec §6.4) not implemented. Invalid action → immediate auto-fold, no retry. The `ValidatorRejection` event is emitted with `retry_allowed=False` despite spec stating retry should be allowed. | `harness/runner.py` |
| 5 | Runner | `_apply_raw_to_table` silently catches all `ValueError` from `apply_fold` and downgrades to `check_or_call`. Too broad — could mask seat-index desync or real pokerkit bugs. No diagnostic log event written. | `harness/runner.py:140-146` |
| 6 | Runner | `Showdown` events never emitted. EV adjustment never triggered. `hand_end.stack_deltas` always reflects actual runout — EV-adjustment promise broken. | `harness/runner.py` |
| 7 | Validator | TDA Rule 40 min-raise formula is wrong: `current_bet * 2` doesn't equal "current_bet + last_raise_size". Only coincidentally correct on fresh-street opens. | `engine/validator.py:57` |
| 8 | Types | Duplicate/divergent `ActionName`/`ActionKind`/`Street`/`ChatKind` aliases in four modules (schema.py, validator.py, agents/base.py, chat/protocol.py) invite drift. | multiple |
| 9 | Types | `_Base` leaks into public API via `__all__` and is the return type of `parse_event`. Callers lose discriminated-union narrowing. | `events/schema.py` |
| 10 | Types | `RawDecision` (validator.py) and `ActionResponse` (schema.py) validate the same invariants (kind→action, kind→message) separately, so representations can drift. | `engine/validator.py`, `events/schema.py` |
| 11 | Types | `TournamentConfig` has no `__post_init__` validation. Empty seats dict, malformed "Seat1" keys, zero hand_cap, negative seeds all pass silently. | `harness/runner.py:61-78` |
| 12 | Tests | 3 spec §12.1 Layer-2 property invariants are absent: side-pot sum, showdown non-empty, canonical-log token-count ≤ cache budget. | `tests/property/` |
| 13 | Tests | 5 of 15 event types never instantiated anywhere: `Deal`, `CommunityDeal`, `ActionRequest`, `Showdown`, `ProbeResponseRequest`, `BudgetCircuitBreak`. Runner coverage at 34%. | `tests/` |
| 14 | Tests | No test asserts the runner emits every required event type in valid order (TournamentStart→SessionStart→...→TournamentEnd with proper nesting). | `tests/integration/test_runner.py` |

## [P2] Should-fix

- `cards_hash` (spec §5.3 #4) commits only hole cards, not the full deal sequence. Post-hoc board substitution undetectable.
- `AutoFold` emitted then unconditional `table.apply_fold()` → log/state desync risk if apply raises afterwards.
- Probe budget pre-check (`< 80 remaining`) missing; spec §7.3 wants it.
- `_git_sha` bare `except Exception` silences all errors; returns `"unknown"` with no warning.
- `TightPassiveAgent` fallback to `ctx.legal[0]` without logging when invariant violated.
- Baseline agents don't validate `ctx.hole` has 2 cards / `ctx.legal` non-empty.
- `Showdown.winners` typed as `list[dict[str, str|int]]` — should be a nested `Winner` model.
- `ValidatorRejection.original_response: dict` is an escape hatch; acceptable but loose.
- `Manifest` dataclass is serialized to JSON but has no pydantic round-trip validator on read.
- `TableConfig.__post_init__` missing: `small_blind > 0`, `ante >= 0`, `starting_stacks[i] > 0`.
- `DecisionContext.legal: list[...]` should be `tuple[...]` for immutability parity with `board`/`hole`.
- `Pricing.cost_usd` allows negative token counts silently.
- Transitional "Phase 0 placeholder / refined once..." comments violate project rules (comments shouldn't reference current task/phase).
- Misleading "TOURNAMENT mode" reference in runner docstring — no such mode configured.
- `SLF001` noqa documents the silencer, not the WHY (`table._state` access).
- `TightPassive._compute_raise` returns magic `60` assuming BB=20.
- No Hypothesis fuzz test on `validate_content` — spec §12.1 point 6 wants it.
- Determinism test lacks a negative control (same-seed-two-runs ≠ different-seed-two-runs).
- Golden hand 02 ("check_around") only exercises preflop state — never runs to showdown.
- No shared contract test suite across the four baselines.
- `replay()` has no `JSONDecodeError` wrapping for file:line context on corrupted logs.
- `TightPassive` / `GTOApprox` / `CannedTalk` re-implement their own `_ctx()` helper.
- Adding a new event type requires edits in 4 places (class, `__all__`, `Event` union, `_EVENT_TYPES`); generate dispatch from `typing.get_args(Event)` instead.

## [P3] Nice-to-have

- Redundant step-by-step comments in `tight_passive.py` ("Check for free play", "Premium hands: raise if possible", etc.).
- "Postflop stub: check/call only" section-header comment in `gto_approx.py`.
- "Documented in docs/prompting.md" dangling link in `tokenizer.py`.
- `deck.py` docstring parrots the signature.
- `chat/protocol.py` "Lifecycle" arrow chain of limited value.
- `cli.py` `raw[...]` KeyError bubbles up without file/field context.
- Coverage for `schema.py:_check_kind_shape` probe_reply-without-message path.
- `test_attached_message_decrements_budget` asserts `< BUDGET` not exact value.

## Positive findings

- `engine/table.py` module docstring is the gold-standard for pokerkit adaptation documentation.
- `chat/content.py` policy note (card claims allowed but logged) is the canonical good WHY comment.
- `scoring/multi_way_elo.py` cites the Plackett-Luce paper cleanly.
- `_canary_uuid`, `_anon_salt` have zero comments because the names carry meaning — philosophy working.
- Event schema uses `ConfigDict(extra="forbid", frozen=True)` uniformly.
- `test_byte_identical_on_reseed` exercises the full CLI via subprocess — a real determinism test.
- `test_invalid_action_names_always_rejected` uses `st.text()` — genuine adversarial input.
- Dockerfile `# TODO: pin digest before v0.1.0 release` is the canonical good TODO (concrete trigger + action).

## Fix plan

Before tagging v0.1.0, fix all 14 P1 findings. Defer P2/P3 to follow-up issues. The P1 fixes are grouped into three tranches:

- **Tranche A (runner + validator correctness):** findings 1-7. Biggest chunk. Requires runner surgery to wire chat protocol, emit missing events, implement retry-once, narrow exception catch, emit `Showdown` + EV adjustment, fix min-raise arithmetic.
- **Tranche B (type consolidation):** findings 8-11. Single source of truth for literal aliases; `_Base` → `Event`; unified `RawDecision`/`ActionResponse` invariants; `TournamentConfig` validation.
- **Tranche C (test expansion):** findings 12-14. Missing property invariants, event-shape assertion, all 15 event types constructed.

Each tranche: fix tests first (reproduce the bug), then the fix, then verify all prior tests still pass, then commit.
