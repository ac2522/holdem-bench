# Phase 0 Follow-ups

Issues deferred from the Phase 0 review audit (see
`2026-04-22-phase-0-review.md`). Each item should become a GitHub issue
tagged `phase-0-followup` when the repo is made public.

## Deferred P1s (tranche postponed from v0.1.0)

### P1-A — Narrow `_apply_raw_to_table` `ValueError` catch
**File:** `src/holdembench/harness/runner.py:140-146`
**Why deferred:** The broad catch works for the known pokerkit "can't fold without aggression" edge case. All tests pass. The risk is that future pokerkit versions could throw a different `ValueError` from the same call site and be silently coerced to a check.
**Fix:** Narrow to the specific pokerkit exception type (or inspect the error message). Emit a log event on the downgrade.
**✅ Resolved in Phase 1.1 Task 17** — all three `apply_fold` catches in the runner now require the message `"no reason for this player to fold"`; any other ValueError propagates.  New test `test_unexpected_fold_error_message_is_reraised` pins the behaviour.

### P1-B — Emit `Showdown` events + wire EV adjustment
**File:** `src/holdembench/harness/runner.py`, `src/holdembench/engine/ev_adjustment.py`
**Why deferred:** The Phase 0 plan explicitly allowed `Showdown` emission to slip to Phase 1. Stub baselines rarely reach showdown, so tests pass without it. `ev_adjustment.py` is fully implemented but never called by the runner.
**Fix:** At each `hand_end`, detect whether an all-in occurred pre-river with ≥2 players; if so, compute Monte Carlo equity, produce an EV-adjusted `stack_deltas`, and emit `Showdown(all_in_ev_adjusted=True)` with the actual deltas in `stack_deltas_actual`.

### P1-C — Consolidate duplicate literal aliases
**Files:** `events/schema.py`, `engine/validator.py`, `agents/base.py`, `chat/protocol.py`
**Why deferred:** Refactor, no functional bug today. Current tests pass because the aliases happen to match.
**Fix:** Create `src/holdembench/types.py` with `ActionName`, `Street`, `ActionKind`, `ChatKind` as PEP 695 type aliases. All four modules import from there.
**✅ Resolved in Phase 1.0 Task 1** — canonical `src/holdembench/types.py` now exports `ActionName`, `ActionKind`, `ChatKind`, `Street`; schema / validator / chat protocol / agent base all import from there.

### P1-D — Fix `_Base` leakage; `parse_event` return type
**File:** `events/schema.py`
**Why deferred:** Functional; the narrowing loss is cosmetic to external consumers.
**Fix:** Remove `_Base` from `__all__`. Change `parse_event` signature to return `Event` (the TypeAlias). Rely on `TypeAdapter(Event)` for dispatch.
**✅ Resolved in Phase 1.0 Task 1** — `_Base` removed from `__all__`; `parse_event` now returns `Event` (with a single `# type: ignore[return-value]` where the registry-dict lookup defeats narrowing).

### P1-E — Merge `RawDecision` / `ActionResponse` invariant validation
**Files:** `engine/validator.py`, `events/schema.py`
**Why deferred:** The duplication is not causing drift yet.
**Fix:** Make `RawDecision` a frozen pydantic `BaseModel` with the same `kind→action`/`kind→message` validator as `ActionResponse`. Construct `ActionResponse` from a `RawDecision` field-for-field where possible.
**✅ Resolved in Phase 1.0 Task 1** — `RawDecision` is now a frozen pydantic `BaseModel` with the same `@model_validator` invariant as `ActionResponse`; invalid actions are now rejected at construction (pydantic Literal + model_validator), making the invariant stronger than before.

### P1-F — Missing property-test invariants (spec §12.1 Layer 2)
**File:** `tests/property/`
**Why deferred:** 5 property tests already exist; these are 3 additional invariants that weren't wired.
**Fix:** Add:
- `test_side_pot_sum_equals_committed_chips` — after any action sequence, `sum(pots) == total committed chips`.
- `test_showdown_winners_non_empty_and_sum_to_pot` — on every showdown, winners list non-empty; chip distribution sums to pot.
- `test_canonical_action_log_within_token_budget` — serialized action log for any hand fits in the cache-breakpoint token budget.

### P1-G — All 15 event types constructed in tests
**Files:** `tests/**`
**Why deferred:** 10 of 15 are already constructed; the uncovered ones require Showdown/EV-adjustment wiring (P1-B) and/or the content of specific events.
**Fix:** After P1-B lands, add unit tests that directly construct `Deal`, `CommunityDeal`, `ActionRequest`, `Showdown`, `ProbeResponseRequest`, `BudgetCircuitBreak`. Add an integration assertion that counts event types in a full tournament run.

### P1-H — Event-shape-and-order assertion in integration test
**File:** `tests/integration/test_runner.py`
**Why deferred:** First-and-last-event assertion catches gross breakage.
**Fix:** Add a regex-like state machine that validates events follow the grammar: `tournament_start (session_start (hand_start deal* (action_request action_response)* community_deal* showdown? hand_end)+ session_end)+ tournament_end`. Fail the test on any ordering violation.

## P2s (collected from reviewers; prioritize for v0.2.0)

1. `cards_hash` should commit the full deal sequence (not just hole cards) — `harness/runner.py:211`.
2. `AutoFold` event → unconditional `apply_fold()` can desync log from state if pokerkit raises — `harness/runner.py:254-255`. Note: runner currently falls back to `check_or_call` when fold is illegal, but the `AutoFold` event is already emitted.
3. Probe budget pre-check (`remaining < per_action_cap`) missing — `chat/protocol.py:82-90`.
4. `_git_sha` bare `except Exception` — `harness/runner.py:112-116`.
5. `TightPassiveAgent` fallback to `ctx.legal[0]` without asserting invariant — `baselines/tight_passive.py:64-65`.
6. Baseline agents don't validate `ctx.hole` has 2 cards or `ctx.legal` non-empty.
7. `Showdown.winners` should be a nested `Winner` pydantic model, not `list[dict[str, str|int]]`.
8. `Manifest` dataclass should be pydantic for round-trip validation.
9. `TableConfig.__post_init__` missing: `small_blind > 0`, `ante >= 0`, `starting_stacks[i] > 0`.
10. `DecisionContext.legal` should be `tuple[...]` for immutability parity.
11. `Pricing.cost_usd` should reject negative token counts.
12. Remove transitional "Phase 0 placeholder / refined once..." comments.
13. Fix misleading "TOURNAMENT mode" reference in `harness/runner.py:137` docstring.
14. Replace `table._state` (SLF001) with a public `Table.final_stacks()` accessor.
15. `TightPassive._compute_raise` returns magic `60`; parameterize on BB.
16. Hypothesis fuzz test on `validate_content`.
17. Determinism test negative control (different seeds produce different logs).
18. Golden hand 02 should actually run to showdown, not just preflop.
19. Shared contract test suite across the four baselines.
20. `replay()` `JSONDecodeError` wrapping with file:line context.
21. Unify `_ctx()` helper across baseline tests.
22. Generate `_EVENT_TYPES` dispatch from `typing.get_args(Event)` to avoid 4-edit-point drift risk.

## P3s (cosmetic / doc rot)

1. Remove redundant step-by-step comments in `tight_passive.py` (keep only the invariant guard).
2. Drop "Postflop stub: check/call only" section header in `gto_approx.py`.
3. Remove "Documented in docs/prompting.md" dangling link in `tokenizer.py`.
4. Strip parroting docstring in `deck.py`.
5. Trim `ChatProtocol` Lifecycle arrow chain.
6. `cli.py` `raw[...]` KeyError should wrap with file/field context.
7. Tighten `test_raise_below_min_always_rejected` to `max_value=39`.
8. Convert `test_hand_end_roundtrip` to use `tmp_path` fixture.
9. `test_attached_message_decrements_budget` should assert exact budget value.
10. `test_eventlog_sha256_hex_matches_file` coverage for probe_reply-without-message validator path.
