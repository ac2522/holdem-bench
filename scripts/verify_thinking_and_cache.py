"""Verify thinking-token capture and prompt-cache behaviour against real models.

Sends the same large prompt twice to each candidate model:
  - call 1 (cold): expect cache_read_tokens == 0
  - call 2 (warm): expect cache_read_tokens > 0 if the provider supports
    automatic prompt caching for this model

Each call is sent twice — once with reasoning_effort=None and once with
reasoning_effort="medium" — so we can compare:
  - thinking-token counts in usage
  - cost contribution of thinking tokens
  - whether cache hits land regardless of thinking on/off

This script costs real money — typically $0.01-0.10 per model.  It uses
the OpenRouter unified API via the existing OpenAIAgent path.

Run:
    uv run python scripts/verify_thinking_and_cache.py
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass

from openai import AsyncOpenAI

from holdembench.agents.openai import OpenAIAgent
from holdembench.agents.pricing_sheet import PRICING_SHEET, lookup_pricing
from holdembench.agents.prompt import SessionContext, TournamentContext
from holdembench.agents.base import DecisionContext
from holdembench.credentials import get_provider_credentials, load_dotenv_from_repo

# Cheap thinking-capable representative for each provider.  All routed
# through OpenRouter so a single key covers them.
_CANDIDATE_MODELS: tuple[str, ...] = (
    "openrouter:anthropic/claude-haiku-4.5",
    "openrouter:google/gemini-2.5-flash",
    "openrouter:openai/gpt-4o-mini",  # no thinking — sanity baseline
    "openrouter:deepseek/deepseek-chat-v3.1",
    "openrouter:x-ai/grok-3-mini",
)


@dataclass
class _Trial:
    model_id: str
    reasoning_effort: str | None
    call_no: int  # 1 = cold, 2 = warm
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int
    thinking_tokens: int
    cost_usd: float
    text_preview: str


def _build_decision_context() -> DecisionContext:
    return DecisionContext(
        seat="Seat1",
        hand_id="probe-h001",
        street="preflop",
        legal=("fold", "call", "raise"),
        stacks={"Seat1": 1000, "Seat2": 1000, "Seat3": 1000},
        board=(),
        hole=("As", "Kd"),
        budget_remaining=400,
        is_probe_reply=False,
        deadline_s=60.0,
        min_raise_to=40,
        # Pad the action_log to make the prompt long enough that prompt
        # caching has something useful to cache on the warm call.  ~5KB
        # of repeated history.
        canonical_action_log="\n".join(
            f"s1h{i:03d} preflop Seat{(i % 3) + 1} {'fold' if i % 4 else 'raise 40'}"
            for i in range(200)
        ),
    )


async def _run_one(
    model_id: str, *, reasoning_effort: str | None, call_no: int
) -> _Trial:
    creds = get_provider_credentials("openrouter")
    client = AsyncOpenAI(api_key=creds.api_key, base_url=creds.base_url)
    agent = OpenAIAgent(
        model_id=model_id, client=client, reasoning_effort=reasoning_effort  # type: ignore[arg-type]
    )
    agent.set_context(
        tournament=TournamentContext(tournament_id="probe", seat="Seat1", seat_count=3),
        session=SessionContext(
            session_id=1,
            small_blind=10,
            big_blind=20,
            ante=0,
            starting_stack_bb=50,
            orbit_budget_tokens=400,
        ),
    )
    raw = await agent.decide(_build_decision_context())
    u = agent.last_usage
    assert u is not None
    text = ""
    if raw.kind == "action" and raw.action:
        text = f"{raw.action}{f' {raw.amount}' if raw.amount else ''}"
    return _Trial(
        model_id=model_id,
        reasoning_effort=reasoning_effort,
        call_no=call_no,
        input_tokens=u.input_tokens,
        output_tokens=u.output_tokens,
        cache_read_tokens=u.cache_read_tokens,
        thinking_tokens=u.thinking_tokens,
        cost_usd=agent.last_cost_usd,
        text_preview=text,
    )


async def _trial_pair(model_id: str, *, reasoning_effort: str | None) -> list[_Trial]:
    """Send two consecutive calls; second should warm the cache."""
    cold = await _run_one(model_id, reasoning_effort=reasoning_effort, call_no=1)
    warm = await _run_one(model_id, reasoning_effort=reasoning_effort, call_no=2)
    return [cold, warm]


async def _run_model(model_id: str) -> list[_Trial]:
    if model_id not in PRICING_SHEET:
        print(f"  SKIP {model_id} — no pricing entry, would crash on construct")
        return []
    p = lookup_pricing(model_id)
    print(f"  {model_id}: ${p.input_per_mtok}/MTok in, ${p.output_per_mtok}/MTok out")
    trials: list[_Trial] = []
    print("    no-thinking pair...")
    trials += await _trial_pair(model_id, reasoning_effort=None)
    print("    thinking=medium pair...")
    trials += await _trial_pair(model_id, reasoning_effort="medium")
    return trials


def _format_row(t: _Trial) -> str:
    eff = t.reasoning_effort or "off"
    return (
        f"  call {t.call_no} eff={eff:6s} | "
        f"in={t.input_tokens:5d} out={t.output_tokens:4d} "
        f"think={t.thinking_tokens:4d} cache_read={t.cache_read_tokens:5d} | "
        f"${t.cost_usd:.6f} | {t.text_preview}"
    )


async def _main() -> None:
    load_dotenv_from_repo()
    if not os.environ.get("OPENROUTER_API_KEY", "").strip():
        msg = "OPENROUTER_API_KEY not set in env or .env"
        raise RuntimeError(msg)
    print("Verifying thinking-token capture + prompt cache for cheap thinking models")
    print("=" * 78)
    grand_total = 0.0
    for model_id in _CANDIDATE_MODELS:
        print(f"\n{model_id}")
        try:
            trials = await _run_model(model_id)
        except Exception as e:  # noqa: BLE001  — diagnostic script
            print(f"  ERROR: {type(e).__name__}: {e}")
            continue
        for t in trials:
            print(_format_row(t))
            grand_total += t.cost_usd
        # Sanity checks per model
        if trials:
            no_think = [t for t in trials if t.reasoning_effort is None]
            think = [t for t in trials if t.reasoning_effort is not None]
            think_total = sum(t.thinking_tokens for t in think)
            print(
                f"    -> thinking_tokens with eff=None: "
                f"{sum(t.thinking_tokens for t in no_think)} "
                f"(should be 0 unless the model thinks by default)"
            )
            print(f"    -> thinking_tokens with eff=medium: {think_total}")
            warm_no_think = [t for t in no_think if t.call_no == 2]
            warm_think = [t for t in think if t.call_no == 2]
            for t in warm_no_think + warm_think:
                marker = "✓" if t.cache_read_tokens > 0 else "✗"
                print(
                    f"    {marker} warm call (eff={t.reasoning_effort or 'off'}) "
                    f"cache_read={t.cache_read_tokens} "
                    f"({'cache hit' if t.cache_read_tokens > 0 else 'NO CACHE HIT'})"
                )
    print("\n" + "=" * 78)
    print(f"TOTAL SPEND: ${grand_total:.6f}")


if __name__ == "__main__":
    asyncio.run(_main())
