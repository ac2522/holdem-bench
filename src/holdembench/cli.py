"""CLI entry point — ``holdembench run --config CONFIG``."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import click
import yaml

from holdembench.agents.base import Agent
from holdembench.baselines import (
    CannedTalkAgent,
    GTOApproxAgent,
    RandomAgent,
    TightPassiveAgent,
)
from holdembench.credentials import (
    Credentials,
    CredentialsFileMissing,
    ProviderCredentials,
    load_credentials,
)
from holdembench.harness.runner import TournamentConfig, run_tournament

_STUB_REGISTRY: dict[str, type[Agent]] = {
    "stub:random": RandomAgent,
    "stub:tight_passive": TightPassiveAgent,
    "stub:gto_approx": GTOApproxAgent,
    "stub:canned_talk": CannedTalkAgent,
}


@click.group()
def cli() -> None:
    """HoldEmBench — poker benchmark runner."""


@cli.command()
@click.option("--config", "config_path", type=click.Path(exists=True), required=True)
@click.option("--results-dir", type=click.Path(), default="results")
@click.option("--seed", type=int, default=None, help="Override master_seed from config")
@click.option(
    "--deterministic-time/--real-time",
    default=True,
    help="Use deterministic time (default) or real wall-clock time",
)
@click.option(
    "--credentials",
    "credentials_path",
    type=click.Path(),
    default=None,
    help="Path to credentials.toml (default: ~/.holdembench/credentials.toml)",
)
def run(
    config_path: str,
    results_dir: str,
    seed: int | None,
    deterministic_time: bool,
    credentials_path: str | None,
) -> None:
    """Run a tournament from a YAML config file."""
    raw: dict[str, Any] = yaml.safe_load(Path(config_path).read_text())
    cfg = TournamentConfig(
        tournament_id=raw["tournament_id"],
        seats=raw["seats"],
        small_blind=raw["small_blind"],
        big_blind=raw["big_blind"],
        ante=raw["ante"],
        starting_stack=raw["starting_stack"],
        hand_cap=raw["hand_cap"],
        session_count=raw["session_count"],
        master_seed=seed if seed is not None else raw["master_seed"],
        results_dir=Path(results_dir),
        deterministic_time=deterministic_time,
        budget_ceilings_usd=raw.get("budget_ceilings_usd"),
    )
    creds = _load_credentials_if_available(credentials_path)
    agents = _build_agents(set(raw["seats"].values()), creds=creds)
    _wire_llm_contexts(cfg, agents)
    result = asyncio.run(run_tournament(cfg, agents))
    click.echo(f"Done — log: {result.log_path}")
    click.echo(f"Final chip totals: {result.final_chip_totals}")
    click.echo(f"Total cost: ${result.total_cost_usd:.4f}")
    for mid, stats in sorted(result.per_model_cost.items()):
        click.echo(
            f"  {mid}: ${stats['usd_total']:.4f} "
            f"(in {int(stats['input_tokens'])}tok, out {int(stats['output_tokens'])}tok, "
            f"cache_read {int(stats['cache_read_tokens'])}tok)"
        )


def _load_credentials_if_available(path: str | None) -> Credentials | None:
    try:
        return load_credentials(Path(path)) if path else load_credentials()
    except CredentialsFileMissing:
        return None


def _build_agents(model_ids: set[str], *, creds: Credentials | None) -> dict[str, Agent]:
    built: dict[str, Agent] = {}
    for mid in model_ids:
        if mid.startswith("stub:"):
            cls = _STUB_REGISTRY.get(mid)
            if cls is None:
                raise click.ClickException(f"unknown stub agent: {mid}")
            built[mid] = cls()
            continue
        built[mid] = _build_llm_agent(mid, creds=creds)
    return built


def _build_llm_agent(model_id: str, *, creds: Credentials | None) -> Agent:
    provider = model_id.split(":", 1)[0]
    if creds is None or not creds.has(provider):
        raise click.ClickException(
            f"no credentials for provider {provider!r}; add a [{provider}] section to "
            "~/.holdembench/credentials.toml (or pass --credentials)"
        )
    p = creds.get(provider)
    if provider == "anthropic":
        return _anthropic(model_id, p)
    if provider == "openai":
        return _openai(model_id, p)
    if provider == "google":
        return _google(model_id, p)
    if provider == "xai":
        return _xai(model_id, p)
    if provider == "moonshot":
        return _moonshot(model_id, p)
    if provider == "openrouter":
        return _openrouter(model_id, p)
    raise click.ClickException(f"unknown provider: {provider}")


def _anthropic(model_id: str, p: ProviderCredentials) -> Agent:
    from anthropic import AsyncAnthropic  # noqa: PLC0415

    from holdembench.agents.anthropic import AnthropicAgent  # noqa: PLC0415

    client = AsyncAnthropic(api_key=p.api_key)
    return AnthropicAgent(model_id=model_id, client=client)  # type: ignore[arg-type]


def _openai(model_id: str, p: ProviderCredentials) -> Agent:
    from openai import AsyncOpenAI  # noqa: PLC0415

    from holdembench.agents.openai import OpenAIAgent  # noqa: PLC0415

    client = AsyncOpenAI(api_key=p.api_key, base_url=p.base_url) if p.base_url else AsyncOpenAI(
        api_key=p.api_key
    )
    return OpenAIAgent(model_id=model_id, client=client)  # type: ignore[arg-type]


def _google(model_id: str, p: ProviderCredentials) -> Agent:
    from google import genai  # noqa: PLC0415

    from holdembench.agents.google import GoogleAgent  # noqa: PLC0415

    client = genai.Client(api_key=p.api_key).aio.models
    return GoogleAgent(model_id=model_id, client=client)  # type: ignore[arg-type]


def _xai(model_id: str, p: ProviderCredentials) -> Agent:
    from openai import AsyncOpenAI  # noqa: PLC0415

    from holdembench.agents.xai import DEFAULT_XAI_BASE_URL, XAIAgent  # noqa: PLC0415

    client = AsyncOpenAI(api_key=p.api_key, base_url=p.base_url or DEFAULT_XAI_BASE_URL)
    return XAIAgent(model_id=model_id, client=client)  # type: ignore[arg-type]


def _moonshot(model_id: str, p: ProviderCredentials) -> Agent:
    from openai import AsyncOpenAI  # noqa: PLC0415

    from holdembench.agents.moonshot import (  # noqa: PLC0415
        DEFAULT_MOONSHOT_BASE_URL,
        MoonshotAgent,
    )

    client = AsyncOpenAI(api_key=p.api_key, base_url=p.base_url or DEFAULT_MOONSHOT_BASE_URL)
    return MoonshotAgent(model_id=model_id, client=client)  # type: ignore[arg-type]


def _openrouter(model_id: str, p: ProviderCredentials) -> Agent:
    from openai import AsyncOpenAI  # noqa: PLC0415

    from holdembench.agents.openrouter import (  # noqa: PLC0415
        DEFAULT_OPENROUTER_BASE_URL,
        OpenRouterAgent,
    )

    client = AsyncOpenAI(api_key=p.api_key, base_url=p.base_url or DEFAULT_OPENROUTER_BASE_URL)
    return OpenRouterAgent(model_id=model_id, client=client)  # type: ignore[arg-type]


def _wire_llm_contexts(cfg: TournamentConfig, agents: dict[str, Agent]) -> None:
    """Populate each LLM adapter with a :class:`TournamentContext` + :class:`SessionContext`.

    Current limitation (P1.1-B): when multiple seats share a single model_id
    the same agent instance serves all of them and the set_context call will
    only reflect the first seat.  See docs/reviews/follow-ups.md.
    """
    from holdembench.agents.prompt import SessionContext, TournamentContext  # noqa: PLC0415

    seats_by_model: dict[str, str] = {}
    for seat, model_id in cfg.seats.items():
        seats_by_model.setdefault(model_id, seat)
    for model_id, first_seat in seats_by_model.items():
        agent = agents[model_id]
        if not hasattr(agent, "set_context"):
            continue
        tournament = TournamentContext(
            tournament_id=cfg.tournament_id,
            seat=first_seat,
            seat_count=len(cfg.seats),
        )
        session = SessionContext(
            session_id=1,
            small_blind=cfg.small_blind,
            big_blind=cfg.big_blind,
            ante=cfg.ante,
            starting_stack_bb=max(1, cfg.starting_stack // cfg.big_blind),
            orbit_budget_tokens=400,
        )
        agent.set_context(tournament=tournament, session=session)  # type: ignore[attr-defined]
