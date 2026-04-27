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
    MissingCredentialError,
    ProviderCredentials,
    get_provider_credentials,
    load_dotenv_from_repo,
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
    "--env-file",
    "env_file",
    type=click.Path(),
    default=None,
    help="Path to a .env file (default: <repo-root>/.env)",
)
def run(
    config_path: str,
    results_dir: str,
    seed: int | None,
    deterministic_time: bool,
    env_file: str | None,
) -> None:
    """Run a tournament from a YAML config file."""
    if env_file is not None:
        from dotenv import load_dotenv  # noqa: PLC0415

        load_dotenv(Path(env_file), override=False)
    else:
        load_dotenv_from_repo()

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
        reasoning_effort=raw.get("reasoning_effort"),
    )
    agents = _build_agents(
        set(raw["seats"].values()),
        reasoning_effort=cfg.reasoning_effort,
    )
    # Runner refreshes TournamentContext + SessionContext at each session
    # boundary (see runner._refresh_adapter_contexts) so session_id is
    # correctly threaded for multi-session tournaments.
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


def _build_agents(
    model_ids: set[str], *, reasoning_effort: str | None = None
) -> dict[str, Agent]:
    built: dict[str, Agent] = {}
    for mid in model_ids:
        if mid.startswith("stub:"):
            cls = _STUB_REGISTRY.get(mid)
            if cls is None:
                raise click.ClickException(f"unknown stub agent: {mid}")
            built[mid] = cls()
            continue
        built[mid] = _build_llm_agent(mid, reasoning_effort=reasoning_effort)
    return built


def _build_llm_agent(
    model_id: str, *, reasoning_effort: str | None = None
) -> Agent:
    provider = model_id.split(":", 1)[0]
    try:
        p = get_provider_credentials(provider)
    except MissingCredentialError as e:
        raise click.ClickException(str(e)) from e
    if provider == "anthropic":
        return _anthropic(model_id, p)
    if provider == "openai":
        return _openai(model_id, p, reasoning_effort=reasoning_effort)
    if provider == "google":
        return _google(model_id, p)
    if provider == "xai":
        return _xai(model_id, p, reasoning_effort=reasoning_effort)
    if provider == "moonshot":
        return _moonshot(model_id, p, reasoning_effort=reasoning_effort)
    if provider == "openrouter":
        return _openrouter(model_id, p, reasoning_effort=reasoning_effort)
    raise click.ClickException(f"unknown provider: {provider}")


def _anthropic(model_id: str, p: ProviderCredentials) -> Agent:
    from anthropic import AsyncAnthropic  # noqa: PLC0415

    from holdembench.agents.anthropic import AnthropicAgent  # noqa: PLC0415

    client = AsyncAnthropic(api_key=p.api_key)
    return AnthropicAgent(model_id=model_id, client=client)  # type: ignore[arg-type]


def _openai(
    model_id: str, p: ProviderCredentials, *, reasoning_effort: str | None = None
) -> Agent:
    from openai import AsyncOpenAI  # noqa: PLC0415

    from holdembench.agents.openai import OpenAIAgent  # noqa: PLC0415

    client = AsyncOpenAI(api_key=p.api_key, base_url=p.base_url) if p.base_url else AsyncOpenAI(
        api_key=p.api_key
    )
    return OpenAIAgent(  # type: ignore[arg-type]
        model_id=model_id, client=client, reasoning_effort=reasoning_effort
    )


def _google(model_id: str, p: ProviderCredentials) -> Agent:
    from google import genai  # noqa: PLC0415

    from holdembench.agents.google import GoogleAgent  # noqa: PLC0415

    client = genai.Client(api_key=p.api_key).aio.models
    return GoogleAgent(model_id=model_id, client=client)  # type: ignore[arg-type]


def _xai(
    model_id: str, p: ProviderCredentials, *, reasoning_effort: str | None = None
) -> Agent:
    from openai import AsyncOpenAI  # noqa: PLC0415

    from holdembench.agents.xai import XAIAgent  # noqa: PLC0415

    client = AsyncOpenAI(api_key=p.api_key, base_url=p.base_url)
    return XAIAgent(  # type: ignore[arg-type]
        model_id=model_id, client=client, reasoning_effort=reasoning_effort
    )


def _moonshot(
    model_id: str, p: ProviderCredentials, *, reasoning_effort: str | None = None
) -> Agent:
    from openai import AsyncOpenAI  # noqa: PLC0415

    from holdembench.agents.moonshot import MoonshotAgent  # noqa: PLC0415

    client = AsyncOpenAI(api_key=p.api_key, base_url=p.base_url)
    return MoonshotAgent(  # type: ignore[arg-type]
        model_id=model_id, client=client, reasoning_effort=reasoning_effort
    )


def _openrouter(
    model_id: str, p: ProviderCredentials, *, reasoning_effort: str | None = None
) -> Agent:
    from openai import AsyncOpenAI  # noqa: PLC0415

    from holdembench.agents.openrouter import OpenRouterAgent  # noqa: PLC0415

    client = AsyncOpenAI(api_key=p.api_key, base_url=p.base_url)
    return OpenRouterAgent(  # type: ignore[arg-type]
        model_id=model_id, client=client, reasoning_effort=reasoning_effort
    )
