"""CLI entry point — `holdembench run --config CONFIG`."""
from __future__ import annotations

import asyncio
from pathlib import Path

import click
import yaml

from holdembench.agents.base import Agent
from holdembench.baselines import (
    CannedTalkAgent,
    GTOApproxAgent,
    RandomAgent,
    TightPassiveAgent,
)
from holdembench.harness.runner import TournamentConfig, run_tournament

_AGENT_REGISTRY = {
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
def run(
    config_path: str, results_dir: str, seed: int | None, deterministic_time: bool
) -> None:
    """Run a tournament from a YAML config file."""
    raw = yaml.safe_load(Path(config_path).read_text())
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
    )
    agents = _build_agents(set(raw["seats"].values()))
    result = asyncio.run(run_tournament(cfg, agents))
    click.echo(f"Done — log: {result.log_path}")
    click.echo(f"Final chip totals: {result.final_chip_totals}")


def _build_agents(model_ids: set[str]) -> dict[str, Agent]:
    built: dict[str, Agent] = {}
    for mid in model_ids:
        cls = _AGENT_REGISTRY.get(mid)
        if cls is None:
            raise click.ClickException(f"unknown agent: {mid}")
        built[mid] = cls()
    return built
