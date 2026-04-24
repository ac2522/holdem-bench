"""Agent protocol + supporting types for the adapter layer."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from holdembench.engine.validator import RawDecision
from holdembench.types import ActionName, Street


@dataclass(frozen=True)
class Pricing:
    """Per-million-token USD prices for a given model."""

    input_per_mtok: float
    output_per_mtok: float
    cache_read_per_mtok: float = 0.0
    cache_write_per_mtok: float = 0.0

    def cost_usd(
        self,
        *,
        input_tokens: int,
        output_tokens: int,
        cache_read_tokens: int = 0,
        cache_write_tokens: int = 0,
    ) -> float:
        return (
            input_tokens * self.input_per_mtok / 1_000_000
            + output_tokens * self.output_per_mtok / 1_000_000
            + cache_read_tokens * self.cache_read_per_mtok / 1_000_000
            + cache_write_tokens * self.cache_write_per_mtok / 1_000_000
        )


@dataclass(frozen=True)
class DecisionContext:
    """Everything an agent needs to make one decision."""

    seat: str
    hand_id: str
    street: Street
    legal: tuple[ActionName, ...]
    stacks: dict[str, int]
    board: tuple[str, ...]
    hole: tuple[str, ...]
    budget_remaining: int
    is_probe_reply: bool
    deadline_s: float
    chat_log: tuple[str, ...] = field(default_factory=tuple)


class Agent(Protocol):
    """A black-box poker-playing agent."""

    model_id: str
    pricing: Pricing

    async def decide(self, ctx: DecisionContext) -> RawDecision: ...
