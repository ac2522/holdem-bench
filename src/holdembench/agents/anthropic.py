"""AnthropicAgent — Claude via the ``anthropic`` async SDK.

Cache discipline (spec §8.4):
  * ``cache_control={"type": "ephemeral"}`` on both system blocks
  * User block (session log + volatile) re-sent each call
Native optional ``thinking`` extended-reasoning support (5000-token budget).
"""

from __future__ import annotations

from typing import Any, Protocol

from holdembench.agents.base import DecisionContext
from holdembench.agents.base_adapter import BaseAdapter, ProviderCall, Usage
from holdembench.agents.output_schema import AgentOutputParseError

EXTENDED_THINKING_BUDGET_TOKENS = 5000


class _AnthropicMessagesProto(Protocol):
    async def create(self, **kwargs: Any) -> Any: ...


class AnthropicClientProtocol(Protocol):
    @property
    def messages(self) -> _AnthropicMessagesProto: ...


class AnthropicAgent(BaseAdapter):
    """Claude adapter with explicit cache_control on system blocks."""

    def __init__(
        self,
        *,
        model_id: str,
        client: AnthropicClientProtocol,
        enable_thinking: bool = False,
    ) -> None:
        super().__init__(model_id=model_id, client=client)
        self._enable_thinking = enable_thinking

    async def _call_provider(
        self,
        ctx: DecisionContext,
        *,
        retry_reason: str | None,
    ) -> ProviderCall:
        bundle = self._render(ctx)
        system_blocks: list[dict[str, Any]] = [
            {
                "type": "text",
                "text": bundle.system_tournament,
                "cache_control": {"type": "ephemeral"},
            },
            {
                "type": "text",
                "text": bundle.system_session,
                "cache_control": {"type": "ephemeral"},
            },
        ]
        user_content = bundle.user_session_log + "\n\n" + bundle.user_volatile
        if retry_reason:
            user_content += (
                f"\n\nNOTE: your previous response failed validation: {retry_reason}.  "
                "Reply with a fresh JSON object conforming to the schema."
            )
        msg_kwargs: dict[str, Any] = {
            "model": self._sdk_model_name(),
            "max_tokens": 1024,
            "system": system_blocks,
            "messages": [{"role": "user", "content": user_content}],
        }
        if self._enable_thinking:
            msg_kwargs["thinking"] = {
                "type": "enabled",
                "budget_tokens": EXTENDED_THINKING_BUDGET_TOKENS,
            }
        client: AnthropicClientProtocol = self._client  # type: ignore[assignment]
        resp = await client.messages.create(**msg_kwargs)
        text = _first_text_block(resp)
        usage_obj = resp.usage
        usage = Usage(
            input_tokens=int(getattr(usage_obj, "input_tokens", 0)),
            output_tokens=int(getattr(usage_obj, "output_tokens", 0)),
            cache_read_tokens=int(getattr(usage_obj, "cache_read_input_tokens", 0)),
            cache_write_tokens=int(getattr(usage_obj, "cache_creation_input_tokens", 0)),
        )
        return ProviderCall(text=text, usage=usage, latency_ms=0)

    def _sdk_model_name(self) -> str:
        return self.model_id.split(":", 1)[1]


def _first_text_block(resp: Any) -> str:
    for block in resp.content:
        if getattr(block, "type", None) == "text":
            return block.text
    # Extended-thinking responses can exhaust their budget without producing
    # a text block; surface as AgentOutputParseError so the retry-then-autofold
    # path in BaseAdapter.decide takes over instead of crashing the runner.
    raise AgentOutputParseError("anthropic response had no text block")
