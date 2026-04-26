"""OpenAIAgent — GPT via the ``openai`` async SDK.

OpenAI handles prompt caching automatically; the adapter reports
``cache_read_tokens`` from ``usage.prompt_tokens_details.cached_tokens``.
Structured output via ``response_format={"type": "json_schema", ...}`` —
this reinforces :class:`AgentOutput` invariants server-side and cuts
retry rate.
"""

from __future__ import annotations

from typing import Any, Protocol

from holdembench.agents.base import DecisionContext
from holdembench.agents.base_adapter import BaseAdapter, ProviderCall, Usage
from holdembench.types import ActionName


class _CompletionsProto(Protocol):
    async def create(self, **kwargs: Any) -> Any: ...


class _ChatProto(Protocol):
    @property
    def completions(self) -> _CompletionsProto: ...


class OpenAIClientProtocol(Protocol):
    @property
    def chat(self) -> _ChatProto: ...


def build_openai_action_schema(legal: tuple[ActionName, ...]) -> dict[str, Any]:
    """JSON Schema for one decision, with ``action`` enum narrowed to ``legal``.

    Nullable fields use ``anyOf`` (not the ``type: [..., "null"]`` shorthand)
    because Anthropic-via-OpenRouter rejects the latter when paired with
    ``enum``.  All providers accept ``anyOf``.

    Narrowing the enum per-call lets the provider reject illegal action names
    at the protocol layer, eliminating the most common ``ValidatorRejection``
    class entirely.  Amount bounds are still enforced by ``TDAValidator`` at
    apply time — JSON Schema can't express min_raise without threading
    table state through ``DecisionContext``.
    """
    return {
        "name": "agent_output",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "kind": {"type": "string", "enum": ["action", "probe", "probe_reply"]},
                "action": {
                    "anyOf": [
                        {"type": "string", "enum": list(legal)},
                        {"type": "null"},
                    ],
                },
                "amount": {"anyOf": [{"type": "integer"}, {"type": "null"}]},
                "message": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                "thinking": {"anyOf": [{"type": "string"}, {"type": "null"}]},
            },
            "required": ["kind", "action", "amount", "message", "thinking"],
        },
    }


class OpenAIAgent(BaseAdapter):
    def __init__(
        self,
        *,
        model_id: str,
        client: OpenAIClientProtocol,
        reasoning_effort: str | None = None,
    ) -> None:
        super().__init__(model_id=model_id, client=client)
        self._reasoning_effort = reasoning_effort

    async def _call_provider(
        self,
        ctx: DecisionContext,
        *,
        retry_reason: str | None,
    ) -> ProviderCall:
        bundle = self._render(ctx)
        user = bundle.user_session_log + "\n\n" + bundle.user_volatile
        if retry_reason:
            user += f"\n\nRETRY: previous output failed validation: {retry_reason}"
        kwargs: dict[str, Any] = {
            "model": self._sdk_model_name(),
            "messages": [
                {"role": "system", "content": bundle.system_tournament},
                {"role": "system", "content": bundle.system_session},
                {"role": "user", "content": user},
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": build_openai_action_schema(ctx.legal),
            },
            "max_tokens": 1024,
        }
        if self._reasoning_effort:
            kwargs["reasoning_effort"] = self._reasoning_effort
        client: OpenAIClientProtocol = self._client  # type: ignore[assignment]
        resp = await client.chat.completions.create(**kwargs)
        text = resp.choices[0].message.content or ""
        u = resp.usage
        details = getattr(u, "prompt_tokens_details", None)
        cache_read = int(getattr(details, "cached_tokens", 0)) if details else 0
        prompt_tokens = int(getattr(u, "prompt_tokens", 0))
        usage = Usage(
            input_tokens=max(0, prompt_tokens - cache_read),
            output_tokens=int(getattr(u, "completion_tokens", 0)),
            cache_read_tokens=cache_read,
        )
        return ProviderCall(text=text, usage=usage, latency_ms=0)

    def _sdk_model_name(self) -> str:
        return self.model_id.split(":", 1)[1]
