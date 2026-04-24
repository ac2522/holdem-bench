"""GoogleAgent — Gemini via the ``google-genai`` async SDK.

Gemini 1.5+ Pro class models support implicit caching.  We surface the
cached slice of the prompt via ``usage_metadata.cached_content_token_count``
so cost/cache-hit-rate accounting matches the other providers.
"""

from __future__ import annotations

from typing import Any, Protocol

from holdembench.agents.base import DecisionContext
from holdembench.agents.base_adapter import BaseAdapter, ProviderCall, Usage
from holdembench.agents.output_schema import AgentOutputParseError


class GoogleClientProtocol(Protocol):
    async def generate_content(self, **kwargs: Any) -> Any: ...


_AGENT_OUTPUT_GENAI_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "kind": {"type": "string", "enum": ["action", "probe", "probe_reply"]},
        "action": {
            "type": "string",
            "enum": ["fold", "check", "call", "raise"],
            "nullable": True,
        },
        "amount": {"type": "integer", "nullable": True},
        "message": {"type": "string", "nullable": True},
        "thinking": {"type": "string", "nullable": True},
    },
    # Mirror the OpenAI schema: every field is required (nullable allowed).
    # Without this Gemini will happily return {"kind": "action"} missing
    # "action", forcing a retry when the adapter re-validates.
    "required": ["kind", "action", "amount", "message", "thinking"],
}


class GoogleAgent(BaseAdapter):
    def __init__(
        self,
        *,
        model_id: str,
        client: GoogleClientProtocol,
    ) -> None:
        super().__init__(model_id=model_id, client=client)

    async def _call_provider(
        self,
        ctx: DecisionContext,
        *,
        retry_reason: str | None,
    ) -> ProviderCall:
        bundle = self._render(ctx)
        prompt = (
            bundle.system_tournament
            + "\n\n"
            + bundle.system_session
            + "\n\n"
            + bundle.user_session_log
            + "\n\n"
            + bundle.user_volatile
        )
        if retry_reason:
            prompt += f"\n\nRETRY: {retry_reason}"
        client: GoogleClientProtocol = self._client  # type: ignore[assignment]
        resp = await client.generate_content(
            model=self._sdk_model_name(),
            contents=prompt,
            config={
                "response_mime_type": "application/json",
                "response_schema": _AGENT_OUTPUT_GENAI_SCHEMA,
                "max_output_tokens": 1024,
            },
        )
        text = _first_text(resp)
        u = resp.usage_metadata
        cache_read = int(getattr(u, "cached_content_token_count", 0) or 0)
        prompt_tokens = int(getattr(u, "prompt_token_count", 0))
        usage = Usage(
            input_tokens=max(0, prompt_tokens - cache_read),
            output_tokens=int(getattr(u, "candidates_token_count", 0)),
            cache_read_tokens=cache_read,
        )
        return ProviderCall(text=text, usage=usage, latency_ms=0)

    def _sdk_model_name(self) -> str:
        return self.model_id.split(":", 1)[1]


def _first_text(resp: Any) -> str:
    """Extract the first text part from a Gemini response, or raise parse error."""
    candidates: list[Any] = getattr(resp, "candidates", None) or []
    for cand in candidates:
        parts: list[Any] = getattr(getattr(cand, "content", None), "parts", None) or []
        for part in parts:
            text: str | None = getattr(part, "text", None)
            if text:
                return text
    raise AgentOutputParseError("google response had no text part")
