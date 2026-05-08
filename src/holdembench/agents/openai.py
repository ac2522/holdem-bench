"""OpenAIAgent — GPT via the ``openai`` async SDK.

OpenAI handles prompt caching automatically; the adapter reports
``cache_read_tokens`` from ``usage.prompt_tokens_details.cached_tokens``.
Structured output via ``response_format={"type": "json_schema", ...}`` —
this reinforces :class:`AgentOutput` invariants server-side and cuts
retry rate.
"""

from __future__ import annotations

import hashlib
from typing import Any, Protocol

from holdembench.agents.base import DecisionContext
from holdembench.agents.base_adapter import BaseAdapter, ProviderCall, Usage
from holdembench.types import ActionName

_SEED_MODULUS = 2**31


class _CompletionsProto(Protocol):
    async def create(self, **kwargs: Any) -> Any: ...


class _ChatProto(Protocol):
    @property
    def completions(self) -> _CompletionsProto: ...


class OpenAIClientProtocol(Protocol):
    @property
    def chat(self) -> _ChatProto: ...


def build_openai_action_schema(
    legal: tuple[ActionName, ...],
    *,
    min_raise_to: int | None = None,  # noqa: ARG001 — kept for API symmetry; see note
    is_probe_reply: bool = False,
) -> dict[str, Any]:
    """JSON Schema for one decision, narrowing ``action`` enum to ``legal``.

    Nullable fields use ``anyOf`` (not the ``type: [..., "null"]`` shorthand)
    because Anthropic-via-OpenRouter rejects the latter when paired with
    ``enum``.  All providers accept ``anyOf``.

    ``min_raise_to`` is accepted for symmetry with the prompt path but not
    expressed as a JSON-Schema ``minimum`` — Anthropic-via-OpenRouter
    rejects ``minimum`` on integer types ("property 'minimum' is not
    supported"), and we can't tell at request time whether OpenRouter
    will route to Anthropic.  Sub-min raises are caught post-hoc by
    ``TDAValidator`` instead.

    ``kind`` is narrowed by context: when ``is_probe_reply`` is False
    the model may emit ``action`` or ``probe`` only; when True the
    model is replying to another seat's probe and may emit only
    ``probe_reply``.  Without this narrowing models have been observed
    to spontaneously emit ``probe_reply`` at themselves.
    """
    kind_enum = ["probe_reply"] if is_probe_reply else ["action", "probe"]
    return {
        "name": "agent_output",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "kind": {"type": "string", "enum": kind_enum},
                "action": {
                    "anyOf": [
                        {"type": "string", "enum": list(legal)},
                        {"type": "null"},
                    ],
                },
                "amount": {"anyOf": [{"type": "integer"}, {"type": "null"}]},
                # ~80 tokens budget per chat message → ~480 chars in English.
                # Hard schema cap stops verbose models from auto-folding on
                # over-budget chat (which would measure verbosity, not poker).
                "message": {
                    "anyOf": [{"type": "string", "maxLength": 480}, {"type": "null"}],
                },
            },
            "required": ["kind", "action", "amount", "message"],
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
        # Plain-string content for all messages.  Anthropic-style
        # cache_control is provider-specific and our short ~334-token
        # system prompt is below Haiku 4.5's 4096-token cache minimum,
        # so the hint never fires.  Sending content-arrays with
        # cache_control to non-Anthropic OR routes is undocumented and
        # has caused 400s on some providers — keep this clean.
        kwargs: dict[str, Any] = {
            "model": self._sdk_model_name(),
            "messages": [
                {"role": "system", "content": bundle.system_tournament},
                {"role": "system", "content": bundle.system_session},
                {"role": "user", "content": user},
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": build_openai_action_schema(
                    ctx.legal,
                    min_raise_to=ctx.min_raise_to,
                    is_probe_reply=ctx.is_probe_reply,
                ),
            },
            # max_tokens covers BOTH visible JSON output AND server-side
            # reasoning (provider-specific).  At 1024 with reasoning on,
            # thinking eats the whole budget and visible JSON is empty —
            # auto-fold storm.  4096 leaves >=1k for visible output even
            # under heavy reasoning.
            "max_tokens": 4096,
            # Reproducibility.  All OR routes default temperature=1.0 if
            # absent; pinning explicitly so a future provider change
            # doesn't silently shift sampling.  `seed` is best-effort
            # honoured by most providers and lets us replay a single
            # decision deterministically given the same prompt.
            "temperature": 1.0,
            "seed": _seed_from_ctx(self.model_id, ctx),
        }
        if self._reasoning_effort:
            # OpenRouter unified form.  Sending BOTH `reasoning_effort`
            # (top-level) and `reasoning.effort` (extra_body) is rejected
            # by some OR-routed reasoning models with a 400 ("only one
            # of 'reasoning' and 'reasoning_effort' may be provided").
            # Use only the OR canonical form here; native OpenAI o-series
            # would need a separate code path if added later.
            kwargs["extra_body"] = {
                "reasoning": {"effort": self._reasoning_effort},
            }
        client: OpenAIClientProtocol = self._client  # type: ignore[assignment]
        resp = await client.chat.completions.create(**kwargs)
        # Defensive: providers occasionally return choices=[] on certain
        # content-filter / upstream-error paths.  Treat that as an empty
        # response so the BaseAdapter retry loop sees AgentOutputParseError
        # rather than an IndexError that would crash the run mid-hand.
        if not resp.choices:
            return ProviderCall(text="", usage=Usage(0, 0), latency_ms=0)
        msg = resp.choices[0].message
        text = msg.content or ""
        # OpenRouter canonical reasoning text lives on `message.reasoning`
        # (a string).  When present we capture it for telemetry only —
        # never asked for in the schema and never echoed in next-turn
        # prompts.  Native OpenAI o-series doesn't expose reasoning text
        # at all, so this stays None there.
        reasoning_text: str | None = getattr(msg, "reasoning", None) or None
        u = resp.usage
        details = getattr(u, "prompt_tokens_details", None)
        cache_read = int(getattr(details, "cached_tokens", 0) or 0) if details else 0
        prompt_tokens = int(getattr(u, "prompt_tokens", 0))
        completion_tokens = int(getattr(u, "completion_tokens", 0))
        # reasoning_tokens is a SUBSET of completion_tokens for OpenAI o-series
        # and (via OpenRouter normalisation) for any model that supports
        # thinking.  We split them so visible output and reasoning tokens
        # are billed at the right rate even when those rates differ.
        comp_details = getattr(u, "completion_tokens_details", None)
        reasoning_tokens = (
            int(getattr(comp_details, "reasoning_tokens", 0) or 0) if comp_details else 0
        )
        usage = Usage(
            input_tokens=max(0, prompt_tokens - cache_read),
            output_tokens=max(0, completion_tokens - reasoning_tokens),
            cache_read_tokens=cache_read,
            thinking_tokens=reasoning_tokens,
        )
        return ProviderCall(
            text=text, usage=usage, latency_ms=0, reasoning_text=reasoning_text
        )

    def _sdk_model_name(self) -> str:
        return self.model_id.split(":", 1)[1]


def _seed_from_ctx(model_id: str, ctx: DecisionContext) -> int:
    """Stable per-decision seed derived from model_id + hand_id + seat.

    Using the same seed across re-runs of the same hand lets us reproduce
    a model's decision when investigating a log; using different seeds
    across hands prevents the model from accidentally giving the same
    response to every decision.
    """
    h = hashlib.sha256(f"{model_id}|{ctx.hand_id}|{ctx.seat}".encode())
    return int.from_bytes(h.digest()[:4], "big") % _SEED_MODULUS
