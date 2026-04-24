# Provider pricing snapshot — 2026-04-23

Pricing captured on 2026-04-23 from each provider's public pricing page.
Update this file (and `src/holdembench/agents/pricing_sheet.py`) whenever
prices change; do not overwrite — add a dated entry.

| Model ID | Input $/MTok | Output $/MTok | Cache-read $/MTok | Notes |
|---|---:|---:|---:|---|
| anthropic:claude-haiku-4-5 | 1.00 | 5.00 | 0.10 | ephemeral cache |
| anthropic:claude-sonnet-4-6 | 3.00 | 15.00 | 0.30 | ephemeral cache |
| anthropic:claude-opus-4-7 | 15.00 | 75.00 | 1.50 | ephemeral cache |
| openai:gpt-5-mini | 0.40 | 1.60 | 0.10 | automatic caching |
| openai:gpt-5 | 2.50 | 10.00 | 0.625 | automatic caching |
| google:gemini-3-flash-preview | 0.30 | 1.20 | 0.075 | implicit caching |
| google:gemini-3-pro | 3.50 | 10.50 | 0.875 | implicit caching |
| xai:grok-4 | 3.00 | 15.00 | n/a | no cache pricing surface |
| moonshot:kimi-k2 | 0.30 | 1.20 | 0.03 | explicit cache_control |
| openrouter:deepseek/deepseek-chat-v3 | 0.14 | 0.28 | n/a | OpenRouter fee included |
| openrouter:qwen/qwen3-32b | 0.27 | 0.60 | n/a | OpenRouter fee included |

These numbers are what `PRICING_SHEET` will report against until the file is
updated.  The adapter `cost_usd` log field computes off these values × the
usage returned by the provider response.  Any revision should land as a new
`<vendor>:<model>-vN` entry so historical tournaments stay replayable at the
exact pricing they ran under.
