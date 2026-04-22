# Governance

## Decision making

Maintainer (Alex Carter) has final say on schema changes, tournament configurations, and roster policy until the project matures to a stewardship model.

## Versioning

- `schema_version` bumps on any breaking change to `src/holdembench/events/schema.py`. Old logs remain readable.
- `holdembench.__version__` follows SemVer.
- Each tournament release is tagged `t-<tournament_id>` with a GitHub Release carrying `events.jsonl` and `manifest.json` as assets.

## Amendments

Governance changes require a PR + 7-day comment period.
