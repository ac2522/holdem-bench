# Methodology (Phase 0)

Phase 0 validates the harness, engine, chat protocol, scoring, and event log against deterministic baselines (`RandomAgent`, `TightPassiveAgent`, `GTOApproxAgent`, `CannedTalkAgent`). No real model calls, no chat enabled yet (baseline agents are silent except `CannedTalkAgent`'s rotating canned messages).

Phase 1 adds real model adapters, a Django leaderboard, and first public weekly tournament. See the design spec in `docs/design/` for the full methodology roadmap.
