# Reproducibility

## Re-running a published tournament

```bash
git clone https://github.com/ac2522/holdem-bench && cd holdem-bench
git checkout t-<tournament_id>
uv sync --frozen
docker build -t holdem-bench:<version> .
docker run --rm -v $PWD/results:/app/results holdem-bench:<version> \
  run --config evals/<config>.yaml --seed <seed> --deterministic-time
```

The generated `events.jsonl` should be byte-identical to the one in the release. Verify via:

```bash
shasum -a 256 results/<tid>/events.jsonl
# compare against manifest.json["events_sha256"]
```

## What's pinned

- Python 3.12 (via `.python-version`)
- All dependencies via `uv.lock`
- Base image digest-pinned in `Dockerfile`
- Master RNG seed logged in `tournament_start.master_seed`
- Deal-pack seed per session derived deterministically from master seed
- Canonical tokenizer (cl100k_base) pinned via `tiktoken` version
