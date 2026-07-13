# router-relay

An OpenAI-compatible API relay with smart routing (inspired by OpenSquilla's
`SquillaRouter` + Ensemble). **P0** is a transparent passthrough: opencode (or any
OpenAI-compatible client) points at this server, and every request is forwarded
verbatim to an upstream OpenAI-compatible provider (OpenAI, OpenRouter, …),
streaming included.

Later phases add: **P1** per-turn routing (LightGBM classifier → tier → model),
**P2** B5 ensemble fusion, **P3** a self-learning loop. The routing/fusion seams
are already marked in `src/relay/upstream.py` (`_chat_stream` is the hook point).

## Layout

```
router-relay/
├── pyproject.toml
├── .env.example
└── src/relay/
    ├── __init__.py
    ├── __main__.py     # python -m relay  /  router-relay
    ├── app.py          # FastAPI app + routes (/v1/chat/completions, /v1/models, /healthz)
    ├── auth.py         # Bearer token dependency
    ├── config.py       # env-driven settings (pydantic-settings)
    ├── errors.py       # RelayError → OpenAI-shaped error envelope
    └── upstream.py     # httpx client + SSE passthrough
```

## Setup

Requires Python ≥ 3.10 and `uv` (recommended) or pip.

```sh
cd E:\PY_CODE\router-relay

# 1. install (creates .venv)
uv sync
#   — or —
python -m venv .venv && .venv/Scripts/activate && pip install -e .

# 2. configure
cp .env .env
#   then edit .env: set RELAY_API_KEYS, UPSTREAM_BASE_URL, UPSTREAM_API_KEY

# 3. run
uv run router-relay
#   — or —
uv run python -m relay
#   — or —
uv run uvicorn relay.app:app --port 8787
```

Server listens on `http://127.0.0.1:8787` by default.

## Configure opencode

Put this in `opencode.json` (project root or `~/.config/opencode/opencode.json`).
The `model` keys **must match the ids your upstream accepts** (e.g. OpenAI model
ids, or `openai/...` for OpenRouter).

```json
{
  "$schema": "https://opencode.ai/config.json",
  "model": "relay/gpt-4o-mini",
  "provider": {
    "relay": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "Router Relay (local)",
      "options": {
        "baseURL": "http://127.0.0.1:8787/v1",
        "apiKey": "sk-relay-dev-change-me"
      },
      "models": {
        "gpt-4o-mini": { "name": "GPT-4o mini (via relay)" },
        "gpt-4o": { "name": "GPT-4o (via relay)" }
      }
    }
  }
}
```

Then run `opencode` and select the `relay/...` model.

## Verify it works (curl)

```sh
# health (no auth)
curl http://127.0.0.1:8787/healthz
# {"status":"ok"}

# auth gate (expect 401 without token)
curl http://127.0.0.1:8787/v1/models

# list models (proxies upstream GET /v1/models)
curl -H "Authorization: Bearer sk-relay-dev-change-me" \
     http://127.0.0.1:8787/v1/models

# non-stream chat
curl -H "Authorization: Bearer sk-relay-dev-change-me" \
     -H "Content-Type: application/json" \
     -d '{"model":"gpt-4o-mini","messages":[{"role":"user","content":"ping"}]}' \
     http://127.0.0.1:8787/v1/chat/completions

# streaming chat (SSE)
curl -N -H "Authorization: Bearer sk-relay-dev-change-me" \
     -H "Content-Type: application/json" \
     -d '{"model":"gpt-4o-mini","stream":true,"messages":[{"role":"user","content":"count to 3"}]}' \
     http://127.0.0.1:8787/v1/chat/completions
```

## Configuration reference (`.env`)

| Var | Default | Purpose |
| --- | --- | --- |
| `RELAY_API_KEYS` | — | Comma-separated bearer tokens clients must send. Empty = open relay (dev only). |
| `UPSTREAM_BASE_URL` | `https://api.openai.com/v1` | Upstream OpenAI-compatible base URL. |
| `UPSTREAM_API_KEY` | — | Upstream API key (sent as `Authorization: Bearer`). |
| `UPSTREAM_ORGANIZATION` | — | Optional `OpenAI-Organization` header. |
| `DEFAULT_MODEL` | — | Fallback `model` when a request omits it. |
| `LISTEN_HOST` | `127.0.0.1` | Bind host. |
| `LISTEN_PORT` | `8787` | Bind port. |
| `UPSTREAM_TIMEOUT` | `600` | Upstream request timeout (seconds). |
| `LOG_LEVEL` | `info` | uvicorn log level. |

## P1 Routing

Enable per-turn routing: each request is classified by a deterministic rule
scorer (length / language / code ratio / keyword buckets / context size) into a
difficulty tier `c0..c3`, a policy chain (`confidence_gate` → `large_context_floor`
→ `sticky`/anti-downgrade) finalizes the tier, and the tier's model overrides the
one the client sent. Scoring runs in a worker thread under a timeout; on timeout
or error the relay transparently passes the client's model through (never blocks).

This mirrors OpenSquilla's SquillaRouter (`engine/steps/squilla_router.py`) at a
fraction of the size: handcrafted features + policy seam, no trained model yet.
A P3 LightGBM head can drop in at the same `score_features` call site.

| Tier | Default model (marketingforce preset) | Use |
| --- | --- | --- |
| c0 | qwen3-max | cheap / fast — simple Q&A, chitchat |
| c1 | deepseek-r1 | medium |
| c2 | gpt-5.4 | strong — engineering, design |
| c3 | claude-opus-4-8 | strongest |

Enable in `.env`:

```
ROUTER_ENABLED=true
# ROUTER_OBSERVE_ONLY=true   # record decisions but DON'T override — validate first
# ROUTER_TIERS={"c3":{"model":"claude-opus-4-8"}}  # override a tier (JSON)
```

Inspect live routing:

```sh
curl -H "Authorization: Bearer $RELAY_KEY" \
     "http://127.0.0.1:8787/v1/router/decisions?limit=20"
```

Decisions are kept in an in-memory ring buffer (last 100). Set `ROUTER_DECISION_DB`
to a SQLite path to also persist them (aggregate features only — never prompt text).

Run the scoring unit test:

```sh
uv run python tests/test_router_scoring.py
```

## P2 Ensemble

B5 fusion (inspired by OpenSquilla's `provider/ensemble.py`): for complex turns
the routed model plus configured proposers run **in parallel** (non-stream) to
produce draft answers; an aggregator LLM then fuses the drafts via a
`<CANDIDATE N>` prompt into one final answer. Only `b5_fusion` mode (no voting /
best-of-N). Only fires when the routed tier rank ≥ `ENSEMBLE_MIN_TIER` (default
`c2`), so easy turns stay single-model.

Enable in `.env` (requires `ROUTER_ENABLED=true`):

```
ENSEMBLE_ENABLED=true
ENSEMBLE_PROPOSERS=qwen3-max,deepseek-r1   # comma-separated; routed anchor is auto-prepended
ENSEMBLE_AGGREGATOR=qwen3-max              # empty = use routed anchor as aggregator
ENSEMBLE_MIN_TIER=c2                       # only fuse complex turns
ENSEMBLE_MIN_SUCCESSFUL=2                   # quorum before aggregation
```

Behavior:
- **Quorum**: if fewer than `ENSEMBLE_MIN_SUCCESSFUL` proposers succeed, fall back
  to single-model passthrough on the routed anchor (`fallback_single`).
- **Non-stream**: returns a JSON response with usage summed across all proposers
  + the aggregator.
- **Stream**: the aggregator's SSE is forwarded verbatim. The proposer phase is
  awaited first (no heartbeat during that window) so proposer errors surface as
  clean HTTP errors rather than broken mid-stream SSE.
- **Tools**: requests carrying `tools` skip ensemble (P2 does not fuse
  tool-calling) and fall through to single-model.
- **Cost caveat**: reasoning models (e.g. `deepseek-r1`) as proposers can burn
  many reasoning tokens that ignore `max_tokens`. Prefer non-reasoning models in
  the proposer pool for cost control.

## What P0/P1/P2 do NOT do yet

- No trained model / self-learning loop — P3 (LightGBM on captured decisions).
- No SSE heartbeat during the ensemble proposer phase (could time out slow
  clients) — a future P2.5.
- Ensemble does not fuse tool-calling turns (skips when `tools` present).

## Roadmap

- **P2.5**: SSE heartbeat comment-frames during the proposer phase so streaming
  clients don't time out; optional `router_dynamic` proposer selection.
- **P3** self-learning: capture features → feedback join → label realignment →
  incremental retrain → session-holdout CV + cost-ceiling gate → atomic swap +
  live rollback.
