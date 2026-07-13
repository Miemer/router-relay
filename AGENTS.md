# AGENTS.md

Guidance for ZCode agents working in `router-relay`. Read the `README.md` for the
full human-facing doc (config reference, rollout path, opencode/ZCode wiring);
this file captures what the code expects that isn't obvious from a glance.

## What this is

An OpenAI-compatible **and** Anthropic-compatible API relay with smart per-turn
routing. Clients (opencode via OpenAI format, ZCode via Anthropic format) point
at this service; the relay forwards to an upstream OpenAI-compatible provider
(OpenAI / OpenRouter / marketingforce / …) and optionally:

- **P1** rule-scores each turn's difficulty → `c0..c3` tier → overrides `model`.
- **P2** B5 ensemble fusion (parallel proposers → aggregator LLM) for complex tiers.
- **P3-prep** date-partitioned JSONL capture of aggregate features for future
  LightGBM self-learning (P3 itself is not yet implemented).

Inspired by OpenSquilla's `SquillaRouter` + Ensemble.

## Layout

```
src/relay/
├── __main__.py        # entry: `python -m relay` / `router-relay` console script
├── app.py             # FastAPI app + _handle_completion (shared OpenAI/Anthropic)
├── auth.py            # Bearer token dependency (RELAY_API_KEYS)
├── config.py          # pydantic-settings, lru_cached get_settings()
├── upstream.py        # httpx AsyncClient; SSE raw-byte passthrough
├── ensemble.py        # P2 B5 fusion (proposer → aggregator)
├── capture.py         # P3-prep: date-partitioned JSONL (logs/router-samples-YYYY-MM-DD.jsonl)
├── errors.py          # RelayError → OpenAI-shaped error envelope
└── router/
    ├── features.py    # handcrafted feature extraction (pure, no I/O, no prompt text)
    ├── scorer.py      # rule scorer → tier + confidence (P3 replacement point)
    ├── policy.py      # confidence_gate → complaint_upgrade → large_context_floor → sticky
    ├── tiers.py       # DEFAULT_TIERS preset + resolve_model (source of truth)
    └── runtime.py     # RoutingDecision, RoutingHistory, bounded apply_router
tests/test_router_scoring.py  # standalone scoring smoke test (NOT pytest)
```

## Commands

```sh
uv sync                                         # install deps into .venv
uv run router-relay                             # run server (127.0.0.1:8787)
uv run python -m relay                           # alt entry
uv run uvicorn relay.app:app --port 8787         # alt entry (reload-friendly)
uv run python tests/test_router_scoring.py       # scoring smoke test (prints table, exit code)
curl http://127.0.0.1:8787/healthz                # liveness (no auth)
curl -H "Authorization: Bearer <token>" "http://127.0.0.1:8787/v1/router/decisions?limit=10"
```

No linter/typechecker is configured in `pyproject.toml` (`.ruff_cache` /
`.mypy_cache` are gitignored but there is no config section — run ad-hoc if
needed). There is no pytest suite; the one test file is a script with `main()`.

## Architecture boundaries (matter for edits)

- **`app.py::_handle_completion`** is the shared handler for both
  `/v1/chat/completions` (OpenAI, `allow_ensemble=True`, forwards to
  `/chat/completions`) and `/v1/messages` (Anthropic, `allow_ensemble=False`,
  forwards to `/messages`). **Ensemble is skipped on the Anthropic path** — it
  calls `/chat/completions` (OpenAI-shaped). Routing applies to both paths
  because it only reads `messages` and overrides `model`.
- **`DEFAULT_TIERS` (tiers.py) is the source of truth** for tier→model. For both
  paths to work, the four models must be **dual-endpoint** (OpenAI + Anthropic)
  on the upstream (marketingforce supports both). If only one protocol is in
  use, single-endpoint models (e.g. `qwen3-max`, OpenAI-only) may be cheaper.
  > ⚠️ Model ids are **case-sensitive** on the marketingforce `new_api`
  > distributor and must match the upstream catalog exactly. Verified working
  > on `/messages`: `claude-3-5-haiku`, `qwen3.7-plus`, `claude-sonnet-4.5`
  > (not `4.6` — that id has no channel → 503 `model_not_found`),
  > `gpt-5.5`. The entry model ZCode sends must also be upstream-valid when
  > `ROUTER_OBSERVE_ONLY=true` (observe-only forwards the raw client model).
- **`apply_router` (runtime.py) is the single routing entry point** the gateway
  calls per turn. It is **bounded**: feature extraction + scoring run in a
  worker thread (`asyncio.to_thread`) under `asyncio.wait_for`. On timeout or
  any exception it returns `None` and the caller **transparently passes the
  client's original model through** — never block or fail a request for routing.
  This seam is deliberately forward-compatible with a P3 LightGBM/ONNX head.
- **P2 ensemble wraps AFTER routing**, and only fires when: OpenAI path,
  `ENSEMBLE_ENABLED=true`, a decision exists, **not** `router_observe_only`,
  request has **no `tools`** (P2 does not fuse tool-calling), and routed tier
  rank ≥ `ENSEMBLE_MIN_TIER`.
- **Scoring + feature extraction are pure** (no I/O, no embeddings, no prompt
  text). The `FeatureBundle` fields are the P3 training substrate — **keep them
  stable and aggregate-only**. Never store raw prompt text anywhere (features,
  decision records, capture JSONL, SQLite) — this mirrors OpenSquilla's privacy
  stance and is enforced by `to_snapshot()` / `to_record()`.
- **Session stickiness** is derived from the **first user message**
  (`sha1[:16]`). This works because the OpenAI/Anthropic protocols are stateless
  and clients resend full history each turn — no client cooperation needed.

## Conventions

- `from __future__ import annotations` at the top of every module.
- Logger names: `relay` (app/gateway), `relay.router` (routing internals).
- Errors: raise `RelayError(status_code, {"error": {"message": ..., "type": ...}})`
  — the OpenAI-shaped envelope clients expect. The app-level exception handler
  converts it to a `JSONResponse`. Upstream HTTP failures are mapped in
  `upstream._raise_for_status`.
- Settings: `pydantic-settings` + `get_settings()` (`lru_cache` — read it, don't
  construct `Settings()` ad hoc). Env var = uppercased field name. CSV/list
  fields (`relay_api_keys`, `ensemble_proposers`) accept comma-separated **or**
  JSON; `router_tiers` accepts a JSON object string and **fails loud on
  malformed JSON** (intentional).
- Fire-and-forget persistence: decision store + capture writes are scheduled via
  `asyncio.create_task` (off the request loop).
- SSE: raw-byte passthrough with `_SSE_HEADERS` (`Cache-Control: no-cache`,
  `X-Accel-Buffering: no`, …). The upstream stream is opened before our own SSE
  response begins so a clean HTTP error can be returned instead of a broken SSE.
- Decimal rounding: feature floats are `round(..., 3/4)` in snapshots.

## Gotchas

- **Restart the server after code/config changes.** The README calls this out
  for the `/v1/messages` addition: a running uvicorn holds old code, so edits
  won't take effect until `uv run router-relay` is restarted.
- **The capture JSONL records routing decisions, not upstream outcomes.**
  `executed_kind` is hardcoded `"single"` and there is no status/success field,
  so the file cannot tell you *why* a call failed — only what model the router
  *would have* picked (and in observe-only, that is not what was actually sent
  upstream). To debug a 503/upstream failure, reproduce the call and read the
  error body; `RelayError` carries the upstream's `{error: {...}}` verbatim.
- **`ROUTER_TIERS` must be valid JSON or empty** — a malformed string raises in
  the `field_validator` (fail loud by design).
- **Reasoning models (e.g. `deepseek-r1`) as ensemble proposers burn inference
  tokens not capped by `max_tokens`** — prefer non-reasoning models in the
  proposer pool.
- **Recommended rollout order**: `ROUTER_OBSERVE_ONLY=true` first (record
  decisions, don't override) → validate `/v1/router/decisions` → flip
  `observe_only=false` → optionally enable ensemble. Never go straight to
  override + ensemble on real traffic.
- `logs/`, `*.db`, `*.sqlite` are runtime artifacts — gitignored, do not commit.

## Files to read before sensitive edits

- `README.md` — full config reference, opencode/ZCode wiring, rollout path.
- `src/relay/router/runtime.py` — the bounded `apply_router` seam; P3 LightGBM
  replaces `score_features` at this call site.
- `src/relay/router/scorer.py` — `TIER_BOUNDS` / weights are tunable; this is
  the P3 replacement target.
- `src/relay/router/features.py` — field set must stay stable (P3 trains on it).
