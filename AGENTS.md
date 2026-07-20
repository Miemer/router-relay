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
- **P3** LightGBM self-learning: JSONL capture → realign/judge labeling →
  time-holdout training → registry-promoted ML head (hot reload, no restart).

Inspired by OpenSquilla's `SquillaRouter` + Ensemble.

## Layout

```
src/relay/
├── __main__.py        # entry: `python -m relay` / `router-relay` console script
├── app.py             # FastAPI app + _handle_completion + outcome capture helpers
├── auth.py            # Bearer token dependency (RELAY_API_KEYS)
├── config.py          # pydantic-settings, lru_cached get_settings()
├── upstream.py        # httpx AsyncClient; SSE raw-byte passthrough
├── ensemble.py        # P2 B5 fusion (proposer → aggregator)
├── converters.py      # OpenAI ↔ Anthropic shape adapters (dual-path ensemble)
├── capture.py         # P3-prep: decisions + outcomes + raw triple JSONL
├── errors.py          # RelayError → OpenAI-shaped error envelope
└── router/
    ├── features.py    # handcrafted feature extraction (pure, no I/O, no prompt text)
    ├── scorer.py      # rule scorer → tier + confidence (P3 replacement point)
    ├── ml_head.py     # P3 ML head: registry-driven hot-reload LightGBM → ScoreResult
    ├── registry.py    # P3 model registry: models/registry.json versions + active pointer
    ├── policy.py      # confidence_gate → complaint_upgrade → large_context_floor → sticky
    ├── tiers.py       # DEFAULT_TIERS preset + resolve_model (source of truth)
    └── runtime.py     # RoutingDecision, RoutingHistory, bounded apply_router, _derive_source
scripts/realign_labels.py  # offline label realignment (decisions + outcomes + judge → labeled)
scripts/judge_labels.py    # LLM-as-judge absolute difficulty labeling (user msg → optimal_tier)
scripts/train_p3.py        # LightGBM training (cost-sensitive + time-window holdout + gate vs active model)
scripts/self_learn.py      # self-learning orchestration (realign→judge→train→gate→promote)
tests/test_router_scoring.py  # standalone scoring + routing smoke test (NOT pytest)
```

## Commands

```sh
uv sync                                         # install deps into .venv
uv run router-relay                             # run server (127.0.0.1:8787)
uv run python -m relay                           # alt entry
uv run uvicorn relay.app:app --port 8787         # alt entry (reload-friendly)
uv run python tests/test_router_scoring.py       # scoring + routing smoke test (prints table, exit code)
uv run python scripts/realign_labels.py --date YYYY-MM-DD  # offline label realignment (--dry-run to preview)
curl http://127.0.0.1:8787/healthz                # liveness (no auth)
curl -H "Authorization: Bearer <token>" "http://127.0.0.1:8787/v1/router/decisions?limit=10"
```

No linter/typechecker is configured in `pyproject.toml` (`.ruff_cache` /
`.mypy_cache` are gitignored but there is no config section — run ad-hoc if
needed). There is no pytest suite; the one test file is a script with `main()`.

## Architecture boundaries (matter for edits)

- **`app.py::_handle_completion`** is the shared handler for both
  `/v1/chat/completions` (OpenAI, `request_format="openai"`) and
  `/v1/messages` (Anthropic, `request_format="anthropic"`). **Both paths run
  ensemble** — the OpenAI request is translated in (`src/relay/converters.py`)
  for the proposer/aggregator calls (which hit `/chat/completions`) and the
  OpenAI-shaped result is translated back out to Anthropic SSE/JSON. Routing
  applies to both paths because it only reads `messages` and overrides `model`.
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
- **P2 ensemble wraps AFTER routing**, and only fires when: `ENSEMBLE_ENABLED=true`,
  a decision exists, **not** `router_observe_only`, request has **no `tools`**
  (P2 does not fuse tool-calling), and routed tier rank ≥ `ENSEMBLE_MIN_TIER`.
  **Fires on BOTH OpenAI and Anthropic paths** — `converters.py` translates the
  request/response in and out; the internal proposer/aggregator calls are always
  OpenAI `/chat/completions`.
- **Scoring + feature extraction are pure** (no I/O, no embeddings, no prompt
  text). The `FeatureBundle` fields are the P3 training substrate — **keep them
  stable and aggregate-only**. Never store raw prompt text anywhere (features,
  decision records, capture JSONL, SQLite) — this mirrors OpenSquilla's privacy
  stance and is enforced by `to_snapshot()` / `to_record()`.
- **Session stickiness** is derived from the **first user message**
  (`sha1[:16]`). This works because the OpenAI/Anthropic protocols are stateless
  and clients resend full history each turn — no client cooperation needed.
- **Outcome capture is a second write path, post-call.** `app.py` wraps the
  upstream/ensemble call with timing + response extraction and fires
  `capture.write_outcome(decision_id, …)` via `asyncio.create_task`. Outcomes
  land in a **sidecar file** (`router-outcomes-YYYY-MM-DD.jsonl`), joined to
  decisions by `decision_id`. For streaming responses only `stream_started` +
  time-to-first-response is captured (raw-byte SSE passthrough makes parsing
  usage mid-stream fragile). `label_hint` is a per-turn heuristic
  (`under_routed`/`over_routed`/`appropriate`) derived from status + token
  counts; the offline realignment script combines it with complaint_followup
  signals to produce the final `label`.
- **Complaint retrospective backfill** (OpenSquilla `retrospective_under_routing`):
  when turn *N*'s `complaint_detected` is True, `app.py` calls
  `history.previous_decision(session_key, exclude_id=…)` to find turn *N−1*'s
  decision and writes an outcome record with `label_hint="under_routed"` for it.
  `exclude_id` is required because `apply_router` records the current decision
  into the ring buffer *before* returning.
- **`source` is derived from the policy trail** (`runtime._derive_source`), not
  hardcoded. Values: `rule_scorer` (no policy fired), `rule_scorer:confidence_gate`,
  `rule_scorer:complaint_upgrade`, `rule_scorer:large_context_floor`,
  `rule_scorer:sticky`, `passthrough` (router timeout/exception). When the P3 ML
  head replaces `score_features`, set `source="ml_head"` at that call site.
- **P3 self-learning deploy path** (added 2026-07-20): the serving model is the
  registry's *active* version, not `ROUTER_ML_MODEL_PATH`. `runtime.apply_router`
  calls `ml_head.get_active_ml_head(settings)`, which reads
  `models/registry.json` on every decision (registry active → hot-reload; else
  `ROUTER_ML_MODEL_PATH` → rule scorer). `scripts/self_learn.py` is the
  orchestrator: on first run it **bootstraps the incumbent
  `models/p3_lightgbm.txt` as the baseline version** (so a candidate must beat
  it), then realign → judge → `train_p3 --holdout-days N` (time-window holdout,
  compares candidate vs active on the future window) → promote via registry if
  `promote_ok`. Schedule it with cron / Task Scheduler once daily. Hot reload is
  keyed by `(path, mtime)` — a registry promote is picked up on the next request,
  no restart. Rollback = flip `active` back in `registry.json` (older versions
  stay in `models/versions/`). Ops endpoints: `POST /v1/router/reload`,
  `GET /v1/router/registry`.
- **`signals` (scorer sub-scores) are captured** in `to_record()` and the SQLite
  store. These are *additional* features (not part of `FeatureBundle`), safe to
  add — they don't affect the stable FeatureBundle contract.
- **Raw content capture is opt-in** (`CAPTURE_RAW_CONTENT=true`, default false).
  When enabled, `capture.write_raw()` stores the last user message text to a
  sidecar `router-raw-*.jsonl` for LLM-as-judge offline labeling. Only the last
  user message is stored (not full history, not response) — the judge assesses
  absolute difficulty from the task description alone. Privacy-preserving by
  default; the raw file can be deleted after judging.
- **LLM-as-judge absolute labels** (`scripts/judge_labels.py`): a strong LLM
  reads the user message and independently classifies `optimal_tier` (c0..c3),
  breaking the dependence on the rule scorer's own pick. Judge labels live in
  `router-judge-*.jsonl`. The realignment script merges them with outcome
  signals — priority: complaint > judge (absolute) > upstream_error > label_hint.

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
- **Capture has two files: decisions + outcomes.** The decision file
  (`router-samples-*.jsonl`) records what the router *would have* picked at
  routing time; the outcome file (`router-outcomes-*.jsonl`) records what
  actually happened upstream (status/usage/latency/finish_reason + `label_hint`).
  They are joined by `decision_id` via `scripts/realign_labels.py`, which
  populates `label` + `optimal_tier` in a third file (`router-labeled-*.jsonl`).
  In observe-only mode, the decision file's `model` is the router's pick but the
  outcome reflects the client's original model — interpret labels accordingly.
- **Streaming outcomes are now captured.** For `stream=true` requests,
  `upstream._chat_stream` parses SSE events from the raw byte stream (while
  still forwarding them verbatim to the client) to extract `usage` and
  `finish_reason`. A fire-and-forget callback (`_make_stream_callback` in
  `app.py`) fires in the generator's `finally` block after the last byte is
  sent, writing the full outcome (`success`/`stream_error` + usage + latency
  + `label_hint`) to the sidecar file. This handles both OpenAI
  (`choices[0].finish_reason` + `usage`) and Anthropic (`message_delta` +
  `message_start` usage) SSE shapes. The SSE parser adds zero overhead when
  capture is off (fast-path raw passthrough). The callback uses
  `asyncio.create_task` so it never delays closing the client connection.
  Note: if the upstream stream is interrupted (client disconnect / network
  drop), `stream_completed=False` and no `finish_reason` → outcome is
  `stream_error`.
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
  replaces `score_features` at this call site. Also `_derive_source` (trail→source)
  and `previous_decision` (complaint backfill seam).
- `src/relay/router/scorer.py` — `TIER_BOUNDS` / weights are tunable; this is
  the P3 replacement target.
- `src/relay/router/features.py` — field set must stay stable (P3 trains on it).
- `src/relay/capture.py` — dual-file JSONL (decisions + outcomes); `write_outcome`
  and `write_passthrough` are the post-call capture entry points.
- `src/relay/app.py` — outcome capture helpers (`_extract_outcome`,
  `_derive_label_hint`, `_error_outcome`) and the complaint backfill wiring in
  `_handle_completion`.
- `scripts/realign_labels.py` — label priority logic (complaint > judge > error
  > hint) and `optimal_tier` derivation; adjust heuristics here, not in capture.
  When judge labels exist, `optimal_tier` is absolute (judge's pick, not ±1).
- `scripts/judge_labels.py` — LLM-as-judge prompt design + response parsing;
  adjust the tier descriptions and examples here.
