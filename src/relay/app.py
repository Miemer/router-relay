"""FastAPI app: OpenAI-compatible relay surface (P0 passthrough)."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse

from .auth import verify_token
from .capture import CaptureStore
from .config import Settings, get_settings
from .ensemble import run_ensemble
from .errors import RelayError
from .router import DecisionStore, RoutingHistory, apply_router
from .router.features import extract_features, _last_user_text
from .router.tiers import tier_rank
from .upstream import UpstreamClient

logger = logging.getLogger("relay")

# Heuristics for the per-turn `label_hint` derived from the upstream outcome.
# Tunable; the offline realignment script may override these with richer logic.
_OVER_ROUTE_MIN_TIER = "c2"      # tiers at/above this with trivial output → over_routed
_OVER_ROUTE_MAX_TOKENS = 50      # completion_tokens below this on a strong tier → over_routed
_UNDER_ROUTE_MAX_TIER = "c1"     # tiers at/below this with very long output → under_routed
_UNDER_ROUTE_MIN_TOKENS = 2000   # completion_tokens above this on a cheap tier → under_routed


def _derive_label_hint(
    decision, usage: dict, upstream_status: int
) -> str | None:
    """Classify the turn as under/over/appropriate from the upstream outcome.

    These are *hints* — the offline realignment script combines them with
    complaint_followup signals (which take priority) to produce the final
    `label`. Returns None when there isn't enough signal to judge.
    """
    if upstream_status >= 400:
        # Upstream error: the chosen model couldn't serve the request → likely too weak.
        return "under_routed"
    if decision is None:
        return None
    completion_tokens = int(usage.get("completion_tokens") or 0)
    if completion_tokens <= 0:
        return None
    rank = tier_rank(decision.tier)
    if rank >= tier_rank(_OVER_ROUTE_MIN_TIER) and completion_tokens < _OVER_ROUTE_MAX_TOKENS:
        return "over_routed"
    if rank <= tier_rank(_UNDER_ROUTE_MAX_TIER) and completion_tokens > _UNDER_ROUTE_MIN_TOKENS:
        return "under_routed"
    return "appropriate"


def _extract_outcome(response, decision, t0: float, executed_kind: str) -> dict:
    """Build an outcome record from the upstream/ensemble response.

    For non-streaming JSONResponse: extracts usage + finish_reason + status.
    For streaming StreamingResponse: only records that the stream started
    (status < 400); mid-stream outcomes are not captured (raw-byte passthrough
    makes parsing SSE for usage fragile). The latency is time-to-first-response.
    """
    latency_ms = int((time.monotonic() - t0) * 1000)
    if isinstance(response, StreamingResponse):
        return {
            "outcome": "stream_started",
            "executed_kind": executed_kind,
            "latency_ms": latency_ms,
            "usage": None,
            "finish_reason": None,
            "upstream_status": 200,
            "label_hint": None,
        }
    # JSONResponse: body is the JSON-encoded bytes set at construction time.
    status = getattr(response, "status_code", 200)
    data: dict = {}
    body = getattr(response, "body", None)
    if body:
        try:
            data = json.loads(body) if isinstance(body, (bytes, bytearray)) else body
        except (json.JSONDecodeError, TypeError):
            data = {}
    usage = data.get("usage") or {}
    finish_reason = None
    try:
        finish_reason = data["choices"][0]["finish_reason"]
    except (KeyError, IndexError, TypeError):
        pass
    outcome = "success" if status < 400 else "upstream_error"
    return {
        "outcome": outcome,
        "executed_kind": executed_kind,
        "latency_ms": latency_ms,
        "usage": usage,
        "finish_reason": finish_reason,
        "upstream_status": status,
        "label_hint": _derive_label_hint(decision, usage, status),
    }


def _error_outcome(decision, t0: float, executed_kind: str, status_code: int) -> dict:
    """Outcome record for an upstream error (RelayError raised)."""
    return {
        "outcome": "upstream_error",
        "executed_kind": executed_kind,
        "latency_ms": int((time.monotonic() - t0) * 1000),
        "usage": None,
        "finish_reason": None,
        "upstream_status": status_code,
        "label_hint": "under_routed",
    }


def _make_stream_callback(
    capture: CaptureStore, decision, t0: float, executed_kind: str
):
    """Build a fire-and-forget callback for streaming-completion outcome capture.

    The upstream SSE parser populates ``captured`` with ``usage``,
    ``finish_reason``, ``upstream_status``, and ``stream_completed``. This
    callback turns that into an outcome record and writes it to the sidecar
    file. It runs after the last byte has been forwarded to the client (the
    generator's ``finally`` block schedules it via ``asyncio.create_task``).
    """

    async def _on_stream_done(captured: dict) -> None:
        usage = captured.get("usage") or {}
        upstream_status = captured.get("upstream_status", 200)
        finish = captured.get("finish_reason")
        stream_ok = captured.get("stream_completed", False)
        # finish_reason is the authoritative signal that the model completed its
        # output. If it's set, the turn succeeded — even if the client
        # disconnected before reading the trailing [DONE] / message_stop sentinel
        # (stream_completed would be False in that case, but the work is done).
        if finish:
            outcome_type = "success"
        elif stream_ok:
            outcome_type = "stream_completed_no_finish"
        else:
            outcome_type = "stream_error"
        outcome = {
            "outcome": outcome_type,
            "executed_kind": executed_kind,
            "latency_ms": int((time.monotonic() - t0) * 1000),
            "usage": usage or None,
            "finish_reason": finish,
            "upstream_status": upstream_status,
            "label_hint": _derive_label_hint(decision, usage, upstream_status),
        }
        await capture.write_outcome(decision.decision_id, outcome)

    return _on_stream_done


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    client = UpstreamClient(settings)
    client.open()
    app.state.upstream = client
    app.state.settings = settings
    # P1 routing state.
    app.state.router_history = RoutingHistory(settings.router_sticky_turns)
    store = DecisionStore(settings.router_decision_db) if settings.router_decision_db else None
    if store is not None:
        await store.open()
    app.state.router_store = store
    # P3-prep capture (date-partitioned JSONL).
    capture = CaptureStore(settings.router_capture_dir) if settings.router_capture_dir else None
    if capture is not None:
        await capture.open()
        logger.info("capture: writing samples to %s", settings.router_capture_dir)
    app.state.capture_store = capture
    if settings.router_enabled:
        logger.info("router enabled (observe_only=%s)", settings.router_observe_only)
    yield
    if capture is not None:
        await capture.close()
    if store is not None:
        await store.close()
    await client.close()


app = FastAPI(title="router-relay", version="0.1.0", lifespan=lifespan)


def get_upstream(request: Request) -> UpstreamClient:
    return request.app.state.upstream


@app.exception_handler(RelayError)
async def _relay_error_handler(_, exc: RelayError):
    return JSONResponse(status_code=exc.status_code, content=exc.body)


@app.exception_handler(RequestValidationError)
async def _validation_error_handler(_, exc: RequestValidationError):
    return JSONResponse(
        status_code=422,
        content={"error": {"message": "invalid request body", "type": "invalid_request_error"}},
    )


@app.get("/")
async def root():
    return {"service": "router-relay", "version": "0.1.0"}


@app.get("/healthz")
async def healthz():
    return {"status": "ok"}


@app.get("/v1/models", dependencies=[Depends(verify_token)])
async def list_models(upstream: UpstreamClient = Depends(get_upstream)):
    data = await upstream.list_models()
    return JSONResponse(content=data)


async def _handle_completion(
    request: Request, upstream: UpstreamClient, forward_path: str, allow_ensemble: bool
):
    """Shared handler for OpenAI (/chat/completions) and Anthropic (/messages) paths.

    ``forward_path`` is the upstream endpoint path; ``allow_ensemble`` is False on
    the Anthropic path because ensemble calls /chat/completions (OpenAI-shaped).
    Routing applies to both paths (it only reads `messages` + overrides `model`).
    """
    try:
        body = await request.json()
    except Exception:
        raise RelayError(
            400, {"error": {"message": "request body must be JSON", "type": "invalid_request_error"}}
        ) from None
    if not isinstance(body, dict):
        raise RelayError(
            400, {"error": {"message": "request body must be a JSON object", "type": "invalid_request_error"}}
        )

    settings: Settings = get_settings()
    if not body.get("model") and settings.default_model:
        body["model"] = settings.default_model
    if not body.get("model"):
        raise RelayError(
            400, {"error": {"message": "missing 'model' field", "type": "invalid_request_error"}}
        )

    stream = bool(body.get("stream", False))
    capture: CaptureStore | None = request.app.state.capture_store

    # ── P1 routing: classify difficulty and override model per tier ──
    decision = None
    if settings.router_enabled:
        history: RoutingHistory = request.app.state.router_history
        decision = await apply_router(body, settings, history)
        client_model = str(body.get("model") or "")
        if decision is not None:
            if settings.router_log_decisions:
                logger.info("router: %s", decision.summarize())
            if not settings.router_observe_only and decision.model:
                body["model"] = decision.model
            store = request.app.state.router_store
            if store is not None:
                asyncio.create_task(store.write(decision))  # fire-and-forget, off-loop
            if capture is not None:
                asyncio.create_task(capture.write(decision))  # P3-prep training data

            # ── P3-prep: raw content capture for LLM-as-judge ──
            # Opt-in (CAPTURE_RAW_CONTENT=true). Stores the last user message
            # text so the offline judge can independently assess absolute difficulty.
            if capture is not None and settings.capture_raw_content:
                user_text = _last_user_text(body.get("messages") or [])
                if user_text:
                    asyncio.create_task(capture.write_raw(decision, user_text))

            # ── P3-prep: retrospective complaint backfill ──
            # If this turn's user message detects a complaint, the PREVIOUS turn
            # in this session was likely under-routed. Mark it so the realignment
            # script can populate label="under_routed" for that decision.
            # (OpenSquilla retrospective_under_routing; see features.py comment.)
            if decision.feature_snapshot.get("complaint_detected") and capture is not None:
                prev = history.previous_decision(
                    decision.session_key, exclude_id=decision.decision_id
                )
                if prev is not None:
                    asyncio.create_task(capture.write_outcome(prev.decision_id, {
                        "outcome": "complaint_followup",
                        "executed_kind": "complaint_backfill",
                        "latency_ms": None,
                        "usage": None,
                        "finish_reason": None,
                        "upstream_status": None,
                        "label_hint": "under_routed",
                    }))
        elif capture is not None:
            # apply_router returned None (timeout/exception) → passthrough.
            # Record a minimal passthrough decision so these failure cases are
            # visible in the capture log and available to P3 training.
            try:
                features = extract_features(body)
                pk = features.session_key
                pfeat = features.to_snapshot()
            except Exception:
                pk = "default"
                pfeat = {}
            asyncio.create_task(capture.write_passthrough(
                pk, client_model, "scoring_unavailable", pfeat
            ))

    # ── P2 ensemble: B5 fusion for complex tiers (OpenAI path only) ──
    t0 = time.monotonic()
    can_capture = capture is not None and decision is not None
    # Streaming outcome callbacks (fire when the stream finishes, not now).
    on_done_single = (
        _make_stream_callback(capture, decision, t0, "single")
        if stream and can_capture else None
    )
    on_done_ensemble = (
        _make_stream_callback(capture, decision, t0, "ensemble")
        if stream and can_capture else None
    )
    if (
        allow_ensemble
        and settings.ensemble_enabled
        and decision is not None
        and not settings.router_observe_only
        and not body.get("tools")  # P2 doesn't fuse tool-calling; skip when tools present
        and tier_rank(decision.tier) >= tier_rank(settings.ensemble_min_tier)
    ):
        try:
            ensemble_resp = await run_ensemble(
                body, settings, upstream, decision, stream,
                on_stream_done=on_done_ensemble,
            )
        except RelayError:
            if can_capture:
                asyncio.create_task(capture.write_outcome(
                    decision.decision_id,
                    _error_outcome(decision, t0, "ensemble", 502),
                ))
            raise
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("ensemble: failed (%s); single passthrough", exc)
            ensemble_resp = None
        if ensemble_resp is not None:
            logger.info(
                "ensemble: executed tier=%s anchor=%s", decision.tier, decision.model
            )
            # Non-streaming: capture outcome now. Streaming: the callback fires
            # when the stream finishes (on_done_ensemble handles it).
            if not stream and can_capture:
                outcome = _extract_outcome(ensemble_resp, decision, t0, "ensemble")
                asyncio.create_task(capture.write_outcome(decision.decision_id, outcome))
            return ensemble_resp

    # ── Single-model upstream path ──
    try:
        response = await upstream.chat_completions(
            body, stream=stream, path=forward_path, on_stream_done=on_done_single,
        )
    except RelayError as exc:
        if can_capture:
            asyncio.create_task(capture.write_outcome(
                decision.decision_id,
                _error_outcome(decision, t0, "single", exc.status_code),
            ))
        raise
    # Non-streaming: capture outcome now. Streaming: the callback fires when
    # the stream finishes (on_done_single handles it).
    if not stream and can_capture:
        outcome = _extract_outcome(response, decision, t0, "single")
        asyncio.create_task(capture.write_outcome(decision.decision_id, outcome))
    return response


@app.post("/v1/chat/completions", dependencies=[Depends(verify_token)])
async def chat_completions(request: Request, upstream: UpstreamClient = Depends(get_upstream)):
    """OpenAI-compatible endpoint (opencode, openai clients, /v1/chat/completions)."""
    return await _handle_completion(request, upstream, "/chat/completions", allow_ensemble=True)


@app.post("/v1/messages", dependencies=[Depends(verify_token)])
async def anthropic_messages(request: Request, upstream: UpstreamClient = Depends(get_upstream)):
    """Anthropic-compatible endpoint (ZCode, /v1/messages).

    Forwards to upstream /messages. Routing still applies (override model per tier);
    the tier models must be Anthropic-capable on the upstream (see DEFAULT_TIERS).
    Ensemble is skipped here — it calls /chat/completions (OpenAI-shaped).
    """
    return await _handle_completion(request, upstream, "/messages", allow_ensemble=False)


@app.get("/v1/router/decisions", dependencies=[Depends(verify_token)])
async def router_decisions(request: Request, limit: int = 20):
    """Return the most recent routing decisions from the in-memory ring buffer."""
    history: RoutingHistory = request.app.state.router_history
    items = [d.to_record() for d in history.recent_decisions(limit)]
    return JSONResponse(content={"decisions": items, "count": len(items)})


@app.post("/v1/router/reload", dependencies=[Depends(verify_token)])
async def router_reload(request: Request):
    """Force a hot reload of the serving ML model from the registry / env path.

    Operational escape hatch: normally the registry active pointer is re-read on
    every routing decision, so a self-learn promote is picked up automatically.
    This endpoint lets an operator (or the self-learn loop) force an immediate
    refresh and confirm which model is now serving. Never blocks the request
    path — on failure it reports the previous state and the rule scorer stays up.
    """
    from .router.ml_head import reload_ml_head
    from .router.registry import load_registry

    settings: Settings = get_settings()
    head = reload_ml_head(settings)
    reg = load_registry(settings.router_models_dir)
    active = reg.active
    return JSONResponse(content={
        "reloaded": head is not None,
        "serving": "ml_head" if head is not None else "rule_scorer",
        "active_version": active.version if active else None,
        "active_model_path": active.model_path if active else None,
        "versions": len(reg.list()),
    })


@app.get("/v1/router/registry", dependencies=[Depends(verify_token)])
async def router_registry(request: Request):
    """Return the model registry (versions + active pointer) for inspection."""
    from .router.registry import load_registry

    settings: Settings = get_settings()
    reg = load_registry(settings.router_models_dir)
    return JSONResponse(content=reg.to_dict())
