"""FastAPI app: OpenAI-compatible relay surface (P0 passthrough)."""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from .auth import verify_token
from .capture import CaptureStore
from .config import Settings, get_settings
from .ensemble import run_ensemble
from .errors import RelayError
from .router import DecisionStore, RoutingHistory, apply_router
from .router.tiers import tier_rank
from .upstream import UpstreamClient

logger = logging.getLogger("relay")


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


@app.post("/v1/chat/completions", dependencies=[Depends(verify_token)])
async def chat_completions(request: Request, upstream: UpstreamClient = Depends(get_upstream)):
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

    # ── P1 routing: classify difficulty and override model per tier ──
    decision = None
    if settings.router_enabled:
        history: RoutingHistory = request.app.state.router_history
        decision = await apply_router(body, settings, history)
        if decision is not None:
            if settings.router_log_decisions:
                logger.info("router: %s", decision.summarize())
            if not settings.router_observe_only and decision.model:
                body["model"] = decision.model
            store = request.app.state.router_store
            if store is not None:
                asyncio.create_task(store.write(decision))  # fire-and-forget, off-loop
            capture = request.app.state.capture_store
            if capture is not None:
                asyncio.create_task(capture.write(decision))  # P3-prep training data

    # ── P2 ensemble: B5 fusion for complex tiers (wraps after routing) ──
    if (
        settings.ensemble_enabled
        and decision is not None
        and not settings.router_observe_only
        and not body.get("tools")  # P2 doesn't fuse tool-calling; skip when tools present
        and tier_rank(decision.tier) >= tier_rank(settings.ensemble_min_tier)
    ):
        try:
            ensemble_resp = await run_ensemble(body, settings, upstream, decision, stream)
        except RelayError:
            raise
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("ensemble: failed (%s); single passthrough", exc)
            ensemble_resp = None
        if ensemble_resp is not None:
            logger.info(
                "ensemble: executed tier=%s anchor=%s", decision.tier, decision.model
            )
            return ensemble_resp

    return await upstream.chat_completions(body, stream=stream)


@app.get("/v1/router/decisions", dependencies=[Depends(verify_token)])
async def router_decisions(request: Request, limit: int = 20):
    """Return the most recent routing decisions from the in-memory ring buffer."""
    history: RoutingHistory = request.app.state.router_history
    items = [d.to_record() for d in history.recent_decisions(limit)]
    return JSONResponse(content={"decisions": items, "count": len(items)})
