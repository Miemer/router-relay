"""P2 ensemble: B5 fusion (N parallel proposers → 1 aggregator LLM).

Mirrors OpenSquilla's `provider/ensemble.py` at a fraction of the size. Only the
`b5_fusion` mode is supported (parallel proposers → prompt-based aggregation);
no voting, no best-of-N selection.

Pipeline:
  1. N proposer models run in parallel, NON-streaming (we need full drafts to fuse).
  2. Quorum check (min_successful_proposers); on failure → fallback_single (stream
     or return the routed anchor model directly).
  3. Build aggregator messages with each draft wrapped in <CANDIDATE N> tags.
  4. The aggregator LLM streams (stream=true) or returns JSON (stream=false).

For stream=true the proposer phase is awaited before returning the StreamingResponse
for the aggregator, so proposer errors surface as clean HTTP errors (no mid-stream
broken SSE). The trade-off is no heartbeat during the proposer phase — P2.5 can add
SSE comment-frame heartbeats if clients time out.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

from fastapi.responses import JSONResponse

from .errors import RelayError

if TYPE_CHECKING:
    from .config import Settings
    from .router.runtime import RoutingDecision
    from .upstream import StreamDoneCallback, UpstreamClient

logger = logging.getLogger("relay.ensemble")


def _proposer_models(settings: "Settings", anchor_model: str) -> list[str]:
    """Configured proposer lineup, deduped, with the routed anchor prepended."""
    models: list[str] = []
    seen: set[str] = set()
    if anchor_model and anchor_model not in seen:
        models.append(anchor_model)
        seen.add(anchor_model)
    for m in settings.ensemble_proposers:
        if m and m not in seen:
            models.append(m)
            seen.add(m)
    return models


async def _run_one_proposer(
    client, model: str, messages: list, max_tokens, timeout: float
) -> dict:
    """Call one proposer non-stream; return {model, ok, text, usage} or {ok:false}."""
    req_body: dict = {"model": model, "messages": messages, "stream": False}
    if max_tokens is not None:
        req_body["max_tokens"] = max_tokens
    try:
        resp = await client.post("/chat/completions", json=req_body, timeout=timeout)
        if resp.status_code >= 400:
            return {"model": model, "ok": False, "error": f"upstream {resp.status_code}"}
        data = resp.json()
        try:
            text = data["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError):
            text = ""
        return {"model": model, "ok": True, "text": text, "usage": data.get("usage") or {}}
    except Exception as exc:  # defensive: one proposer failing must not break the others
        return {"model": model, "ok": False, "error": str(exc)}


def _build_aggregator_messages(messages: list, candidates: list[dict], max_chars: int) -> list:
    """Append candidate drafts as a synthetic user message for the aggregator."""
    lines = [
        "You are the aggregator in a multi-model ensemble fusion.",
        "Synthesize the best answer from the original conversation and the candidate drafts below.",
        "Do not mention the ensemble, candidates, or model names unless the user explicitly asks.",
        "Otherwise, answer the user directly with the strongest fused result.",
        "",
        "Candidate drafts:",
    ]
    for i, cand in enumerate(candidates, start=1):
        draft = (cand.get("text") or "").strip() or "[empty]"
        if len(draft) > max_chars:
            draft = draft[:max_chars] + " …[truncated]"
        lines.append(f"\n<CANDIDATE {i}>\n{draft}\n</CANDIDATE {i}>")
    return [*messages, {"role": "user", "content": "\n".join(lines)}]


def _sum_usage(proposer_results: list[dict], agg_usage: dict) -> dict:
    total = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    for r in proposer_results:
        u = r.get("usage") or {}
        total["prompt_tokens"] += int(u.get("prompt_tokens") or 0)
        total["completion_tokens"] += int(u.get("completion_tokens") or 0)
        total["total_tokens"] += int(u.get("total_tokens") or 0)
    total["prompt_tokens"] += int(agg_usage.get("prompt_tokens") or 0)
    total["completion_tokens"] += int(agg_usage.get("completion_tokens") or 0)
    total["total_tokens"] += int(agg_usage.get("total_tokens") or 0)
    return total


def _error_body(resp) -> dict:
    try:
        data = resp.json()
    except Exception:
        return {"error": {"message": resp.text or "upstream error", "type": "upstream_error"}}
    if isinstance(data, dict) and "error" in data:
        return data
    return {"error": {"message": str(data), "type": "upstream_error"}}


async def run_ensemble(
    body: dict,
    settings: "Settings",
    upstream: "UpstreamClient",
    decision: "RoutingDecision",
    stream: bool,
    on_stream_done: "StreamDoneCallback | None" = None,
):
    """Run B5 fusion. Returns a JSONResponse / StreamingResponse, or None to let the
    caller fall back to single-model passthrough.

    ``on_stream_done`` is forwarded to the upstream streaming call so the caller
    can capture the full streaming outcome (usage/finish_reason) when the
    aggregator (or fallback) stream completes.
    """
    messages = body.get("messages") or []
    if not messages:
        return None

    max_tokens = body.get("max_tokens")
    anchor = decision.model or body.get("model") or ""
    proposers = _proposer_models(settings, anchor)
    aggregator = settings.ensemble_aggregator or anchor

    if len(proposers) < 2:
        logger.info("ensemble: fewer than 2 proposers (%s); single passthrough", proposers)
        return None

    # Phase 1: proposers in parallel (non-stream).
    results = await asyncio.gather(*[
        _run_one_proposer(
            upstream.client, m, messages, max_tokens, settings.ensemble_proposer_timeout
        )
        for m in proposers
    ])
    successful = [r for r in results if r["ok"]]
    logger.info(
        "ensemble: %d/%d proposers ok (aggregator=%s)",
        len(successful), len(results), aggregator,
    )

    if len(successful) < settings.ensemble_min_successful:
        # Quorum not met → fallback_single: call the routed anchor model directly.
        logger.warning(
            "ensemble: quorum not met (%d<%d); fallback single %s",
            len(successful), settings.ensemble_min_successful, anchor,
        )
        fb_body = dict(body)
        fb_body["model"] = anchor
        return await upstream.chat_completions(fb_body, stream=stream, on_stream_done=on_stream_done)

    # Phase 2: aggregator fuses the drafts.
    agg_messages = _build_aggregator_messages(
        messages, successful, settings.ensemble_candidate_max_chars
    )
    agg_body: dict = {"model": aggregator, "messages": agg_messages, "stream": stream}
    if max_tokens is not None:
        agg_body["max_tokens"] = max_tokens

    if stream:
        # Reuse the upstream streaming path; it forwards the aggregator's SSE verbatim.
        return await upstream.chat_completions(agg_body, stream=True, on_stream_done=on_stream_done)

    # Non-stream: call the aggregator and fold proposer usage into the response.
    try:
        resp = await upstream.client.post(
            "/chat/completions",
            json=agg_body,
            timeout=settings.ensemble_aggregator_timeout,
        )
    except Exception as exc:
        raise RelayError(
            502, {"error": {"message": f"aggregator unreachable: {exc}", "type": "upstream_error"}}
        ) from exc
    if resp.status_code >= 400:
        raise RelayError(resp.status_code, _error_body(resp))
    data = resp.json()
    data["usage"] = _sum_usage(results, data.get("usage") or {})
    data["model"] = aggregator
    return JSONResponse(content=data, status_code=200)
