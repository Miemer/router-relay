"""P2 ensemble: B5 fusion + best-of-N (N parallel proposers → aggregate or pick).

Mirrors OpenSquilla's `provider/ensemble.py` and extends it with a best-of-N
mode. Two modes are supported:

  - ``b5_fusion`` (default): proposer drafts → aggregator LLM fuses them into a
    single synthesized answer. Best for tasks where multiple perspectives add value.
  - ``best_of_n``: proposer drafts → scorer LLM picks the single best draft by
    quality. Best for tasks with a clear correct answer (debugging, factual Q&A).

Pipeline:
  1. N proposer models run in parallel, NON-streaming (we need full drafts to fuse).
     SSE comment-frame heartbeats are sent every 5s during this phase for streaming
     clients so they don't time out.
  2. Quorum check (min_successful_proposers); on failure → fallback_single (stream
     or return the routed anchor model directly).
  3. Mode dispatch:
     - b5_fusion → aggregator fuses drafts via <CANDIDATE> prompt.
     - best_of_n → scorer picks best draft, returned directly (synthesized SSE for
       streaming clients — no second model call needed for the output itself).
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import TYPE_CHECKING

from fastapi.responses import JSONResponse, StreamingResponse

from .errors import RelayError

if TYPE_CHECKING:
    from .config import Settings
    from .router.runtime import RoutingDecision
    from .upstream import StreamDoneCallback, UpstreamClient

logger = logging.getLogger("relay.ensemble")

_SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
}

_HEARTBEAT_INTERVAL = 5.0  # seconds between SSE comment-frame heartbeats


def _proposer_models(settings: "Settings", anchor_model: str) -> list[str]:
    """Configured proposer lineup, deduped, with the routed anchor prepended.

    Capped at ``ensemble_max_proposers`` to bound cost.
    """
    models: list[str] = []
    seen: set[str] = set()
    if anchor_model and anchor_model not in seen:
        models.append(anchor_model)
        seen.add(anchor_model)
    for m in settings.ensemble_proposers:
        if m and m not in seen:
            models.append(m)
            seen.add(m)
    # Cap the number of proposers (cost control).
    cap = max(2, settings.ensemble_max_proposers)
    if len(models) > cap:
        models = models[:cap]
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


# ── B5 fusion: aggregator prompt ──

_B5_SYSTEM = (
    "You are the aggregator in a multi-model ensemble fusion. "
    "Synthesize the best answer from the original conversation and the candidate drafts below. "
    "Do not mention the ensemble, candidates, or model names unless the user explicitly asks. "
    "Otherwise, answer the user directly with the strongest fused result."
)


def _build_aggregator_messages(messages: list, candidates: list[dict], max_chars: int) -> list:
    """Build messages for the aggregator LLM.

    System message carries the aggregation instructions (avoids the consecutive
    user-message problem when the conversation ends with a user turn). Candidate
    drafts are appended as a separate user message after the conversation.
    """
    lines = [_B5_SYSTEM, ""]
    lines.append("Candidate drafts:")
    for i, cand in enumerate(candidates, start=1):
        draft = (cand.get("text") or "").strip() or "[empty]"
        if len(draft) > max_chars:
            draft = draft[:max_chars] + " …[truncated]"
        lines.append(f"\n<CANDIDATE {i}>\n{draft}\n</CANDIDATE {i}>")
    system_msg = {"role": "system", "content": "\n".join(lines[:1])}  # just the instruction
    candidates_msg = {"role": "user", "content": "\n".join(lines[1:])}
    return [system_msg, *messages, candidates_msg]


# ── best-of-N: scorer prompt ──

_SCORER_SYSTEM = (
    "You are a quality judge. Given the user's question and N candidate answers, "
    "pick the single best answer by quality, correctness, and completeness. "
    "Respond ONLY with a JSON object: {\"best\": <number>} where <number> is 1..N."
)


def _build_scorer_messages(
    messages: list, candidates: list[dict], max_chars: int
) -> list:
    """Build messages for the best-of-N scorer LLM."""
    # Extract the last user message (the actual question).
    user_question = ""
    for msg in reversed(messages):
        if isinstance(msg, dict) and msg.get("role") == "user":
            content = msg.get("content")
            if isinstance(content, str):
                user_question = content
                break
            if isinstance(content, list):
                user_question = "\n".join(
                    p.get("text", "") for p in content
                    if isinstance(p, dict) and isinstance(p.get("text"), str)
                )
                break

    lines = [f"User question:\n{user_question[:max_chars]}\n"]
    for i, cand in enumerate(candidates, start=1):
        draft = (cand.get("text") or "").strip() or "[empty]"
        if len(draft) > max_chars:
            draft = draft[:max_chars] + " …[truncated]"
        lines.append(f"\n<CANDIDATE {i}>\n{draft}\n</CANDIDATE {i}>")
    lines.append(f"\nPick the best candidate (1..{len(candidates)}). Respond ONLY with JSON.")

    return [
        {"role": "system", "content": _SCORER_SYSTEM},
        {"role": "user", "content": "\n".join(lines)},
    ]


def _parse_best_of_n(text: str, n: int) -> int:
    """Extract the chosen candidate number from the scorer's response. Returns 0 on failure."""
    # Try JSON parse.
    try:
        data = json.loads(text)
        best = int(data.get("best", 0))
        if 1 <= best <= n:
            return best
    except (json.JSONDecodeError, TypeError, ValueError):
        pass
    # Regex fallback: find "best": N or best=N or just a bare number.
    m = re.search(r'"?best"?\s*[:=]\s*(\d+)', text, re.IGNORECASE)
    if m:
        val = int(m.group(1))
        if 1 <= val <= n:
            return val
    # Last resort: first bare number 1..N in the text.
    m = re.search(r'\b([1-9])\b', text)
    if m:
        val = int(m.group(1))
        if 1 <= val <= n:
            return val
    return 0


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


def _synthesize_sse(text: str, model: str):
    """Wrap a plain text response as an OpenAI-shaped SSE stream (for best-of-N streaming).

    Returns an async generator that yields SSE-formatted bytes.
    """
    import time

    ts = int(time.time())

    async def gen():
        # chunk: content delta
        chunk = {
            "id": "chatcmpl-ensemble",
            "object": "chat.completion.chunk",
            "created": ts,
            "model": model,
            "choices": [{"index": 0, "delta": {"role": "assistant", "content": text}, "finish_reason": None}],
        }
        yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"

        # final chunk with finish_reason
        final = {
            "id": "chatcmpl-ensemble",
            "object": "chat.completion.chunk",
            "created": ts,
            "model": model,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        }
        yield f"data: {json.dumps(final, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"

    return gen()


async def run_ensemble(
    body: dict,
    settings: "Settings",
    upstream: "UpstreamClient",
    decision: "RoutingDecision",
    stream: bool,
    on_stream_done: "StreamDoneCallback | None" = None,
):
    """Run ensemble (b5_fusion or best_of_n). Returns JSONResponse / StreamingResponse,
    or None to let the caller fall back to single-model passthrough.
    """
    messages = body.get("messages") or []
    if not messages:
        return None

    max_tokens = body.get("max_tokens")
    anchor = decision.model or body.get("model") or ""
    proposers = _proposer_models(settings, anchor)
    aggregator = settings.ensemble_aggregator or anchor
    mode = settings.ensemble_mode

    if len(proposers) < 2:
        logger.info("ensemble: fewer than 2 proposers (%s); single passthrough", proposers)
        return None

    # ── Streaming with heartbeat: return StreamingResponse immediately ──
    # The generator runs proposers inline, sends heartbeats while waiting,
    # then either fuses (aggregator stream) or synthesizes (best-of-N).
    if stream:
        return await _run_ensemble_stream(
            body, settings, upstream, decision, messages, proposers,
            aggregator, anchor, mode, max_tokens, on_stream_done,
        )

    # ── Non-streaming path ──
    # Phase 1: proposers in parallel (non-stream).
    results = await asyncio.gather(*[
        _run_one_proposer(
            upstream.client, m, messages, max_tokens, settings.ensemble_proposer_timeout
        )
        for m in proposers
    ])
    successful = [r for r in results if r["ok"]]
    logger.info(
        "ensemble: %d/%d proposers ok (mode=%s, aggregator=%s)",
        len(successful), len(results), mode, aggregator,
    )

    if len(successful) < settings.ensemble_min_successful:
        logger.warning(
            "ensemble: quorum not met (%d<%d); fallback single %s",
            len(successful), settings.ensemble_min_successful, anchor,
        )
        fb_body = dict(body)
        fb_body["model"] = anchor
        return await upstream.chat_completions(fb_body, stream=stream, on_stream_done=on_stream_done)

    # Phase 2: mode dispatch.
    if mode == "best_of_n":
        return await _run_best_of_n_nonstream(
            upstream, settings, messages, successful, aggregator, results, on_stream_done
        )
    return await _run_b5_fusion_nonstream(
        upstream, settings, messages, successful, aggregator, results, max_tokens
    )


async def _run_b5_fusion_nonstream(
    upstream, settings, messages, successful, aggregator, results, max_tokens,
):
    """B5 fusion non-streaming: aggregator fuses drafts."""
    agg_messages = _build_aggregator_messages(
        messages, successful, settings.ensemble_candidate_max_chars
    )
    agg_body: dict = {"model": aggregator, "messages": agg_messages, "stream": False}
    if max_tokens is not None:
        agg_body["max_tokens"] = max_tokens

    try:
        resp = await upstream.client.post(
            "/chat/completions", json=agg_body,
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


async def _run_best_of_n_nonstream(
    upstream, settings, messages, successful, aggregator, results, on_stream_done,
):
    """best-of-N non-streaming: scorer picks best draft, returned directly."""
    scorer_model = settings.ensemble_scorer_model or aggregator
    scorer_messages = _build_scorer_messages(
        messages, successful, settings.ensemble_candidate_max_chars
    )
    try:
        resp = await upstream.client.post(
            "/chat/completions",
            json={"model": scorer_model, "messages": scorer_messages, "stream": False, "temperature": 0},
            timeout=settings.ensemble_aggregator_timeout,
        )
    except Exception as exc:
        raise RelayError(
            502, {"error": {"message": f"scorer unreachable: {exc}", "type": "upstream_error"}}
        ) from exc
    if resp.status_code >= 400:
        raise RelayError(resp.status_code, _error_body(resp))

    scorer_data = resp.json()
    try:
        scorer_text = scorer_data["choices"][0]["message"]["content"] or ""
    except (KeyError, IndexError, TypeError):
        scorer_text = ""

    best_idx = _parse_best_of_n(scorer_text, len(successful))
    if best_idx == 0:
        # Scorer failed to pick → fall back to first candidate.
        logger.warning("ensemble: best_of_n scorer parse failed; using candidate 1")
        best_idx = 1

    chosen = successful[best_idx - 1]
    logger.info("ensemble: best_of_n picked candidate %d/%d (%s)", best_idx, len(successful), chosen.get("model"))

    # Build an OpenAI-shaped response with the chosen draft.
    data = {
        "id": "chatcmpl-ensemble-bon",
        "object": "chat.completion",
        "model": chosen.get("model", "ensemble"),
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": chosen.get("text", "")},
            "finish_reason": "stop",
        }],
        "usage": _sum_usage(results, scorer_data.get("usage") or {}),
    }
    return JSONResponse(content=data, status_code=200)


async def _run_ensemble_stream(
    body, settings, upstream, decision, messages, proposers,
    aggregator, anchor, mode, max_tokens, on_stream_done,
):
    """Streaming ensemble with SSE heartbeats during the proposer phase.

    Returns a StreamingResponse immediately. The generator:
    1. Sends comment-frame heartbeats while proposers run.
    2. On quorum: dispatches to b5_fusion (aggregator stream) or best_of_n
       (synthesized SSE from chosen draft).
    3. On quorum failure: falls back to single-model streaming.
    """
    import time

    proposer_task: asyncio.Task | None = None
    captured: dict = {
        "usage": None, "finish_reason": None,
        "upstream_status": 200, "stream_completed": False,
    }

    async def gen():
        nonlocal proposer_task
        # ── Phase 1: proposers + heartbeat ──
        # Send an initial comment so the client knows we're alive.
        yield b": ensemble: starting proposer phase\n\n"
        last_heartbeat = time.monotonic()

        proposer_task = asyncio.ensure_future(asyncio.gather(*[
            _run_one_proposer(
                upstream.client, m, messages, max_tokens,
                settings.ensemble_proposer_timeout,
            )
            for m in proposers
        ]))

        # Heartbeat loop: yield comment frames while proposers run.
        while not proposer_task.done():
            await asyncio.sleep(0.5)
            now = time.monotonic()
            if now - last_heartbeat >= _HEARTBEAT_INTERVAL:
                yield b": heartbeat\n\n"
                last_heartbeat = now

        # Proposers done. Retrieve results.
        try:
            results = proposer_task.result()
        except Exception as exc:
            logger.warning("ensemble: proposer phase failed (%s); fallback single", exc)
            # Fallback: single-model stream.
            async for chunk in _fallback_single_stream(upstream, body, anchor, on_stream_done, captured):
                yield chunk
            return

        successful = [r for r in results if r["ok"]]
        logger.info(
            "ensemble: %d/%d proposers ok (mode=%s, stream)",
            len(successful), len(results), mode,
        )

        if len(successful) < settings.ensemble_min_successful:
            logger.warning(
                "ensemble: quorum not met (%d<%d); fallback single %s",
                len(successful), settings.ensemble_min_successful, anchor,
            )
            async for chunk in _fallback_single_stream(upstream, body, anchor, on_stream_done, captured):
                yield chunk
            return

        # ── Phase 2: mode dispatch ──
        if mode == "best_of_n":
            # Scorer picks best draft (non-stream call), then synthesize SSE.
            async for chunk in _best_of_n_stream(
                upstream, settings, messages, successful, aggregator, results, captured
            ):
                yield chunk
        else:
            # B5 fusion: stream the aggregator's output.
            async for chunk in _b5_fusion_stream(
                upstream, settings, messages, successful, aggregator, results, max_tokens, captured
            ):
                yield chunk

        captured["stream_completed"] = True
        # Fire-and-forget outcome callback.
        if on_stream_done is not None:
            asyncio.create_task(on_stream_done(captured))

    return StreamingResponse(gen(), media_type="text/event-stream", headers=_SSE_HEADERS)


async def _b5_fusion_stream(
    upstream, settings, messages, successful, aggregator, results, max_tokens, captured,
):
    """B5 fusion streaming: open aggregator stream, forward SSE verbatim."""
    agg_messages = _build_aggregator_messages(
        messages, successful, settings.ensemble_candidate_max_chars
    )
    agg_body: dict = {"model": aggregator, "messages": agg_messages, "stream": True}
    if max_tokens is not None:
        agg_body["max_tokens"] = max_tokens

    # Open the aggregator stream.
    request = upstream.client.build_request("POST", "/chat/completions", json=agg_body)
    try:
        resp = await upstream.client.send(request, stream=True)
    except Exception as exc:
        raise RelayError(
            502, {"error": {"message": f"aggregator unreachable: {exc}", "type": "upstream_error"}}
        ) from exc
    if resp.status_code >= 400:
        try:
            await resp.aread()
            raise RelayError(resp.status_code, _error_body(resp))
        finally:
            await resp.aclose()

    buffer = b""
    try:
        async for chunk in resp.aiter_bytes():
            if chunk:
                buffer += chunk
                while b"\n\n" in buffer:
                    event, buffer = buffer.split(b"\n\n", 1)
                    _parse_sse_for_captured(event, captured)
                yield chunk
        if buffer:
            _parse_sse_for_captured(buffer, captured)
    finally:
        await resp.aclose()


async def _best_of_n_stream(
    upstream, settings, messages, successful, aggregator, results, captured,
):
    """best-of-N streaming: scorer picks best draft (non-stream), synthesize SSE."""
    from .upstream import _parse_sse_event  # reuse the SSE parser for outcome capture

    scorer_model = settings.ensemble_scorer_model or aggregator
    scorer_messages = _build_scorer_messages(
        messages, successful, settings.ensemble_candidate_max_chars
    )
    try:
        resp = await upstream.client.post(
            "/chat/completions",
            json={"model": scorer_model, "messages": scorer_messages, "stream": False, "temperature": 0},
            timeout=settings.ensemble_aggregator_timeout,
        )
    except Exception as exc:
        raise RelayError(
            502, {"error": {"message": f"scorer unreachable: {exc}", "type": "upstream_error"}}
        ) from exc
    if resp.status_code >= 400:
        raise RelayError(resp.status_code, _error_body(resp))

    scorer_data = resp.json()
    try:
        scorer_text = scorer_data["choices"][0]["message"]["content"] or ""
    except (KeyError, IndexError, TypeError):
        scorer_text = ""

    best_idx = _parse_best_of_n(scorer_text, len(successful))
    if best_idx == 0:
        logger.warning("ensemble: best_of_n scorer parse failed; using candidate 1")
        best_idx = 1

    chosen = successful[best_idx - 1]
    logger.info("ensemble: best_of_n picked candidate %d/%d (%s, stream)", best_idx, len(successful), chosen.get("model"))

    # Populate captured for outcome tracking.
    captured["usage"] = _sum_usage(results, scorer_data.get("usage") or {})
    captured["finish_reason"] = "stop"

    # Synthesize SSE from the chosen draft text.
    gen = _synthesize_sse(chosen.get("text", ""), chosen.get("model", "ensemble"))
    async for chunk in gen:
        yield chunk.encode("utf-8") if isinstance(chunk, str) else chunk


async def _fallback_single_stream(upstream, body, anchor, on_stream_done, captured):
    """Fallback to single-model streaming when quorum fails."""
    fb_body = dict(body)
    fb_body["model"] = anchor

    # Open the stream.
    request = upstream.client.build_request("POST", "/chat/completions", json=fb_body)
    try:
        resp = await upstream.client.send(request, stream=True)
    except Exception:
        return  # caller will handle
    if resp.status_code >= 400:
        try:
            await resp.aread()
        finally:
            await resp.aclose()
        return

    buffer = b""
    try:
        async for chunk in resp.aiter_bytes():
            if chunk:
                buffer += chunk
                while b"\n\n" in buffer:
                    event, buffer = buffer.split(b"\n\n", 1)
                    _parse_sse_for_captured(event, captured)
                yield chunk
        if buffer:
            _parse_sse_for_captured(buffer, captured)
    finally:
        await resp.aclose()
        captured["stream_completed"] = True
        if on_stream_done is not None:
            asyncio.create_task(on_stream_done(captured))


def _parse_sse_for_captured(event_bytes: bytes, captured: dict) -> None:
    """Parse one SSE event block for usage/finish_reason (lightweight version)."""
    from .upstream import _parse_sse_event
    _parse_sse_event(event_bytes, captured)
