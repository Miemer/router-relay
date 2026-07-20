"""Format adapters between OpenAI and Anthropic chat-message shapes.

Ensemble is implemented in OpenAI terms: proposers and the aggregator call the
upstream ``/chat/completions`` endpoint, and the synthesized response is
OpenAI-shaped. To expose ensemble on the Anthropic ``/v1/messages`` path, the
relay translates the request *in* (Anthropic → OpenAI) and the response *out*
(OpenAI → Anthropic).

Non-tool conversations only — ensemble never fires when ``tools`` are present,
so content blocks are assumed to be plain ``text`` (no tool_use/tool_result
blocks to translate).

Reference shapes
----------------
OpenAI request:  {"model", "messages":[{"role","content":str}], "stream", "max_tokens", ...}
Anthropic request: {"model", "messages":[{"role","content":[blocks]}], "system", "max_tokens", "stream", ...}

OpenAI response:  {"id","object":"chat.completion","model","choices":[{"message":{"role","content"},"finish_reason"}],"usage":{prompt/completion/total_tokens}}
Anthropic response: {"id","type":"message","role","model","content":[{"type":"text","text"}],"stop_reason","usage":{input/output_tokens}}
"""

from __future__ import annotations

import json

from fastapi.responses import StreamingResponse

_SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
}


# ── request: Anthropic → OpenAI ──

def _content_to_text(content) -> str:
    """Flatten an Anthropic ``content`` (str or list of blocks) to plain text."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            b.get("text", "")
            for b in content
            if isinstance(b, dict) and b.get("type") == "text" and isinstance(b.get("text"), str)
        )
    return str(content)


# OpenAI-recognized generation params we pass through from the Anthropic request.
_PASSTHROUGH_KEYS = ("temperature", "top_p", "max_tokens", "stop", "stream", "top_k")


def anthropic_request_to_openai(body: dict) -> dict:
    """Convert an Anthropic ``/v1/messages`` request body to OpenAI ``/chat/completions`` shape.

    The upstream proposer/aggregator calls are always OpenAI-shaped, so the
    ensemble needs the conversation in that form. Assumes a non-tool exchange.
    """
    messages: list[dict] = []

    system = body.get("system")
    if system:
        sys_text = system if isinstance(system, str) else _content_to_text(system)
        if sys_text:
            messages.append({"role": "system", "content": sys_text})

    for msg in body.get("messages", []):
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        if role not in ("user", "assistant"):
            continue
        text = _content_to_text(msg.get("content"))
        messages.append({"role": role, "content": text})

    out: dict = {"model": body.get("model"), "messages": messages}
    for key in _PASSTHROUGH_KEYS:
        if key in body:
            out[key] = body[key]
    return out


# ── response: OpenAI → Anthropic (non-streaming) ──

def _finish_to_stop_reason(finish: str | None) -> str:
    return {
        "stop": "end_turn",
        "length": "max_tokens",
        "tool_calls": "tool_use",
        "content_filter": "content_filter",
    }.get(finish, "end_turn")


def openai_completion_to_anthropic(data: dict, model: str | None = None) -> dict:
    """Convert a single (non-streaming) OpenAI chat.completion into an Anthropic message."""
    choices = data.get("choices") or []
    choice = choices[0] if choices else {}
    message = choice.get("message", {}) if isinstance(choice, dict) else {}
    text = message.get("content") or "" if isinstance(message, dict) else ""
    finish = choice.get("finish_reason") if isinstance(choice, dict) else None
    usage = data.get("usage") or {}

    return {
        "id": data.get("id") or "msg_ensemble",
        "type": "message",
        "role": "assistant",
        "model": model or data.get("model") or "",
        "content": [{"type": "text", "text": text}],
        "stop_reason": _finish_to_stop_reason(finish),
        "stop_sequence": None,
        "usage": {
            "input_tokens": int(usage.get("prompt_tokens") or 0),
            "output_tokens": int(usage.get("completion_tokens") or 0),
        },
    }


# ── response: OpenAI → Anthropic (streaming) ──

def _anthropic_event(event: str, data: dict) -> bytes:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n".encode("utf-8")


async def openai_stream_to_anthropic(openai_resp: StreamingResponse, model: str) -> StreamingResponse:
    """Wrap an OpenAI SSE stream as an Anthropic SSE stream.

    Parses each incoming ``data: {…}`` chunk, forwards text deltas as
    ``content_block_delta`` events, then closes with the standard Anthropic
    trailing events (``content_block_stop`` / ``message_delta`` / ``message_stop``).
    Comment frames (``: heartbeat``) and ``[DONE]`` are ignored.
    """
    orig = openai_resp.body_iterator

    async def gen():
        yield _anthropic_event("message_start", {
            "type": "message_start",
            "message": {
                "id": "msg_ensemble",
                "type": "message",
                "role": "assistant",
                "model": model,
                "content": [],
                "usage": {"input_tokens": 0, "output_tokens": 0},
            },
        })
        yield _anthropic_event("content_block_start", {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "text", "text": ""},
        })

        finish_reason = None
        saw_content = False
        buffer = b""

        async for chunk in orig:
            if isinstance(chunk, str):
                chunk = chunk.encode("utf-8")
            buffer += chunk
            while b"\n\n" in buffer:
                event, buffer = buffer.split(b"\n\n", 1)
                for line in event.split(b"\n"):
                    if not line.startswith(b"data: "):
                        continue
                    payload = line[len(b"data: "):].strip()
                    if payload == b"[DONE]":
                        continue
                    try:
                        obj = json.loads(payload)
                    except json.JSONDecodeError:
                        continue
                    choices = obj.get("choices") or []
                    if not choices:
                        continue
                    delta = choices[0].get("delta") or {}
                    text = delta.get("content") or ""
                    if text:
                        saw_content = True
                        yield _anthropic_event("content_block_delta", {
                            "type": "content_block_delta",
                            "index": 0,
                            "delta": {"type": "text_delta", "text": text},
                        })
                    fr = choices[0].get("finish_reason")
                    if fr:
                        finish_reason = fr

        if saw_content:
            yield _anthropic_event("content_block_stop", {
                "type": "content_block_stop",
                "index": 0,
            })
        yield _anthropic_event("message_delta", {
            "type": "message_delta",
            "delta": {"stop_reason": _finish_to_stop_reason(finish_reason), "stop_sequence": None},
            "usage": {"output_tokens": 0},
        })
        yield _anthropic_event("message_stop", {"type": "message_stop"})

    return StreamingResponse(gen(), media_type="text/event-stream", headers=_SSE_HEADERS)
