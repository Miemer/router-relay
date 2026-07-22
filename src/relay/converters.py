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
_PASSTHROUGH_KEYS = ("temperature", "top_p", "max_tokens", "stop", "stream", "top_k",
                     "reasoning_effort", "stream_options")


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

    # ── thinking intensity passthrough (best-effort Anthropic → OpenAI) ──
    # OpenAI-native reasoning keys: forward verbatim when present.
    if "reasoning" in body:
        out["reasoning"] = body["reasoning"]
    # Anthropic `thinking` (budget_tokens) → OpenAI `reasoning_effort`. This is a
    # loose mapping (GLM budget → effort tier); verify against the marketingforce
    # GLM catalog. Only applied when no explicit OpenAI reasoning key is present.
    thinking = body.get("thinking")
    if isinstance(thinking, dict) and "reasoning_effort" not in out:
        budget = thinking.get("budget_tokens")
        if isinstance(budget, int):
            out["reasoning_effort"] = (
                "high" if budget >= 8000 else "medium" if budget >= 2000 else "low"
            )
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

    # Forward the REAL token accounting (verbatim passthrough of upstream usage).
    # OpenAI reports cache hits under `prompt_tokens_details.cached_tokens`;
    # map those to Anthropic's cache_read_input_tokens so ZCode's cache-hit rate
    # is populated instead of being zeroed.
    prompt_tokens = int(usage.get("prompt_tokens") or 0)
    completion_tokens = int(usage.get("completion_tokens") or 0)
    details = usage.get("prompt_tokens_details") or {}
    cache_read = int(details.get("cached_tokens") or 0)
    cache_creation = int(details.get("cache_creation_tokens") or 0)

    return {
        "id": data.get("id") or "msg_ensemble",
        "type": "message",
        "role": "assistant",
        "model": model or data.get("model") or "",
        "content": [{"type": "text", "text": text}],
        "stop_reason": _finish_to_stop_reason(finish),
        "stop_sequence": None,
        "usage": {
            "input_tokens": prompt_tokens,
            "output_tokens": completion_tokens,
            "cache_read_input_tokens": cache_read,
            "cache_creation_input_tokens": cache_creation,
        },
    }


# ── response: OpenAI → Anthropic (streaming) ──

def _anthropic_event(event: str, data: dict) -> bytes:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n".encode("utf-8")


# ── reasoning_content / thinking-block consistency normalization ──
#
# Zhipu GLM (and similar thinking models) require that once an assistant
# message carries ``reasoning_content``, ALL subsequent assistant messages in
# the conversation must carry it too. Per-turn routing can mix thinking
# (glm-5.2) and non-thinking models within a single conversation, producing a
# history where a thinking assistant message is followed by a non-thinking one
# (no ``reasoning_content``). The upstream then rejects with a 400
# "The reasoning_content ... must be passed back to the API".
#
# We cannot synthesize missing reasoning, so the only safe repair is to make the
# history uniformly non-thinking: when the assistant messages are
# reasoning-inconsistent (some have it, some don't), strip ``reasoning_content``
# from ALL of them. A history that is already consistent (all-have or all-lack)
# is left untouched — preserving a pure-thinking conversation's reasoning context.

def strip_reasoning_when_inconsistent(messages: list) -> bool:
    """OpenAI request shape. Mutates ``messages`` in place; returns True if stripped."""
    assistant = [m for m in messages if isinstance(m, dict) and m.get("role") == "assistant"]
    if len(assistant) < 2:
        return False
    has = [bool(m.get("reasoning_content")) for m in assistant]
    if all(has) or not any(has):
        return False  # already consistent — preserve
    changed = False
    for m in assistant:
        if m.pop("reasoning_content", None) is not None:
            changed = True
    return changed


def strip_thinking_when_inconsistent(messages: list) -> bool:
    """Anthropic request shape: ``thinking`` blocks inside assistant ``content``
    lists. Mutates ``messages`` in place; returns True if stripped."""
    assistant = [m for m in messages if isinstance(m, dict) and m.get("role") == "assistant"]
    if len(assistant) < 2:
        return False

    def has_thinking(m: dict) -> bool:
        content = m.get("content")
        return isinstance(content, list) and any(
            isinstance(b, dict) and b.get("type") == "thinking" for b in content
        )

    flags = [has_thinking(m) for m in assistant]
    if all(flags) or not any(flags):
        return False  # already consistent — preserve
    changed = False
    for m in assistant:
        content = m.get("content")
        if isinstance(content, list):
            kept = [b for b in content if not (isinstance(b, dict) and b.get("type") == "thinking")]
            if len(kept) != len(content):
                m["content"] = kept
                changed = True
    return changed


def strip_all_thinking(messages: list, request_format: str) -> bool:
    """Non-thinking target: remove ALL thinking from the assistant history,
    regardless of consistency. Handles both Anthropic (thinking blocks) and
    OpenAI (reasoning_content) shapes. Mutates ``messages`` in place.
    """
    changed = False
    for m in messages:
        if not isinstance(m, dict) or m.get("role") != "assistant":
            continue
        if request_format == "anthropic":
            content = m.get("content")
            if isinstance(content, list):
                kept = [b for b in content if not (isinstance(b, dict) and b.get("type") == "thinking")]
                if len(kept) != len(content):
                    m["content"] = kept
                    changed = True
        else:
            if m.pop("reasoning_content", None) is not None:
                changed = True
    return changed


def strip_top_level_thinking(body: dict, request_format: str) -> bool:
    """Remove the request-level thinking params (intensity / budget) so a
    non-thinking target doesn't choke on them. Anthropic uses ``thinking``;
    OpenAI uses ``reasoning`` / ``reasoning_effort``. Mutates ``body`` in place.
    """
    keys = ("thinking",) if request_format == "anthropic" else ("reasoning", "reasoning_effort")
    changed = False
    for k in keys:
        if k in body:
            del body[k]
            changed = True
    return changed


def normalize_thinking_for_target(body: dict, request_format: str, target_thinks: bool) -> None:
    """Target-aware thinking normalization, called once per turn after routing
    has overridden ``body["model"]``.

    * ``target_thinks=True``  (executed model is thinking-capable, e.g. glm-5.2 /
      gpt-5.6-terra): keep the top-level thinking param (intensity passes through
      to the upstream) and only repair a *mixed* assistant history — GLM rejects
      a history where some assistant messages carry reasoning_content and others
      don't ("must be passed back"). A consistent history is left untouched so a
      pure-thinking conversation keeps its reasoning context.
    * ``target_thinks=False`` (non-thinking target, e.g. deepseek-*): strip ALL
      thinking content blocks AND the top-level thinking param — a non-thinking
      model rejects either, with a 400.
    """
    msgs = body.get("messages")
    if not isinstance(msgs, list):
        return
    if target_thinks:
        if request_format == "anthropic":
            strip_thinking_when_inconsistent(msgs)
        else:
            strip_reasoning_when_inconsistent(msgs)
    else:
        strip_all_thinking(msgs, request_format)
        strip_top_level_thinking(body, request_format)


async def openai_stream_to_anthropic(openai_resp: StreamingResponse, model: str) -> StreamingResponse:
    """Wrap an OpenAI SSE stream as an Anthropic SSE stream.

    Parses each incoming ``data: {…}`` chunk, forwards text deltas as
    ``content_block_delta`` events, then closes with the standard Anthropic
    trailing events (``content_block_stop`` / ``message_delta`` / ``message_stop``).
    Comment frames (``: heartbeat``) and ``[DONE]`` are ignored.
    """
    orig = openai_resp.body_iterator

    async def gen():
        # Accumulate real token accounting from the OpenAI SSE stream. The
        # `usage` object (incl. prompt_tokens_details.cached_tokens) arrives in
        # the final chunk when the client requested stream usage; forward it to
        # Anthropic's message_delta so cache_read_input_tokens is populated.
        usage_acc = {"input_tokens": 0, "output_tokens": 0,
                     "cache_read": 0, "cache_creation": 0}

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
                    # Capture usage when present (usually the terminal chunk).
                    u = obj.get("usage")
                    if isinstance(u, dict):
                        usage_acc["input_tokens"] = int(u.get("prompt_tokens") or 0)
                        usage_acc["output_tokens"] = int(u.get("completion_tokens") or 0)
                        d = u.get("prompt_tokens_details") or {}
                        usage_acc["cache_read"] = int(d.get("cached_tokens") or 0)
                        usage_acc["cache_creation"] = int(d.get("cache_creation_tokens") or 0)
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
            "usage": {
                "input_tokens": usage_acc["input_tokens"],
                "output_tokens": usage_acc["output_tokens"],
                "cache_read_input_tokens": usage_acc["cache_read"],
                "cache_creation_input_tokens": usage_acc["cache_creation"],
            },
        })
        yield _anthropic_event("message_stop", {"type": "message_stop"})

    return StreamingResponse(gen(), media_type="text/event-stream", headers=_SSE_HEADERS)
