"""Upstream OpenAI-compatible client: passthrough for chat completions + models."""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator
from typing import Awaitable, Callable

import httpx
from fastapi.responses import JSONResponse, StreamingResponse

from .config import Settings
from .errors import RelayError

logger = logging.getLogger("relay.upstream")

# Headers an SSE stream must send so proxies/clients don't buffer.
_SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",  # disable nginx buffering if proxied
}

# Type for the streaming-completion callback. Receives a dict with
# usage/finish_reason/upstream_status/stream_completed populated by the SSE
# parser. Fire-and-forget (scheduled via asyncio.create_task in the generator's
# finally block, after the last byte has been sent to the client).
StreamDoneCallback = Callable[[dict], Awaitable[None]]


def _parse_sse_event(event_bytes: bytes, captured: dict) -> None:
    """Parse one SSE event block (delimited by ``\\n\\n``) for usage/finish_reason.

    Handles both OpenAI (``/chat/completions``) and Anthropic (``/messages``)
    streaming shapes. Mutates ``captured`` in place.
    """
    for line in event_bytes.split(b"\n"):
        line = line.strip()
        if not line.startswith(b"data:"):
            continue
        data_str = line[5:].strip()  # bytes after "data:"
        if not data_str or data_str == b"[DONE]":
            continue
        try:
            data = json.loads(data_str)
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        _extract_sse_metadata(data, captured)


def _extract_sse_metadata(data: dict, captured: dict) -> None:
    """Extract usage + finish_reason from one SSE ``data:`` payload.

    OpenAI shape: ``{"choices": [{"finish_reason": "stop"}], "usage": {...}}``
    Anthropic shape: ``{"type": "message_start|message_delta|...", ...}``
    """
    # ── OpenAI format ──
    if "choices" in data:
        try:
            fr = data["choices"][0].get("finish_reason")
            if fr:
                captured["finish_reason"] = fr
        except (KeyError, IndexError, TypeError):
            pass
        usage = data.get("usage")
        if usage:
            captured["usage"] = usage
        return
    # ── Anthropic format ──
    msg_type = data.get("type", "")
    if msg_type == "message_start":
        msg = data.get("message") or {}
        u = msg.get("usage") or {}
        if u:
            # `captured["usage"]` starts as None; promote to a dict on first write.
            if not isinstance(captured.get("usage"), dict):
                captured["usage"] = {}
            captured["usage"]["prompt_tokens"] = u.get("input_tokens", 0)
            captured["usage"]["total_tokens"] = u.get("input_tokens", 0)
    elif msg_type == "message_delta":
        u = data.get("usage") or {}
        if u:
            if not isinstance(captured.get("usage"), dict):
                captured["usage"] = {}
            captured["usage"]["completion_tokens"] = u.get("output_tokens", 0)
            captured["usage"]["total_tokens"] = (
                captured["usage"].get("total_tokens", 0) + u.get("output_tokens", 0)
            )
        delta = data.get("delta") or {}
        stop = delta.get("stop_reason")
        if stop:
            captured["finish_reason"] = stop


class UpstreamClient:
    """A long-lived httpx client wrapping the configured upstream provider."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client: httpx.AsyncClient | None = None

    def open(self) -> None:
        headers = {"Accept": "application/json"}
        if self._settings.upstream_api_key:
            headers["Authorization"] = f"Bearer {self._settings.upstream_api_key}"
        if self._settings.upstream_organization:
            headers["OpenAI-Organization"] = self._settings.upstream_organization
        self._client = httpx.AsyncClient(
            base_url=self._settings.upstream_base_url.rstrip("/"),
            headers=headers,
            timeout=httpx.Timeout(self._settings.upstream_timeout, connect=10.0),
        )

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None:
            raise RuntimeError("upstream client is not open")
        return self._client

    async def list_models(self) -> dict:
        try:
            resp = await self.client.get("/models")
        except httpx.RequestError as exc:
            raise RelayError(
                502,
                {"error": {"message": f"upstream unreachable: {exc}", "type": "upstream_error"}},
            ) from exc
        self._raise_for_status(resp)
        return resp.json()

    async def chat_completions(
        self,
        body: dict,
        stream: bool,
        path: str = "/chat/completions",
        on_stream_done: StreamDoneCallback | None = None,
    ):
        """Forward a completion request. Returns a JSONResponse or StreamingResponse.

        ``path`` is the upstream endpoint path: ``/chat/completions`` (OpenAI) or
        ``/messages`` (Anthropic). The body shape must match the path — the caller
        is responsible for routing the right protocol to the right path.

        ``on_stream_done`` (streaming only) is a fire-and-forget callback fired
        after the stream completes, receiving a dict with ``usage``,
        ``finish_reason``, ``upstream_status``, and ``stream_completed`` populated
        by the SSE parser. Use it to capture the full streaming outcome (which
        is not known when the StreamingResponse object is first returned).
        """
        if stream:
            return await self._chat_stream(body, path, on_stream_done)
        try:
            resp = await self.client.post(path, json=body)
        except httpx.RequestError as exc:
            raise RelayError(
                502,
                {"error": {"message": f"upstream unreachable: {exc}", "type": "upstream_error"}},
            ) from exc
        self._raise_for_status(resp)
        return JSONResponse(content=resp.json(), status_code=resp.status_code)

    async def _chat_stream(
        self,
        body: dict,
        path: str = "/chat/completions",
        on_stream_done: StreamDoneCallback | None = None,
    ) -> StreamingResponse:
        # Open the upstream stream first so we can surface a clean error before
        # beginning our own SSE response.
        request = self._client.build_request("POST", path, json=body)
        try:
            resp = await self._client.send(request, stream=True)
        except httpx.RequestError as exc:
            raise RelayError(
                502,
                {"error": {"message": f"upstream unreachable: {exc}", "type": "upstream_error"}},
            ) from exc
        if resp.status_code >= 400:
            try:
                await resp.aread()
                self._raise_for_status(resp)
            finally:
                await resp.aclose()

        captured: dict = {
            "usage": None,
            "finish_reason": None,
            "upstream_status": resp.status_code,
            "stream_completed": False,
        }

        async def gen() -> AsyncIterator[bytes]:
            # Raw byte passthrough: upstream is already text/event-stream, so we
            # forward each chunk verbatim. This is the lowest-latency path and the
            # seam where P1 routing / P2 ensemble fusion will hook in.
            #
            # When on_stream_done is set, we also parse SSE events from the byte
            # stream to extract usage/finish_reason for outcome capture. The
            # parsing is purely additive — the raw bytes are still yielded
            # unchanged, so the client sees exactly what the upstream sent.
            if on_stream_done is None:
                # Fast path: no capture, zero parsing overhead.
                try:
                    async for chunk in resp.aiter_bytes():
                        if chunk:
                            yield chunk
                finally:
                    await resp.aclose()
                return

            # Capture path: parse SSE events while forwarding raw bytes.
            buffer = b""
            try:
                async for chunk in resp.aiter_bytes():
                    if chunk:
                        buffer += chunk
                        # Parse complete SSE events (delimited by \n\n) from the
                        # buffer. The remainder (incomplete event) stays buffered.
                        while b"\n\n" in buffer:
                            event, buffer = buffer.split(b"\n\n", 1)
                            _parse_sse_event(event, captured)
                        yield chunk
                # Parse any trailing event that wasn't \n\n-terminated.
                if buffer:
                    _parse_sse_event(buffer, captured)
                captured["stream_completed"] = True
            finally:
                await resp.aclose()
                # Fire-and-forget: schedule the outcome callback off the request
                # loop so it never delays closing the client connection.
                if on_stream_done is not None:
                    asyncio.create_task(on_stream_done(captured))

        return StreamingResponse(gen(), media_type="text/event-stream", headers=_SSE_HEADERS)

    @staticmethod
    def _raise_for_status(resp: httpx.Response) -> None:
        """Convert an upstream failure into an OpenAI-shaped RelayError."""
        if resp.status_code < 400:
            return
        try:
            data = resp.json()
        except Exception:
            data = None
        if isinstance(data, dict) and "error" in data:
            body = data
        elif isinstance(data, dict):
            body = {"error": {"message": str(data), "type": "upstream_error"}}
        else:
            body = {
                "error": {"message": resp.text or "upstream error", "type": "upstream_error"}
            }
        raise RelayError(resp.status_code, body)
