"""Upstream OpenAI-compatible client: passthrough for chat completions + models."""

from __future__ import annotations

from collections.abc import AsyncIterator

import httpx
from fastapi.responses import JSONResponse, StreamingResponse

from .config import Settings
from .errors import RelayError

# Headers an SSE stream must send so proxies/clients don't buffer.
_SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",  # disable nginx buffering if proxied
}


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

    async def chat_completions(self, body: dict, stream: bool, path: str = "/chat/completions"):
        """Forward a completion request. Returns a JSONResponse or StreamingResponse.

        ``path`` is the upstream endpoint path: ``/chat/completions`` (OpenAI) or
        ``/messages`` (Anthropic). The body shape must match the path — the caller
        is responsible for routing the right protocol to the right path.
        """
        if stream:
            return await self._chat_stream(body, path)
        try:
            resp = await self.client.post(path, json=body)
        except httpx.RequestError as exc:
            raise RelayError(
                502,
                {"error": {"message": f"upstream unreachable: {exc}", "type": "upstream_error"}},
            ) from exc
        self._raise_for_status(resp)
        return JSONResponse(content=resp.json(), status_code=resp.status_code)

    async def _chat_stream(self, body: dict, path: str = "/chat/completions") -> StreamingResponse:
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

        async def gen() -> AsyncIterator[bytes]:
            # Raw byte passthrough: upstream is already text/event-stream, so we
            # forward each chunk verbatim. This is the lowest-latency path and the
            # seam where P1 routing / P2 ensemble fusion will hook in.
            try:
                async for chunk in resp.aiter_bytes():
                    if chunk:
                        yield chunk
            finally:
                await resp.aclose()

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
