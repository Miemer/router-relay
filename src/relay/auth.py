"""Bearer token auth. Empty ``relay_api_keys`` means an open relay (dev only)."""

from __future__ import annotations

import secrets

from fastapi import Request

from .config import get_settings
from .errors import RelayError


async def verify_token(request: Request) -> str | None:
    """Dependency for /v1/* routes. Returns the validated token or None (open relay)."""
    settings = get_settings()
    keys = settings.relay_api_keys
    if not keys:
        # No tokens configured → open relay. Only acceptable for local dev.
        return None

    auth = request.headers.get("authorization", "")
    if not auth.lower().startswith("bearer "):
        raise RelayError(
            401,
            {"error": {"message": "missing bearer token", "type": "authentication_error"}},
        )
    token = auth.split(" ", 1)[1].strip()
    if not any(secrets.compare_digest(token, key) for key in keys):
        raise RelayError(
            401,
            {"error": {"message": "invalid api key", "type": "authentication_error"}},
        )
    return token
