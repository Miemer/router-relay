"""Relay error carrying an OpenAI-shaped error body."""

from __future__ import annotations


class RelayError(Exception):
    """Raise to short-circuit a request with an OpenAI-shaped error envelope.

    ``body`` should be ``{"error": {"message": ..., "type": ...}}`` to match the
    OpenAI error schema that opencode/clients expect.
    """

    def __init__(self, status_code: int, body: dict) -> None:
        self.status_code = status_code
        self.body = body
        super().__init__(f"relay error {status_code}: {body}")
