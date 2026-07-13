"""Entry point: `python -m relay` or the `router-relay` console script."""

from __future__ import annotations

import uvicorn

from .config import get_settings


def main() -> None:
    settings = get_settings()
    uvicorn.run(
        "relay.app:app",
        host=settings.listen_host,
        port=settings.listen_port,
        log_level=settings.log_level,
    )


if __name__ == "__main__":
    main()
