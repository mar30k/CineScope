"""
Host-friendly entry point: ``python -m api``.

Render (and most PaaS hosts) inject the port the service must listen on via
``PORT``; the development command ``uvicorn api.main:app --port 8000`` would
ignore it and the health check would fail.

A single worker is used deliberately. The recommender is loaded once per
process and holds ~190 MB, so each extra worker would multiply memory for a
workload this small; concurrency is handled by the async server loop and the
per-request cost is a single sparse matrix pass.
"""

from __future__ import annotations

import os

import uvicorn

DEFAULT_PORT = 8000
DEFAULT_HOST = "0.0.0.0"


def resolve_port(value: str | None = None) -> int:
    """Port from ``PORT`` (host default) or ``DEFAULT_PORT`` for development."""
    raw = (value if value is not None else os.environ.get("PORT", "")).strip()
    if not raw:
        return DEFAULT_PORT
    try:
        port = int(raw)
    except ValueError:
        raise SystemExit(f"PORT must be an integer, got {raw!r}") from None
    if not 1 <= port <= 65535:
        raise SystemExit(f"PORT must be between 1 and 65535, got {port}")
    return port


def main() -> None:
    uvicorn.run(
        "api.main:app",
        host=DEFAULT_HOST,
        port=resolve_port(),
        workers=1,
        log_level="info",
    )


if __name__ == "__main__":
    main()
