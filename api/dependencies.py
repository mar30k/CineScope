"""
Shared FastAPI dependencies.

Kept in their own module so both ``api.main`` (which owns the lifespan) and
``api.routes`` (which consumes the model) can import the accessor without a
circular import.
"""

from __future__ import annotations

from fastapi import Request

from src.models.recommender import MovieRecommender


def get_model(request: Request) -> MovieRecommender:
    """Return the process-wide model loaded once during application startup.

    The instance lives on ``app.state.model`` and is created by the lifespan
    handler; this dependency never loads or rebuilds anything.
    """
    model = getattr(request.app.state, "model", None)
    if model is None:
        raise RuntimeError(
            "The recommendation model is not loaded. This indicates the "
            "application lifespan did not run."
        )
    return model
