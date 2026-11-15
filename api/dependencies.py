"""
Shared FastAPI dependencies.

Kept in their own module so both ``api.main`` (which owns the lifespan) and
``api.routes`` (which consumes the model) can import the accessor without a
circular import.
"""

from __future__ import annotations

from fastapi import Depends, Request

from api.movie_cast import MovieCastService
from api.search import MovieSearchIndex
from api.tmdb_images import TmdbImageService
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


def get_image_service(request: Request) -> TmdbImageService | None:
    """Return the optional TMDB image service created during startup.

    ``None`` is expected when ``TMDB_API_KEY`` is absent; routes treat that as
    "serve the catalogue without artwork" rather than an error.
    """
    return getattr(request.app.state, "image_service", None)


def get_cast_service(request: Request) -> MovieCastService | None:
    """Return the credits lookup loaded during startup, when available."""
    return getattr(request.app.state, "cast_service", None)


def get_search_index(
    request: Request,
    model: MovieRecommender = Depends(get_model),
) -> MovieSearchIndex:
    """Return the process-wide title search index.

    Built once during startup. The lazy fallback exists so a host that skips
    the lifespan (for example a test that overrides the model) still gets a
    working index; it is cached on ``app.state`` so it is still built once.
    ``get_model`` arrives through ``Depends`` so dependency overrides apply.
    """
    index = getattr(request.app.state, "search_index", None)
    if index is None:
        index = MovieSearchIndex.build(model.movies)
        request.app.state.search_index = index
    return index
