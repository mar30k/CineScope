"""
HTTP routes.

Recommendation ranking is delegated entirely to ``MovieRecommender.recommend()``
and title search reads the already-loaded catalogue; this module only
translates HTTP input into model calls and the results into the public
response schemas. No scoring, feature or loading logic is duplicated here.
"""

from __future__ import annotations

import logging
import math
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Path, Query

from api.dependencies import (
    get_cast_service,
    get_image_service,
    get_model,
    get_search_index,
)
from api.movie_cast import MovieCastService
from api.search import MovieSearchIndex
from api.schemas import (
    CastMember,
    MovieDetails,
    MovieImage,
    MovieSearchResponse,
    MovieSummary,
    RecommendationItem,
    RecommendationResponse,
)
from api.tmdb_images import ImageAssets, MovieIdentity, TmdbImageService
from src.models.recommender import MovieRecommender

router = APIRouter()
logger = logging.getLogger("cinescope.api.routes")

DEFAULT_N = 10
MAX_N = 50

MIN_QUERY_LENGTH = 2
DEFAULT_SEARCH_LIMIT = 10
MAX_SEARCH_LIMIT = 50


def _finite(value) -> float | None:
    """NaN/inf are not valid JSON, so they become null rather than 'NaN'."""
    if value is None or not math.isfinite(float(value)):
        return None
    return float(value)


def _int_or_none(value) -> int | None:
    """Whole-number field, null when missing or non-finite."""
    number = _finite(value)
    return None if number is None else int(number)


def _text_or_none(value) -> str | None:
    """Trimmed string, null when missing, blank or a pandas NaN placeholder."""
    if value is None:
        return None
    if isinstance(value, float) and not math.isfinite(value):
        return None
    text = str(value).strip()
    return text or None


def _movie_image(assets: ImageAssets | None) -> MovieImage | None:
    """Translate the enrichment layer's value into the public schema."""
    if assets is None:
        return None
    return MovieImage(
        poster_url=assets.poster_url,
        backdrop_url=assets.backdrop_url,
    )


def _images_for(
    service: TmdbImageService | None, movies: list[MovieIdentity]
) -> dict[int, ImageAssets]:
    """Resolve images without allowing TMDB to affect the catalogue request."""
    if service is None or not movies:
        return {}
    try:
        return service.get_images_for_movies(movies)
    except Exception:
        # Artwork is additive; an unexpected enrichment failure must not turn
        # a valid search, details or recommendation response into a 500.
        logger.exception("TMDB image enrichment failed; continuing without artwork.")
        return {}


def _cast_for(
    service: MovieCastService | None, movie_id: int
) -> list[CastMember]:
    """Resolve cast without allowing credits data to break movie details."""
    if service is None:
        return []
    try:
        credits = service.get_cast(movie_id)
    except Exception:
        logger.exception("Cast lookup failed for movie %s; returning no cast.", movie_id)
        return []
    return [
        CastMember(name=credit.name, character=credit.character)
        for credit in credits
    ]


@router.get(
    "/recommendations/{movie_id}",
    response_model=RecommendationResponse,
    tags=["recommendations"],
    summary="Recommend movies similar to a given movie",
    responses={
        404: {"description": "No movie with that id in the catalogue."},
        422: {"description": "Invalid movie_id or n."},
    },
)
def get_recommendations(
    movie_id: Annotated[
        int,
        Path(ge=1, description="CineScope catalogue id of the query movie."),
    ],
    n: Annotated[
        int,
        Query(
            ge=1,
            le=MAX_N,
            description=f"Number of recommendations to return (1-{MAX_N}).",
        ),
    ] = DEFAULT_N,
    model: MovieRecommender = Depends(get_model),
    image_service: TmdbImageService | None = Depends(get_image_service),
) -> RecommendationResponse:
    """Top-N content-similar movies for ``movie_id``, excluding the movie itself."""
    try:
        frame = model.recommend(movie_id=movie_id, n_recommendations=n)
    except KeyError as error:
        # Raised by MovieRecommender.resolve() for an id that is not in the
        # catalogue. Reported as an empty resource, not a server fault.
        raise HTTPException(
            status_code=404,
            detail=f"Movie {movie_id} was not found in the catalogue.",
        ) from error
    except ValueError as error:
        # Reachable when the query movie has no usable feature blocks.
        raise HTTPException(status_code=422, detail=str(error)) from error

    rows = list(frame.itertuples(index=False))
    identities = [
        MovieIdentity(
            movie_id=int(row.movie_id),
            title=str(row.title),
            year=_int_or_none(row.year),
        )
        for row in rows
    ]
    images = _images_for(image_service, identities)

    items = [
        RecommendationItem(
            movie_id=int(row.movie_id),
            title=row.title,
            year=_int_or_none(row.year),
            rank=int(row.rank),
            score=_finite(row.score) or 0.0,
            coverage=_finite(row.coverage) or 0.0,
            agreement=_finite(row.agreement),
            image=_movie_image(images.get(int(row.movie_id))),
        )
        for row in rows
    ]

    return RecommendationResponse(
        movie_id=movie_id,
        count=len(items),
        recommendations=items,
    )


@router.get(
    "/movies/search",
    response_model=MovieSearchResponse,
    tags=["movies"],
    summary="Search the catalogue by title",
    responses={422: {"description": "Empty query or invalid limit."}},
)
def search_movies(
    q: Annotated[
        str,
        Query(
            min_length=MIN_QUERY_LENGTH,
            description=f"Title fragment to match (at least {MIN_QUERY_LENGTH} characters).",
        ),
    ],
    limit: Annotated[
        int,
        Query(
            ge=1,
            le=MAX_SEARCH_LIMIT,
            description=f"Maximum number of matches to return (1-{MAX_SEARCH_LIMIT}).",
        ),
    ] = DEFAULT_SEARCH_LIMIT,
    model: MovieRecommender = Depends(get_model),
    image_service: TmdbImageService | None = Depends(get_image_service),
    search_index: MovieSearchIndex = Depends(get_search_index),
) -> MovieSearchResponse:
    """Ranked title search over the loaded catalogue.

    Matching and ranking live in ``api.search``: exact, prefix, substring,
    order-independent token and fuzzy matches are tried in that order, so a
    typo-tolerant hit can never outrank a direct title match. The index is
    built once at startup, so a request only scans pre-normalized titles.
    Duplicate titles are returned as separate entries because each is a
    distinct catalogue row with its own id.
    """
    term = q.strip()
    if not term:
        # ``min_length`` counts characters, so "   " passes it; a search with
        # no actual term is still a client error.
        raise HTTPException(
            status_code=422,
            detail="Query must contain at least one non-whitespace character.",
        )

    selected = [model.get_movie(movie_id) for movie_id in search_index.search(term, limit)]

    identities = [
        MovieIdentity(
            movie_id=int(info["movie_id"]),
            title=str(info["title"]),
            year=_int_or_none(info["year"]),
        )
        for info in selected
    ]
    images = _images_for(image_service, identities)

    results = [
        MovieSummary(
            movie_id=info["movie_id"],
            title=info["title"],
            year=info["year"],
            genres=info["genres"],
            image=_movie_image(images.get(int(info["movie_id"]))),
        )
        for info in selected
    ]

    return MovieSearchResponse(query=q, count=len(results), results=results)


@router.get(
    "/movies/{movie_id}",
    response_model=MovieDetails,
    tags=["movies"],
    summary="Metadata for one movie",
    responses={404: {"description": "No movie with that id in the catalogue."}},
)
def get_movie_details(
    movie_id: Annotated[
        int, Path(ge=1, description="CineScope catalogue movie id.")
    ],
    model: MovieRecommender = Depends(get_model),
    image_service: TmdbImageService | None = Depends(get_image_service),
    cast_service: MovieCastService | None = Depends(get_cast_service),
) -> MovieDetails:
    """Describe a single catalogue entry.

    Registered after ``/movies/search`` so the static path is matched first;
    a parameterised route declared earlier would swallow it and reject
    "search" as a non-integer id.

    Identity, title, year, genres and director come from the model's public
    ``get_movie()`` accessor; the remaining fields are read from the loaded
    ``model.movies`` frame. Cast credits come from the independently loaded
    ``MovieCastService`` because the model keeps only the top five actor names
    needed for its feature block.
    """
    try:
        info = model.get_movie(movie_id)
        row = model.movies.iloc[model.resolve(movie_id=movie_id)]
    except KeyError as error:
        raise HTTPException(
            status_code=404,
            detail=f"Movie {movie_id} was not found in the catalogue.",
        ) from error

    # 35 movies record runtime as 0.0, which means "unknown", not "zero
    # minutes" - reporting 0 would be a data error shown to the user.
    runtime = _int_or_none(row["runtime"])
    if runtime is not None and runtime <= 0:
        runtime = None

    identity = MovieIdentity(
        movie_id=int(info["movie_id"]),
        title=str(info["title"]),
        year=_int_or_none(info["year"]),
    )
    images = _images_for(image_service, [identity])

    return MovieDetails(
        movie_id=info["movie_id"],
        title=info["title"],
        year=info["year"],
        overview=_text_or_none(row["overview"]),
        genres=info["genres"],
        runtime=runtime,
        vote_average=_finite(row["vote_average"]),
        vote_count=_int_or_none(row["vote_count"]),
        director=info["director"],
        cast=_cast_for(cast_service, identity.movie_id),
        image=_movie_image(images.get(identity.movie_id)),
    )
