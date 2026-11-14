"""
HTTP routes.

Recommendation ranking is delegated entirely to ``MovieRecommender.recommend()``
and title search reads the already-loaded catalogue; this module only
translates HTTP input into model calls and the results into the public
response schemas. No scoring, feature or loading logic is duplicated here.
"""

from __future__ import annotations

import math
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Path, Query

from api.dependencies import get_model
from api.schemas import (
    MovieDetails,
    MovieSearchResponse,
    MovieSummary,
    RecommendationItem,
    RecommendationResponse,
)
from src.models.recommender import MovieRecommender

router = APIRouter()

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
        int, Path(ge=1, description="TMDB movie id of the query movie.")
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

    items = [
        RecommendationItem(
            movie_id=int(row.movie_id),
            title=row.title,
            year=None if row.year is None else int(row.year),
            rank=int(row.rank),
            score=_finite(row.score) or 0.0,
            coverage=_finite(row.coverage) or 0.0,
            agreement=_finite(row.agreement),
        )
        for row in frame.itertuples(index=False)
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
    summary="Search the catalogue by partial title",
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
) -> MovieSearchResponse:
    """Case-insensitive partial-title search over the loaded catalogue.

    Title matching only - the recommendation engine is not involved, so a
    search costs a string scan over 4,803 rows rather than a similarity
    computation. Duplicate titles are returned as separate entries because
    each is a distinct catalogue row with its own id.
    """
    term = q.strip().lower()
    if not term:
        # ``min_length`` counts characters, so "   " passes it; a search with
        # no actual term is still a client error.
        raise HTTPException(
            status_code=422,
            detail="Query must contain at least one non-whitespace character.",
        )

    titles = model.movies["title"].astype(str).str.lower()
    hits = titles.str.contains(term, regex=False).to_numpy()
    lowered = titles.to_numpy()

    # Order: exact title, then prefix matches, then substring matches;
    # alphabetical within a tier so results are stable across requests.
    matches: list[tuple[int, str, int]] = []
    for position, hit in enumerate(hits):
        if not hit:
            continue
        title = str(lowered[position])
        tier = 0 if title == term else (1 if title.startswith(term) else 2)
        matches.append((tier, title, position))
    matches.sort()

    ids = model.movies["id"].to_numpy()
    results = []
    for _, _, position in matches[:limit]:
        # Hydrated through the model's public accessor so the year handling
        # (including "no release date") stays in one place.
        info = model.get_movie(int(ids[position]))
        results.append(
            MovieSummary(
                movie_id=info["movie_id"],
                title=info["title"],
                year=info["year"],
            )
        )

    return MovieSearchResponse(query=q, count=len(results), results=results)


@router.get(
    "/movies/{movie_id}",
    response_model=MovieDetails,
    tags=["movies"],
    summary="Metadata for one movie",
    responses={404: {"description": "No movie with that id in the catalogue."}},
)
def get_movie_details(
    movie_id: Annotated[int, Path(ge=1, description="TMDB movie id.")],
    model: MovieRecommender = Depends(get_model),
) -> MovieDetails:
    """Describe a single catalogue entry.

    Registered after ``/movies/search`` so the static path is matched first;
    a parameterised route declared earlier would swallow it and reject
    "search" as a non-integer id.

    Identity, title, year, genres and director come from the model's public
    ``get_movie()`` accessor; the remaining fields are read from the loaded
    ``model.movies`` frame. Both read in memory - no CSV, no reload.
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
    )
