"""
Response models for the CineScope API.

These are the public contract. They deliberately expose only what a client
needs to present a result: identity, ranking and the confidence pair that
makes the score interpretable. Internal machinery (block names, matrix
positions, the raw pandas frame) stays server-side.
"""

from pydantic import BaseModel, Field


class MovieImage(BaseModel):
    """Optional artwork resolved from TMDB for a catalogue movie."""

    poster_url: str | None = Field(
        default=None, description="TMDB poster URL at w500, or null when absent."
    )
    backdrop_url: str | None = Field(
        default=None, description="TMDB backdrop URL at w1280, or null when absent."
    )


class CastMember(BaseModel):
    """One billed cast member in the movie details response."""

    name: str = Field(description="Actor name.")
    character: str | None = Field(
        default=None, description="Character name, or null when unavailable."
    )


class RecommendationItem(BaseModel):
    """One ranked recommendation."""

    movie_id: int = Field(description="CineScope catalogue movie id.")
    title: str = Field(description="Movie title.")
    year: int | None = Field(
        default=None, description="Release year, when the catalogue records one."
    )
    rank: int = Field(description="1-based position in this result list.")
    score: float = Field(
        description=(
            "Strategy B content-agreement score in [0, 1]. A ranking score, "
            "not a probability that the user will like the movie."
        )
    )
    coverage: float = Field(
        description=(
            "Share of the configured feature weight the pair actually shares, "
            "in [0, 1]. Low coverage means the score rests on little evidence."
        )
    )
    agreement: float | None = Field(
        default=None,
        description=(
            "Weighted mean similarity across the shared blocks. Null when the "
            "pair shares no feature blocks. Note score = coverage * agreement."
        ),
    )
    image: MovieImage | None = Field(
        default=None,
        description="TMDB artwork when a confident title/year match exists.",
    )


class RecommendationResponse(BaseModel):
    """Recommendations for one query movie."""

    movie_id: int = Field(description="The movie the recommendations are for.")
    count: int = Field(description="Number of recommendations returned.")
    recommendations: list[RecommendationItem] = Field(
        description="Recommendations ordered from most to least similar."
    )


class MovieSummary(BaseModel):
    """A catalogue entry, as returned by search."""

    movie_id: int = Field(
        description="CineScope catalogue movie id. Use this to request recommendations."
    )
    title: str = Field(description="Movie title.")
    year: int | None = Field(
        default=None, description="Release year, when the catalogue records one."
    )
    image: MovieImage | None = Field(
        default=None,
        description="TMDB artwork when a confident title/year match exists.",
    )


class MovieDetails(BaseModel):
    """Full metadata for one selected movie."""

    movie_id: int = Field(description="CineScope catalogue movie id.")
    title: str = Field(description="Movie title.")
    year: int | None = Field(
        default=None, description="Release year, when the catalogue records one."
    )
    overview: str | None = Field(
        default=None, description="Short synopsis, or null when absent."
    )
    genres: list[str] = Field(default_factory=list, description="Genre names.")
    runtime: int | None = Field(
        default=None,
        description="Runtime in minutes. Null when missing or recorded as zero.",
    )
    vote_average: float | None = Field(
        default=None, description="TMDB mean rating out of 10, or null when absent."
    )
    vote_count: int | None = Field(
        default=None, description="Number of TMDB votes, or null when absent."
    )
    director: list[str] = Field(default_factory=list, description="Director names.")
    cast: list[CastMember] = Field(
        default_factory=list,
        description="Up to eight billed cast members, in credits order.",
    )
    image: MovieImage | None = Field(
        default=None,
        description="TMDB artwork when a confident title/year match exists.",
    )


class MovieSearchResponse(BaseModel):
    """Case-insensitive partial-title matches."""

    query: str = Field(description="The search string as submitted.")
    count: int = Field(description="Number of matches returned.")
    results: list[MovieSummary] = Field(
        description=(
            "Matches, exact title first, then titles starting with the query, "
            "then titles containing it; alphabetical within each group."
        )
    )
