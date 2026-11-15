"""Cast credits for the movie details response.

The recommendation artifact keeps only the five billed actor names needed by
the cast feature block. The details response needs billing order, character
names and up to eight entries, so this module reads those fields from the same
TMDB credits CSV used to build the artifact. It is loaded once during startup
and never affects ranking.
"""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass
from pathlib import Path

from src.data.processor import parse_json_column

logger = logging.getLogger("cinescope.api.cast")

CREDITS_PATH = Path("data/tmdb-5000/tmdb_5000_credits.csv")
MAX_CAST_MEMBERS = 8


@dataclass(frozen=True)
class CastCredit:
    """One billed cast member from the credits dataset."""

    name: str
    character: str | None


def _text(value) -> str | None:
    """Trimmed string, or None when the source value is absent or blank."""
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def parse_cast(value, limit: int = MAX_CAST_MEMBERS) -> list[CastCredit]:
    """Return billed cast in source order, safely ignoring malformed entries."""
    try:
        items = parse_json_column(value)
    except Exception:
        return []

    if not isinstance(items, list):
        return []

    cast: list[CastCredit] = []
    for item in items:
        if len(cast) >= limit:
            break
        if not isinstance(item, dict):
            continue
        name = _text(item.get("name"))
        if name is None:
            continue
        cast.append(CastCredit(name=name, character=_text(item.get("character"))))
    return cast


class MovieCastService:
    """Read-only movie-id to cast lookup loaded from the credits dataset."""

    def __init__(self, credits: dict[int, tuple[CastCredit, ...]]) -> None:
        self._credits = credits

    @property
    def n_movies(self) -> int:
        """Number of movies with at least one usable cast credit."""
        return len(self._credits)

    def get_cast(self, movie_id: int) -> list[CastCredit]:
        """Cast for one movie, or an empty list when none is available."""
        try:
            key = int(movie_id)
        except (TypeError, ValueError):
            return []
        return list(self._credits.get(key, ()))

    @classmethod
    def load(cls, path: str | Path = CREDITS_PATH) -> "MovieCastService":
        """Load the existing credits CSV, tolerating missing or malformed rows."""
        credits: dict[int, tuple[CastCredit, ...]] = {}
        try:
            with Path(path).open(encoding="utf-8", newline="") as handle:
                for row in csv.DictReader(handle):
                    try:
                        movie_id = int(row.get("movie_id") or "")
                    except (TypeError, ValueError):
                        continue
                    cast = parse_cast(row.get("cast"))
                    if cast:
                        credits[movie_id] = tuple(cast)
        except (OSError, UnicodeError, csv.Error) as error:
            logger.warning(
                "Cast metadata unavailable at %s (%s). Details will return an "
                "empty cast list.",
                path,
                error,
            )
            return cls({})

        return cls(credits)
