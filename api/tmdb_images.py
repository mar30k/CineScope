"""
TMDB poster and backdrop enrichment.

The catalogue stores its own movie ids, which are not TMDB ids. Images are
therefore resolved by an exact, conservative title/year match against TMDB's
search endpoint. Ambiguous results produce no image rather than a guess.

The service is intentionally optional: without ``TMDB_API_KEY`` it reports
``None`` for every lookup and the rest of the API continues to work. Network
failures, malformed responses and rate limits are also non-fatal. Successful
matches, confirmed misses and transient failures are cached separately so a
TMDB outage cannot add a timeout to every request.
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
import unicodedata
from collections.abc import Callable, Iterable, Mapping
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen

logger = logging.getLogger("cinescope.api.tmdb")

TMDB_SEARCH_URL = "https://api.themoviedb.org/3/search/movie"
TMDB_IMAGE_BASE_URL = "https://image.tmdb.org/t/p"
POSTER_SIZE = "w500"
BACKDROP_SIZE = "w1280"

DEFAULT_TIMEOUT_SECONDS = 3.0
DEFAULT_POSITIVE_TTL_SECONDS = 24 * 60 * 60
DEFAULT_NEGATIVE_TTL_SECONDS = 6 * 60 * 60
DEFAULT_FAILURE_TTL_SECONDS = 2 * 60
DEFAULT_MAX_CACHE_ENTRIES = 2_000
DEFAULT_MAX_WORKERS = 8


@dataclass(frozen=True, slots=True)
class MovieIdentity:
    """The catalogue fields needed to find a TMDB counterpart."""

    movie_id: int
    title: str
    year: int | None


@dataclass(frozen=True, slots=True)
class ImageAssets:
    """Image URLs for one movie. Either side may be absent in TMDB."""

    poster_url: str | None = None
    backdrop_url: str | None = None

    @property
    def has_images(self) -> bool:
        return bool(self.poster_url or self.backdrop_url)


@dataclass(frozen=True, slots=True)
class _CacheEntry:
    assets: ImageAssets
    expires_at: float


Transport = Callable[[str, float], Mapping[str, Any]]


def _normalize_title(title: Any) -> str:
    """Fold case, accents and punctuation so exact titles compare reliably."""
    if not isinstance(title, str):
        return ""

    folded = unicodedata.normalize("NFKD", title)
    folded = "".join(char for char in folded if not unicodedata.combining(char))
    folded = folded.casefold().replace("&", " and ")
    folded = re.sub(r"[\W_]+", " ", folded, flags=re.UNICODE)
    return " ".join(folded.split())


def _release_year(value: Any) -> int | None:
    """Extract a release year from TMDB's ``YYYY-MM-DD`` field."""
    if not isinstance(value, str):
        return None
    match = re.match(r"^(\d{4})", value.strip())
    return int(match.group(1)) if match else None


def _image_url(path: Any, size: str) -> str | None:
    if not isinstance(path, str):
        return None
    cleaned = path.strip()
    if not cleaned:
        return None
    if not cleaned.startswith("/"):
        cleaned = f"/{cleaned}"
    return f"{TMDB_IMAGE_BASE_URL}/{size}{cleaned}"


def _urllib_transport(url: str, timeout: float) -> Mapping[str, Any]:
    """Fetch JSON using the standard library, keeping the API key server-side."""
    request = Request(
        url,
        headers={
            "Accept": "application/json",
            "User-Agent": "CineScope/1.0",
        },
    )
    with urlopen(request, timeout=timeout) as response:
        payload = response.read()
    data = json.loads(payload)
    if not isinstance(data, Mapping):
        raise ValueError("TMDB response was not a JSON object.")
    return data


class TmdbImageService:
    """Resolve TMDB images with conservative matching and a small TTL cache."""

    def __init__(
        self,
        api_key: str | None,
        *,
        transport: Transport | None = None,
        clock: Callable[[], float] = time.monotonic,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        positive_ttl: float = DEFAULT_POSITIVE_TTL_SECONDS,
        negative_ttl: float = DEFAULT_NEGATIVE_TTL_SECONDS,
        failure_ttl: float = DEFAULT_FAILURE_TTL_SECONDS,
        max_cache_entries: int = DEFAULT_MAX_CACHE_ENTRIES,
        max_workers: int = DEFAULT_MAX_WORKERS,
    ) -> None:
        self._api_key = (api_key or "").strip() or None
        self._transport = transport or _urllib_transport
        self._clock = clock
        self._timeout = timeout
        self._positive_ttl = positive_ttl
        self._negative_ttl = negative_ttl
        self._failure_ttl = failure_ttl
        self._max_cache_entries = max(1, max_cache_entries)
        self._max_workers = max(1, max_workers)
        self._cache: dict[tuple[int, str, int | None], _CacheEntry] = {}
        self._cache_lock = threading.Lock()

    @classmethod
    def from_environment(cls, **kwargs: Any) -> TmdbImageService:
        return cls(os.environ.get("TMDB_API_KEY"), **kwargs)

    @property
    def enabled(self) -> bool:
        return self._api_key is not None

    def get_image(self, movie: MovieIdentity) -> ImageAssets | None:
        """Return image assets, or ``None`` when enrichment is disabled.

        An enabled service always returns an ``ImageAssets`` value. Empty
        assets mean "no confident match or no usable paths", not an error.
        """
        if not self.enabled:
            return None

        key = (movie.movie_id, _normalize_title(movie.title), movie.year)
        cached = self._get_cached(key)
        if cached is not None:
            return cached

        if not key[1]:
            self._store(key, ImageAssets(), self._negative_ttl)
            return ImageAssets()

        try:
            payload = self._transport(self._search_url(movie), self._timeout)
            match = self._match(movie, payload)
        except Exception as error:
            # Deliberately broad: TMDB must never break a catalogue request.
            # The exception type is safe to log; the URL contains the key.
            logger.debug(
                "TMDB lookup failed for movie %s (%s).",
                movie.movie_id,
                type(error).__name__,
            )
            assets = ImageAssets()
            self._store(key, assets, self._failure_ttl)
            return assets

        if match is None:
            assets = ImageAssets()
            self._store(key, assets, self._negative_ttl)
            return assets

        assets = ImageAssets(
            poster_url=_image_url(match.get("poster_path"), POSTER_SIZE),
            backdrop_url=_image_url(match.get("backdrop_path"), BACKDROP_SIZE),
        )
        self._store(
            key,
            assets,
            self._positive_ttl if assets.has_images else self._negative_ttl,
        )
        return assets

    def get_images_for_movies(
        self, movies: Iterable[MovieIdentity]
    ) -> dict[int, ImageAssets]:
        """Resolve a batch in parallel, deduplicated by catalogue id."""
        if not self.enabled:
            return {}

        unique: dict[int, MovieIdentity] = {}
        for movie in movies:
            unique.setdefault(movie.movie_id, movie)
        if not unique:
            return {}

        if len(unique) == 1:
            movie = next(iter(unique.values()))
            assets = self.get_image(movie)
            return {} if assets is None else {movie.movie_id: assets}

        results: dict[int, ImageAssets] = {}
        with ThreadPoolExecutor(
            max_workers=min(self._max_workers, len(unique)),
            thread_name_prefix="tmdb-image",
        ) as executor:
            futures = {
                executor.submit(self.get_image, movie): movie.movie_id
                for movie in unique.values()
            }
            for future in as_completed(futures):
                movie_id = futures[future]
                try:
                    assets = future.result()
                except Exception:
                    assets = ImageAssets()
                if assets is not None:
                    results[movie_id] = assets
        return results

    def _search_url(self, movie: MovieIdentity) -> str:
        parameters: dict[str, str] = {
            "api_key": self._api_key or "",
            "query": movie.title,
            "include_adult": "false",
            "language": "en-US",
        }
        if movie.year is not None:
            parameters["year"] = str(movie.year)
        return f"{TMDB_SEARCH_URL}?{urlencode(parameters)}"

    def _match(
        self, movie: MovieIdentity, payload: Mapping[str, Any]
    ) -> Mapping[str, Any] | None:
        results = payload.get("results")
        if not isinstance(results, list):
            return None

        normalized_title = _normalize_title(movie.title)
        if not normalized_title:
            return None

        matches: list[Mapping[str, Any]] = []
        for candidate in results:
            if not isinstance(candidate, Mapping):
                continue

            titles = {
                _normalize_title(candidate.get("title")),
                _normalize_title(candidate.get("original_title")),
            }
            if normalized_title not in titles:
                continue

            if movie.year is not None:
                if _release_year(candidate.get("release_date")) != movie.year:
                    continue

            matches.append(candidate)

        # No match is safer than displaying the wrong film's poster. This also
        # rejects remakes and duplicate search entries, even when year is absent.
        return matches[0] if len(matches) == 1 else None

    def _get_cached(self, key: tuple[int, str, int | None]) -> ImageAssets | None:
        now = self._clock()
        with self._cache_lock:
            entry = self._cache.get(key)
            if entry is None:
                return None
            if entry.expires_at <= now:
                del self._cache[key]
                return None
            return entry.assets

    def _store(
        self, key: tuple[int, str, int | None], assets: ImageAssets, ttl: float
    ) -> None:
        expires_at = self._clock() + ttl
        with self._cache_lock:
            if key not in self._cache and len(self._cache) >= self._max_cache_entries:
                self._prune_locked()
            self._cache[key] = _CacheEntry(assets=assets, expires_at=expires_at)

    def _prune_locked(self) -> None:
        now = self._clock()
        expired = [
            key for key, entry in self._cache.items() if entry.expires_at <= now
        ]
        for key in expired:
            del self._cache[key]

        if len(self._cache) >= self._max_cache_entries:
            oldest = min(
                self._cache,
                key=lambda key: self._cache[key].expires_at,
            )
            del self._cache[oldest]
