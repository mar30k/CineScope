"""
CineScope production recommender.

Ranks the catalogue for a query movie with **Strategy B** (fixed-denominator
weighted content agreement) over the feature weights in ``src/config.py``.

Design notes
------------
* Movie **IDs** are the primary query identifier. Titles are supported as a
  convenience but raise if the title is ambiguous, because the dataset
  contains three duplicate-title groups (e.g. two films called "Batman").
* Missing feature blocks contribute zero to the numerator while the
  denominator stays at the full configured weight, so a metadata-poor movie
  cannot outrank a complete one on a single shared block.
* The score is a content-similarity ranking score in [0, 1]. It is **not** a
  probability that a user will like the movie.
* ``agreement`` and ``coverage`` are reported next to the score because the
  score is their product. Reading a score without its coverage is misleading.

Build and save the artifact
---------------------------
    python -m src.models.recommender build

Quick use
---------
    from src.models.recommender import MovieRecommender

    model = MovieRecommender.load()          # fast: reads artifacts/*.joblib
    model.recommend(movie_id=155, n_recommendations=10)
"""

from __future__ import annotations

import datetime as _dt
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import (
    ARTIFACT_FORMAT_VERSION,
    CONFIG_NAME,
    FEATURE_BLOCKS,
    FEATURE_WEIGHTS,
    MODEL_NAME,
    MODEL_VERSION,
    RECOMMENDATION_COLUMNS,
    SCORING_STRATEGY,
)
from src.data.processor import process_movies
from src.features.feature_builder import build_feature_pipeline
from src.models.persistence import load_artifact, save_artifact
from src.models.scoring import BlockScorer

# Columns returned by recommend(): the configured fields plus 1-based rank.
OUTPUT_COLUMNS = tuple(RECOMMENDATION_COLUMNS) + ("rank",)


class MovieRecommender:
    """Top-N content-similarity recommender for the TMDB 5000 catalogue.

    Parameters
    ----------
    movies : pandas.DataFrame, optional
        Processed movie metadata (see ``src.data.processor.process_movies``).
        Built from the CSV files when omitted.
    features : dict, optional
        Raw feature matrices per block. Built when omitted.
    encoders : dict, optional
        Fitted vectorizers / binarisers, kept so an artifact can carry them.
    weights : dict, optional
        Block weights. Defaults to the configuration in ``src/config.py``.
    """

    def __init__(self, movies=None, features=None, encoders=None, weights=None):
        # Never import ``src.experiments``: production must not depend on it.
        if movies is None or features is None:
            movies = process_movies()
            features, encoders = build_feature_pipeline(movies)

        self.movies = movies
        self.features = features
        self.encoders = encoders or {}
        self.weights = dict(FEATURE_WEIGHTS if weights is None else weights)
        self.blocks = [b for b in FEATURE_BLOCKS if b in self.weights]
        self.total_weight = float(sum(self.weights.values()))

        self.scorer = BlockScorer(self.features, self.weights)
        self.n_movies = len(self.movies)
        self.version = MODEL_VERSION

        # Positional lookups: the row position, not the DataFrame index, is
        # what every downstream array is indexed by.
        ids = self.movies["id"].to_numpy()
        self.id_to_position = {int(mid): pos for pos, mid in enumerate(ids)}
        self._titles_lower = (
            self.movies["title"].astype(str).str.strip().str.lower().to_numpy()
        )
        years = pd.to_datetime(self.movies["release_date"], errors="coerce").dt.year
        self._years = years.to_numpy()

    # ------------------------------------------------------------- lookup
    def resolve(self, movie_id=None, title=None) -> int:
        """Return the row position for a movie id (preferred) or exact title."""
        if movie_id is not None:
            try:
                mid = int(movie_id)
            except (TypeError, ValueError):
                raise ValueError(f"movie_id must be an integer, got {movie_id!r}")
            position = self.id_to_position.get(mid)
            if position is None:
                raise KeyError(f"Unknown movie id: {mid}")
            return position

        if title is None:
            raise ValueError("Provide movie_id (preferred) or title.")

        key = str(title).strip().lower()
        positions = np.flatnonzero(self._titles_lower == key)
        if len(positions) == 0:
            raise KeyError(f"Movie not found: {title}")
        if len(positions) > 1:
            ids = [int(self.movies.iloc[p]["id"]) for p in positions]
            raise ValueError(
                f"Title {title!r} is ambiguous (ids {ids}); query by movie_id."
            )
        return int(positions[0])

    def get_movie(self, movie_id: int) -> dict:
        """Metadata for one movie, for echoing the query back to a client."""
        position = self.resolve(movie_id=movie_id)
        movie = self.movies.iloc[position]
        year = self._years[position]
        return {
            "movie_id": int(movie["id"]),
            "title": str(movie["title"]),
            "year": int(year) if not pd.isna(year) else None,
            "genres": list(movie["genres"]),
            "director": list(movie["director"]),
        }

    # ---------------------------------------------------------- recommend
    def recommend(self, movie_id=None, n_recommendations: int = 10, *,
                  title: str | None = None, exclude_self: bool = True) -> pd.DataFrame:
        """Top-N recommendations for one movie.

        Returns a DataFrame with ``RECOMMENDATION_COLUMNS``. The query movie is
        excluded. ``n_recommendations`` is clamped to the catalogue size.
        """
        position = self.resolve(movie_id=movie_id, title=title)

        if self.scorer.active_weight[position] <= 0:
            raise ValueError(
                "This movie has no usable feature blocks, so no similarity can "
                "be computed."
            )

        limit = max(int(n_recommendations), 0)
        available = self.n_movies - (1 if exclude_self else 0)
        limit = min(limit, available)
        if limit == 0:
            return pd.DataFrame(columns=list(OUTPUT_COLUMNS))

        scores = self.scorer.score_all(position)["score"].copy()
        if exclude_self:
            scores[position] = -np.inf

        # Stable sort keeps ties deterministic (ordered by catalogue position).
        order = np.argsort(-scores, kind="stable")[:limit]

        rows = []
        for rank, candidate in enumerate(order, start=1):
            candidate = int(candidate)
            if exclude_self and candidate == position:
                continue

            pair = self.scorer.pair_scores(position, candidate)
            movie = self.movies.iloc[candidate]
            year = self._years[candidate]

            rows.append({
                "movie_id": int(movie["id"]),
                "title": str(movie["title"]),
                "year": int(year) if not pd.isna(year) else None,
                "genres": ", ".join(movie["genres"]),
                "director": ", ".join(movie["director"]),
                "score": float(scores[candidate]),
                "agreement": pair["agreement"],
                "coverage": pair["coverage"],
                "n_shared_blocks": pair["n_shared_blocks"],
                "shared_blocks": "+".join(pair["shared_blocks"]),
                "rank": rank,
            })

        frame = pd.DataFrame(rows)
        frame = frame.reindex(columns=list(OUTPUT_COLUMNS))
        return frame.reset_index(drop=True)

    # -------------------------------------------------------- persistence
    def to_artifact(self) -> dict:
        """Serialisable snapshot of everything needed to rank."""
        return {
            "artifact_format_version": ARTIFACT_FORMAT_VERSION,
            "model_name": MODEL_NAME,
            "model_version": self.version,
            "config_name": CONFIG_NAME,
            "scoring_strategy": SCORING_STRATEGY,
            "feature_blocks": list(FEATURE_BLOCKS),
            "weights": dict(self.weights),
            "total_weight": self.total_weight,
            "movies": self.movies,
            "features": self.features,
            "encoders": self.encoders,
            "n_movies": self.n_movies,
            "created_at": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        }

    @classmethod
    def from_artifact(cls, payload: dict) -> "MovieRecommender":
        return cls(
            movies=payload["movies"],
            features=payload["features"],
            encoders=payload.get("encoders"),
            weights=payload.get("weights"),
        )

    def save(self, path: str | Path | None = None) -> Path:
        """Write the artifact; returns the path actually written."""
        return save_artifact(self.to_artifact(), path)

    @classmethod
    def load(cls, path: str | Path | None = None) -> "MovieRecommender":
        """Load a saved artifact. Falls back to building nothing: it raises."""
        return cls.from_artifact(load_artifact(path))


# --------------------------------------------------------------- CLI demo
def _cli(argv: list[str]) -> int:
    if len(argv) > 1 and argv[1] == "build":
        model = MovieRecommender()
        path = model.save()
        print(f"Built {MODEL_NAME} {MODEL_VERSION} ({model.n_movies} movies)")
        print(f"Saved to {path}")
        return 0

    model = MovieRecommender.load()
    recommendations = model.recommend(movie_id=19995, n_recommendations=10)
    print(f"\n{MODEL_NAME} - recommendations for id 19995 (Avatar)")
    print("-" * 72)
    print(recommendations[["movie_id", "title", "year", "score", "coverage"]]
          .to_string(index=False))
    return 0


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    raise SystemExit(_cli(sys.argv))
