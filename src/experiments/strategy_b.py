"""
Experimental A/B recommender for CineScope (Phase 3 experiments).

This module is retained so the Strategy A vs Strategy B comparison in
``reports/strategy_ab_evaluation_report.md`` stays reproducible. It is NOT
used by production: ``src/models/recommender.py`` is the production entry
point and ranks with the same ``BlockScorer`` imported below.

The scoring maths lives in ``src.models.scoring`` and is shared, so the
experiment and production cannot drift apart. What is kept here is only the
*comparison harness*: building both an A model and a B model from the same
matrices and returning their ranked lists.

Definitions
-----------
Let ``B`` be the configured blocks, ``w_b`` their weights, ``A_b(x)`` the
indicator that block ``b`` is non-empty for movie ``x``, and ``c_b(x, y)`` the
cosine similarity of the two block rows (0 when either row is empty).

    numerator(x, y) = SUM_b  w_b * A_b(x) * A_b(y) * c_b(x, y)
    total_weight    = SUM_b  w_b                         (constant)

    Strategy B  score(x, y)    = numerator / total_weight
    Strategy C  agreement(x,y) = numerator / SUM_{shared b} w_b
    coverage(x, y)             = SUM_{shared b} w_b / total_weight

Hence ``score = coverage * agreement`` and ``agreement`` is the weighted mean
of the shared blocks' cosines.  ``score`` is comparable across all pairs
because its denominator is constant; ``agreement`` is not, and must never be
read without its coverage.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from src.config import FEATURE_WEIGHTS
from src.data.processor import process_movies
from src.features.feature_builder import build_features, combine_features
from src.models.scoring import BlockScorer

# Kept for backwards compatibility with the Phase 3 scripts and reports.
# The values now come from the single configuration module rather than being
# duplicated here.
CONFIG_L = dict(FEATURE_WEIGHTS)

RESULT_COLUMNS = [
    "movie_id", "title", "score", "agreement", "coverage",
    "n_shared_blocks", "shared_blocks",
]

__all__ = ["BlockScorer", "ExperimentalRecommender", "CONFIG_L", "RESULT_COLUMNS"]


class ExperimentalRecommender:
    """Selectable A / B recommender used only for evaluation.

    Parameters
    ----------
    strategy : "A" or "B"
    weights  : block weight mapping (defaults to the production configuration)
    """

    def __init__(self, strategy: str = "B", weights: dict | None = None,
                 movies=None, features=None):
        strategy = str(strategy).upper()
        if strategy not in ("A", "B"):
            raise ValueError("strategy must be 'A' or 'B'")

        self.strategy = strategy
        self.weights = dict(CONFIG_L if weights is None else weights)
        self.movies = process_movies() if movies is None else movies
        self.features = build_features(self.movies) if features is None else features
        self.n_movies = len(self.movies)

        self.scorer = BlockScorer(self.features, self.weights)

        # Strategy A uses the project's existing construction path verbatim.
        self._combined = None
        self._model = None
        if strategy == "A":
            self._combined = combine_features(self.features, weights=self.weights)
            self._model = NearestNeighbors(metric="cosine", algorithm="brute")
            self._model.fit(self._combined)

        self.id_to_index = {
            int(mid): idx for idx, mid in enumerate(self.movies["id"].values)
        }

    # ------------------------------------------------------------- lookup
    def resolve_index(self, movie_id=None, title=None) -> int:
        """Resolve a row index by movie ID (preferred) or exact title.

        IDs are unambiguous; titles are not (three duplicate-title groups
        exist in this dataset), so ``title`` is a convenience that raises on
        ambiguity rather than silently picking the first match.
        """
        if movie_id is not None:
            mid = int(movie_id)
            if mid not in self.id_to_index:
                raise ValueError(f"Unknown movie id: {movie_id}")
            return self.id_to_index[mid]

        if title is None:
            raise ValueError("provide movie_id or title")

        matches = self.movies.index[
            self.movies["title"].str.lower() == str(title).strip().lower()
        ]
        if len(matches) == 0:
            raise ValueError(f"Movie not found: {title}")
        if len(matches) > 1:
            ids = [int(self.movies.loc[m, "id"]) for m in matches]
            raise ValueError(
                f"Title '{title}' is ambiguous (ids {ids}); query by movie ID."
            )
        return int(matches[0])

    # ---------------------------------------------------------- recommend
    def recommend(self, movie_id=None, title=None, n_recommendations: int = 10,
                  exclude_self: bool = True) -> pd.DataFrame:
        """Top-N recommendations under the selected strategy.

        Requests larger than the available number of candidates are clamped
        rather than raising.
        """
        index = self.resolve_index(movie_id=movie_id, title=title)

        limit = max(int(n_recommendations), 0)
        available = self.n_movies - (1 if exclude_self else 0)
        limit = min(limit, available)
        if limit == 0:
            return pd.DataFrame(columns=RESULT_COLUMNS)

        if self.strategy == "A":
            return self._recommend_a(index, limit, exclude_self)
        return self._recommend_b(index, limit, exclude_self)

    # ------------------------------------------------------------- helpers
    def _decorate(self, index: int, order: np.ndarray, scores: np.ndarray,
                  exclude_self: bool) -> pd.DataFrame:
        rows = []
        for j in order:
            j = int(j)
            if exclude_self and j == index:
                continue
            pair = self.scorer.pair_scores(index, j)
            movie = self.movies.iloc[j]
            rows.append({
                "movie_id": int(movie["id"]),
                "title": movie["title"],
                "score": float(scores[j]),
                "agreement": pair["agreement"],
                "coverage": pair["coverage"],
                "n_shared_blocks": pair["n_shared_blocks"],
                "shared_blocks": pair["shared_blocks"] and
                                 "+".join(pair["shared_blocks"]) or "",
            })
        return pd.DataFrame(rows, columns=RESULT_COLUMNS)

    def _recommend_a(self, index: int, limit: int, exclude_self: bool):
        # Ask for one extra neighbour, exactly as the existing experiments do,
        # so the query itself can be dropped.
        k = min(limit + (1 if exclude_self else 0), self.n_movies)
        distances, indices = self._model.kneighbors(
            self._combined[index], n_neighbors=k
        )
        order = indices[0]
        scores = 1.0 - distances[0]
        full = np.zeros(self.n_movies)
        full[order] = scores
        return self._decorate(index, order, full, exclude_self).head(limit)

    def _recommend_b(self, index: int, limit: int, exclude_self: bool):
        out = self.scorer.score_all(index)
        scores = out["score"].copy()
        if exclude_self:
            scores[index] = -np.inf
        # stable sort => deterministic tie-breaking by catalogue position
        order = np.argsort(-scores, kind="stable")[:limit]
        return self._decorate(index, order, scores, exclude_self)
