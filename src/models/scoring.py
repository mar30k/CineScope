"""
Content similarity scoring for CineScope.

Strategy B - fixed-denominator weighted content agreement
---------------------------------------------------------
Let ``B`` be the configured feature blocks, ``w_b`` their weights, ``A_b(x)``
the indicator that block ``b`` is non-empty for movie ``x``, and ``c_b(x, y)``
the cosine similarity of the two block rows (0 when either row is empty).

    N(x, y)      = SUM_b  w_b * A_b(x) * A_b(y) * c_b(x, y)
    S_B(x, y)    = N(x, y) / SUM_b w_b                    (fixed denominator)
    agreement    = N(x, y) / SUM_{shared b} w_b          (Strategy C view)
    coverage     = SUM_{shared b} w_b / SUM_b w_b

so that ``S_B = coverage * agreement``.  A block that is missing for either
movie contributes **zero to the numerator** while the denominator stays at the
full configured weight; that is what stops metadata-poor movies from
accumulating inflated similarity.

Contrast with Strategy A (the previous production behaviour), which divides by
each movie's own row norm:

    S_A(x, y) = N(x, y) / ( sqrt(SUM_b w_b A_b(x)) * sqrt(SUM_b w_b A_b(y)) )

For two movies with complete metadata, ``S_A == S_B`` exactly.
"""

from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize


class BlockScorer:
    """Per-block cosine scoring with the Strategy B fixed denominator.

    Block matrices are L2-normalised per row, exactly as ``combine_features``
    does before applying ``sqrt(w)``.  Scaling a block by a constant does not
    change a cosine, so these are the same quantities that enter the
    concatenated (Strategy A) vector.

    Parameters
    ----------
    features : dict[str, scipy.sparse matrix]
        One raw (un-normalised) matrix per feature block.
    weights : dict[str, float]
        Block weights.  Blocks absent from ``features`` are ignored.
    """

    def __init__(self, features: dict, weights: dict):
        if not weights:
            raise ValueError("weights must not be empty")

        self.weights = {b: float(w) for b, w in weights.items() if b in features}
        if not self.weights:
            raise ValueError("none of the requested blocks exist in features")
        if any(w <= 0 for w in self.weights.values()):
            raise ValueError("all weights must be positive")
        if any(features[b].shape[0] != next(iter(features.values())).shape[0]
               for b in self.weights):
            raise ValueError("all feature blocks must have the same number of rows")

        self.blocks = list(self.weights)
        self.total_weight = float(sum(self.weights.values()))
        self.block_matrix = {}
        self.active = {}
        for b in self.blocks:
            mat = normalize(csr_matrix(features[b]), norm="l2", axis=1).tocsr()
            self.block_matrix[b] = mat
            self.active[b] = np.asarray(mat.getnnz(axis=1)).ravel() > 0

        self.n_movies = next(iter(self.block_matrix.values())).shape[0]

        # Weight each movie actually carries (Strategy A denominator squared).
        self.active_weight = np.zeros(self.n_movies)
        for b in self.blocks:
            self.active_weight += self.weights[b] * self.active[b]

    # ---------------------------------------------------------------- pair
    def pair_scores(self, i: int, j: int) -> dict:
        """Exact block-level decomposition of a single pair."""
        numerator = 0.0
        shared_weight = 0.0
        shared = []
        per_block = {}
        contributions = {}

        for b in self.blocks:
            if not (self.active[b][i] and self.active[b][j]):
                per_block[b] = 0.0
                contributions[b] = 0.0
                continue

            x = self.block_matrix[b][i]
            y = self.block_matrix[b][j]
            cos = float(x.multiply(y).sum())
            cos = max(min(cos, 1.0), 0.0)          # guard floating-point drift
            per_block[b] = cos
            contributions[b] = self.weights[b] * cos
            numerator += contributions[b]
            shared_weight += self.weights[b]
            shared.append(b)

        agreement = (numerator / shared_weight) if shared_weight > 0 else float("nan")
        return {
            "per_block": per_block,
            "contributions": contributions,
            "numerator": numerator,
            "shared_weight": shared_weight,
            "total_weight": self.total_weight,
            "coverage": shared_weight / self.total_weight,
            "shared_blocks": shared,
            "n_shared_blocks": len(shared),
            "score": numerator / self.total_weight,
            "agreement": agreement,
        }

    # ----------------------------------------------------------- all pairs
    def score_all(self, i: int) -> dict:
        """Score movie ``i`` against the whole catalogue using sparse ops."""
        n = self.n_movies
        numerator = np.zeros(n)
        shared_weight = np.zeros(n)
        n_shared = np.zeros(n, dtype=np.int16)

        for b in self.blocks:
            if not self.active[b][i]:
                continue
            row = self.block_matrix[b][i]
            dots = np.asarray((row @ self.block_matrix[b].T).todense()).ravel()
            mask = self.active[b]
            numerator += self.weights[b] * np.where(mask, dots, 0.0)
            shared_weight += self.weights[b] * mask
            n_shared += mask.astype(np.int16)

        agreement = np.full(n, np.nan)
        has = shared_weight > 0
        agreement[has] = numerator[has] / shared_weight[has]

        return {
            "score": numerator / self.total_weight,
            "agreement": agreement,
            "coverage": shared_weight / self.total_weight,
            "n_shared_blocks": n_shared,
            "numerator": numerator,
            "shared_weight": shared_weight,
        }

    # ------------------------------------------------------------ helpers
    def shared_block_names(self, i: int, j: int) -> str:
        return "+".join(b for b in self.blocks
                        if self.active[b][i] and self.active[b][j])

    def concatenated_cosine(self, i: int, j: int) -> float:
        """Strategy A score for one pair, kept as the comparison baseline."""
        num = 0.0
        for b in self.blocks:
            if self.active[b][i] and self.active[b][j]:
                x = self.block_matrix[b][i]
                y = self.block_matrix[b][j]
                num += self.weights[b] * float(x.multiply(y).sum())
        denom = np.sqrt(self.active_weight[i]) * np.sqrt(self.active_weight[j])
        return num / denom if denom > 0 else 0.0
