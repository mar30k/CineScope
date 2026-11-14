"""
CineScope model configuration - the single source of truth.

Both the production recommender and the experimental scripts read the selected
weights from here, so the configuration cannot drift between them again.
Changing a weight is a deliberate, one-line edit in this file.
"""

# ---------------------------------------------------------------- identity
MODEL_NAME = "CineScope"
MODEL_VERSION = "1.0.0"
ARTIFACT_FORMAT_VERSION = 1

# --------------------------------------------------------------- features
# Order is meaningful: it is the order in which blocks are concatenated and
# the order in which shared blocks are reported.
FEATURE_BLOCKS = ("genres", "keywords", "overview", "cast", "director")

# The selected configuration ("L - Light Director"), chosen during Phase 3
# and evaluated in reports/strategy_ab_evaluation_report.md.
FEATURE_WEIGHTS = {
    "genres": 1.0,
    "keywords": 1.0,
    "overview": 1.0,
    "cast": 0.5,
    "director": 0.25,
}

CONFIG_NAME = "L - Light Director"

# Ranking strategy used in production. Strategy B is the fixed-denominator
# weighted content-agreement score; see src/models/scoring.py.
SCORING_STRATEGY = "B"

TOTAL_WEIGHT = sum(FEATURE_WEIGHTS[b] for b in FEATURE_BLOCKS)

# ----------------------------------------------------------------- paths
ARTIFACT_DIRECTORY = "artifacts"
MODEL_ARTIFACT_NAME = "cinescope_model.joblib"

# ---------------------------------------------------------------- output
# Columns returned by MovieRecommender.recommend().
RECOMMENDATION_COLUMNS = (
    "movie_id",
    "title",
    "year",
    "genres",
    "director",
    "score",
    "agreement",
    "coverage",
    "n_shared_blocks",
    "shared_blocks",
)
