"""
Feature engineering for CineScope.

Each feature block is built independently and kept as a sparse matrix:

    genres    MultiLabelBinarizer
    keywords  TfidfVectorizer (stop words removed)
    overview  TfidfVectorizer (stop words removed, max_features=5000)
    cast      MultiLabelBinarizer (top five billed actors)
    director  MultiLabelBinarizer

``build_features`` returns only the matrices, which is all the recommendation
pipeline needs.  ``build_feature_pipeline`` additionally returns the fitted
encoders so they can be persisted with the model and reused at load time
instead of being re-fitted.
"""

import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from scipy.sparse import csr_matrix, hstack
from sklearn.preprocessing import normalize


def build_features(movies):
    """
    Convert processed movie information into numerical feature matrices.

    Returns a dictionary containing one matrix per feature group.
    """

    features, _ = build_feature_pipeline(movies)

    return features


def build_feature_pipeline(movies):
    """
    Build the feature matrices and keep the fitted encoders.

    Returns
    -------
    (features, encoders)
        ``features`` is the same dictionary ``build_features`` returns;
        ``encoders`` maps each block name to its fitted transformer.
    """

    features = {}
    encoders = {}

    # 1. Genres
    genre_encoder = MultiLabelBinarizer(sparse_output=True)
    features["genres"] = genre_encoder.fit_transform(movies["genres"])
    encoders["genres"] = genre_encoder

    # 2. Keywords
    keyword_text = movies["keywords"].apply(
        lambda items: " ".join(items)
    )

    keyword_vectorizer = TfidfVectorizer(
        stop_words="english"
    )

    features["keywords"] = keyword_vectorizer.fit_transform(
        keyword_text
    )
    encoders["keywords"] = keyword_vectorizer

    # 3. Movie overviews
    overview_vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=5000
    )

    features["overview"] = overview_vectorizer.fit_transform(
        movies["overview"].fillna("")
    )
    encoders["overview"] = overview_vectorizer

    # 4. Cast
    cast_encoder = MultiLabelBinarizer(sparse_output=True)
    features["cast"] = cast_encoder.fit_transform(movies["cast"])
    encoders["cast"] = cast_encoder

    # 5. Directors
    director_encoder = MultiLabelBinarizer(sparse_output=True)
    features["director"] = director_encoder.fit_transform(
        movies["director"]
    )
    encoders["director"] = director_encoder

    return features, encoders


def combine_features(features, weights=None):
    """
    Concatenate the blocks into one weighted, L2-normalised matrix.

    Each block is normalised to unit row length and then scaled by
    ``sqrt(weight)``, so the concatenated cosine is the weighted average of the
    per-block cosines.  This is the Strategy A representation and is retained
    for comparison; production ranking uses ``src.models.scoring.BlockScorer``.
    """

    if weights is None:
        weights = {
            name: 1.0 for name in features
        }

    combined_features = []

    for name, weight in weights.items():

        if weight <= 0:
            continue

        matrix = csr_matrix(features[name])

        matrix = normalize(
            matrix,
            norm="l2",
            axis=1
        )

        matrix = matrix.multiply(weight ** 0.5)

        combined_features.append(matrix)

    if not combined_features:
        raise ValueError("At least one feature group is required.")

    return hstack(
        combined_features,
        format="csr"
    )
