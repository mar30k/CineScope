
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from src.data.processor import process_movies
from src.features.feature_builder import (
    build_features,
    combine_features
)


# Each experiment tests a different combination of movie features.


EXPERIMENTS = {
    "J - Content Baseline": {
        "genres": 1.0,
        "keywords": 1.0,
        "overview": 1.0
    },

    "K - Add Cast": {
        "genres": 1.0,
        "keywords": 1.0,
        "overview": 1.0,
        "cast": 0.5
    },

    "L - Light Director": {
        "genres": 1.0,
        "keywords": 1.0,
        "overview": 1.0,
        "cast": 0.5,
        "director": 0.25
    },

    "M - Moderate Director": {
        "genres": 1.0,
        "keywords": 1.0,
        "overview": 1.0,
        "cast": 0.5,
        "director": 0.5
    },

    "N - Full Director": {
        "genres": 1.0,
        "keywords": 1.0,
        "overview": 1.0,
        "cast": 0.5,
        "director": 1.0
    }
}


TEST_MOVIES = [
    "Avatar",
    "The Dark Knight",
    "Finding Nemo"
]


def build_experiment_model(features, weights):

    combined = combine_features(
        features,
        weights=weights
    )

    model = NearestNeighbors(
        metric="cosine",
        algorithm="brute"
    )

    model.fit(combined)

    return model, combined


def get_recommendations(
    movies,
    model,
    combined_features,
    movie_title,
    n_recommendations=5
):

    matches = movies.index[
        movies["title"].str.lower() == movie_title.lower()
    ]

    if len(matches) == 0:
        raise ValueError(f"Movie not found: {movie_title}")

    movie_index = matches[0]

    distances, indices = model.kneighbors(
        combined_features[movie_index],
        n_neighbors=min(
            n_recommendations + 1,
            len(movies)
        )
    )

    results = []

    for index, distance in zip(indices[0], distances[0]):

        # Exclude the selected movie itself.
        if index == movie_index:
            continue

        movie = movies.iloc[index]

        results.append({
            "title": movie["title"],
            "similarity": round(1 - distance, 4)
        })

        if len(results) == n_recommendations:
            break

    return pd.DataFrame(results)


def main():

    print("Loading and processing movies...")

    movies = process_movies()

    print("Building feature matrices...")

    features = build_features(movies)

    print("\nStarting experiments...")

    for experiment_name, weights in EXPERIMENTS.items():

        print("\n" + "=" * 65)
        print(experiment_name)
        print("=" * 65)

        model, combined = build_experiment_model(
            features,
            weights
        )

        print("Combined matrix shape:", combined.shape)

        for movie_title in TEST_MOVIES:

            try:
                recommendations = get_recommendations(
                    movies,
                    model,
                    combined,
                    movie_title
                )

                print(f"\nRecommendations for: {movie_title}")
                print(recommendations.to_string(index=False))

            except ValueError as error:
                print(f"\n{error}")


if __name__ == "__main__":
    main()
