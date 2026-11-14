
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from src.data.processor import process_movies
from src.features.feature_builder import (
    build_features,
    combine_features
)


# Compare three representative configurations.
EXPERIMENTS = {
    "J - Content Baseline": {
        "genres": 1.0,
        "keywords": 1.0,
        "overview": 1.0
    },
    "L - Light Director": {
        "genres": 1.0,
        "keywords": 1.0,
        "overview": 1.0,
        "cast": 0.5,
        "director": 0.25
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
    "Finding Nemo",
    "The Godfather",
    "Jurassic Park"
]


def generate_recommendations(
    movies,
    model,
    matrix,
    movie_title,
    experiment_name,
    limit=10
):

    matches = movies.index[
        movies["title"].str.lower() == movie_title.lower()
    ]

    if len(matches) == 0:
        print(f"Skipping missing movie: {movie_title}")
        return []

    movie_index = matches[0]

    distances, indices = model.kneighbors(
        matrix[movie_index],
        n_neighbors=min(limit + 1, len(movies))
    )

    results = []

    for index, distance in zip(indices[0], distances[0]):

        if index == movie_index:
            continue

        movie = movies.iloc[index]

        results.append({
            "query_movie": movie_title,
            "experiment": experiment_name,
            "rank": len(results) + 1,
            "recommended_movie": movie["title"],
            "genres": ", ".join(movie["genres"]),
            "director": ", ".join(movie["director"]),
            "similarity": round(1 - distance, 4),
            "relevance": ""
        })

        if len(results) == limit:
            break

    return results


def main():

    print("Loading movies...")
    movies = process_movies()

    print("Building feature matrices...")
    features = build_features(movies)

    all_results = []

    for experiment_name, weights in EXPERIMENTS.items():

        print(f"\nRunning {experiment_name}...")

        matrix = combine_features(
            features,
            weights=weights
        )

        model = NearestNeighbors(
            metric="cosine",
            algorithm="brute"
        )

        model.fit(matrix)

        for movie_title in TEST_MOVIES:

            results = generate_recommendations(
                movies,
                model,
                matrix,
                movie_title,
                experiment_name
            )

            all_results.extend(results)

    results_df = pd.DataFrame(all_results)

    output_path = "data/recommendation_review.csv"

    results_df.to_csv(
        output_path,
        index=False
    )

    print("\nEvaluation dataset created.")
    print("Total recommendations:", len(results_df))
    print("Saved to:", output_path)


if __name__ == "__main__":
    main()
