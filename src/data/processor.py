import pandas as pd
import ast


DATA_PATH = "data/tmdb-5000/"


def load_datasets():
    movies = pd.read_csv(DATA_PATH + "tmdb_5000_movies.csv")
    credits = pd.read_csv(DATA_PATH + "tmdb_5000_credits.csv")

    return movies, credits


def parse_json_column(value):
    if pd.isna(value):
        return []

    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return []


def extract_names(value):
    items = parse_json_column(value)

    return [item["name"] for item in items if "name" in item]

def extract_director(value):
    crew = parse_json_column(value)

    directors = [
        person["name"]
        for person in crew
        if person.get("job") == "Director"
    ]

    return directors

def process_movies():
    movies, credits = load_datasets()

    # Extract structured information.
    for column in ["genres", "keywords"]:
        movies[column] = movies[column].apply(extract_names)

    # Combine movie information with credits.
    movies = movies.merge(
        credits[["movie_id", "cast", "crew"]],
        left_on="id",
        right_on="movie_id",
        how="left"
    )

    # Extract cast and crew information.
    movies["cast"] = movies["cast"].apply(
    lambda value: extract_names(value)[:5]
    )
    movies["director"] = movies["crew"].apply(extract_director)
    movies.drop(columns=["crew", "movie_id"], inplace=True)

    return movies



if __name__ == "__main__":
    processed_movies = process_movies()

    print("Processed dataset shape:", processed_movies.shape)

    print("\nFirst movie:")
    print(processed_movies.iloc[0][
        ["title", "genres", "keywords", "cast", "director"]
    ].to_string())