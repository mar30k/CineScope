import pandas as pd

movies = pd.read_csv(
    "data/tmdb-5000/tmdb_5000_movies.csv"
)

credits = pd.read_csv(
    "data/tmdb-5000/tmdb_5000_credits.csv"
)

print("MOVIES DATASET")
print("Shape:", movies.shape)
print("Columns:", movies.columns.tolist())

print("\nCREDITS DATASET")
print("Shape:", credits.shape)
print("Columns:", credits.columns.tolist())

print("\nMOVIE SAMPLE")
print(
    movies[
        ["title", "overview", "genres", "keywords"]
    ].head(3).to_string()
)