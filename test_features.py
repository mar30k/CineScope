
from src.data.processor import process_movies
from src.features.feature_builder import build_features, combine_features


# Load processed movie data.
movies = process_movies()

# Generate numerical feature matrices.
features = build_features(movies)
combined = combine_features(features)

print("\nFEATURE MATRIX SUMMARY")
print("----------------------")

for name, matrix in features.items():
    print(f"{name}: {matrix.shape}")
print("\nCOMBINED FEATURE MATRIX")
print("-----------------------")
print("Shape:", combined.shape)
print("Format:", combined.format)
print("Non-zero values:", combined.nnz)
print("Total possible values:", combined.shape[0] * combined.shape[1])