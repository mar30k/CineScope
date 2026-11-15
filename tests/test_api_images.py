import unittest

import pandas as pd
from fastapi.testclient import TestClient

from api.dependencies import get_image_service, get_model
from api.main import app
from api.tmdb_images import ImageAssets


class FakeModel:
    def __init__(self) -> None:
        self.movies = pd.DataFrame(
            [
                {
                    "id": 1,
                    "title": "The Matrix",
                    "year": 1999,
                    "overview": "A hacker learns the truth.",
                    "genres": "Action",
                    "runtime": 136.0,
                    "vote_average": 8.2,
                    "vote_count": 25000,
                    "director": "Lana Wachowski, Lilly Wachowski",
                }
            ]
        )

    def get_movie(self, movie_id: int) -> dict:
        if movie_id != 1:
            raise KeyError(movie_id)
        return {
            "movie_id": 1,
            "title": "The Matrix",
            "year": 1999,
            "genres": ["Action", "Science Fiction"],
            "director": ["Lana Wachowski", "Lilly Wachowski"],
        }

    def resolve(self, movie_id: int) -> int:
        if movie_id != 1:
            raise KeyError(movie_id)
        return 0

    def recommend(self, movie_id: int, n_recommendations: int) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "movie_id": 2,
                    "title": "The Matrix Reloaded",
                    "year": 2003,
                    "rank": 1,
                    "score": 0.82,
                    "coverage": 1.0,
                    "agreement": 0.82,
                }
            ]
        )


class FakeImageService:
    def get_images_for_movies(self, movies) -> dict[int, ImageAssets]:
        return {
            movie.movie_id: ImageAssets(
                poster_url=f"https://image.tmdb.org/t/p/w500/{movie.movie_id}.jpg",
                backdrop_url=f"https://image.tmdb.org/t/p/w1280/{movie.movie_id}-bg.jpg",
            )
            for movie in movies
        }


class ApiImageEnrichmentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.model = FakeModel()
        self.images = FakeImageService()
        app.dependency_overrides[get_model] = lambda: self.model
        app.dependency_overrides[get_image_service] = lambda: self.images
        self.client = TestClient(app)

    def tearDown(self) -> None:
        app.dependency_overrides.clear()

    def test_search_includes_images(self) -> None:
        response = self.client.get("/movies/search", params={"q": "matrix"})

        self.assertEqual(response.status_code, 200)
        item = response.json()["results"][0]
        self.assertEqual(
            item["image"]["poster_url"],
            "https://image.tmdb.org/t/p/w500/1.jpg",
        )

    def test_details_includes_poster_and_backdrop(self) -> None:
        response = self.client.get("/movies/1")

        self.assertEqual(response.status_code, 200)
        image = response.json()["image"]
        self.assertEqual(image["poster_url"], "https://image.tmdb.org/t/p/w500/1.jpg")
        self.assertEqual(
            image["backdrop_url"],
            "https://image.tmdb.org/t/p/w1280/1-bg.jpg",
        )

    def test_recommendations_include_images(self) -> None:
        response = self.client.get("/recommendations/1", params={"n": 1})

        self.assertEqual(response.status_code, 200)
        item = response.json()["recommendations"][0]
        self.assertEqual(
            item["image"]["poster_url"],
            "https://image.tmdb.org/t/p/w500/2.jpg",
        )


if __name__ == "__main__":
    unittest.main()
