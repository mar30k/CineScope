import unittest

import pandas as pd
from fastapi.testclient import TestClient

from api.dependencies import get_cast_service, get_image_service, get_model
from api.main import app
from api.movie_cast import CastCredit


class FakeModel:
    def __init__(self) -> None:
        self.movies = pd.DataFrame(
            [
                {
                    "id": 1,
                    "title": "The Matrix",
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


class FakeCastService:
    def __init__(self, credits: dict[int, list[CastCredit]]) -> None:
        self.credits = credits

    def get_cast(self, movie_id: int) -> list[CastCredit]:
        return self.credits.get(movie_id, [])


class FailingCastService:
    def get_cast(self, movie_id: int) -> list[CastCredit]:
        raise RuntimeError("broken credits")


class ApiCastTests(unittest.TestCase):
    def setUp(self) -> None:
        self.model = FakeModel()
        app.dependency_overrides[get_model] = lambda: self.model
        app.dependency_overrides[get_image_service] = lambda: None
        self.client = TestClient(app)

    def tearDown(self) -> None:
        app.dependency_overrides.clear()

    def test_details_returns_cast_in_order_and_keeps_director(self) -> None:
        app.dependency_overrides[get_cast_service] = lambda: FakeCastService(
            {
                1: [
                    CastCredit("Keanu Reeves", "Neo"),
                    CastCredit("Carrie-Anne Moss", "Trinity"),
                    CastCredit("Hugo Weaving", None),
                ]
            }
        )

        response = self.client.get("/movies/1")

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(
            body["cast"],
            [
                {"name": "Keanu Reeves", "character": "Neo"},
                {"name": "Carrie-Anne Moss", "character": "Trinity"},
                {"name": "Hugo Weaving", "character": None},
            ],
        )
        self.assertEqual(body["director"], ["Lana Wachowski", "Lilly Wachowski"])

    def test_details_returns_empty_cast_when_movie_has_none(self) -> None:
        app.dependency_overrides[get_cast_service] = lambda: FakeCastService({})

        response = self.client.get("/movies/1")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["cast"], [])

    def test_details_survives_cast_lookup_failure(self) -> None:
        app.dependency_overrides[get_cast_service] = lambda: FailingCastService()

        response = self.client.get("/movies/1")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["cast"], [])

    def test_details_without_cast_service_returns_empty_cast(self) -> None:
        app.dependency_overrides[get_cast_service] = lambda: None

        response = self.client.get("/movies/1")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["cast"], [])


if __name__ == "__main__":
    unittest.main()
