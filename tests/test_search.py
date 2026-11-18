import unittest

import pandas as pd
from fastapi.testclient import TestClient

from api.dependencies import get_image_service, get_model, get_search_index
from api.main import app
from api.search import MovieSearchIndex, normalize_title

CATALOGUE = [
    (1, "The Dark Knight", "2008-07-16"),
    (2, "The Dark Knight Rises", "2012-07-16"),
    (3, "The Lord of the Rings: The Fellowship of the Ring", "2001-12-18"),
    (4, "The Lord of the Rings: The Two Towers", "2002-12-18"),
    (5, "Planet of the Apes", "1968-02-07"),
    (6, "Planet of the Apes", "2001-07-25"),
    (7, "Batman Begins", "2005-06-10"),
    (8, "Fast & Furious", "2009-04-03"),
    (9, "Rise of the Planet of the Apes", "2011-08-05"),
    (10, "Dawn of the Planet of the Apes", "2014-06-26"),
    (11, "War for the Planet of the Apes", "2017-07-11"),
    (12, "Avatar", "2009-12-15"),
    (13, "The Vatican Tapes", "2015-07-24"),
]

GENRES = {
    1: ["Action", "Crime"],
    2: ["Action", "Crime"],
    3: ["Adventure", "Fantasy"],
    4: ["Adventure", "Fantasy"],
    5: ["Science Fiction"],
    6: ["Science Fiction"],
    7: ["Action", "Crime"],
    8: ["Action", "Crime"],
    9: ["Science Fiction"],
    10: ["Science Fiction"],
    11: ["Science Fiction"],
    12: ["Science Fiction"],
    13: ["Horror"],
}


def build_frame():
    return pd.DataFrame(
        [
            {"id": movie_id, "title": title, "release_date": released}
            for movie_id, title, released in CATALOGUE
        ]
    )


class FakeModel:
    """Minimal stand-in exposing only what the search route reads."""

    def __init__(self) -> None:
        self.movies = build_frame()

    def get_movie(self, movie_id: int) -> dict:
        row = self.movies[self.movies["id"] == movie_id].iloc[0]
        year = str(row["release_date"])[:4]
        return {
            "movie_id": int(row["id"]),
            "title": str(row["title"]),
            "year": int(year),
            "genres": GENRES[int(row["id"])],
            "director": [],
        }


class NormalizationTests(unittest.TestCase):
    def test_case_punctuation_and_whitespace_are_folded(self) -> None:
        self.assertEqual(normalize_title("  The   Dark Knight  "), "the dark knight")
        self.assertEqual(normalize_title("PLANET OF THE APES"), "planet of the apes")
        self.assertEqual(normalize_title("Fast & Furious"), "fast and furious")
        self.assertEqual(normalize_title("Schindler's List"), "schindlers list")


class SearchIndexTests(unittest.TestCase):
    def setUp(self) -> None:
        self.index = MovieSearchIndex.build(build_frame())

    def test_exact_title_ranks_first(self) -> None:
        self.assertEqual(self.index.search("The Dark Knight", 10)[0], 1)

    def test_partial_title(self) -> None:
        results = self.index.search("dark kn", 10)
        self.assertEqual(results[0], 1)
        self.assertIn(2, results)

    def test_missing_stop_words(self) -> None:
        results = self.index.search("lord rings", 10)
        self.assertIn(3, results)
        self.assertIn(4, results)

    def test_different_word_order(self) -> None:
        self.assertEqual(self.index.search("knight dark", 10)[0], 1)

    def test_typo_fuzzy_match(self) -> None:
        results = self.index.search("planent of the apes", 10)
        self.assertIn(5, results)
        self.assertIn(6, results)

    def test_case_variation(self) -> None:
        self.assertIn(5, self.index.search("PLANET OF THE APES", 10))

    def test_leading_trailing_and_repeated_whitespace(self) -> None:
        self.assertEqual(
            self.index.search("   dark    kn  ", 10),
            self.index.search("dark kn", 10),
        )

    def test_duplicate_titles_stay_separate_records(self) -> None:
        results = self.index.search("planet of the apes", 10)
        self.assertEqual(results[:2], [5, 6])

    def test_franchise_titles_follow_exact_duplicates(self) -> None:
        results = self.index.search("planet of the apes", 10)
        self.assertEqual(results[:2], [5, 6])
        self.assertEqual(set(results[2:]), {9, 10, 11})

    def test_word_start_phrase_beats_mid_word_fragment(self) -> None:
        results = self.index.search("apes", 10)
        self.assertEqual(results[:2], [5, 6])
        self.assertEqual(set(results[:5]), {5, 6, 9, 10, 11})
        self.assertIn(13, results)
        self.assertGreater(results.index(13), 4)

    def test_fuzzy_match_finds_misspelled_title(self) -> None:
        self.assertEqual(self.index.search("avatr", 10), [12])

    def test_very_short_query_is_not_fuzzy(self) -> None:
        self.assertEqual(self.index.search("zz", 10), [])

    def test_no_matches(self) -> None:
        self.assertEqual(self.index.search("zzzzzzzzzz", 10), [])

    def test_year_separates_same_titled_movies(self) -> None:
        self.assertEqual(self.index.search("planet of the apes 2001", 10)[0], 6)
        self.assertEqual(self.index.search("planet of the apes 1968", 10)[0], 5)

    def test_exact_match_outranks_prefix_and_fuzzy(self) -> None:
        results = self.index.search("batman", 10)
        self.assertEqual(results, [7])

    def test_weak_fuzzy_match_does_not_surface(self) -> None:
        # "zzzz" is length 4, so fuzzy matching runs but nothing is close.
        self.assertEqual(self.index.search("zzzz", 10), [])

    def test_empty_query_returns_nothing(self) -> None:
        self.assertEqual(self.index.search("   ", 10), [])

    def test_limit_is_respected(self) -> None:
        self.assertEqual(len(self.index.search("the", 1)), 1)


class SearchEndpointTests(unittest.TestCase):
    def setUp(self) -> None:
        self.model = FakeModel()
        self.index = MovieSearchIndex.build(self.model.movies)
        app.dependency_overrides[get_model] = lambda: self.model
        app.dependency_overrides[get_image_service] = lambda: None
        app.dependency_overrides[get_search_index] = lambda: self.index
        self.client = TestClient(app)

    def tearDown(self) -> None:
        app.dependency_overrides.clear()

    def test_response_contract_includes_genres(self) -> None:
        response = self.client.get("/movies/search", params={"q": "dark kn"})

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["query"], "dark kn")
        self.assertEqual(body["count"], len(body["results"]))
        self.assertEqual(
            set(body["results"][0]),
            {"movie_id", "title", "year", "genres", "image"},
        )
        self.assertEqual(body["results"][0]["movie_id"], 1)
        self.assertEqual(body["results"][0]["genres"], ["Action", "Crime"])

    def test_no_match_returns_empty_results(self) -> None:
        response = self.client.get("/movies/search", params={"q": "zzzzzzzzzz"})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"query": "zzzzzzzzzz", "count": 0, "results": []})

    def test_blank_query_is_rejected(self) -> None:
        response = self.client.get("/movies/search", params={"q": "   "})

        self.assertEqual(response.status_code, 422)

    def test_query_shorter_than_minimum_is_rejected(self) -> None:
        response = self.client.get("/movies/search", params={"q": "a"})

        self.assertEqual(response.status_code, 422)

    def test_invalid_limits_are_rejected(self) -> None:
        for limit in (0, 51):
            with self.subTest(limit=limit):
                response = self.client.get(
                    "/movies/search", params={"q": "dark kn", "limit": limit}
                )
                self.assertEqual(response.status_code, 422)

    def test_duplicate_titles_keep_their_own_ids(self) -> None:
        response = self.client.get("/movies/search", params={"q": "planet of the apes"})

        ids = [item["movie_id"] for item in response.json()["results"]]
        self.assertEqual(ids[:2], [5, 6])
        self.assertEqual(set(ids[2:]), {9, 10, 11})

    def test_fuzzy_suggestion_is_returned_by_endpoint(self) -> None:
        response = self.client.get("/movies/search", params={"q": "avatr"})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            [item["movie_id"] for item in response.json()["results"]], [12]
        )

    def test_deterministic_across_identical_requests(self) -> None:
        first = self.client.get("/movies/search", params={"q": "the"}).json()
        second = self.client.get("/movies/search", params={"q": "the"}).json()

        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
