import csv
import json
import tempfile
import unittest
from pathlib import Path

from api.movie_cast import CastCredit, MovieCastService, parse_cast


def credit(name, character=None):
    return {"name": name, "character": character}


class ParseCastTests(unittest.TestCase):
    def test_preserves_source_order_and_limits_to_eight(self) -> None:
        raw = json.dumps([credit(f"Actor {index}", f"Role {index}") for index in range(10)])

        cast = parse_cast(raw)

        self.assertEqual(len(cast), 8)
        self.assertEqual([member.name for member in cast], [f"Actor {index}" for index in range(8)])
        self.assertEqual(cast[0], CastCredit("Actor 0", "Role 0"))

    def test_keeps_character_when_present_and_null_when_blank(self) -> None:
        raw = json.dumps([credit("Actor One", "Role One"), credit("Actor Two", "  ")])

        cast = parse_cast(raw)

        self.assertEqual(cast[0].character, "Role One")
        self.assertIsNone(cast[1].character)

    def test_skips_malformed_entries_but_keeps_valid_ones(self) -> None:
        raw = json.dumps(
            [
                "not a person",
                {"character": "No name"},
                credit("Actor Three", "Role Three"),
            ]
        )

        cast = parse_cast(raw)

        self.assertEqual(cast, [CastCredit("Actor Three", "Role Three")])

    def test_missing_or_malformed_cast_returns_empty(self) -> None:
        for value in (None, "", "not valid python", "{}", json.dumps([])):
            with self.subTest(value=value):
                self.assertEqual(parse_cast(value), [])


class MovieCastServiceTests(unittest.TestCase):
    def test_loads_credits_in_csv_order_and_returns_empty_for_unknown_movie(self) -> None:
        rows = [
            {
                "movie_id": "19995",
                "title": "Avatar",
                "cast": json.dumps(
                    [
                        credit("Sam Worthington", "Jake Sully"),
                        credit("Zoe Saldana", "Neytiri"),
                    ]
                ),
                "crew": "[]",
            },
            {
                "movie_id": "1",
                "title": "No Cast",
                "cast": "[]",
                "crew": "[]",
            },
        ]

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "credits.csv"
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)

            service = MovieCastService.load(path)

        self.assertEqual(service.n_movies, 1)
        self.assertEqual(
            service.get_cast(19995),
            [
                CastCredit("Sam Worthington", "Jake Sully"),
                CastCredit("Zoe Saldana", "Neytiri"),
            ],
        )
        self.assertEqual(service.get_cast(1), [])
        self.assertEqual(service.get_cast(99999), [])

    def test_missing_file_returns_an_empty_service(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            service = MovieCastService.load(Path(directory) / "missing.csv")

        self.assertEqual(service.n_movies, 0)
        self.assertEqual(service.get_cast(19995), [])


if __name__ == "__main__":
    unittest.main()
