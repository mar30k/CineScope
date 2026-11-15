import threading
import unittest
from urllib.parse import parse_qs, urlparse

from api.tmdb_images import ImageAssets, MovieIdentity, TmdbImageService


class FakeClock:
    def __init__(self, now: float = 1_000.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class RecordingTransport:
    def __init__(self, handler) -> None:
        self.handler = handler
        self.calls: list[tuple[str, float]] = []
        self.lock = threading.Lock()

    def __call__(self, url: str, timeout: float):
        with self.lock:
            self.calls.append((url, timeout))
        return self.handler(url)


def tmdb_result(
    title: str,
    *,
    year: str = "1999-03-31",
    original_title: str | None = None,
    poster: str | None = "/poster.jpg",
    backdrop: str | None = "/backdrop.jpg",
) -> dict:
    return {
        "title": title,
        "original_title": original_title or title,
        "release_date": year,
        "poster_path": poster,
        "backdrop_path": backdrop,
    }


def service_with(handler, *, api_key: str | None = "test-key", **kwargs):
    clock = kwargs.pop("clock", None) or FakeClock()
    transport = RecordingTransport(handler)
    service = TmdbImageService(
        api_key,
        transport=transport,
        clock=clock,
        **kwargs,
    )
    return service, transport, clock


class TmdbImageServiceTests(unittest.TestCase):
    def test_exact_title_and_year_returns_sized_image_urls(self) -> None:
        movie = MovieIdentity(1, "The Matrix", 1999)
        service, transport, _ = service_with(
            lambda _url: {"results": [tmdb_result("The Matrix")]}
        )

        assets = service.get_image(movie)

        self.assertEqual(
            assets,
            ImageAssets(
                poster_url="https://image.tmdb.org/t/p/w500/poster.jpg",
                backdrop_url="https://image.tmdb.org/t/p/w1280/backdrop.jpg",
            ),
        )
        self.assertEqual(len(transport.calls), 1)
        self.assertEqual(transport.calls[0][1], 3.0)

    def test_original_title_is_matched(self) -> None:
        movie = MovieIdentity(1, "Spirited Away", 2001)
        service, _, _ = service_with(
            lambda _url: {
                "results": [
                    tmdb_result(
                        "Sen to Chihiro no kamikakushi",
                        original_title="Spirited Away",
                        year="2001-07-20",
                    )
                ]
            }
        )

        self.assertTrue(service.get_image(movie).has_images)

    def test_accents_and_punctuation_are_normalized(self) -> None:
        movie = MovieIdentity(1, "Amelie", 2001)
        service, _, _ = service_with(
            lambda _url: {
                "results": [
                    tmdb_result("Amélie", year="2001-04-25")
                ]
            }
        )

        self.assertTrue(service.get_image(movie).has_images)

    def test_year_mismatch_returns_no_image(self) -> None:
        movie = MovieIdentity(1, "The Matrix", 1999)
        service, _, _ = service_with(
            lambda _url: {"results": [tmdb_result("The Matrix", year="1998-01-01")]}
        )

        assets = service.get_image(movie)

        self.assertEqual(assets, ImageAssets())

    def test_ambiguous_results_return_no_image(self) -> None:
        movie = MovieIdentity(1, "The Matrix", 1999)
        service, _, _ = service_with(
            lambda _url: {
                "results": [
                    tmdb_result("The Matrix"),
                    tmdb_result("The Matrix"),
                ]
            }
        )

        self.assertEqual(service.get_image(movie), ImageAssets())

    def test_positive_result_is_cached(self) -> None:
        movie = MovieIdentity(1, "The Matrix", 1999)
        service, transport, clock = service_with(
            lambda _url: {"results": [tmdb_result("The Matrix")]},
            positive_ttl=100,
        )

        first = service.get_image(movie)
        clock.advance(99)
        second = service.get_image(movie)
        clock.advance(2)
        third = service.get_image(movie)

        self.assertEqual(first, second)
        self.assertEqual(second, third)
        self.assertEqual(len(transport.calls), 2)

    def test_negative_result_is_cached(self) -> None:
        movie = MovieIdentity(1, "Unknown Film", 2020)
        service, transport, clock = service_with(
            lambda _url: {"results": []},
            negative_ttl=50,
        )

        self.assertEqual(service.get_image(movie), ImageAssets())
        clock.advance(49)
        self.assertEqual(service.get_image(movie), ImageAssets())
        self.assertEqual(len(transport.calls), 1)
        clock.advance(2)
        service.get_image(movie)
        self.assertEqual(len(transport.calls), 2)

    def test_failure_is_cached_and_swallowed(self) -> None:
        movie = MovieIdentity(1, "The Matrix", 1999)

        def fail(_url):
            raise TimeoutError("TMDB timed out")

        service, transport, clock = service_with(fail, failure_ttl=10)

        self.assertEqual(service.get_image(movie), ImageAssets())
        self.assertEqual(service.get_image(movie), ImageAssets())
        self.assertEqual(len(transport.calls), 1)
        clock.advance(11)
        service.get_image(movie)
        self.assertEqual(len(transport.calls), 2)

    def test_missing_key_disables_lookups(self) -> None:
        movie = MovieIdentity(1, "The Matrix", 1999)
        service, transport, _ = service_with(lambda _url: {}, api_key=None)

        self.assertFalse(service.enabled)
        self.assertIsNone(service.get_image(movie))
        self.assertEqual(service.get_images_for_movies([movie]), {})
        self.assertEqual(transport.calls, [])

    def test_batch_deduplicates_and_returns_images_by_id(self) -> None:
        def handler(url: str):
            title = parse_qs(urlparse(url).query)["query"][0]
            year = "1999-03-31" if title == "The Matrix" else "2010-07-16"
            return {"results": [tmdb_result(title, year=year)]}

        service, transport, _ = service_with(handler, max_workers=2)
        movies = [
            MovieIdentity(1, "The Matrix", 1999),
            MovieIdentity(2, "Inception", 2010),
            MovieIdentity(1, "The Matrix", 1999),
        ]

        images = service.get_images_for_movies(movies)

        self.assertEqual(set(images), {1, 2})
        self.assertTrue(images[1].has_images)
        self.assertTrue(images[2].has_images)
        self.assertEqual(len(transport.calls), 2)

    def test_missing_poster_paths_become_null_urls(self) -> None:
        movie = MovieIdentity(1, "The Matrix", 1999)
        service, _, _ = service_with(
            lambda _url: {
                "results": [
                    tmdb_result("The Matrix", poster=None, backdrop=None)
                ]
            }
        )

        self.assertEqual(service.get_image(movie), ImageAssets())


if __name__ == "__main__":
    unittest.main()
