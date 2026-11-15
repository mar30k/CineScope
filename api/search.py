"""
Title search for the movie catalogue.

The search is a single in-memory pass over pre-computed normalized titles.
Everything that does not depend on the query - normalization, tokenization,
the year of each movie - is computed once when the index is built during
application startup, so a search costs one scan over 4,803 rows and never
rebuilds an index per request.

Matching levels, strongest first
--------------------------------
0. Exact normalized title (``The Dark Knight`` == query).
1. Prefix of the title, ignoring leading stop words (``dark kn``).
2. Contiguous phrase inside the title (``of the apes``).
3. Every meaningful query token appears in the title, in any order and
   regardless of stop words (``lord rings``, ``knight dark``).
4. Fuzzy token match for misspellings (``planent of the apes``).

A weaker level can never outrank a stronger one: the level is the first
component of the sort key. Within a level the ordering is deterministic
(shorter/better matches first, then title, then movie id), so duplicate
titles stay separate and results are stable across requests.

Ranking scores are relevance tiers only. Nothing here touches the KNN
features, the recommendation scorer or the catalogue itself.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher

# Words that carry no discriminating power in a title search. Query tokens
# matching only these are ignored when comparing token sets, so "lord rings"
# still finds "The Lord of the Rings".
STOPWORDS = frozenset(
    """
    a an and are as at be but by for from has have in into is it its of on or
    over that the their then there these they this to was were will with
    el la le les los las il lo gli un una une du des der die das
    """.split()
)

# Queries shorter than this only get prefix/exact treatment: substring and
# fuzzy matching on two characters would return most of the catalogue.
SHORT_QUERY_LENGTH = 3

# Fuzzy matching needs enough material to be meaningful.
FUZZY_MIN_QUERY_LENGTH = 4

# Every query token must match a title token at least this closely, and the
# mean over the query tokens must reach MIN_FUZZY_SCORE.
MIN_TOKEN_SIMILARITY = 0.78
MIN_FUZZY_SCORE = 0.85

_YEAR_TOKEN = re.compile(r"^(?:19|20)\d{2}$")
_NON_ALPHANUMERIC = re.compile(r"[^a-z0-9]+")
_APOSTROPHES = str.maketrans({"\u2019": "'", "\u2018": "'", "`": "'"})


def normalize_title(value: object) -> str:
    """Fold case, accents and punctuation, and collapse whitespace.

    ``"The  Dark Knight's"`` -> ``"the dark knights"``, ``"Fast & Furious"``
    -> ``"fast and furious"``. The result is what every match level compares
    against, so punctuation and spacing never affect a match.
    """
    text = unicodedata.normalize("NFKD", str(value))
    text = "".join(char for char in text if not unicodedata.combining(char))
    text = text.translate(_APOSTROPHES).casefold()
    text = text.replace("&", " and ").replace("'", "")
    text = _NON_ALPHANUMERIC.sub(" ", text)
    return " ".join(text.split())


def meaningful_tokens(normalized: str) -> tuple[str, ...]:
    """Tokens of a normalized string with stop words removed."""
    return tuple(token for token in normalized.split() if token not in STOPWORDS)


def _strip_leading_stopwords(normalized: str) -> str:
    """Drop leading articles so "the dark knight" also matches "dark kn"."""
    tokens = normalized.split()
    start = 0
    while start < len(tokens) - 1 and tokens[start] in STOPWORDS:
        start += 1
    return " ".join(tokens[start:])


@dataclass(frozen=True, slots=True)
class _Entry:
    """One catalogue row, with everything the matcher needs pre-computed."""

    movie_id: int
    title: str
    normalized: str
    core: str
    tokens: frozenset[str]
    token_list: tuple[str, ...]
    token_count: int
    year: int | None


def _years(movies) -> list[int | None]:
    """Release year per row, or ``None`` everywhere when unavailable."""
    try:
        release_dates = movies["release_date"]
    except (KeyError, TypeError):
        return [None] * len(movies)

    years: list[int | None] = []
    for value in release_dates:
        text = "" if value is None else str(value)
        match = _YEAR_TOKEN.match(text.strip()[:4]) if text.strip() else None
        years.append(int(match.group(0)) if match else None)
    return years


class MovieSearchIndex:
    """Immutable, pre-normalized index over the catalogue's movie titles."""

    def __init__(self, entries: tuple[_Entry, ...]) -> None:
        self._entries = entries

    @property
    def n_titles(self) -> int:
        return len(self._entries)

    @classmethod
    def build(cls, movies) -> "MovieSearchIndex":
        """Build the index from the loaded catalogue frame (once, at startup)."""
        ids = list(movies["id"])
        titles = list(movies["title"].astype(str))
        years = _years(movies)

        entries: list[_Entry] = []
        for movie_id, title, year in zip(ids, titles, years):
            normalized = normalize_title(title)
            tokens = meaningful_tokens(normalized)
            entries.append(
                _Entry(
                    movie_id=int(movie_id),
                    title=title,
                    normalized=normalized,
                    core=_strip_leading_stopwords(normalized),
                    tokens=frozenset(tokens),
                    token_list=tokens,
                    token_count=len(tokens),
                    year=year,
                )
            )
        return cls(tuple(entries))

    # ------------------------------------------------------------- search
    def search(self, query: str, limit: int = 10) -> list[int]:
        """Ranked movie ids for ``query``; at most ``limit`` of them."""
        if limit <= 0:
            return []

        normalized, year = _parse_query(query)
        if not normalized:
            return []

        query_tokens = meaningful_tokens(normalized)
        query_token_set = frozenset(query_tokens)

        short_query = len(normalized) < SHORT_QUERY_LENGTH
        allow_phrase_tier = not short_query
        allow_token_tier = not short_query and len(query_tokens) >= 2
        allow_fuzzy = (
            not short_query
            and len(normalized) >= FUZZY_MIN_QUERY_LENGTH
            and len(query_tokens) >= 1
        )

        ranked: list[tuple[int, int, int, int, str, int]] = []
        for entry in self._entries:
            tier = self._tier(
                entry,
                normalized,
                query_token_set,
                query_tokens,
                allow_phrase_tier=allow_phrase_tier,
                allow_token_tier=allow_token_tier,
                allow_fuzzy=allow_fuzzy,
            )
            if tier is None:
                continue

            # A query year never filters; it only separates matching titles.
            year_miss = 0 if year is None or entry.year == year else 1

            if tier == 4:
                score = self._fuzzy_score(query_tokens, entry)
                secondary = -int(round(score * 1000))
                tertiary = entry.token_count
            elif tier == 3:
                secondary = entry.token_count - len(query_token_set)
                tertiary = len(entry.normalized)
            else:
                secondary = len(entry.normalized)
                tertiary = entry.token_count

            ranked.append(
                (
                    tier,
                    year_miss,
                    secondary,
                    tertiary,
                    entry.normalized,
                    entry.movie_id,
                )
            )

        ranked.sort()
        return [row[5] for row in ranked[:limit]]

    # ------------------------------------------------------------ helpers
    def _tier(
        self,
        entry: _Entry,
        normalized: str,
        query_token_set: frozenset[str],
        query_tokens: tuple[str, ...],
        *,
        allow_phrase_tier: bool,
        allow_token_tier: bool,
        allow_fuzzy: bool,
    ) -> int | None:
        """Strongest matching level for one entry, or ``None`` for no match."""
        if entry.normalized == normalized or entry.core == normalized:
            return 0
        if entry.normalized.startswith(normalized) or entry.core.startswith(normalized):
            return 1
        if allow_phrase_tier and (
            normalized in entry.normalized
            or (entry.core and normalized in entry.core)
        ):
            return 2
        if allow_token_tier and query_token_set and query_token_set <= entry.tokens:
            return 3
        if allow_fuzzy and self._fuzzy_score(query_tokens, entry) >= MIN_FUZZY_SCORE:
            return 4
        return None

    @staticmethod
    def _fuzzy_score(query_tokens: tuple[str, ...], entry: _Entry) -> float:
        """Mean best per-token similarity, or 0 when any token looks unrelated.

        Comparing token to token rather than whole title to whole query keeps
        "planent of the apes" close to "planet of the apes" without letting a
        long, unrelated title win on length alone.

        Two cheap prefilters keep this off the hot path. A multi-word query
        must share at least one exact token with the title ("apes" in
        "planent of the apes"), and a single-word query must be able to match
        a title token that starts the same way and is about the same length.
        Only then is the expensive per-token similarity computed.
        """
        if not query_tokens or not entry.token_list:
            return 0.0

        if len(query_tokens) > 1:
            if not any(token in entry.tokens for token in query_tokens):
                return 0.0
        else:
            token = query_tokens[0]
            prefix = token[:2]
            if not any(
                title_token.startswith(prefix)
                and abs(len(title_token) - len(token)) <= 3
                for title_token in entry.token_list
            ):
                return 0.0

        total = 0.0
        for query_token in query_tokens:
            best = 0.0
            for title_token in entry.token_list:
                if (
                    query_token[0] != title_token[0]
                    and query_token[-1] != title_token[-1]
                ):
                    continue
                ratio = SequenceMatcher(None, query_token, title_token).ratio()
                if ratio > best:
                    best = ratio
                    if best == 1.0:
                        break
            if best < MIN_TOKEN_SIMILARITY:
                return 0.0
            total += best

        return total / len(query_tokens)


def _parse_query(query: str) -> tuple[str, int | None]:
    """Normalize a query and pull out a trailing release year, when present.

    A year is only separated when something other than stop words remains, so
    a film actually called "1917" still searches by its title.
    """
    normalized = normalize_title(query)
    if not normalized:
        return "", None

    tokens = normalized.split()
    year_tokens = [token for token in tokens if _YEAR_TOKEN.match(token)]
    if not year_tokens:
        return normalized, None

    candidate = year_tokens[-1]
    remaining = [token for token in tokens if token != candidate]
    if not any(token not in STOPWORDS for token in remaining):
        return normalized, None

    return " ".join(remaining), int(candidate)
