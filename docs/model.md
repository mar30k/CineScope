# CineScope — Model Documentation

CineScope is a content-based movie recommender: given one movie, it ranks the
rest of the catalogue by how similar their *content* is. It uses no user data,
so it has no cold-start problem for users and no personalisation.

- **Model version:** 1.0.0
- **Catalogue:** 4,803 movies (TMDB 5000)
- **Configuration:** `L - Light Director`
- **Scoring:** Strategy B (fixed-denominator weighted content agreement)

---

## 1. Model overview

### Dataset

`data/tmdb-5000/tmdb_5000_movies.csv` joined to `tmdb_5000_credits.csv` on
movie id. The join is complete in both directions (no orphans). After
processing: 4,803 rows × 22 columns. Three pairs of movies share a title
(e.g. *Batman* 1989 and *Batman* 1966), which is why movie **id** is the
primary query identifier everywhere.

### Features

| Block | Technique | Dimensions | Coverage |
|---|---|---:|---:|
| genres | MultiLabelBinarizer | 20 | 99.4% |
| keywords | TF-IDF, English stop words | 7,069 | 91.4% |
| overview | TF-IDF, English stop words, `max_features=5000` | 5,000 | 99.9% |
| cast | MultiLabelBinarizer, top 5 billed | 9,390 | 99.1% |
| director | MultiLabelBinarizer | 2,577 | 99.4% |
| **combined** | legendre-free sparse `hstack` | **24,056** | — |

The combined matrix has 190,053 non-zeros (0.16% dense). Every block is
L2-normalised per row before use, so each block contributes on a comparable
scale regardless of how many features a movie has in it.

### Selected weights

| Block | Weight |
|---|---:|
| genres | 1.00 |
| keywords | 1.00 |
| overview | 1.00 |
| cast | 0.50 |
| director | 0.25 |
| **total** | **3.75** |

Defined once, in `src/config.py`. Production and the experimental scripts both
read them from there.

### Nearest-neighbour search

Ranking is a **brute-force k-nearest-neighbour scan** over the catalogue using
cosine similarity per block — no approximate index, no fitted tree. For 4,803
movies a full scan takes ~1.5 ms (~9 ms end-to-end including per-result
agreement and coverage), so an index would add complexity without changing
the result. `sklearn.neighbors.NearestNeighbors(metric="cosine",
algorithm="brute")` is retained only in the Strategy A comparison harness.

---

## 2. Recommendation scoring (Strategy B)

Let `b` range over the feature blocks, `w_b` be the weight of block `b`,
`A_b(x) ∈ {0,1}` indicate whether movie `x` has any data in that block, and
`c_b(x,y) ∈ [0,1]` be the cosine similarity of the two block rows.

```
N(x,y) = Σ_b w_b · A_b(x) · A_b(y) · c_b(x,y)

S_B(x,y) = N(x,y) / Σ_b w_b
```

**In plain language.** For each of the five feature groups, compare the two
movies and get a similarity between 0 and 1. Multiply each by that group's
weight and add them up. Then divide by the total weight — always 3.75,
whatever the two movies happen to contain.

**Why the denominator is fixed.** Each movie's block is normalised to length
one, so a movie with three genres and a movie with twenty keywords both look
"equally full" to the maths. If the denominator were built from what the two
movies actually share, a movie with almost no metadata would divide by a tiny
number and collect an inflated score. Keeping the denominator at the full
3.75 means **a missing block contributes zero to the numerator while the
denominator stays whole** — the similarity can only go down, never up.

**Two numbers are reported with every score**, because the score is their
product:

```
agreement = N(x,y) / Σ_{shared b} w_b      how well the shared blocks match
coverage  = Σ_{shared b} w_b / Σ_b w_b     how much evidence there is

S_B = coverage × agreement
```

A score of 0.30 with coverage 1.00 means "weak but fully evidenced match"; the
same 0.30 with coverage 0.27 means "one shared block out of five". They are
not the same claim, which is why coverage is always shown.

The score is a **ranking score in [0, 1], not a probability** that a user will
like the movie.

> For two movies with complete metadata, Strategy B gives exactly the same
> number as the old concatenated cosine (Strategy A). The two differ only when
> metadata is missing — verified to 5.6e-17 over 200 random complete pairs.
> 4,375 of the 4,803 movies are complete; 428 are partial and 15 have a single
> block.

---

## 3. Evaluation

### Method

Five query movies (Avatar, The Dark Knight, Finding Nemo, The Godfather,
Jurassic Park) produced a top-10 under each strategy. The union of both lists
was judged on a 0/1/2 rubric by a single rater: **3 judgements from the project
owner, 56 AI-assisted.** Both strategies are scored against the same pooled
ideal ranking, so neither is rewarded for defining the pool.

- Precision@k: fraction of the top *k* with relevance ≥ 1.
- NDCG@k: gain `2^relevance − 1`, discounted by `log₂(rank + 1)`.

### Results (macro average over 5 queries)

| Metric | Strategy A | Strategy B |
|---|---:|---:|
| Precision@5 | 0.76 | **0.88** |
| Precision@10 | 0.62 | **0.70** |
| NDCG@5 | 0.8465 | **0.9166** |
| NDCG@10 | 0.8737 | **0.9361** |

Strategy B is identical to A on the two queries whose lists already contained
only complete-metadata films, and better on the other three. Across the three
affected queries B removed 9 items (mean relevance 0.33, all with incomplete
metadata) and added 9 (mean relevance 0.89). The largest single contributor
was *Amidst the Devil's Wings*, a film with a 29-character synopsis, no
keywords, no cast and no director, which A placed at rank 3 for
*The Dark Knight* and rank 2 for *The Godfather*.

Full detail: `reports/strategy_ab_evaluation_report.md`.

### Honest status of these numbers

The comparison is **preliminary**. 56 of 59 judgements were produced by an AI
reading short dataset synopses, not by people watching the films. There is one
rater and no inter-rater agreement figure. The improvement is consistent in
direction but small in absolute terms, and part of it rests on a single
low-confidence judgement about a film the dataset barely describes.

---

## 4. Limitations

**Catalogue and data**
- 4,803 movies from a fixed 2017 snapshot; no new releases, no streaming
  availability, no ratings or popularity in the ranking.
- Metadata is uneven: 8.6% of movies have no keywords, 0.9% no cast, 0.6% no
  director and 0.1% no overview.
- `overview` is capped at 5,000 TF-IDF features. Five movies lose every
  overview term to that cap and end up with an empty overview block.

**Ranking**
- Content-based similarity assumes "similar content is what you want next".
  It cannot produce serendipity or novelty, and it over-specialises: the top
  match for *Avatar* is *Star Trek Into Darkness*, and for *The Godfather* it
  is its own sequel.
- Cast and director blocks are 63–66% one-movie-only dimensions, so much of
  their signal can never match anything.
- Keywords are phrase labels ("based on novel") that TF-IDF splits into words,
  so the block's most frequent dimensions are boilerplate.
- Scores are not comparable to user-facing expectations; 0.36 is a *typical*
  top-1 match, not a poor one.

**Evaluation**
- Five queries, all well-known mainstream films. Nothing here says how the
  model behaves on documentaries, foreign-language films or obscure entries —
  exactly the population where metadata is thinnest.
- Judgements are AI-assisted and subjective; a different rater could move the
  numbers by more than the difference between the strategies.
- Only re-ranking of a fixed candidate pool is measured; recall is not.
- A better NDCG does not establish a better recommender overall.

**Product**
- There is no user model, so "personalised recommendation" is not a claim this
  system can make. It is an item-to-item similarity engine.

---

## 5. Using the model

### Build the artifact (once, after any data or feature change)

```bash
python -m src.models.recommender build
```

Writes `artifacts/cinescope_model.joblib` (~7 MB): processed metadata, the five
feature matrices, the fitted TF-IDF vectorisers and MultiLabelBinarizers, and
the configuration. Rebuilding from CSV takes ~7 s; loading takes ~0.2 s.

### Recommend

```python
from src.models.recommender import MovieRecommender

model = MovieRecommender.load()              # or MovieRecommender() to rebuild
model.recommend(movie_id=155, n_recommendations=10)

model.get_movie(155)                          # query echo: title, year, genres
```

Returns a DataFrame with `movie_id, title, year, genres, director, score,
agreement, coverage, n_shared_blocks, shared_blocks, rank`.

Behaviour guaranteed by the validation suite (`python -m
src.experiments.verify_finalization`, 31 checks):

- the query movie is always excluded;
- `n_recommendations` is clamped to the 4,802 available;
- unknown ids raise `KeyError`, ambiguous titles raise `ValueError`;
- scores, agreement and coverage are finite and inside [0, 1];
- metadata-poor movies still return valid results;
- a saved artifact reproduces the in-memory ranking exactly.
