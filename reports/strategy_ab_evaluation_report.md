# CineScope — Strategy A vs Strategy B: AI-Assisted Relevance Evaluation

**Status:** preliminary. The relevance judgements in this report are **AI-assisted, not human ground truth.**
3 of 59 judgements come from the project owner; the other 56 were produced by an AI assistant reading the
TMDB overview, genres, keywords and cast present in the local dataset. No judgement here should be cited as
independent human evaluation.

**Reproduce:** `python -m src.experiments.evaluate_strategy_ab`

---

## 1. Scope and inputs

| Artifact | Role | Rows |
|---|---|---|
| `data/strategy_ab_blind_candidates.csv` | unlabelled candidate pool (no strategy, rank or score) | 59 |
| `data/strategy_ab_blind_key.csv` | hidden A/B rank per candidate (read **after** judging) | 59 |
| `data/strategy_ab_comparison_L.csv` | A/B scores and coverage, config L | 100 |
| `data/strategy_ab_ai_assisted_judgments.csv` | **output** — judgements + provenance | 59 |
| `data/strategy_ab_evaluation_results.csv` | **output** — judgements joined to A/B ranks and scores | 59 |
| `data/strategy_ab_evaluation_summary.csv` | **output** — per-query and macro metrics | 12 |

Nothing in `src/models/`, `src/features/` or `src/data/` was modified. The blind candidates file, the hidden
key and `data/recommendation_review.csv` were read-only throughout; the latter still has 150 empty relevance
cells and was not touched.

The pool is the **deduplicated union of both strategies' top-10 lists** per query:

| Query | id | Candidates in pool |
|---|---|---|
| Avatar | 19995 | 10 (A and B returned the same 10) |
| The Dark Knight | 155 | 13 (7 shared + 3 A-only + 3 B-only) |
| Finding Nemo | 12 | 13 (7 shared + 3 A-only + 3 B-only) |
| The Godfather | 238 | 13 (7 shared + 3 A-only + 3 B-only) |
| Jurassic Park | 329 | 10 (A and B returned the same 10) |
| **Total** | | **59** |

---

## 2. Method

### 2.1 Rubric

Each candidate was judged as *"would this be a meaningful next watch for someone who enjoyed the query?"*

| Score | Meaning |
|---|---|
| 0 | Not relevant |
| 1 | Somewhat relevant |
| 2 | Highly relevant |

Judged on story and narrative shape, themes and subject matter, genre/subgenre, setting and world-building,
tone and viewing experience, and character/conflict structure. A shared director, a shared franchise or a
shared broad genre **never** earned a 2 on its own — where a film shares a world but a very different tone
(the two Joel Schumacher Batman films), it was graded 1, not 2.

Judgements were formed from the dataset's own `overview`, `genres`, `keywords` and `cast` fields. No external
plot summaries were fabricated; where the dataset carries essentially no information (e.g. *Amidst the
Devil's Wings*, whose entire synopsis is `'Prequel to "5th of a Degree."'`), the judgement is recorded at
`low` confidence.

### 2.2 Provenance

| Source | Count | How marked |
|---|---|---|
| `human` — supplied by the project owner | 3 | `judgment_source = human` |
| `AI-assisted` — generated in this evaluation | 56 | `judgment_source = AI-assisted` |

The three human judgements are exactly:

| Query | Candidate | Relevance |
|---|---|---:|
| Avatar | Star Trek Into Darkness (2013) | 2 |
| Avatar | Jupiter Ascending (2015) | 2 |
| Avatar | Aliens (1986) | 1 |

They are preserved verbatim. **Note:** all three fall on Avatar, and Avatar's two lists are identical — so the
human judgements affect both strategies equally and cannot tilt the A/B comparison either way.

### 2.3 Metrics

For a ranked list of 10 candidates with graded relevance `r₁…r₁₀`:

```
P@k    = |{ i ≤ k : rᵢ ≥ 1 }| / k
gainᵢ  = 2^rᵢ − 1
DCG@k  = Σ_{i≤k} gainᵢ / log₂(i + 1)
NDCG@k = DCG@k / IDCG@k
```

`IDCG@k` is computed from the **pooled** candidates for that query (all 10–13 judged items), so both
strategies are normalised against the same ideal ordering. This is the fair choice when judgements are
collected pool-wise: it prevents a strategy from being rewarded merely because the ideal list was built from
its own output.

Both strategies return exactly 10 ranked candidates for every query, so no slot is empty and no missing
recommendation can be mistaken for a relevant one.

---

## 3. Results

### 3.1 Macro averages (5 queries)

| Strategy | P@5 | P@10 | NDCG@5 | NDCG@10 | Mean graded relevance |
|---|---:|---:|---:|---:|---:|
| **A** — concatenated cosine | 0.76 | 0.62 | 0.8465 | 0.8737 | 0.88 |
| **B** — fixed denominator | **0.88** | **0.70** | **0.9166** | **0.9361** | **0.98** |
| Difference | +0.12 | +0.08 | +0.0701 | +0.0624 | +0.10 |

### 3.2 Per query

| Query | Strategy | P@5 | P@10 | NDCG@5 | NDCG@10 |
|---|---|---:|---:|---:|---:|
| Avatar | A | 1.00 | 0.70 | 0.9323 | 0.9341 |
| Avatar | B | 1.00 | 0.70 | 0.9323 | 0.9341 |
| The Dark Knight | A | 0.60 | 0.70 | 0.7383 | 0.7770 |
| The Dark Knight | B | **0.80** | **0.80** | **0.8016** | **0.9061** |
| Finding Nemo | A | 0.80 | 0.70 | 0.9377 | 0.9085 |
| Finding Nemo | B | **1.00** | **0.80** | **1.0000** | **0.9521** |
| The Godfather | A | 0.60 | 0.50 | 0.6838 | 0.7687 |
| The Godfather | B | **0.80** | **0.70** | **0.9089** | **0.9079** |
| Jurassic Park | A | 0.80 | 0.50 | 0.9403 | 0.9804 |
| Jurassic Park | B | 0.80 | 0.50 | 0.9403 | 0.9804 |

B is never worse than A and is strictly better on the three queries where the two lists differ. On the two
queries where A's list already contained only complete-metadata films, the strategies are numerically
identical — which is the expected consequence of `S_A = S_B` whenever both sides of every pair are complete.

### 3.3 What changed

| Query | Removed by B (rank, relevance, coverage under A) | Added by B (rank, relevance) |
|---|---|---|
| The Dark Knight | Amidst the Devil's Wings (3, **0**, cov 0.27); Faster (9, 0, 0.73); In Too Deep (10, 1, 0.73) | Law Abiding Citizen (8, **2**); Batman & Robin (9, 1); Hitman (10, **0**) |
| Finding Nemo | Pooh's Heffalump Movie (4, 1, 0.73); Hey Arnold! The Movie (5, 0, 0.73); Roadside Romeo (9, 0, 0.67) | Space Chimps (8, **0**); The Tigger Movie (9, 1); A Turtle's Tale (10, 1) |
| The Godfather | Amidst the Devil's Wings (2, **0**, cov 0.27); El Rey de Najayo (8, 1, 0.73); Black Water Transit (10, 0, 0.73) | In Cold Blood (8, 1); American Gangster (9, 1); Blow (10, 1) |

**Nine items removed, nine added.**

| | n | Mean relevance | Mean coverage under A |
|---|---:|---:|---:|
| Removed by B | 9 | **0.33** | 0.62 |
| Added by B | 9 | **0.89** | — |

All 9 removed items had coverage below 1.00; 2 of the 9 had coverage 0.27 (one shared block).

---

## 4. Findings

**4.1 B removed exactly the items the earlier investigations predicted.** Every removed candidate was
metadata-incomplete, and the two 0.27-coverage removals are the `Amidst the Devil's Wings` hub that
Investigation 2 identified as the catalogue's #1 rank-1 hog. It occupied **rank 2** for *The Godfather* and
**rank 3** for *The Dark Knight* under A. Removing it alone accounts for most of the NDCG@5 gain on both
queries, because a 0-relevance item in the top 3 is heavily penalised by the log₂ discount.

**4.2 The gain is not uniform.** B also introduced two 0-relevance films — *Hitman* (Dark Knight, rank 10) and
*Space Chimps* (Finding Nemo, rank 8). B improves the *list* on average; it does not make every individual
replacement better. Any claim that "B returns better films" is an aggregate statement, not a per-item one.

**4.3 Where A's list was already clean, B changes nothing at all.** Avatar and Jurassic Park have identical
lists and identical metrics. This is the strongest single piece of evidence that B is a *targeted* correction
rather than a different recommender: it only alters what completeness distortion was affecting.

**4.4 The metrics are driven by coverage, but relevance is driven by content.** Jurassic Park scores 0.9403
NDCG@5 under *both* strategies while placing *A.I. Artificial Intelligence* (judged 0) at rank 4 — a content
failure, not a scoring failure. Strategy B fixes the metadata artefact; it does nothing about the underlying
representational blindness (Spielberg + "science fiction" tokens).

**4.5 Precision and NDCG tell slightly different stories.** P@5 improves by 0.12 while NDCG@5 improves by
0.07. NDCG already gave A credit for its good top-2 items (Part II, Dark Knight Rises, WALL·E, Ponyo), so the
binary precision metric reacts more sharply to a single bad item in the top 5.

---

## 5. Limitations

1. **These are not human judgements.** 56 of 59 are AI-generated. There is one rater, no second rater, and no
   inter-rater agreement statistic. The AI grades were formed from short dataset synopses, which flatten tone
   and quality — the exact dimensions a viewer cares most about.
2. **Confidence is uneven.** 25 judgements are `high`, 27 `medium`, 7 `low`. All 7 low-confidence ones come
   from sparse or absent metadata (e.g. *Amidst the Devil's Wings*, *El Rey de Najayo*, *Black Water Transit*,
   *Mi America*, *Roadside Romeo*), and one of them — *Amidst the Devil's Wings* — is load-bearing for the
   result: it is judged 0 with low confidence, and it is the item whose removal drives the Dark Knight and
   Godfather improvements.
3. **Five queries only.** All five are extremely well-known mainstream films. Nothing here says how the
   strategies behave for foreign-language, documentary, or obscure catalogue entries — the population where
   missing metadata is most common.
4. **The pool is closed.** Because only the union of A's and B's top-10 was judged, the evaluation measures
   *re-ranking of a fixed candidate set*, not recall. A third, better recommendation that neither strategy
   returned is invisible to these metrics.
5. **Relevance is subjective.** A different judge could reasonably grade *Batman & Robin* 0 rather than 1,
   *A.I.* 1 rather than 0, or *Planet of the Apes* 1 rather than 2. The margins here (0.07 NDCG) are of the
   same order as plausible human disagreement.
6. **Higher metrics do not prove a better recommender.** NDCG@10 = 0.9361 on five queries with one AI rater is
   a signal about ordering pathology, not a claim of overall superiority. Coverage, diversity, cold-start
   behaviour and user satisfaction are not measured at all.

---

## 6. Recommendation for the next stage

1. **Do not adopt B as the default yet.** Adopt it behind a flag, alongside A, and log both scores plus
   coverage during normal use. The evidence supports "B removes one demonstrated harm", not "B is better".
2. **Validate the load-bearing judgement first.** Have a human judge the 9 removed and 9 added items, blind to
   strategy. If *Amidst the Devil's Wings* is judged 0 by a human too, the finding stands; if not, the whole
   gain shrinks. This is 18 judgements — under an hour of work — and it is the cheapest way to convert an
   AI-assisted result into a defensible one.
3. **Add a second rater and report Cohen's κ** on the 59 items before any of these numbers appear in a
   dissertation.
4. **Expand the query set** to ~20 films stratified across genre, era, and metadata completeness (deliberately
   including keyword-empty and low-coverage queries), then re-run this exact script.
5. **Fix the data, not only the denominator.** B's known cost is that keyword-empty films are pushed to 0.25×
   their population share. The productive follow-on is metadata repair (backfill keywords/cast from the
   MovieLens `tmdbId` join or the TMDB API) with the scoring rule held fixed, then re-measure P@10 — if repair
   recovers those films' relevance, B's suppression is a temporary artefact rather than a permanent trade-off.
6. **Investigate the content failures separately.** *A.I.* at rank 4 for Jurassic Park and *Hitman*/*Space
   Chimps* entering B's lists are representation problems (shared genre/crew tokens), outside the scope of
   denominator choice.

---

## Appendix A — Judgement distribution

| Query | Candidates | 2 | 1 | 0 | human | high | medium | low |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Avatar | 10 | 3 | 4 | 3 | 3 | 5 (+3 human) | 2 | 0 |
| The Dark Knight | 13 | 3 | 6 | 4 | 0 | 5 | 7 | 1 |
| Finding Nemo | 13 | 2 | 7 | 4 | 0 | 3 | 8 | 2 |
| The Godfather | 13 | 3 | 5 | 5 | 0 | 3 | 6 | 4 |
| Jurassic Park | 10 | 3 | 2 | 5 | 0 | 6 | 4 | 0 |
| **Total** | **59** | **14** | **24** | **21** | **3** | **25** | **27** | **7** |

## Appendix B — Validation

`python -m src.experiments.evaluate_strategy_ab` reports **9/9 checks passed**:

- exactly 59 candidates judged;
- one judgement per (query, candidate);
- judged set identical to the blind candidate set;
- no relevance outside 0–2;
- all five query movies represented;
- the three human judgements preserved with their exact values;
- the remaining 56 marked `AI-assisted`;
- no duplicate or omitted candidate;
- the key's ranks agree with the independently recomputed comparison ranks.
