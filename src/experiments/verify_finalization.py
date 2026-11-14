"""
Finalization checks for CineScope.

    python -m src.experiments.verify_finalization

Two blocks:

A. Evaluation audit - re-derives the headline metrics independently from the
   saved artifacts and checks that the judgments, ranks and scores are
   internally consistent. Read-only.

B. Production validation - exercises the final production recommender against
   the Phase 3 experimental output and around the persistence boundary.

Exits non-zero if any check fails.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import FEATURE_BLOCKS, FEATURE_WEIGHTS, RECOMMENDATION_COLUMNS
from src.models.recommender import MovieRecommender

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

pd.set_option("display.width", 200)
pd.set_option("display.max_colwidth", 46)

CHECKS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    CHECKS.append((name, bool(ok), detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  - {detail}" if detail else ""))


def hr(title: str) -> None:
    print("\n" + "=" * 98)
    print(title)
    print("=" * 98)


# ============================================================ A. audit
hr("A. EVALUATION ARTIFACT AUDIT")

results = pd.read_csv("data/strategy_ab_evaluation_results.csv")
summary = pd.read_csv("data/strategy_ab_evaluation_summary.csv")
judgments = pd.read_csv("data/strategy_ab_ai_assisted_judgments.csv")
blind = pd.read_csv("data/strategy_ab_blind_candidates.csv")
key = pd.read_csv("data/strategy_ab_blind_key.csv")
comparison = pd.read_csv("data/strategy_ab_comparison_L.csv")

# --- A1 judgments integrity
check("Judgments: 59 rows, unique per (query, candidate)",
      len(judgments) == 59
      and not judgments.duplicated(["query_id", "candidate_id"]).any())
check("Judgments: cover exactly the blind candidate set",
      set(zip(judgments.query_id, judgments.candidate_id))
      == set(zip(blind.query_id, blind.candidate_id)))
check("Judgments: relevance within 0-2",
      judgments["relevance"].between(0, 2).all())
check("Judgments: 3 human + 56 AI-assisted",
      judgments.judgment_source.value_counts().to_dict()
      == {"AI-assisted": 56, "human": 3},
      str(judgments.judgment_source.value_counts().to_dict()))
human_rows = judgments[judgments.judgment_source == "human"].set_index("candidate_id")["relevance"]
check("Judgments: human values preserved (54138=2, 76757=2, 679=1)",
      human_rows.to_dict() == {54138: 2, 76757: 2, 679: 1}, str(human_rows.to_dict()))

# --- A2 rank structure
rank_ok = True
for (qid, strat), grp in results.melt(
        id_vars=["query_id", "candidate_id"],
        value_vars=["A_rank", "B_rank"]).dropna(subset=["value"]).groupby(["query_id", "variable"]):
    ranks = sorted(int(v) for v in grp["value"])
    if ranks != list(range(1, 11)):
        rank_ok = False
check("Ranks: each (query, strategy) ranks exactly 1..10 with no gaps",
      rank_ok)

check("Key: 9 candidates are unique to one strategy",
      int(key["A_rank"].isna().sum()) == 9 and int(key["B_rank"].isna().sum()) == 9)

# --- A3 independent metric recomputation
def precision_at(rels, k):
    return sum(1 for r in rels[:k] if r >= 1) / k


def ndcg_at(rels, pool_gains, k):
    dcg = sum((2 ** r - 1) / np.log2(i + 2) for i, r in enumerate(rels[:k]))
    idcg = sum(g / np.log2(i + 2)
               for i, g in enumerate(sorted(pool_gains, reverse=True)[:k]))
    return dcg / idcg if idcg > 0 else 0.0


recomputed = []
for qid, grp in results.groupby("query_id"):
    pool_gains = [2 ** r - 1 for r in grp.relevance]
    for strat, col in (("A", "A_rank"), ("B", "B_rank")):
        sub = grp[grp[col].notna()].sort_values(col)
        rels = sub.relevance.astype(int).tolist()
        recomputed.append({
            "query_id": qid, "strategy": strat, "n": len(rels),
            "P@5": precision_at(rels, 5), "P@10": precision_at(rels, 10),
            "NDCG@5": ndcg_at(rels, pool_gains, 5),
            "NDCG@10": ndcg_at(rels, pool_gains, 10),
        })

recomputed = pd.DataFrame(recomputed)
stored = summary[summary.scope == "per_query"][
    ["query_id", "strategy", "P@5", "P@10", "NDCG@5", "NDCG@10"]]
merged = recomputed.merge(stored, on=["query_id", "strategy"], suffixes=("_calc", "_stored"))
max_delta = 0.0
for metric in ["P@5", "P@10", "NDCG@5", "NDCG@10"]:
    max_delta = max(max_delta,
                    float(np.max(np.abs(merged[f"{metric}_calc"] - merged[f"{metric}_stored"]))))
check("Metrics: independent recomputation matches the stored per-query values",
      max_delta < 5e-5, f"max |delta| = {max_delta:.2e}")

# --- A4 stored macro averages
macro = summary[summary.scope == "macro_average"].set_index("strategy")
macro_ok = True
for strat in ("A", "B"):
    per = summary[(summary.scope == "per_query") & (summary.strategy == strat)]
    for metric in ["P@5", "P@10", "NDCG@5", "NDCG@10"]:
        if abs(per[metric].mean() - float(macro.loc[strat, metric])) > 1e-4:
            macro_ok = False
check("Metrics: stored macro averages equal the mean of the per-query values", macro_ok)

# --- A5 score identity S_B = coverage * agreement
b_rows = comparison[comparison.strategy == "B"]
identity = float(np.max(np.abs(b_rows["score"]
                               - b_rows["coverage"] * b_rows["agreement"])))
check("Scores: Strategy B equals coverage x agreement on every row",
      identity < 1e-12, f"max |S_B - cov*agr| = {identity:.2e}")

# --- A6 scores in range
check("Scores: all Strategy B scores within [0, 1]",
      bool(b_rows["score"].between(0, 1).all()),
      f"range [{b_rows.score.min():.4f}, {b_rows.score.max():.4f}]")

# --- A7 the review file was not overwritten
review = pd.read_csv("data/recommendation_review.csv")
check("data/recommendation_review.csv untouched (150 rows, all relevance empty)",
      len(review) == 150 and int(review["relevance"].notna().sum()) == 0)

# --- A8 reported headline numbers
headline = summary[summary.scope == "macro_average"].set_index("strategy")
print("\n  Headline macro averages (as stored):")
print(headline[["P@5", "P@10", "NDCG@5", "NDCG@10"]].to_string())

# ====================================================== B. production
hr("B. PRODUCTION MODEL VALIDATION")

model = MovieRecommender()
print(f"  built in-process: {model.n_movies} movies, "
      f"weights={model.weights}, total={model.total_weight:.2f}")

check("Configuration: production uses the L weights",
      model.weights == FEATURE_WEIGHTS,
      str(model.weights))
check("Feature blocks: all five present",
      all(b in model.features for b in FEATURE_BLOCKS), str(list(model.features)))

QUERIES = [
    (19995, "Avatar"),
    (155, "The Dark Knight"),
    (12, "Finding Nemo"),
    (238, "The Godfather"),
    (329, "Jurassic Park"),
]

comparison_b = comparison[comparison.strategy == "B"]

rows = []
all_ok = {"self": True, "count": True, "range": True, "finite": True,
          "ids": True, "rank_match": True}
for mid, expected_title in QUERIES:
    recs = model.recommend(movie_id=mid, n_recommendations=10)
    movie = model.get_movie(mid)

    rows.append({
        "query": expected_title, "id": mid, "title": movie["title"],
        "returned": len(recs),
        "top1": recs.iloc[0]["title"][:28],
        "top1_score": round(float(recs.iloc[0]["score"]), 4),
        "mean_cov": round(float(recs["coverage"].mean()), 3),
        "min_cov": round(float(recs["coverage"].min()), 3),
    })

    all_ok["ids"] &= movie["title"] == expected_title
    all_ok["self"] &= mid not in set(int(v) for v in recs["movie_id"])
    all_ok["count"] &= len(recs) == 10
    all_ok["range"] &= bool(recs["score"].between(0, 1).all()
                            and recs["coverage"].between(0, 1).all())
    all_ok["finite"] &= bool(np.isfinite(recs["score"]).all())
    valid_agreement = recs["agreement"].dropna()
    all_ok["range"] &= bool(valid_agreement.between(0, 1).all())

    expected = comparison_b[comparison_b.query_id == mid].sort_values("rank")
    same_ids = list(expected["recommended_id"].astype(int)) == list(recs["movie_id"].astype(int))
    score_delta = float(np.max(np.abs(expected["score"].to_numpy()
                                      - recs["score"].to_numpy())))
    all_ok["rank_match"] &= same_ids and score_delta < 1e-9

check("Five queries resolve to the expected movie by id", all_ok["ids"])
check("Query movie is excluded from its own results", all_ok["self"])
check("Exactly the requested number of results is returned", all_ok["count"])
check("All ids and titles are present and non-null",
      all(bool(recs.notna().all().all()) for recs in [model.recommend(movie_id=q) for q, _ in QUERIES]))
check("Scores are finite and in [0, 1] (score, agreement, coverage)", all_ok["finite"] and all_ok["range"])
check("Output reproduces the Phase 3 experimental Strategy B ranking exactly",
      all_ok["rank_match"], "ids identical, scores within 1e-9")

print()
print(pd.DataFrame(rows).to_string(index=False))

# --- metadata-poor queries still work
poor = [(378237, "Amidst the Devil's Wings"), (361505, "Me You and Five Bucks"),
        (292539, "Food Chains")]
poor_ok = True
for mid, name in poor:
    recs = model.recommend(movie_id=mid, n_recommendations=5)
    poor_ok &= (len(recs) == 5
                and np.isfinite(recs["score"]).all()
                and recs["coverage"].between(0, 1).all())
    print(f"  metadata-poor query {mid} ({name}): top1={recs.iloc[0]['title'][:30]!r} "
          f"score={recs.iloc[0]['score']:.4f} coverage={recs.iloc[0]['coverage']:.2f}")
check("Metadata-poor queries produce valid, finite scores", poor_ok)

# --- error handling and clamping
unknown_ok = False
try:
    model.recommend(movie_id=999999)
except KeyError:
    unknown_ok = True
check("Unknown movie id raises KeyError", unknown_ok)

ambiguous_ok = False
try:
    model.recommend(title="Batman")
except ValueError as error:
    ambiguous_ok = "ambiguous" in str(error).lower()
check("Ambiguous title raises ValueError pointing at ids", ambiguous_ok)

unique_title_ok = model.recommend(title="Finding Nemo", n_recommendations=3).iloc[0]["title"]
check("Unique title still resolves", unique_title_ok == "WALL·E", unique_title_ok)

big = model.recommend(movie_id=155, n_recommendations=10 ** 6)
check("Oversized request clamped to the catalogue", len(big) == model.n_movies - 1,
      f"{len(big)} rows")

zero = model.recommend(movie_id=155, n_recommendations=0)
check("n_recommendations=0 returns an empty frame with the right columns",
      len(zero) == 0 and list(zero.columns) == list(RECOMMENDATION_COLUMNS) + ["rank"])

# --- persistence round trip
hr("C. ARTIFACT PERSISTENCE")

artifact_path = Path("artifacts/cinescope_model.joblib")
model.save()
size_mb = artifact_path.stat().st_size / 1e6
print(f"  saved {artifact_path} ({size_mb:.1f} MB)")
check("Artifact file written", artifact_path.is_file())

reloaded = MovieRecommender.load()
check("Loaded model reproduces the in-memory recommendations",
      reloaded.recommend(movie_id=155, n_recommendations=10)
      .equals(model.recommend(movie_id=155, n_recommendations=10)))
check("Loaded model keeps the configuration",
      reloaded.weights == model.weights and reloaded.n_movies == model.n_movies)
check("Loaded model keeps the fitted encoders",
      set(reloaded.encoders) >= {"genres", "keywords", "overview", "cast", "director"})

missing_ok = False
try:
    MovieRecommender.load("artifacts/does_not_exist.joblib")
except FileNotFoundError as error:
    missing_ok = "No model artifact" in str(error)
check("Loading a missing artifact raises a clear FileNotFoundError", missing_ok)

# ------------------------------------------------------------------ summary
hr("SUMMARY")
failed = [name for name, ok, _ in CHECKS if not ok]
print(f"  {len(CHECKS) - len(failed)}/{len(CHECKS)} checks passed")
if failed:
    print("  FAILED:")
    for name in failed:
        print(f"    - {name}")
raise SystemExit(1 if failed else 0)
