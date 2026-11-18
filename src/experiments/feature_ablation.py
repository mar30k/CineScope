"""
Feature ablation experiment: how much does each feature block contribute?

    python -m src.experiments.feature_ablation

This is an EXPERIMENT. It does not touch the production configuration, the
production artifact, or any previously written evaluation output.

Method
------
The evaluation set, queries, relevance judgments, pooling and metric
definitions are taken unchanged from ``data/strategy_ab_ai_assisted_judgments
.csv`` and ``src/experiments/evaluate_strategy_ab.py``:

* five query movies (Avatar, The Dark Knight, Finding Nemo, The Godfather,
  Jurassic Park);
* the 59 judged (query, candidate) pairs, relevance 0/1/2, no new judgments;
* ``P@k`` = share of the top k with relevance >= 1;
* ``gain = 2**r - 1``, ``DCG@k = sum(gain_i / log2(i + 1))``,
  ``NDCG@k = DCG@k / IDCG@k`` with the ideal ranking derived from the
  query's full judged set.

Six configurations are compared: the production weights as a baseline, and
five variants that each drop exactly one feature block. Dropping a block
removes its weight *and* its contribution to the denominator, so
``denominator = sum of the remaining weights`` (handled by ``BlockScorer``,
whose ``total_weight`` is the sum of the weights it was given).

Because the judged pool is a fixed, closed set, every configuration is ranked
*within that pool* and the top ten of that ranking are scored. This is the
same pooled-evaluation principle the original NDCG already relies on, but it
does mean the metrics are pool-relative: a configuration whose best
full-catalogue candidates were never judged cannot be credited for them. The
``unjudged_in_full_top10`` diagnostic reports how often that happens, and no
unjudged movie is ever scored as relevant or irrelevant.

Never writes: the artifact, ``src/config.py``, or any ``data/strategy_ab_*``
file. All output goes to ``data/feature_ablation/``.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import FEATURE_BLOCKS, FEATURE_WEIGHTS
from src.models.recommender import MovieRecommender
from src.models.scoring import BlockScorer

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

pd.set_option("display.width", 220)
pd.set_option("display.max_colwidth", 60)

JUDGMENTS = "data/strategy_ab_ai_assisted_judgments.csv"
STORED_SUMMARY = "data/strategy_ab_evaluation_summary.csv"
OUT_DIR = Path("data/feature_ablation")

TOP_K = 10            # ranked list length scored for every configuration
NDCG_DEPTH = 10


def hr(title: str) -> None:
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)


# --------------------------------------------------------------- metrics
# Copied verbatim from src/experiments/evaluate_strategy_ab.py so the ablation
# and the official evaluation cannot diverge.
def precision_at(relevances, k, threshold=1):
    top = relevances[:k]
    return sum(1 for r in top if r >= threshold) / k


def ndcg_at(relevances, ideal_gains, k):
    gains = [2 ** r - 1 for r in relevances[:k]]
    dcg = sum(g / np.log2(i + 2) for i, g in enumerate(gains))
    ideal = sorted(ideal_gains, reverse=True)[:k]
    idcg = sum(g / np.log2(i + 2) for i, g in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


# ---------------------------------------------------------- configurations
# (label, dropped block or None)
CONFIGS = [
    ("full", None),
    ("without_genres", "genres"),
    ("without_keywords", "keywords"),
    ("without_overview", "overview"),
    ("without_cast", "cast"),
    ("without_director", "director"),
]


def config_weights(dropped: str | None) -> dict:
    """Keep the remaining weights unchanged; the sum becomes the denominator."""
    if dropped is not None and dropped not in FEATURE_BLOCKS:
        raise ValueError(f"unknown block: {dropped}")
    return {
        b: FEATURE_WEIGHTS[b] for b in FEATURE_BLOCKS if b != dropped
    }


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ------------------------------------------------------------------- setup
hr("SETUP")
judgments = pd.read_csv(JUDGMENTS)
judgments_hash_before = digest(Path(JUDGMENTS))
print(f"judgments      : {judgments.shape[0]} pairs, "
      f"{judgments.query_id.nunique()} queries, "
      f"human={int((judgments.judgment_source == 'human').sum())}, "
      f"AI-assisted={int((judgments.judgment_source == 'AI-assisted').sum())}")

model = MovieRecommender.load()
print(f"model          : {len(model.movies)} movies, loaded read-only from artifact")
print(f"production cfg : {FEATURE_WEIGHTS} (total {sum(FEATURE_WEIGHTS.values())})")

query_ids = [int(q) for q in judgments.query_id.drop_duplicates()]
positions = {q: model.id_to_position[q] for q in query_ids}
print(f"queries        : {query_ids}")

pool = {
    q: [int(c) for c in g.candidate_id]
    for q, g in judgments.groupby("query_id", sort=False)
}
print("pool sizes     : " + ", ".join(f"{q}={len(v)}" for q, v in pool.items()))
if any(len(v) < TOP_K for v in pool.values()):
    raise SystemExit("A judged pool is smaller than 10; P@10 would be undefined.")

relevance = {
    (int(r.query_id), int(r.candidate_id)): int(r.relevance)
    for r in judgments.itertuples()
}
ideal_gains = {
    q: [2 ** r - 1 for r in judgments.loc[judgments.query_id == q, "relevance"]]
    for q in query_ids
}

# ------------------------------------------------------- score + rank pools
config_rows = []
ranking_rows = []
per_query_rows = []

for label, dropped in CONFIGS:
    weights = config_weights(dropped)
    denominator = float(sum(weights.values()))
    scorer = BlockScorer(model.features, weights)

    config_rows.append({
        "configuration": label,
        "dropped_block": dropped or "",
        "blocks": "+".join(weights),
        "weights": "; ".join(f"{b}={w:g}" for b, w in weights.items()),
        "denominator": round(denominator, 4),
        "n_blocks": len(weights),
    })

    for qid in query_ids:
        pos = positions[qid]
        scores = scorer.score_all(pos)["score"].copy()
        scores[pos] = -np.inf

        # Rank inside the judged pool. Ties break by catalogue position, the
        # same rule production uses.
        pool_positions = sorted(
            ((model.id_to_position[c], c) for c in pool[qid]),
            key=lambda t: (-float(scores[t[0]]), t[0]),
        )
        ranked = pool_positions[:TOP_K]

        # Diagnostic only: how many of the model's true catalogue-wide top-10
        # were never judged (so the pool cannot reward this configuration).
        full_order = np.argsort(-scores, kind="stable")[:TOP_K]
        judged_ids = set(pool[qid])
        unjudged = sum(
            1 for p in full_order if int(model.movies.iloc[int(p)]["id"]) not in judged_ids
        )

        rels = []
        for rank, (p, candidate_id) in enumerate(ranked, start=1):
            r = relevance[(qid, candidate_id)]
            rels.append(r)
            ranking_rows.append({
                "configuration": label,
                "query_id": qid,
                "query_title": judgments.loc[judgments.query_id == qid,
                                             "query_title"].iloc[0],
                "rank": rank,
                "candidate_id": candidate_id,
                "candidate_title": model.movies.iloc[p]["title"],
                "score": float(scores[p]),
                "relevance": r,
                "unjudged_in_full_top10": unjudged,
            })

        qtitle = judgments.loc[judgments.query_id == qid, "query_title"].iloc[0]
        per_query_rows.append({
            "configuration": label,
            "query_id": qid,
            "query_title": qtitle,
            "n_ranked": len(rels),
            "n_judged_pool": len(pool[qid]),
            "unjudged_in_full_top10": unjudged,
            "mean_graded_relevance": round(float(np.mean(rels)), 4),
            "P@5": round(precision_at(rels, 5), 4),
            "P@10": round(precision_at(rels, 10), 4),
            "NDCG@5": round(ndcg_at(rels, ideal_gains[qid], 5), 4),
            "NDCG@10": round(ndcg_at(rels, ideal_gains[qid], NDCG_DEPTH), 4),
        })

per_query = pd.DataFrame(per_query_rows)
configurations = pd.DataFrame(config_rows)

summary_rows = []
for label, _ in CONFIGS:
    sub = per_query[per_query.configuration == label]
    summary_rows.append({
        "configuration": label,
        "dropped_block": dict(CONFIGS)[label] or "",
        "denominator": round(sum(config_weights(dict(CONFIGS)[label]).values()), 4),
        "n_queries": len(sub),
        "P@5": round(sub["P@5"].mean(), 4),
        "P@10": round(sub["P@10"].mean(), 4),
        "NDCG@5": round(sub["NDCG@5"].mean(), 4),
        "NDCG@10": round(sub["NDCG@10"].mean(), 4),
        "mean_graded_relevance": round(sub["mean_graded_relevance"].mean(), 4),
        "mean_unjudged_in_full_top10": round(sub["unjudged_in_full_top10"].mean(), 2),
    })
summary = pd.DataFrame(summary_rows)
rankings = pd.DataFrame(ranking_rows)

# ---------------------------------------------------------------- validate
hr("VALIDATION")
checks = []


def check(name, ok, detail=""):
    checks.append((name, bool(ok), detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  - {detail}" if detail else ""))


baseline = summary[summary.configuration == "full"].iloc[0]

# 1. The pool ranking of the baseline must match production's own top-10.
prod_matches = True
prod_detail = []
for qid in query_ids:
    prod = model.recommend(movie_id=qid, n_recommendations=TOP_K)
    prod_ids = [int(v) for v in prod["movie_id"]]
    base_rank = rankings[(rankings.configuration == "full") & (rankings.query_id == qid)]
    base_ids = [int(v) for v in base_rank.sort_values("rank")["candidate_id"]]
    base_scores = [float(v) for v in base_rank.sort_values("rank")["score"]]
    prod_scores = [float(v) for v in prod["score"]]
    same_ids = prod_ids == base_ids
    same_scores = len(prod_scores) == len(base_scores) and all(
        abs(a - b) <= 1e-9 for a, b in zip(prod_scores, base_scores)
    )
    if not (same_ids and same_scores):
        prod_matches = False
        prod_detail.append(qid)
check("Baseline pool ranking equals production recommend() (ids + scores)",
      prod_matches, f"mismatched queries: {prod_detail}" if prod_detail else "")

# 2. Baseline metrics must equal the stored Strategy B macro numbers.
stored = pd.read_csv(STORED_SUMMARY)
stored_b = stored[(stored.scope == "macro_average") & (stored.strategy == "B")].iloc[0]
metric_names = ["P@5", "P@10", "NDCG@5", "NDCG@10"]
metric_diff = {m: abs(float(baseline[m]) - float(stored_b[m])) for m in metric_names}
check("Baseline reproduces stored Strategy B macro metrics",
      all(v <= 1e-9 for v in metric_diff.values()),
      ", ".join(f"{m} {v:.2e}" for m, v in metric_diff.items()))

# 3. Denominators are the sum of the remaining weights.
den_ok = True
for label, dropped in CONFIGS:
    expected = sum(config_weights(dropped).values())
    got = float(summary.loc[summary.configuration == label, "denominator"].iloc[0])
    den_ok &= abs(expected - got) <= 1e-9
check("Denominator equals the sum of the remaining weights", den_ok,
      "full=3.75, drop unit block=2.75, drop cast=3.25, drop director=3.5")

# 4. Every configuration excludes the query movie and scores 10 items.
self_ref = all(
    int(r.candidate_id) != int(r.query_id) for r in rankings.itertuples()
)
check("Query movie never appears in its own results", self_ref)
check("All configurations return 10 ranked judged candidates per query",
      (rankings.groupby(["configuration", "query_id"]).size() == TOP_K).all(),
      f"{len(rankings)} rows = 6 configs x 5 queries x 10")

# 5. Judgments were only read.
check("Judgment file unchanged", digest(Path(JUDGMENTS)) == judgments_hash_before,
      "sha256 identical")

# ------------------------------------------------------------------- write
hr("RESULTS")
print(summary.to_string(index=False))

OUT_DIR.mkdir(parents=True, exist_ok=True)
configurations.to_csv(OUT_DIR / "ablation_configurations.csv", index=False)
rankings.to_csv(OUT_DIR / "ablation_rankings.csv", index=False)
per_query.to_csv(OUT_DIR / "ablation_per_query.csv", index=False)
summary.to_csv(OUT_DIR / "ablation_summary.csv", index=False)

for name in ("ablation_configurations.csv", "ablation_rankings.csv",
             "ablation_per_query.csv", "ablation_summary.csv"):
    print(f"wrote {OUT_DIR / name}")

hr("PER-QUERY METRICS")
print(per_query.to_string(index=False))

hr("VALIDATION SUMMARY")
for name, ok, detail in checks:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  - {detail}" if detail else ""))
passed = sum(1 for _, ok, _ in checks if ok)
print(f"\n  {passed}/{len(checks)} checks passed")

raise SystemExit(0 if passed == len(checks) else 1)
