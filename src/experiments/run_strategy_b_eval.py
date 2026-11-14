"""
Phase 3 driver: validate Strategy B and compare it against Strategy A (config L).

    python -m src.experiments.run_strategy_b_eval

Writes:
    data/strategy_ab_comparison_L.csv       full, labelled comparison (analysis)
    data/strategy_ab_blind_candidates.csv   unlabelled pool for manual judging
    data/strategy_ab_blind_key.csv          which system proposed each candidate

Never writes to data/recommendation_review.csv and never fills in relevance.
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors

from src.data.processor import process_movies
from src.features.feature_builder import build_features, combine_features
from src.experiments.strategy_b import (
    CONFIG_L, ExperimentalRecommender,
)

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

pd.set_option("display.width", 200)
pd.set_option("display.max_colwidth", 40)

BLOCKS = ["genres", "keywords", "overview", "cast", "director"]
CHECKS = []


def hr(t):
    print("\n" + "=" * 100)
    print(t)
    print("=" * 100)


def check(name, ok, detail=""):
    CHECKS.append((name, bool(ok), detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  — {detail}" if detail else ""))


# --------------------------------------------------------------- load once
movies = process_movies()
features = build_features(movies)
N = len(movies)

n_blocks = pd.Series(
    sum((np.asarray(features[b].tocsr().getnnz(axis=1)).ravel() > 0).astype(int)
        for b in BLOCKS),
    index=movies.index,
)
kw_len = movies["keywords"].apply(len)
year = pd.to_datetime(movies["release_date"], errors="coerce").dt.year

A = ExperimentalRecommender("A", CONFIG_L, movies=movies, features=features)
B = ExperimentalRecommender("B", CONFIG_L, movies=movies, features=features)

hr("1. VALIDATION")

# --- V1: Strategy A reproduces the original production output -------
# Reference built exactly the way src/experiments/evaluate_recommendations.py
# builds it, with no reference to the new module's scoring code.
ref_matrix = combine_features(features, weights=CONFIG_L)
ref_model = NearestNeighbors(metric="cosine", algorithm="brute").fit(ref_matrix)

QUERIES = [
    (19995, "Avatar"),
    (155, "The Dark Knight"),
    (12, "Finding Nemo"),
    (238, "The Godfather"),
    (329, "Jurassic Park"),
]

mismatch = 0
max_delta = 0.0
for mid, _ in QUERIES:
    idx = A.resolve_index(movie_id=mid)
    ref = ref_model.kneighbors(ref_matrix[idx], n_neighbors=11)
    ref_ids, ref_scores = [], []
    for j, dist in zip(ref[1][0], ref[0][0]):
        if int(j) == idx:
            continue
        ref_ids.append(int(movies.iloc[j]["id"]))
        ref_scores.append(1 - float(dist))
        if len(ref_ids) == 10:
            break
    got = A.recommend(movie_id=mid, n_recommendations=10)
    got_ids = [int(v) for v in got["movie_id"]]
    got_scores = [float(v) for v in got["score"]]
    if ref_ids != got_ids:
        mismatch += 1
    max_delta = max(max_delta,
                    max(abs(a - b) for a, b in zip(ref_scores, got_scores)))
check("Strategy A reproduces production (id + score, 5 queries)",
      mismatch == 0 and max_delta < 1e-9,
      f"mismatched queries={mismatch}, max |score delta|={max_delta:.2e}")

# --- V2: Strategy B matches an independent pairwise calculation -----
rng = np.random.default_rng(7)
pairs = [(int(rng.integers(0, N)), int(rng.integers(0, N))) for _ in range(300)]
max_diff = 0.0
for i, j in pairs:
    if i == j:
        continue
    manual = 0.0
    for b in BLOCKS:
        if b not in CONFIG_L:
            continue
        xi = features[b][i].toarray().ravel()
        xj = features[b][j].toarray().ravel()
        if not xi.any() or not xj.any():
            continue
        c = float(cosine_similarity(xi.reshape(1, -1), xj.reshape(1, -1))[0, 0])
        manual += CONFIG_L[b] * c
    manual /= sum(CONFIG_L.values())
    got = B.scorer.pair_scores(i, j)["score"]
    max_diff = max(max_diff, abs(manual - got))
check("Strategy B matches independent pairwise cosine", max_diff < 1e-12,
      f"max abs diff over 300 random pairs = {max_diff:.2e}")

# --- V3: complete-metadata pairs give identical A and B -------------
complete = np.where(n_blocks.values == 5)[0]
diffs = []
for _ in range(200):
    i, j = rng.choice(complete, 2, replace=False)
    diffs.append(abs(B.scorer.pair_scores(i, j)["score"]
                     - B.scorer.concatenated_cosine(i, j)))
check("Complete-metadata pairs: A == B", max(diffs) < 1e-12,
      f"max |A-B| over 200 complete pairs = {max(diffs):.2e}")

# --- V4: a missing block contributes exactly zero -------------------
kw_empty = np.where(kw_len.values == 0)[0][:20]
worst = 0.0
for i in kw_empty:
    for j in rng.choice(N, 5, replace=False):
        worst = max(worst, abs(B.scorer.pair_scores(int(i), int(j))["contributions"]["keywords"]))
check("Missing blocks contribute 0 to the B numerator", worst == 0.0,
      f"max keyword contribution from 20 keyword-empty movies = {worst:.1e}")

# --- V5: scores stay inside [0, 1] ---------------------------------
all_scores, all_agree, all_cov = [], [], []
for mid, _ in QUERIES:
    idx = B.resolve_index(movie_id=mid)
    out = B.scorer.score_all(idx)
    all_scores.append(out["score"]); all_agree.append(out["agreement"]); all_cov.append(out["coverage"])
S = np.concatenate(all_scores); AG = np.concatenate(all_agree); CV = np.concatenate(all_cov)
tol = 1e-12
ok_range = (S.min() >= -tol and S.max() <= 1 + tol
            and np.nanmax(AG) <= 1 + tol and np.nanmax(AG) >= -tol
            and CV.min() >= -tol and CV.max() <= 1 + tol)
check("Scores within [0, 1] (fp tolerance 1e-12)", ok_range,
      f"B score [{S.min():.6f}, {S.max():.15f}], agreement max "
      f"{np.nanmax(AG):.15f}, coverage max {CV.max():.6f}")

# --- V6: self is excluded ------------------------------------------
self_hits = []
for mid, _ in QUERIES:
    got = B.recommend(movie_id=mid, n_recommendations=10)
    self_hits.append(mid in set(int(x) for x in got["movie_id"]))
    gotA = A.recommend(movie_id=mid, n_recommendations=10)
    self_hits.append(mid in set(int(x) for x in gotA["movie_id"]))
check("Self-recommendation excluded (A and B)", not any(self_hits))

# --- V7: duplicate titles are resolved by ID -----------------------
DUP_PAIRS = [(268, 2661, "Batman"), (72710, 1255, "The Host"), (39269, 10844, "Out of the Blue")]
dup_ok = True
for a_id, b_id, t in DUP_PAIRS:
    ra = B.recommend(movie_id=a_id, n_recommendations=5)["movie_id"].tolist()
    rb = B.recommend(movie_id=b_id, n_recommendations=5)["movie_id"].tolist()
    dup_ok &= (ra != rb)
    try:
        B.recommend(title=t, n_recommendations=5)
        dup_ok = False          # ambiguous title should have raised
    except ValueError:
        pass
check("Duplicate titles resolved by ID; ambiguous title raises", dup_ok)

# --- V8: oversized request clamped ---------------------------------
big = B.recommend(movie_id=155, n_recommendations=10 ** 6)
check("Request larger than neighbour pool handled safely",
      len(big) == N - 1, f"returned {len(big)} rows for n=10^6")

hr("2. TOP-10 COMPARISON — CONFIG L")

def completeness(frame):
    ids = frame["movie_id"].astype(int).tolist()
    idx = [B.id_to_index[i] for i in ids]
    return (float(np.mean(n_blocks.values[idx])),
            int(sum(1 for k in idx if kw_len.values[k] == 0)),
            float(np.mean(kw_len.values[idx])))


rows = []
blind_rows = []
summary = []

for mid, name in QUERIES:
    ra = A.recommend(movie_id=mid, n_recommendations=10)
    rb = B.recommend(movie_id=mid, n_recommendations=10)
    ta = ra["movie_id"].tolist()
    tb = rb["movie_id"].tolist()
    both = [m for m in ta if m in tb]
    summary.append({
        "query": name, "id": mid,
        "overlap/10": len(both),
        "mean blocks A": completeness(ra)[0], "mean blocks B": completeness(rb)[0],
        "kw-empty in A": completeness(ra)[1], "kw-empty in B": completeness(rb)[1],
        "mean cov A": ra["coverage"].mean(), "mean cov B": rb["coverage"].mean(),
        "mean score A": ra["score"].mean(), "mean score B": rb["score"].mean(),
    })
    for label, frame, strat in (("A", ra, "A"), ("B", rb, "B")):
        for rank, r in enumerate(frame.itertuples(), start=1):
            rows.append({
                "query_id": mid, "query_title": name, "strategy": strat, "rank": rank,
                "recommended_id": r.movie_id, "recommended_title": r.title,
                "score": r.score, "agreement": r.agreement, "coverage": r.coverage,
                "n_shared_blocks": r.n_shared_blocks, "shared_blocks": r.shared_blocks,
                "metadata_blocks": int(n_blocks.values[B.id_to_index[int(r.movie_id)]]),
                "keyword_count": int(kw_len.values[B.id_to_index[int(r.movie_id)]]),
            })
    # unlabelled candidate pool: union of both lists, shuffled
    pool = list(dict.fromkeys(ta + tb))
    rng.shuffle(pool)
    for cid in pool:
        idx = B.id_to_index[int(cid)]
        blind_rows.append({
            "query_id": mid, "query_title": name,
            "candidate_id": int(cid),
            "candidate_title": movies.iloc[idx]["title"],
            "year": int(year.iloc[idx]) if not pd.isna(year.iloc[idx]) else "",
            "genres": ", ".join(movies.iloc[idx]["genres"]),
            "director": ", ".join(movies.iloc[idx]["director"]),
            "relevance": "", "notes": "",
        })

print(pd.DataFrame(summary).to_string(index=False, float_format=lambda v: f"{v:.2f}"))

print("\n  Per-query detail (rank, title, score, agreement, coverage, shared blocks)")
for mid, name in QUERIES:
    ra = A.recommend(movie_id=mid, n_recommendations=10)
    rb = B.recommend(movie_id=mid, n_recommendations=10)
    print(f"\n  --- {name} (id {mid}) ---")
    print(f"  {'rk':<3}{'A: title':<34}{'sc':>7}{'cov':>6}   "
          f"{'B: title':<34}{'sc':>7}{'agr':>7}{'cov':>6}{'blk':>5}  shared")
    for k in range(10):
        a = ra.iloc[k]
        b = rb.iloc[k]
        print(f"  {k+1:<3}{a['title'][:32]:<34}{a['score']:>7.3f}{a['coverage']:>6.2f}   "
              f"{b['title'][:32]:<34}{b['score']:>7.3f}{b['agreement']:>7.3f}"
              f"{b['coverage']:>6.2f}{b['n_shared_blocks']:>5}  {b['shared_blocks']}")

hr("3. DUPLICATE-TITLE CASES UNDER A AND B")

for a_id, b_id, t in DUP_PAIRS:
    print(f"\n  '{t}': id {a_id} vs id {b_id}")
    for mid in (a_id, b_id):
        idx = B.id_to_index[mid]
        yr = year.iloc[idx]
        print(f"    id={mid} ({int(yr) if not pd.isna(yr) else '?'}) "
              f"blocks={n_blocks.iloc[idx]} kw={kw_len.iloc[idx]}")
        ra = A.recommend(movie_id=mid, n_recommendations=5)
        rb = B.recommend(movie_id=mid, n_recommendations=5)
        for label, frame in (("A", ra), ("B", rb)):
            print(f"      {label}: " + " | ".join(
                f"{r.title[:26]} ({r.score:.3f})" for r in frame.itertuples()))

hr("4. WRITING OUTPUTS")

cmp_path = "data/strategy_ab_comparison_L.csv"
pd.DataFrame(rows).to_csv(cmp_path, index=False)
print(f"  wrote {cmp_path}  ({len(rows)} rows)")

blind_path = "data/strategy_ab_blind_candidates.csv"
pd.DataFrame(blind_rows).to_csv(blind_path, index=False)
print(f"  wrote {blind_path}  ({len(blind_rows)} rows, unrevealed)")

key_rows = []
for mid, name in QUERIES:
    ra = A.recommend(movie_id=mid, n_recommendations=10)
    rb = B.recommend(movie_id=mid, n_recommendations=10)
    rank_a = {int(m): r for r, m in enumerate(ra["movie_id"], start=1)}
    rank_b = {int(m): r for r, m in enumerate(rb["movie_id"], start=1)}
    for cid in dict.fromkeys(list(rank_a) + list(rank_b)):
        key_rows.append({
            "query_id": mid, "query_title": name, "candidate_id": cid,
            "A_rank": rank_a.get(cid, ""), "B_rank": rank_b.get(cid, ""),
        })
key_path = "data/strategy_ab_blind_key.csv"
pd.DataFrame(key_rows).to_csv(key_path, index=False)
print(f"  wrote {key_path}  ({len(key_rows)} rows) — keep closed while judging")

print("\n  data/recommendation_review.csv was NOT modified "
      "(150 relevance cells still empty).")

hr("5. VALIDATION SUMMARY")
for name, ok, detail in CHECKS:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  — {detail}" if detail else ""))
print(f"\n  {sum(1 for _, ok, _ in CHECKS if ok)}/{len(CHECKS)} checks passed")
