"""
Phase 3 evaluation: AI-assisted relevance judgments for Strategy A vs Strategy B.

    python -m src.experiments.evaluate_strategy_ab

Reads  : data/strategy_ab_blind_candidates.csv  (59 unlabelled candidates)
         data/strategy_ab_blind_key.csv         (hidden A/B ranks)
         data/strategy_ab_comparison_L.csv      (A/B scores, config L)
Writes : data/strategy_ab_ai_assisted_judgments.csv
         data/strategy_ab_evaluation_results.csv
         data/strategy_ab_evaluation_summary.csv

Never modifies the blind candidate file, the key, or
data/recommendation_review.csv.

Provenance
----------
Three judgments were supplied directly by the project owner and are marked
``human``. Every other judgment was produced by an AI assistant (Buffly)
reading the TMDB overview, genres, keywords and cast present in the local
dataset, and is marked ``AI-assisted``. These are NOT independent human
ground truth and must not be reported as such.

Rubric
------
0 = not relevant, 1 = somewhat relevant, 2 = highly relevant, judged as
"would this be a meaningful next watch for someone who enjoyed the query?".
A shared director, franchise or broad genre alone never earns a 2.
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

pd.set_option("display.width", 200)
pd.set_option("display.max_colwidth", 60)

BLIND = "data/strategy_ab_blind_candidates.csv"
KEY = "data/strategy_ab_blind_key.csv"
COMPARISON = "data/strategy_ab_comparison_L.csv"

OUT_JUDGMENTS = "data/strategy_ab_ai_assisted_judgments.csv"
OUT_RESULTS = "data/strategy_ab_evaluation_results.csv"
OUT_SUMMARY = "data/strategy_ab_evaluation_summary.csv"

# ---------------------------------------------------------------- judgments
# (query_id, candidate_id) -> (relevance, confidence, note)
HUMAN = {
    (19995, 54138): (2, "high",
                     "Owner-judged. Large-scale space adventure with the same "
                     "immersive-spectacle register as the query."),
    (19995, 76757): (2, "high",
                     "Owner-judged. Space opera with elaborate world-building and a "
                     "human protagonist caught in an alien dynastic conflict."),
    (19995, 679): (1, "high",
                   "Owner-judged. Same director and space-marine premise, but a "
                   "horror-thriller tone rather than epic adventure."),
}

AI = {
    # ---- Avatar (19995)
    (19995, 13475): (1, "high",
                     "Space action-adventure with comparable effects spectacle, but a "
                     "franchise origin story rather than a first-contact / colonial narrative."),
    (19995, 333355): (0, "high",
                      "Animated micro-series episodes; shares only the 'space' tag, no "
                      "narrative, thematic or tonal overlap."),
    (19995, 188927): (1, "high",
                      "Same register as the other Star Trek entries: space action-adventure "
                      "without the Pandora / colonial theme."),
    (19995, 869): (2, "medium",
                   "Astronaut stranded on an alien world ruled by another species, with a "
                   "colonial / otherness theme - the closest thematic match in the pool."),
    (19995, 118340): (1, "high",
                      "Space action-adventure with spectacle and a team dynamic, but a "
                      "comedic tone and no ecological theme."),
    (19995, 8077): (0, "medium",
                    "Grim prison-planet horror; shares the franchise and 'space' but not the "
                    "adventure-spectacle register."),
    (19995, 698): (0, "high",
                   "Spy film that happens to end in space; no overlap in story, theme or tone."),
    # ---- The Dark Knight (155)
    (155, 22803): (2, "high",
                   "Vigilante-versus-justice-system thriller; the same moral argument about "
                   "whether the system can be trusted."),
    (155, 1620): (0, "high",
                  "Video-game assassin action; shares 'secret identity' only, no thematic or "
                  "tonal overlap."),
    (155, 7873): (1, "medium",
                  "Gritty LA crime character study in the same register, but smaller in scale "
                  "and without the superhero / morality-parable core."),
    (155, 26039): (1, "medium",
                   "Stylish neo-noir revenge thriller; same crime-and-retribution drive, more "
                   "surreal and 1960s in style."),
    (155, 272): (2, "high",
                 "Direct predecessor: same director, same world, same gritty origin-of-the-"
                 "vigilante story."),
    (155, 415): (1, "medium",
                 "Shares the character and Gotham City, but the campy comedy tone is the "
                 "opposite of the query's."),
    (155, 22314): (1, "medium",
                   "Undercover-cop crime drama in the same street-crime register, without the "
                   "superhero or moral-parable layer."),
    (155, 136797): (0, "high",
                    "Street-racing action; shares only the 'action' and 'crime' tags."),
    (155, 41283): (0, "medium",
                   "Straightforward revenge actioner without the moral or structural complexity."),
    (155, 49026): (2, "high",
                   "Direct sequel in the same trilogy, world and cast."),
    (155, 378237): (0, "low",
                    "No usable synopsis, cast, keywords or director in the dataset. Judged not "
                    "relevant on the available evidence; flagged uncertain."),
    (155, 102362): (1, "medium",
                    "NYC revenge thriller involving a crime lord; adjacent tone, thinner "
                    "character work."),
    (155, 414): (1, "medium",
                 "Same character and world, but a neon-campy register very unlike the query."),
    # ---- Finding Nemo (12)
    (12, 10681): (2, "high",
                  "Same studio and director; wordless, emotionally driven non-human "
                  "protagonists and a strong environmental theme."),
    (12, 49953): (1, "medium",
                  "Ocean-set animated animal journey with a family framing; much weaker craft "
                  "and characterisation."),
    (12, 17710): (0, "high",
                  "Urban kids' TV spin-off; no ocean, adventure or father-son overlap."),
    (12, 10567): (1, "medium",
                  "Animal protagonist separated from family and on a perilous journey; "
                  "different setting and tone."),
    (12, 36696): (0, "medium",
                  "Russian animated kids' comedy; no thematic overlap."),
    (12, 11802): (0, "medium",
                  "Broad kids' comedy; shares only the 'animation / family' umbrella."),
    (12, 15655): (1, "medium",
                  "Gentle family animation built on a search for a missing friend; same "
                  "audience, softer register."),
    (12, 14444): (1, "low",
                  "Missing-child and new-sibling plot echoes the query's separation story, but "
                  "it is a TV-series comedy."),
    (12, 51162): (1, "medium",
                  "Gentle family animation; shares audience and emotional simplicity, not the "
                  "adventure."),
    (12, 20653): (0, "low",
                  "Bollywood animated comedy with sparse metadata; no narrative or tonal overlap."),
    (12, 13682): (1, "medium",
                  "Family animation about searching for a feared outsider; same audience, no "
                  "ocean or adventure."),
    (12, 12429): (2, "medium",
                  "Ocean, a child and a magical fish, and a father's protectiveness - the "
                  "strongest thematic match in the pool."),
    (12, 20352): (1, "high",
                  "Animated family film whose emotional core is a parent learning to raise "
                  "children; comedic tone."),
    # ---- The Godfather (238)
    (238, 364083): (0, "low",
                    "Hate-crime drama with limited metadata; no mafia or family-dynasty overlap."),
    (238, 119657): (1, "low",
                    "Organised-crime rise-and-fall, but a different country, era and production "
                    "scale."),
    (238, 18900): (1, "medium",
                   "Sombre true-crime procedural; crime and moral weight, but not a family saga."),
    (238, 378237): (0, "low",
                    "No usable synopsis in the dataset; not relevant on the available evidence."),
    (238, 242): (2, "high",
                 "Direct continuation of the same family and story."),
    (238, 769): (2, "high",
                 "Italian-American mob life, rise and fall; the same subject matter and register."),
    (238, 4133): (1, "medium",
                  "Drug-trade rise and fall; same arc, no family or patriarch dimension."),
    (238, 4982): (1, "medium",
                  "Crime-empire rise-and-fall with a parallel police storyline and a different "
                  "milieu."),
    (238, 227): (0, "medium",
                 "Teen gang coming-of-age; shares Coppola and the 'crime' tag only."),
    (238, 11699): (0, "medium",
                   "Corrupt-cop character study; no organised-crime dynasty."),
    (238, 261023): (1, "medium",
                    "Organised crime and FBI complicity in a true-crime register, without the "
                    "family saga."),
    (238, 310706): (0, "low",
                    "Post-Katrina crime thriller with sparse metadata; no thematic overlap."),
    (238, 240): (2, "high",
                 "Direct sequel: same family, same saga."),
    # ---- Jurassic Park (329)
    (329, 299687): (0, "high",
                    "YA alien-invasion survival story; no creature-park or science-hubris theme."),
    (329, 330): (2, "high",
                 "Direct sequel: dinosaurs, island, same franchise and spectacle."),
    (329, 262504): (0, "high",
                    "YA dystopia sequel; shares only 'adventure / science fiction'."),
    (329, 74): (1, "medium",
                "Spielberg disaster spectacle with a family in peril from monstrous forces; no "
                "dinosaurs or park."),
    (329, 18162): (1, "medium",
                   "Dinosaurs and a lost-world adventure, but played for comedy."),
    (329, 168): (0, "high",
                 "Light time-travel caper; no creature or park elements."),
    (329, 135397): (2, "high",
                    "Same franchise: a fully operating dinosaur theme park that fails."),
    (329, 13001): (0, "medium",
                   "Direct-to-video franchise science fiction; no overlap."),
    (329, 331): (2, "high",
                 "Same franchise, dinosaurs and island survival."),
    (329, 644): (0, "medium",
                 "Spielberg science fiction, but a melancholy robot fable with no creature or "
                 "adventure overlap."),
}


def hr(t):
    print("\n" + "=" * 100)
    print(t)
    print("=" * 100)


# --------------------------------------------------------------- load inputs
blind = pd.read_csv(BLIND)
key = pd.read_csv(KEY)
comparison = pd.read_csv(COMPARISON)

print(f"blind candidates : {blind.shape}")
print(f"hidden key       : {key.shape}")
print(f"comparison rows  : {comparison.shape}")

# --------------------------------------------------------------- judgments
records = []
for row in blind.itertuples():
    pair = (int(row.query_id), int(row.candidate_id))
    if pair in HUMAN:
        relevance, confidence, note = HUMAN[pair]
        source = "human"
    elif pair in AI:
        relevance, confidence, note = AI[pair]
        source = "AI-assisted"
    else:
        raise SystemExit(f"No judgment for {pair}")

    records.append({
        "query_id": int(row.query_id),
        "query_title": row.query_title,
        "candidate_id": int(row.candidate_id),
        "candidate_title": row.candidate_title,
        "year": row.year,
        "relevance": int(relevance),
        "notes": note,
        "judgment_source": source,
        "confidence": confidence,
    })

judgments = pd.DataFrame(records)
judgments.to_csv(OUT_JUDGMENTS, index=False)
print(f"\nwrote {OUT_JUDGMENTS} ({len(judgments)} rows)")

hr("VALIDATION")

checks = []


def check(name, ok, detail=""):
    checks.append((name, bool(ok), detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  - {detail}" if detail else ""))


check("Exactly 59 candidates judged", len(judgments) == 59, f"n={len(judgments)}")
check("One judgment per (query, candidate)",
      not judgments.duplicated(["query_id", "candidate_id"]).any())
check("Judged set equals the blind candidate set",
      set(zip(judgments.query_id, judgments.candidate_id))
      == set(zip(blind.query_id, blind.candidate_id)))
check("No relevance outside 0-2",
      judgments["relevance"].between(0, 2).all(),
      f"range {judgments.relevance.min()}-{judgments.relevance.max()}")
check("All five query movies represented",
      judgments.query_id.nunique() == 5, f"{sorted(judgments.query_id.unique())}")
check("Three human judgments preserved",
      int((judgments.judgment_source == "human").sum()) == 3
      and judgments.loc[judgments.judgment_source == "human", "relevance"].tolist() == [2, 2, 1],
      f"human rows={int((judgments.judgment_source == 'human').sum())}")
check("All other judgments marked AI-assisted",
      int((judgments.judgment_source == "AI-assisted").sum()) == 56)
check("No candidate duplicated or omitted",
      len(judgments.drop_duplicates(["query_id", "candidate_id"])) == 59)

# ----------------------------------------------------------- join + results
merged = judgments.merge(key, on=["query_id", "query_title", "candidate_id"], how="left")
score_map = {}
for row in comparison.itertuples():
    score_map[(int(row.query_id), row.strategy, int(row.recommended_id))] = (
        float(row.score), float(row.coverage))

merged["A_score"] = [score_map.get((q, "A", c), (np.nan, np.nan))[0]
                     for q, c in zip(merged.query_id, merged.candidate_id)]
merged["B_score"] = [score_map.get((q, "B", c), (np.nan, np.nan))[0]
                     for q, c in zip(merged.query_id, merged.candidate_id)]
merged["A_coverage"] = [score_map.get((q, "A", c), (np.nan, np.nan))[1]
                        for q, c in zip(merged.query_id, merged.candidate_id)]

results = merged[["query_id", "query_title", "candidate_id", "candidate_title", "year",
                  "relevance", "judgment_source", "confidence",
                  "A_rank", "B_rank", "A_score", "B_score", "A_coverage"]].copy()
results.to_csv(OUT_RESULTS, index=False)
print(f"\nwrote {OUT_RESULTS} ({len(results)} rows)")

# rank sanity: the key's rank must agree with the comparison file's rank
rank_ok = True
for row in comparison.itertuples():
    col = "A_rank" if row.strategy == "A" else "B_rank"
    match = results[(results.query_id == row.query_id)
                    & (results.candidate_id == row.recommended_id)]
    if len(match) != 1 or int(match.iloc[0][col]) != row.rank:
        rank_ok = False
check("Key ranks agree with the recomputed comparison ranks", rank_ok)

# --------------------------------------------------------------- metrics
def precision_at(relevances, k, threshold=1):
    top = relevances[:k]
    return sum(1 for r in top if r >= threshold) / k


def ndcg_at(relevances, ideal_gains, k):
    gains = [2 ** r - 1 for r in relevances[:k]]
    dcg = sum(g / np.log2(i + 2) for i, g in enumerate(gains))
    ideal = sorted(ideal_gains, reverse=True)[:k]
    idcg = sum(g / np.log2(i + 2) for i, g in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


rows = []
detail_lines = []

for qid, group in results.groupby("query_id", sort=False):
    qtitle = group.query_title.iloc[0]
    ideal = [2 ** r - 1 for r in group.relevance]

    lists = {}
    for strat, col in (("A", "A_rank"), ("B", "B_rank")):
        sub = group[group[col].notna()].sort_values(col)
        lists[strat] = sub

    for strat in ("A", "B"):
        sub = lists[strat]
        rels = sub.relevance.astype(int).tolist()
        assert len(rels) == 10, f"{qtitle} {strat} has {len(rels)} ranked items"
        rows.append({
            "query_id": qid, "query_title": qtitle, "strategy": strat,
            "n_returned": len(rels),
            "mean_graded_relevance": round(float(np.mean(rels)), 4),
            "P@5": round(precision_at(rels, 5), 4),
            "P@10": round(precision_at(rels, 10), 4),
            "NDCG@5": round(ndcg_at(rels, ideal, 5), 4),
            "NDCG@10": round(ndcg_at(rels, ideal, 10), 4),
        })
    detail_lines.append((qid, qtitle, lists))

per_query = pd.DataFrame(rows)

summary_rows = []
for strat in ("A", "B"):
    sub = per_query[per_query.strategy == strat]
    summary_rows.append({
        "scope": "macro_average", "strategy": strat, "n_queries": len(sub),
        "P@5": round(sub["P@5"].mean(), 4),
        "P@10": round(sub["P@10"].mean(), 4),
        "NDCG@5": round(sub["NDCG@5"].mean(), 4),
        "NDCG@10": round(sub["NDCG@10"].mean(), 4),
        "mean_graded_relevance": round(sub["mean_graded_relevance"].mean(), 4),
    })

COLUMNS = ["scope", "query_id", "query_title", "strategy", "n_queries",
           "n_returned", "mean_graded_relevance", "P@5", "P@10", "NDCG@5", "NDCG@10"]
per_query_out = per_query.copy()
per_query_out.insert(0, "scope", "per_query")
per_query_out["n_queries"] = np.nan
per_query_out = per_query_out.reindex(columns=COLUMNS)
macro_out = pd.DataFrame(summary_rows).reindex(columns=COLUMNS)
summary = pd.concat([per_query_out, macro_out], ignore_index=True)
summary[["P@5", "P@10", "NDCG@5", "NDCG@10"]] = \
    summary[["P@5", "P@10", "NDCG@5", "NDCG@10"]].astype(float)
summary.to_csv(OUT_SUMMARY, index=False)
print(f"wrote {OUT_SUMMARY} ({len(summary)} rows)")

hr("PER-QUERY METRICS")
print(per_query.to_string(index=False))
print("\nMACRO AVERAGE")
print(pd.DataFrame(summary_rows).to_string(index=False))

# ------------------------------------------------ what changed, and why
hr("CHANGES INTRODUCED BY STRATEGY B")

change_rows = []
for qid, qtitle, lists in detail_lines:
    a = set(lists["A"].candidate_id)
    b = set(lists["B"].candidate_id)
    removed = lists["A"][~lists["A"].candidate_id.isin(b)]
    added = lists["B"][~lists["B"].candidate_id.isin(a)]
    print(f"\n  {qtitle}: A removed {len(removed)}, added {len(added)}")
    for r in removed.itertuples():
        print(f"    -A[{int(r.A_rank):>2}] {r.candidate_title[:38]:<38} rel={r.relevance} "
              f"cov={r.A_coverage:.2f}")
    for r in added.itertuples():
        print(f"    +B[{int(r.B_rank):>2}] {r.candidate_title[:38]:<38} rel={r.relevance}")
    change_rows.append({
        "query_title": qtitle,
        "removed_n": len(removed),
        "removed_mean_rel": removed.relevance.mean() if len(removed) else np.nan,
        "removed_mean_cov": removed.A_coverage.mean() if len(removed) else np.nan,
        "added_n": len(added),
        "added_mean_rel": added.relevance.mean() if len(added) else np.nan,
    })

print("\n  Summary of changes")
print(pd.DataFrame(change_rows).to_string(index=False, float_format=lambda v: f"{v:.2f}"))

removed_all = pd.concat([
    lists["A"][~lists["A"].candidate_id.isin(set(lists["B"].candidate_id))]
    for _, _, lists in detail_lines
])
added_all = pd.concat([
    lists["B"][~lists["B"].candidate_id.isin(set(lists["A"].candidate_id))]
    for _, _, lists in detail_lines
])
print(f"\n  All removed by B: n={len(removed_all)}, "
      f"mean relevance={removed_all.relevance.mean():.2f}, "
      f"mean coverage={removed_all.A_coverage.mean():.2f}, "
      f"below-full-coverage={int((removed_all.A_coverage < 1).sum())}")
print(f"  All added by B  : n={len(added_all)}, "
      f"mean relevance={added_all.relevance.mean():.2f}")

hr("VALIDATION SUMMARY")
for name, ok, detail in checks:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  - {detail}" if detail else ""))
print(f"\n  {sum(1 for _, ok, _ in checks if ok)}/{len(checks)} checks passed")
