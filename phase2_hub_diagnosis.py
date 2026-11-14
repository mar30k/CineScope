"""
Phase 2 / Investigation 2 - Recommendation hub diagnosis.

Read-only, reproducible:  python phase2_hub_diagnosis.py

Measures, using the project's own J / L / N weight sets and the unchanged
cosine-brute KNN:
  1. identity of the metadata-poor "hub" movies
  2. their recommendation appearance behaviour across J / L / N
  3. exact per-feature-block decomposition of selected query-hub similarities
  4. controls: ordinary complete-metadata and blockbuster movies
  5. overview-vocabulary truncation caused by max_features=5000
"""

import sys
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize

from src.data.processor import process_movies
from src.features.feature_builder import build_features, combine_features

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

pd.set_option("display.width", 200)
pd.set_option("display.max_colwidth", 40)
pd.set_option("display.max_rows", 200)

RNG = np.random.default_rng(0)
K = 10


def hr(title):
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)


# ------------------------------------------------------------------ setup
movies = process_movies()
features = build_features(movies)
N_MOVIES = len(movies)
BLOCKS = ["genres", "keywords", "overview", "cast", "director"]

CFG = {
    "J": {"genres": 1.0, "keywords": 1.0, "overview": 1.0},
    "L": {"genres": 1.0, "keywords": 1.0, "overview": 1.0, "cast": 0.5, "director": 0.25},
    "N": {"genres": 1.0, "keywords": 1.0, "overview": 1.0, "cast": 0.5, "director": 1.0},
}

# per-block L2-normalised rows (exactly what combine_features does before scaling)
BN = {b: normalize(features[b].tocsr(), norm="l2", axis=1) for b in BLOCKS}

row_active = {b: np.asarray((BN[b] != 0).sum(axis=1)).ravel() > 0 for b in BLOCKS}
active_matrix = pd.DataFrame(row_active)
n_blocks = active_matrix.sum(axis=1).values

genres_e = ~row_active["genres"]
kw_e = ~row_active["keywords"]
ov_e = ~row_active["overview"]
cast_e = ~row_active["cast"]
dir_e = ~row_active["director"]

year = pd.to_datetime(movies["release_date"], errors="coerce").dt.year

# neighbour tables for the three experiments
IND, SIM = {}, {}
for tag, w in CFG.items():
    M = combine_features(features, weights=w)
    mdl = NearestNeighbors(metric="cosine", algorithm="brute").fit(M)
    d, i = mdl.kneighbors(M, n_neighbors=K + 1)
    IND[tag], SIM[tag] = i[:, 1:], 1.0 - d[:, 1:]      # drop self

    assert np.allclose(d[:, 0], 0.0), f"self not at rank 0 for {tag}"
# ------------------------------------------------------------------ 1. hubs
hr("1. IDENTIFICATION OF THE METADATA-POOR HUB MOVIES")

hub_idx = np.where(n_blocks <= 2)[0]
print(f"definition: movies with <= 2 of 5 non-empty feature blocks -> {len(hub_idx)} movies\n")

appear = {t: Counter() for t in CFG}
rank1 = {t: Counter() for t in CFG}
sims_in = {t: defaultdict(list) for t in CFG}
for t in CFG:
    for q in range(N_MOVIES):
        for r, j in enumerate(IND[t][q]):
            j = int(j)
            appear[t][j] += 1
            sims_in[t][j].append(float(SIM[t][q][r]))
            if r == 0:
                rank1[t][j] += 1

rows = []
for i in hub_idx:
    m = movies.iloc[i]
    active = [b for b in BLOCKS if row_active[b][i]]
    rows.append({
        "id": int(m["id"]),
        "title": m["title"][:34],
        "year": int(year.iloc[i]) if not pd.isna(year.iloc[i]) else None,
        "blocks": len(active),
        "active": "+".join(active),
        "genres": len(m["genres"]),
        "kw": len(m["keywords"]),
        "ov_chars": len(str(m["overview"]).strip()),
        "ov_terms": int((features["overview"][i] != 0).sum()),
        "cast": len(m["cast"]),
        "dir": len(m["director"]),
        "J_app": appear["J"][i], "L_app": appear["L"][i], "N_app": appear["N"][i],
        "appear": sum(appear[t][i] for t in CFG),
        "rank1": sum(rank1[t][i] for t in CFG),
        "r1_J": rank1["J"][i], "r1_L": rank1["L"][i], "r1_N": rank1["N"][i],
    })
hubs = pd.DataFrame(rows).sort_values("rank1", ascending=False).reset_index(drop=True)
hubs["share_r1_%"] = (100 * hubs["rank1"] / (len(CFG) * N_MOVIES)).round(2)
print(hubs[["id", "title", "year", "blocks", "active", "genres", "kw", "ov_chars",
            "ov_terms", "cast", "dir"]].to_string(index=False))
print("\nappearance counts per experiment:")
print(hubs[["id", "title", "J_app", "L_app", "N_app", "appear", "r1_J", "r1_L", "r1_N",
            "rank1", "share_r1_%"]].to_string(index=False))

hub_set = set(int(i) for i in hub_idx)
rank1_tot = {i: sum(rank1[t][i] for t in CFG) for i in hub_idx}
top_i = max(hub_idx, key=lambda i: rank1_tot[i])
m = movies.iloc[top_i]
print(f"\n>>> THE 370-RANK-ONE MOVIE (L config): id={m['id']} '{m['title']}'")
print(f"    rank-1 counts: J={rank1['J'][top_i]}  L={rank1['L'][top_i]}  N={rank1['N'][top_i]}"
      f"  pooled={rank1_tot[top_i]} of {len(CFG)*N_MOVIES} rank-1 slots")
print(f"    genres={m['genres']}  keywords={m['keywords']}  cast={m['cast']}  "
      f"director={m['director']}")
print(f"    overview={m['overview']!r}  (chars={len(str(m['overview']))}, "
      f"active overview terms={int((features['overview'][top_i] != 0).sum())})")

hr("1b. FULL CATALOGUE RANK-1 LEADERBOARD (all movies, all configs pooled)")
pooled_r1 = Counter()
for t in CFG:
    pooled_r1.update(rank1[t])
for i, c in pooled_r1.most_common(12):
    m = movies.iloc[i]
    print(f"  {c:5d} ({100*c/(3*N_MOVIES):5.2f}%)  {m['title'][:44]:<44} id={m['id']:<7} "
          f"blocks={n_blocks[i]} kw={len(m['keywords']):>2} ov_terms={int((features['overview'][i]!=0).sum()):>3}")

# -------------------------------------------------- 2. behaviour analysis
hr("2. RECOMMENDATION BEHAVIOUR OF HUB MOVIES")

print("2a. Query genres that pull each hub to the top (top 3 genres by frequency of query)\n")
hubs_lookup = {int(movies.iloc[i]["id"]): i for i in hub_idx}
for _, row in hubs.head(8).iterrows():
    i = hubs_lookup[int(row["id"])]
    queries = set()
    for t in CFG:
        for q in range(N_MOVIES):
            if i in IND[t][q]:
                queries.add(q)
    qg = Counter()
    for q in queries:
        qg.update(set(movies.iloc[q]["genres"]))
    qdir = set()
    for q in queries:
        qdir.update(movies.iloc[q]["director"])
    topg = ", ".join(f"{g}({c})" for g, c in qg.most_common(3))
    print(f"  id={row['id']:<7} {row['title'][:32]:<32} distinct queries={len(queries):<5} "
          f"distinct query genres={len(qg):<3} distinct query directors={len(qdir):<5} | {topg}")

print("\n2b. Does each hub appear across experiments, or is it concentrated in one?\n")
for _, row in hubs.head(10).iterrows():
    per = {t: appear[t][hubs_lookup[int(row['id'])]] for t in CFG}
    print(f"  id={row['id']:<7} {row['title'][:32]:<32} J={per['J']:<5} L={per['L']:<5} N={per['N']:<5}")

print("\n2c. Rank at which hubs are returned (pooled over all three experiments)\n")
print(f"  {'id':<8}{'title':<34}{'rank1':>7}{'rank2':>7}{'rank3':>7}{'rank4-10':>10}{'mean sim':>10}")
rankpos = {i: Counter() for i in hub_idx}
simacc = {i: [] for i in hub_idx}
for t in CFG:
    for q in range(N_MOVIES):
        for r, j in enumerate(IND[t][q]):
            j = int(j)
            if j in rankpos:
                rankpos[j][r + 1] += 1
                simacc[j].append(float(SIM[t][q][r]))
for _, row in hubs.head(12).iterrows():
    i = hubs_lookup[int(row["id"])]
    c = rankpos[i]
    print(f"  {row['id']:<8}{row['title'][:34]:<34}{c[1]:>7}{c[2]:>7}{c[3]:>7}"
          f"{sum(v for k, v in c.items() if k >= 4):>10}{np.mean(simacc[i]):>10.4f}")

print("\n2d. Concrete mis-recommendations: the highest-similarity query->hub edges\n")
edges = []
for t in CFG:
    for q in range(N_MOVIES):
        for r, j in enumerate(IND[t][q]):
            j = int(j)
            if j in hub_set:
                edges.append((t, q, j, r + 1, float(SIM[t][q][r])))
edges.sort(key=lambda e: -e[4])
seen = set()
shown = 0
print(f"  {'cfg':<4}{'query':<38}{'-> hub':<34}{'rank':>5}{'sim':>8}")
for t, q, j, r, s in edges:
    key = (q, j)
    if key in seen:
        continue
    seen.add(key)
    print(f"  {t:<4}{movies.iloc[q]['title'][:36]:<38}"
          f"{movies.iloc[j]['title'][:32]:<34}{r:>5}{s:>8.4f}")
    shown += 1
    if shown >= 14:
        break

hr("2e. ZERO-SIMILARITY TAIL: WHEN FEWER THAN 10 MOVIES SHARE ANY FEATURE")
print("  A query with < 10 non-zero-similarity candidates gets its list padded with")
print("  similarity 0.0 movies returned in ARBITRARY (positional) order.\n")
for t in CFG:
    z = (SIM[t] == 0).sum()
    short = [(1 - SIM[t][q] > 0).sum() < K for q in range(N_MOVIES)]
    print(f"  config {t}: zero-similarity slots={int(z)} ({100*z/(N_MOVIES*K):.2f}% of slots) | "
          f"movies with <10 non-zero neighbours={int(sum(short))}")

zq = []
for t in CFG:
    for q in range(N_MOVIES):
        nz = int((SIM[t][q] > 0).sum())
        if nz < K:
            zq.append((t, q, nz))
print(f"\n  affected (config, query) pairs: {len(zq)}; distinct query movies: "
      f"{len({q for _, q, _ in zq})}")
print("  worst cases (fewest genuine neighbours):")
print(f"  {'cfg':<4}{'query':<40}{'nonzero nbrs':>13}")
for t, q, nz in sorted(zq, key=lambda x: x[2])[:10]:
    print(f"  {t:<4}{movies.iloc[q]['title'][:38]:<40}{nz:>13}")

print("\n  the 5 evaluation queries — how many of their 10 recs are zero-similarity?")
for name in ["Avatar", "The Dark Knight", "Finding Nemo", "The Godfather", "Jurassic Park"]:
    q = movies.index[movies["title"].str.lower() == name.lower()][0]
    s = [f"{t}:{int((SIM[t][q] == 0).sum())}" for t in CFG]
    print(f"    {name:<18} " + "  ".join(s))

print("\n  which movies fill the zero-similarity tail most often?")
zt = Counter()
for t in CFG:
    for q in range(N_MOVIES):
        for r, j in enumerate(IND[t][q]):
            if SIM[t][q][r] == 0:
                zt[int(j)] += 1
for j, c in zt.most_common(5):
    print(f"    {c:4d} zero-sim slots  {movies.iloc[j]['title'][:44]} (id={movies.iloc[j]['id']})")

print("\n  maximum incoming similarity each hub actually achieves:")
print(f"  {'id':<8}{'title':<36}{'max in':>9}{'mean in':>9}{'n edges':>9}")
for _, row in hubs.head(10).iterrows():
    i = hubs_lookup[int(row["id"])]
    v = [s for t in CFG for s in sims_in[t][i]]
    print(f"  {row['id']:<8}{row['title'][:34]:<36}{max(v):>9.4f}{np.mean(v):>9.4f}{len(v):>9}")

# ------------------------------------------------ 3. pair decomposition
hr("3. PER-FEATURE-BLOCK DECOMPOSITION OF SELECTED QUERY -> HUB PAIRS")

print("identity: each block is L2-normalised per row, scaled by sqrt(w_b), then hstacked, so")
print("  cos(x,y) = SUM_b w_b * cos_b(x,y) / ( sqrt(SUM_b w_b*1[x_b!=0]) * sqrt(SUM_b w_b*1[y_b!=0]) )")
print("  The denominators are the row norms; they depend on WHICH blocks are non-empty.\n")


def decompose(q, j, w, tag):
    num = 0.0
    rows = []
    for b in BLOCKS:
        xb, yb = BN[b][q], BN[b][j]
        nx = float(xb.multiply(xb).sum())
        ny = float(yb.multiply(yb).sum())
        dot = float(xb.multiply(yb).sum())
        cosb = dot / np.sqrt(nx * ny) if (nx > 0 and ny > 0) else 0.0
        wb = w.get(b, 0.0)
        contrib = wb * cosb
        num += contrib
        rows.append({
            "block": b, "w": wb,
            "x_active": int(nx > 0), "y_active": int(ny > 0),
            "cos_b": cosb, "w*cos_b": contrib,
        })
    den = np.sqrt(sum(w.get(b, 0.0) for b in BLOCKS if row_active[b][q])) * \
          np.sqrt(sum(w.get(b, 0.0) for b in BLOCKS if row_active[b][j]))
    df = pd.DataFrame(rows)
    return df, num, den, num / den if den else np.nan


pairs = [
    ("The Dark Knight", "Amidst the Devil's Wings"),
    ("The Godfather", "Amidst the Devil's Wings"),
    ("Finding Nemo", "The Helix... Loaded"),
    ("The Dark Knight", "The Dark Knight Rises"),
]

# plus the single strongest incoming edge for the third-largest hub, found dynamically
_h = movies.index[movies["title"].str.lower() == "me you and five bucks"][0]
_best = max(((q, r) for q in range(N_MOVIES) for r, j in enumerate(IND["L"][q])
             if int(j) == _h), key=lambda x: SIM["L"][x[0]][x[1]])
pairs.append((movies.iloc[_best[0]]["title"], "Me You and Five Bucks"))
print(f"\n[strongest incoming edge to id 361505 in L: "
      f"'{movies.iloc[_best[0]]['title']}' sim={SIM['L'][_best[0]][_best[1]]:.4f}]")


for qname, jname in pairs:
    qi = movies.index[movies["title"].str.lower() == qname.lower()][0]
    ji = movies.index[movies["title"].str.lower() == jname.lower()][0]
    print(f"\n### {qname} (id {movies.iloc[qi]['id']})  ->  {jname} (id {movies.iloc[ji]['id']})")
    print(f"    active blocks query={[b for b in BLOCKS if row_active[b][qi]]}")
    print(f"    active blocks hub  ={[b for b in BLOCKS if row_active[b][ji]]}")
    for tag, w in CFG.items():
        df, num, den, total = decompose(qi, ji, w, tag)
        arr = list(IND[tag][qi])
        rank = arr.index(ji) + 1 if ji in arr else None
        obs = float(SIM[tag][qi][rank - 1]) if rank else None
        print(f"\n  config {tag} (weights {w})")
        print(df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
        print(f"    weighted numerator = {num:.6f}")
        print(f"    denominator = sqrt({sum(w.get(b,0) for b in BLOCKS if row_active[b][qi]):.2f})"
              f" * sqrt({sum(w.get(b,0) for b in BLOCKS if row_active[b][ji]):.2f}) = {den:.6f}")
        if obs is None:
            print(f"    combined cosine = {total:.6f}   | NOT IN THIS QUERY'S TOP-10")
        else:
            print(f"    combined cosine = {total:.6f}   | observed = {obs:.4f} at rank {rank}"
                  f"   | delta = {abs(total - obs):.2e}")

hr("3b. WHAT WOULD THIS PAIR SCORE IF THE HUB HAD A FULL PROFILE?")
qi = movies.index[movies["title"].str.lower() == "the dark knight"][0]
ji = movies.index[movies["title"].str.lower() == "amidst the devil's wings"][0]
for tag, w in CFG.items():
    df, num, den, total = decompose(qi, ji, w, tag)
    full_den = np.sqrt(sum(w.values())) * np.sqrt(sum(w.values()))
    print(f"  {tag}: numerator={num:.4f} (unchanged) | current denominator={den:.4f} -> score {total:.4f}"
          f" | if both profiles complete: denominator={full_den:.4f} -> score {num/full_den:.4f}")
print("\n  interpretation: a missing block removes a term from the NUMERATOR but also shrinks the")
print("  DENOMINATOR. The hub's own denominator is 1.0 (genres only), so every shared genre is")
print("  divided only by the query's norm -- the overlap is not diluted by any non-matching content.")

# ------------------------------------------------ 4. ordinary comparison
hr("4. CONTROL COMPARISON: ORDINARY COMPLETE-METADATA MOVIES AND BLOCKBUSTERS")

complete = np.where(n_blocks == 5)[0]
sample = RNG.choice(complete, size=200, replace=False)
blockbusters = movies["vote_count"].sort_values(ascending=False).index[:10].values
mid_pop = np.where((movies["vote_count"] > 500) & (movies["vote_count"] < 3000))[0]
mid_sample = RNG.choice(mid_pop, size=100, replace=False)


def stats_for(idx, label):
    r1 = np.array([sum(rank1[t][i] for t in CFG) for i in idx])
    ap = np.array([sum(appear[t][i] for t in CFG) for i in idx])
    inc = [np.mean(sims_in[t][i]) for t in CFG for i in idx if sims_in[t][i]]
    return {
        "group": label, "n": len(idx),
        "mean rank1": r1.mean(), "median rank1": np.median(r1), "max rank1": r1.max(),
        "mean appearances": ap.mean(), "median appearances": np.median(ap),
        "max appearances": ap.max(),
        "mean incoming sim": np.mean(inc) if inc else np.nan,
    }


ctrl = pd.DataFrame([
    stats_for(hub_idx, "24 metadata-poor hubs"),
    stats_for(sample, "200 random complete (control)"),
    stats_for(blockbusters, "top-10 by vote_count"),
    stats_for(mid_sample, "100 mid-popularity"),
])
print(ctrl.to_string(index=False, float_format=lambda v: f"{v:.2f}"))

print("\n  top-10 by vote_count — do popular films dominate recommendations?")
for i in blockbusters[:10]:
    m = movies.iloc[i]
    print(f"    votes={int(m['vote_count']):>6}  rank1={sum(rank1[t][i] for t in CFG):>4}  "
          f"appearances={sum(appear[t][i] for t in CFG):>4}  {m['title'][:40]}")

print("\n  across ALL movies: does metadata completeness drive rank-1 count?")
tab = pd.DataFrame({
    "n_blocks": n_blocks,
    "kw": movies["keywords"].apply(len).values,
    "ov_terms": np.asarray((features["overview"] != 0).sum(axis=1)).ravel(),
    "rank1": [sum(rank1[t][i] for t in CFG) for i in range(N_MOVIES)],
    "appear": [sum(appear[t][i] for t in CFG) for i in range(N_MOVIES)],
    "votes": movies["vote_count"].values,
})
print(tab.groupby("n_blocks")[["rank1", "appear"]].agg(["mean", "median", "count"])
      .to_string(float_format=lambda v: f"{v:.2f}"))
print("\n  Spearman correlations with rank-1 count:")
for col in ["n_blocks", "kw", "ov_terms", "votes"]:
    print(f"    {col:<10} rho = {tab[col].corr(tab['rank1'], method='spearman'):+.4f}")
print("\n  keyword-count bucket -> share of rank-1 slots:")
tab["kw_bucket"] = pd.cut(tab["kw"], [-1, 0, 2, 5, 10, 20, 200],
                          labels=["0", "1-2", "3-5", "6-10", "11-20", "21+"])
b = tab.groupby("kw_bucket", observed=True).agg(movies=("rank1", "size"), rank1=("rank1", "sum"))
b["share_of_slots_%"] = (100 * b["rank1"] / b["rank1"].sum()).round(2)
b["share_of_catalogue_%"] = (100 * b["movies"] / b["movies"].sum()).round(2)
b["over_representation"] = (b["share_of_slots_%"] / b["share_of_catalogue_%"]).round(2)
print(b.to_string())

# ------------------------------------------- 5. overview vocab truncation
hr("5. OVERVIEW VOCABULARY TRUNCATION (max_features=5000)")

ov_text = movies["overview"].fillna("")
full = TfidfVectorizer(stop_words="english")
full.fit(ov_text)
F = full.transform(ov_text)
pruned_vec = TfidfVectorizer(stop_words="english", max_features=5000)
P = pruned_vec.fit_transform(ov_text)
n_full = np.asarray((F != 0).sum(axis=1)).ravel()
n_pruned = np.asarray((P != 0).sum(axis=1)).ravel()

nonempty_text = ov_text.str.strip().str.len() > 0
print(f"  full overview vocabulary      : {len(full.vocabulary_)} terms")
print(f"  retained vocabulary           : {P.shape[1]} terms "
      f"({100*P.shape[1]/len(full.vocabulary_):.2f}% of the corpus vocabulary)")
print(f"  movies with non-empty overview: {int(nonempty_text.sum())}")
print(f"  ... whose vectorised overview has ZERO active features: "
      f"{int((nonempty_text & (n_pruned == 0)).sum())}")
for i in np.where(nonempty_text & (n_pruned == 0))[0]:
    print(f"      id={movies.iloc[i]['id']} '{movies.iloc[i]['title']}' "
          f"overview={movies.iloc[i]['overview']!r}  full-vocab terms={n_full[i]}")

retention = np.where(n_full > 0, n_pruned / np.maximum(n_full, 1), np.nan)
print(f"\n  token retention per movie (pruned/full): "
      f"median={np.nanmedian(retention):.3f} "
      f"p05={np.nanpercentile(retention,5):.3f} p95={np.nanpercentile(retention,95):.3f}")
for thr in (0.0, 0.25, 0.5, 0.75):
    k = int(np.nansum(retention <= thr))
    print(f"  movies retaining <= {int(thr*100):>2}% of their overview terms: {k}")

rare = np.where((n_full > 0) & (retention < 0.5))[0]
print(f"\n  {len(rare)} movies lose more than half of their overview tokens.")
print("  do low-retention movies get recommended more?")
low = np.zeros(N_MOVIES, bool)
low[rare] = True
r1 = np.array([sum(rank1[t][i] for t in CFG) for i in range(N_MOVIES)])
app = np.array([sum(appear[t][i] for t in CFG) for i in range(N_MOVIES)])
print(f"    low-retention  (n={len(rare)}): mean rank1={r1[low].mean():.3f} "
      f"mean appearances={app[low].mean():.2f}")
print(f"    rest           (n={N_MOVIES-len(rare)}): mean rank1={r1[~low].mean():.3f} "
      f"mean appearances={app[~low].mean():.2f}")

print("\n  short overviews as a hub risk factor (active overview terms):")
buck = pd.cut(n_pruned, [-1, 0, 3, 10, 20, 40, 5000],
              labels=["0", "1-3", "4-10", "11-20", "21-40", "41+"])
t2 = pd.DataFrame({"bucket": buck, "rank1": r1, "appear": app})
g = t2.groupby("bucket", observed=True).agg(movies=("rank1", "size"), rank1=("rank1", "sum"),
                                            mean_appear=("appear", "mean"))
g["share_of_r1_%"] = (100 * g["rank1"] / g["rank1"].sum()).round(2)
g["share_of_cat_%"] = (100 * g["movies"] / g["movies"].sum()).round(2)
g["over_rep"] = (g["share_of_r1_%"] / g["share_of_cat_%"]).round(2)
print(g.to_string(float_format=lambda v: f"{v:.2f}"))

print("\n  is truncation biased by genre or era?")
primary = movies["genres"].apply(lambda g: g[0] if g else "NONE")
by_genre = pd.DataFrame({"g": primary, "ret": retention, "n": 1})
gg = by_genre.groupby("g").agg(movies=("n", "size"), mean_retention=("ret", "mean")).sort_values("mean_retention")
print(gg.to_string(float_format=lambda v: f"{v:.3f}"))
by_dec = pd.DataFrame({"d": (year // 10 * 10), "ret": retention})
print("\n  by release decade:")
print(by_dec.groupby("d").agg(movies=("ret", "size"), mean_retention=("ret", "mean"))
      .to_string(float_format=lambda v: f"{v:.3f}"))

print("\n  DETAIL: what happens to the hub's overview")
i = movies.index[movies["title"].str.lower() == "amidst the devil's wings"][0]
toks = full.build_analyzer()(movies.iloc[i]["overview"])
vocab = full.vocabulary_
print(f"    raw overview      : {movies.iloc[i]['overview']!r}")
print(f"    tokens after stopword removal: {toks}")
for t in toks:
    print(f"      '{t}': full-vocab idf rank-present={t in vocab}, "
          f"retained in top-5000={t in pruned_vec.vocabulary_}")
