"""
Phase 2 / Investigation 3 - Similarity strategy comparison.

Isolated experiment. Does NOT touch src/ or the production recommender.

    python phase2_similarity_strategies.py

Strategies, all over the same five L2-normalised blocks and the same J/L/N weights:

  A (baseline)  concatenated cosine
                S_A = SUM_b w_b c_b / ( sqrt(SUM_b w_b 1[x_b!=0]) * sqrt(SUM_b w_b 1[y_b!=0]) )

  B (fixed denom)  S_B = SUM_b w_b 1[x_b!=0] 1[y_b!=0] c_b  /  SUM_b w_b        (ALL configured weights)

  C (shared denom) S_C = SUM_b w_b 1[x_b!=0] 1[y_b!=0] c_b  /  SUM_{shared b} w_b

Because c_b = 0 whenever a block is inactive on either side, all three share the SAME
numerator; they differ only in the denominator.  S_C = S_B / coverage.
"""

import sys
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize

from src.data.processor import process_movies
from src.features.feature_builder import build_features, combine_features

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

pd.set_option("display.width", 220)
pd.set_option("display.max_colwidth", 46)
pd.set_option("display.max_rows", 300)

K = 10
CHUNK = 1024
BLOCKS = ["genres", "keywords", "overview", "cast", "director"]

CFG = {
    "J": {"genres": 1.0, "keywords": 1.0, "overview": 1.0},
    "L": {"genres": 1.0, "keywords": 1.0, "overview": 1.0, "cast": 0.5, "director": 0.25},
    "N": {"genres": 1.0, "keywords": 1.0, "overview": 1.0, "cast": 0.5, "director": 1.0},
}


def hr(t):
    print("\n" + "=" * 104)
    print(t)
    print("=" * 104)


movies = process_movies()
features = build_features(movies)
N = len(movies)

BN = {b: normalize(features[b].tocsr(), norm="l2", axis=1) for b in BLOCKS}
ACT = {b: (np.asarray((BN[b] != 0).sum(axis=1)).ravel() > 0) for b in BLOCKS}
n_blocks = pd.Series(sum(ACT[b].astype(int) for b in BLOCKS), index=movies.index)
kw_len = movies["keywords"].apply(len).values
yr = pd.to_datetime(movies["release_date"], errors="coerce").dt.year

hub_ids = [378237, 43630, 361505, 292539, 47546, 102840, 114065, 447027]
id2row = {int(movies.iloc[i]["id"]): i for i in range(N)}


def active_list(i):
    return [b for b in BLOCKS if ACT[b][i]]


hr("0. IDENTITY CHECK — STRATEGY A MUST REPRODUCE THE PRODUCTION COSINE")
chk = combine_features(features, weights=CFG["L"])
mdl = NearestNeighbors(metric="cosine", algorithm="brute").fit(chk)
d, i = mdl.kneighbors(chk, n_neighbors=11)
IND_A_L, SIM_A_L = i[:, 1:], 1.0 - d[:, 1:]
print(f"  production L model fitted; mean top-1 similarity = {SIM_A_L[:,0].mean():.4f}")

# per-block cosine for specific pairs, straight from the normalised blocks
def block_cos(i1, i2, b):
    x, y = BN[b][i1], BN[b][i2]
    nx = float(x.multiply(x).sum())
    ny = float(y.multiply(y).sum())
    if nx == 0 or ny == 0:
        return 0.0
    return float(x.multiply(y).sum()) / np.sqrt(nx * ny)


def pair_scores(i1, i2, w):
    tot = sum(w.values())
    num = 0.0
    shared_w = 0.0
    shared = []
    per = {}
    for b in BLOCKS:
        c = block_cos(i1, i2, b)
        per[b] = c
        if b in w and ACT[b][i1] and ACT[b][i2]:
            num += w[b] * c
            shared_w += w[b]
            shared.append(b)
    sa = sum(w[b] for b in w if ACT[b][i1])
    sb = sum(w[b] for b in w if ACT[b][i2])
    denA = np.sqrt(sa) * np.sqrt(sb)
    S_A = num / denA if denA else 0.0
    S_B = num / tot
    S_C = num / shared_w if shared_w else float("nan")
    return dict(per=per, num=num, shared=shared, shared_w=shared_w, tot=tot,
                S_A=S_A, S_B=S_B, S_C=S_C, coverage=shared_w / tot)


# ---------------------------------------------------------------- pair set
hr("1. CONTROLLED PAIR SET")

def row_of(mid):
    return id2row[mid]


comp = n_blocks == 5
kw_empty_all_else = (kw_len == 0) & (n_blocks.values == 4)

# P2: keyword-empty but otherwise complete, most-voted
p2 = int(np.where(kw_empty_all_else)[0][np.argmax(movies["vote_count"].values[kw_empty_all_else])])

# P5: two movies with the SAME genre set and both complete, maximising |overview length diff|
ovlen = movies["overview"].fillna("").str.len().values
gsets = movies["genres"].apply(lambda g: frozenset(g))
best = None
for g, idx in gsets.groupby(gsets).groups.items():
    idx = np.array([row_of(int(movies.iloc[j]["id"])) for j in idx], dtype=int)
    idx = idx[comp.values[idx]]
    if len(idx) < 2 or len(g) < 2:
        continue
    sub = idx[np.argsort(ovlen[idx])]
    for a in range(len(sub)):
        for b in range(a + 1, len(sub)):
            gap = ovlen[sub[b]] - ovlen[sub[a]]
            if best is None or gap > best[0]:
                best = (gap, sub[a], sub[b], tuple(sorted(g)))
if best:
    _, p5a, p5b, p5g = best

# P6: worst overview retention, excluding the hubs already listed
from sklearn.feature_extraction.text import TfidfVectorizer
ovt = movies["overview"].fillna("")
F = TfidfVectorizer(stop_words="english").fit_transform(ovt)
P = TfidfVectorizer(stop_words="english", max_features=5000).fit_transform(ovt)
nf = np.asarray((F != 0).sum(axis=1)).ravel()
np_ = np.asarray((P != 0).sum(axis=1)).ravel()
ret = np.where(nf > 0, np_ / np.maximum(nf, 1), np.nan)
cand = [j for j in np.argsort(ret) if not np.isnan(ret[j]) and 0 < ret[j] < 0.35
        and j not in {row_of(h) for h in hub_ids} and n_blocks.iloc[j] == 5]
p6 = int(cand[0])
p6_top1 = int(IND_A_L[p6][0])   # its own best match under the production config

PAIRS = [
    ("P1 complete + complete",
     row_of(155), row_of(49026)),
    ("P2 complete + keyword-empty (else complete)", row_of(155), p2),
    ("P3 both keyword-empty", row_of(18701), row_of(361505)),
    ("P4 genres-only + complete", row_of(378237), row_of(155)),
    ("P5 same genres, different everything else", int(p5a), int(p5b)),
    ("P6 worst retention + its own top-1 match", p6, p6_top1),
    ("P7 genres-only hub + genres-only hub", row_of(43630), row_of(292539)),
    ("P8 genres-only hub + documentary query", row_of(292539),
     int(np.where(movies["title"].str.lower() == "call + response")[0][0])),
    ("P9 ordinary complete + complete", row_of(238), row_of(240)),
]

def desc(i):
    m = movies.iloc[i]
    return f"{m['title'][:34]} (id {m['id']})"

print(f"  {'pair':<56}{'movie a':<44}{'movie b'}")
for name, a, b in PAIRS:
    print(f"  {name:<56}{desc(a):<44}{desc(b)}")
print("\n  P5 genre set:", " / ".join(p5g))
print("  P6 retention:", f"{ret[p6]:.3f}")

hr("2. PAIRWISE COMPARISON — BLOCK COSINES AND THREE STRATEGY SCORES")

for name, a, b in PAIRS:
    print(f"\n### {name}")
    print(f"    a = {desc(a)}  [blocks present: {n_blocks.iloc[a]}] active={active_list(a)}")
    print(f"    b = {desc(b)}  [blocks present: {n_blocks.iloc[b]}] active={active_list(b)}")
    wl = CFG["L"]
    print(f"    {'block':<10}{'w(L)':>6}{'a active':>9}{'b active':>9}{'cos_b':>9}"
          f"{'w*cos':>9}")
    r = pair_scores(a, b, wl)
    for bname in BLOCKS:
        wv = wl.get(bname, 0.0)
        cs = r["per"][bname]
        print(f"    {bname:<10}{wv:>6.2f}{int(ACT[bname][a]):>9}{int(ACT[bname][b]):>9}"
              f"{cs:>9.4f}{wv*cs:>9.4f}")
    print(f"    {'SUM':<10}{r['tot']:>6.2f}{'':>18}{'num='+format(r['num'],'.4f'):>18}")
    print(f"    shared blocks ({len(r['shared'])}): {r['shared']}  |  "
          f"shared weight {r['shared_w']:.2f} of {r['tot']:.2f} = {100*r['coverage']:.1f}%")
    print(f"    {'cfg':<5}{'A concat':>10}{'B fixed':>10}{'C shared':>10}"
          f"{'coverage':>10}{'B-A':>9}{'C-A':>9}{'C-B':>9}")
    for tag, w in CFG.items():
        rr = pair_scores(a, b, w)
        cov = rr["coverage"]
        cdiff = rr["S_C"] - rr["S_B"] if rr["shared_w"] else float("nan")
        print(f"    {tag:<5}{rr['S_A']:>10.4f}{rr['S_B']:>10.4f}{rr['S_C']:>10.4f}"
              f"{100*cov:>9.1f}%{rr['S_B']-rr['S_A']:>9.4f}{rr['S_C']-rr['S_A']:>9.4f}"
              f"{cdiff:>9.4f}")

# ------------------------------------------------- all-pairs strategy engine
hr("3. FULL-CATALOGUE ENGINE FOR STRATEGIES A / B / C")
print("  streaming over row chunks; one numerator + three denominators per config\n")

RES = {}          # (cfg, strategy) -> dict of aggregates
TOP = {}          # (cfg, strategy) -> top-10 index array
SCORE = {}        # (cfg, strategy) -> top-10 score array
overlap = {}      # (cfg, label) -> list of overlaps

for tag, w in CFG.items():
    Wtot = sum(w.values())
    svec = sum(w[b] * ACT[b].astype(float) for b in w if b in w).astype(float)
    svec_all = np.zeros(N)
    for b in BLOCKS:
        svec_all += w.get(b, 0.0) * ACT[b].astype(float)

    top_idx = {s: np.zeros((N, K), dtype=np.int32) for s in "ABC"}
    top_sc = {s: np.zeros((N, K)) for s in "ABC"}
    maxdiff = 0.0

    for start in range(0, N, CHUNK):
        stop = min(start + CHUNK, N)
        num = np.zeros((stop - start, N))
        shw = np.zeros((stop - start, N))
        for b in BLOCKS:
            wb = w.get(b, 0.0)
            if wb <= 0:
                continue
            Cb = (BN[b][start:stop] @ BN[b].T).toarray()
            num += wb * Cb
            shw += wb * np.outer(ACT[b][start:stop], ACT[b])
        denA = np.sqrt(np.outer(svec_all[start:stop], svec_all))
        denA[denA == 0] = np.inf
        SA = num / denA
        SB = num / Wtot
        SC = np.where(shw > 0, num / np.where(shw > 0, shw, 1), 0.0)

        for q in range(stop - start):
            g = start + q
            for s, M in (("A", SA), ("B", SB), ("C", SC)):
                row = M[q].copy()
                row[g] = -np.inf
                k = np.argpartition(-row, K)[:K]
                k = k[np.argsort(-row[k])]
                top_idx[s][g] = k
                top_sc[s][g] = row[k]

    # reproduce the production cosine for strategy A and compare
    afin = {s: [] for s in "ABC"}
    for s in "ABC":
        c = Counter()
        for q in range(N):
            for j, sc in zip(top_idx[s][q], top_sc[s][q]):
                c[int(j)] += 1
        afin[s] = c

    RES[tag] = {
        "r1": {s: Counter(int(top_idx[s][q][0]) for q in range(N)) for s in "ABC"},
        "app": afin,
        "top_idx": top_idx, "top_sc": top_sc,
    }
    # agreement with sklearn
    a_scores = np.array([top_sc["A"][q][0] for q in range(N)])
    print(f"  config {tag}: mean strategy-A top-1 = {a_scores.mean():.4f} "
          f"(production L check below)")

# verify A against the production model
d, i = mdl.kneighbors(chk, n_neighbors=2)
prod_top1 = 1.0 - d[:, 1]
mine = np.array([RES["L"]["top_sc"]["A"][q][0] for q in range(N)])
print(f"\n  strategy A vs production sklearn (config L): "
      f"max abs diff top-1 = {np.abs(prod_top1 - mine).max():.2e}")

# ------------------------------------------------------------- 4. hubs
hr("4. HUB BEHAVIOUR UNDER EACH STRATEGY")

for tag in CFG:
    print(f"\n  ===== config {tag} =====")
    print(f"  {'id':<8}{'title':<30}{'strat':<7}{'rank-1':>8}{'top10 app':>10}"
          f"{'distinct q':>12}{'mean sc':>9}{'med sc':>8}")
    for hid in hub_ids:
        h = row_of(hid)
        for s in "ABC":
            r1 = RES[tag]["r1"][s].get(h, 0)
            app = RES[tag]["app"][s].get(h, 0)
            queries = [q for q in range(N) if h in set(RES[tag]["top_idx"][s][q])]
            scs = [float(RES[tag]["top_sc"][s][q][k]) for q in queries
                   for k in range(K) if RES[tag]["top_idx"][s][q][k] == h]
            print(f"  {hid:<8}{movies.iloc[h]['title'][:28]:<30}{s:<7}{r1:>8}{app:>10}"
                  f"{len(queries):>12}{np.mean(scs):>9.4f}{np.median(scs):>8.4f}")

hr("5. GLOBAL EFFECT: WHO OCCUPIES THE TOP-10 SLOTS")
rows = []
for tag in CFG:
    for s in "ABC":
        slots = [int(j) for q in range(N) for j in RES[tag]["top_idx"][s][q]]
        blocks_of = n_blocks.values[slots]
        sc = np.concatenate([RES[tag]["top_sc"][s][q] for q in range(N)])
        rows.append({
            "cfg": tag, "strat": s,
            "mean score@1": np.mean([RES[tag]["top_sc"][s][q][0] for q in range(N)]),
            "mean score@10": np.mean([RES[tag]["top_sc"][s][q][-1] for q in range(N)]),
            "slots 1-block %": 100 * (blocks_of == 1).mean(),
            "slots 2-block %": 100 * (blocks_of == 2).mean(),
            "slots 3-block %": 100 * (blocks_of == 3).mean(),
            "slots 4-block %": 100 * (blocks_of == 4).mean(),
            "slots 5-block %": 100 * (blocks_of == 5).mean(),
            "kw-empty %": 100 * (kw_len[slots] == 0).mean(),
            "distinct rank-1 movies": len(RES[tag]["r1"][s]),
        })
print(pd.DataFrame(rows).to_string(index=False, float_format=lambda v: f"{v:.2f}"))

print("\n  24 near-empty records' share of rank-1 slots:")
severe = set(np.where(n_blocks.values <= 2)[0])
for tag in CFG:
    out = []
    for s in "ABC":
        c = sum(v for k, v in RES[tag]["r1"][s].items() if k in severe)
        out.append(f"{s}={c} ({100*c/N:.2f}%)")
    print(f"    {tag}: " + "  ".join(out))

print("\n  keyword-empty records' share of rank-1 slots:")
kw_empty_rows = set(np.where(kw_len == 0)[0])
for tag in CFG:
    out = []
    for s in "ABC":
        c = sum(v for k, v in RES[tag]["r1"][s].items() if k in kw_empty_rows)
        out.append(f"{s}={100*c/N:.2f}%")
    print(f"    {tag}: " + "  ".join(out))

hr("6. TOP-10 LEADERBOARD BY STRATEGY (config L)")
for s in "ABC":
    print(f"\n  strategy {s}:")
    for h, c in RES["L"]["r1"][s].most_common(8):
        print(f"    {c:5d}  {movies.iloc[h]['title'][:46]:<46} id={movies.iloc[h]['id']:<7} "
              f"blocks={n_blocks.iloc[h]} kw={kw_len[h]}")

hr("7. DO THE STRATEGIES RANK THE SAME CANDIDATES? (config L)")
for s in "BC":
    ov = [len(set(RES["L"]["top_idx"]["A"][q]) & set(RES["L"]["top_idx"][s][q]))
          for q in range(N)]
    print(f"  A vs {s}: mean top-10 overlap = {np.mean(ov):.2f}/10  "
          f"(identical lists: {sum(1 for v in ov if v == 10)}/{N})")
comp_rows = [q for q in range(N) if n_blocks.iloc[q] == 5]
for s in "BC":
    ov = [len(set(RES["L"]["top_idx"]["A"][q]) & set(RES["L"]["top_idx"][s][q]))
          for q in comp_rows]
    print(f"  A vs {s} restricted to complete-metadata queries: "
          f"{np.mean(ov):.2f}/10")

hr("8. SIDE-BY-SIDE RECOMMENDATIONS FOR SELECTED QUERIES (config L)")
for qname in ["The Dark Knight", "Finding Nemo", "The Godfather", "Avatar"]:
    q = int(np.where(movies["title"].str.lower() == qname.lower())[0][0])
    print(f"\n  QUERY: {qname} (blocks={n_blocks.iloc[q]})")
    print(f"  {'rank':<5}{'A concat':<40}{'B fixed':<40}{'C shared'}")
    for k in range(K):
        cells = []
        for s in "ABC":
            j = RES["L"]["top_idx"][s][q][k]
            cells.append(f"{movies.iloc[j]['title'][:28]:<30}{RES['L']['top_sc'][s][q][k]:.3f}")
        print(f"  {k+1:<5}{cells[0]:<40}{cells[1]:<40}{cells[2]}")
