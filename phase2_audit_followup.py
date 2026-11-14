"""Phase 2 audit, part 2: targeted follow-ups on the anomalies found in part 1."""

import sys
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfVectorizer

from src.data.processor import process_movies
from src.features.feature_builder import build_features, combine_features

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

pd.set_option("display.width", 170)
pd.set_option("display.max_colwidth", 90)


def hr(t):
    print("\n" + "=" * 80)
    print(t)
    print("=" * 80)


movies = process_movies()
features = build_features(movies)

genres_e = movies["genres"].apply(len) == 0
kw_e = movies["keywords"].apply(len) == 0
ov_len = movies["overview"].fillna("").str.strip().str.len()
ov_e = ov_len == 0
cast_e = movies["cast"].apply(len) == 0
dir_e = movies["director"].apply(len) == 0

CONFIGS = {
    "J": {"genres": 1.0, "keywords": 1.0, "overview": 1.0},
    "L": {"genres": 1.0, "keywords": 1.0, "overview": 1.0, "cast": 0.5, "director": 0.25},
    "N": {"genres": 1.0, "keywords": 1.0, "overview": 1.0, "cast": 0.5, "director": 1.0},
}

hr("A. BLOCK-PRESENCE PROFILE PER CONFIG")
present = {
    "genres": ~genres_e, "keywords": ~kw_e, "overview": ~ov_e,
    "cast": ~cast_e, "director": ~dir_e,
}
n_present = sum(v.astype(int) for v in present.values())
print("  Movies by count of non-empty feature groups (of 5):")
for k, v in n_present.value_counts().sort_index().items():
    print(f"    {k} group(s): {v:5d} ({100*v/len(movies):5.2f}%)")

print("\n  Which combination (out of 5 groups) is missing:")
combo = pd.DataFrame(present)
pattern = combo.apply(lambda r: "+".join([c for c in combo.columns if not r[c]]) or "none", axis=1)
for p, c in pattern.value_counts().items():
    print(f"    missing={p:<30} {c:5d}")

hr("B. SCORE CEILINGS IMPOSED BY MISSING BLOCKS")
for tag, w in CONFIGS.items():
    total = sum(w.values())
    print(f"\n  Config {tag} (sum of weights = {total:.2f}, full-profile norm = {np.sqrt(total):.4f})")
    print(f"    {'missing block':<22}{'movies':>8}{'row norm':>11}{'max sim vs full':>18}")
    for block in ["genres", "keywords", "overview", "cast", "director"]:
        if block not in w:
            continue
        m = (pattern == block)
        n = int(m.sum())
        if n == 0:
            continue
        nrm = np.sqrt(total - w[block])
        ceil = nrm / np.sqrt(total)
        print(f"    {block:<22}{n:>8}{nrm:>11.4f}{ceil:>18.4f}")
    mcombo = (pattern != "none") & (~pattern.isin(list(w.keys())))
    print(f"    movies with any group missing: {int((pattern != 'none').sum())}")

hr("C. HUB MECHANISM: WHO IS OVER-RECOMMENDED AND WHY")

L = combine_features(features, weights=CONFIGS["L"])
model = NearestNeighbors(metric="cosine", algorithm="brute").fit(L)
dist, ind = model.kneighbors(L, n_neighbors=11)

rank1 = ind[:, 1]
h1 = Counter(int(i) for i in rank1)
print("  Top-10 by number of times chosen as the #1 recommendation:")
for idx, c in h1.most_common(10):
    m = movies.iloc[idx]
    print(f"    {c:5d}  {m['title'][:44]:<44} id={m['id']:<7} "
          f"genres={len(m['genres'])} kw={len(m['keywords']):>2} "
          f"ov={len(str(m['overview'])):>4} cast={len(m['cast'])} dir={len(m['director'])}")

print("\n  How many movies have a #1 recommendation with ZERO keywords?",
      f"{int(kw_e.iloc[rank1].sum())} ({100*kw_e.iloc[rank1].mean():.2f}%)")
print("  Share of the catalogue that has zero keywords:",
      f"{int(kw_e.sum())} ({100*kw_e.mean():.2f}%)")
print("  How many movies have a #1 recommendation with >=3 empty groups?",
      f"{int((n_present.iloc[rank1] <= 2).sum())}")

hubs = [i for i, _ in Counter(int(i) for i in ind[:, 1:].ravel()).most_common(10)]
print("\n  Mean keyword count: top-10 hubs = "
      f"{movies.iloc[hubs]['keywords'].apply(len).mean():.1f} vs catalogue = "
      f"{movies['keywords'].apply(len).mean():.2f}")
print("  Hub genre-count mean:", round(movies.iloc[hubs]["genres"].apply(len).mean(), 2),
      "| hub overview length mean:", int(movies.iloc[hubs]["overview"].fillna("").str.len().mean()))
print("  Hub rows with zero keywords:", sum(kw_e.iloc[hubs]), "/ 10")

print("\n  Full profile of the #1 hub:")
h = movies.iloc[h1.most_common(1)[0][0]]
print(f"    id={h['id']} title={h['title']}")
print(f"    genres={h['genres']} cast={h['cast']} director={h['director']}")
print(f"    overview='{h['overview']}'")
print(f"    keywords={h['keywords']}")

hr("D. NEVER-RECOMMENDED MOVIES")
seen = set(int(i) for i in ind[:, 1:].ravel())
never = [i for i in range(len(movies)) if i not in seen]
print(f"  {len(never)} movies never appear in any top-10 list.")
sub = movies.iloc[never][["id", "title"]].copy()
sub["genres"] = movies.iloc[never]["genres"].apply(len)
sub["kw"] = movies.iloc[never]["keywords"].apply(len)
sub["ov"] = movies.iloc[never]["overview"].fillna("").str.len()
print(f"  their mean keyword count: {sub['kw'].mean():.2f} "
      f"| mean overview chars: {int(sub['ov'].mean())} "
      f"| mean genres: {sub['genres'].mean():.2f}")
print("  (catalogue means: keywords 7.54, overview 305, genres 2.53)")
print(sub.sort_values("kw", ascending=False).head(10).to_string(index=False))

hr("E. KEYWORD TOKENISATION DAMAGE")
kw_text = movies["keywords"].apply(lambda items: " ".join(items))
tv = TfidfVectorizer(stop_words="english")
mat = tv.fit_transform(kw_text)
vocab = {v: k for k, v in tv.vocabulary_.items()}
df = np.asarray((mat != 0).sum(axis=0)).ravel()
order = np.argsort(df)[::-1]
print("  Top 15 keyword tokens by document frequency:")
for i in order[:15]:
    print(f"    {vocab[i]:<28} appears in {df[i]:5d} movies "
          f"({100*df[i]/len(movies):5.2f}%)")

all_kw = {k for items in movies["keywords"] for k in items}
spaced = {k for k in all_kw if " " in k}
tokens_of_spaced = {t for k in spaced for t in k.lower().split()}
standalone = {k.lower() for k in all_kw if " " not in k}
frag = tokens_of_spaced - standalone
print(f"\n  distinct keyword strings      : {len(all_kw)}")
print(f"  of which multi-word phrases   : {len(spaced)}")
print(f"  word-fragments created by splitting phrases that are NOT "
      f"standalone keywords: {len(frag)}")
print(f"  e.g. {sorted(frag)[:12]}")

hr("F. WHICH MOVIE LOSES ITS ENTIRE OVERVIEW TO max_features=5000?")
full = TfidfVectorizer(stop_words="english").fit_transform(movies["overview"].fillna(""))
pruned = TfidfVectorizer(stop_words="english", max_features=5000).fit_transform(
    movies["overview"].fillna(""))
a = np.asarray((full != 0).sum(axis=1)).ravel() > 0
b = np.asarray((pruned != 0).sum(axis=1)).ravel() > 0
only_pruned_empty = np.where(a & ~b)[0]
print(f"  rows with 0 terms using full vocab : {int((~a).sum())}")
print(f"  rows with 0 terms after pruning    : {int((~b).sum())}")
for i in only_pruned_empty:
    m = movies.iloc[i]
    print(f"  id={m['id']} '{m['title']}' overview='{m['overview']}'")

hr("G. CAST / DIRECTOR SINGLETON WEIGHT")
for name in ["cast", "director"]:
    m = features[name]
    col = np.asarray((m != 0).sum(axis=0)).ravel()
    singletons = set(np.where(col == 1)[0])
    row_counts = np.asarray((m != 0).sum(axis=1)).ravel()
    share = []
    for i in range(m.shape[0]):
        if row_counts[i] == 0:
            continue
        idx = m[i].indices
        share.append(np.isin(idx, list(singletons)).sum() / row_counts[i])
    print(f"  {name:<9} singleton columns={len(singletons)}/{m.shape[1]} "
          f"({100*len(singletons)/m.shape[1]:.1f}%) | mean share of a movie's "
          f"{name} coordinates that are one-off = {100*np.mean(share):.1f}%")

hr("H. DUPLICATE TITLES: WHAT `matches[0]` ACTUALLY RESOLVES TO")
for t in ["Batman", "The Host", "Out of the Blue"]:
    matches = movies.index[movies["title"].str.lower() == t.lower()]
    print(f"\n  recommend('{t}') -> {len(matches)} matches, uses index {matches[0]}")
    for i in matches:
        m = movies.iloc[i]
        print(f"    index={i:<5} id={m['id']:<7} {m['title']:<22} "
              f"release={str(m['release_date'])[:10]:<11} "
              f"vote_count={int(m['vote_count']):<6} {m['genres']}")
    i = matches[0]
    d, ix = model.kneighbors(L[i], n_neighbors=4)
    recs = [(movies.iloc[int(j)]["title"][:34], round(1 - dd, 4))
            for j, dd in zip(ix[0], d[0]) if int(j) != i][:3]
    print("    its top-3: " + " | ".join(f"{a} ({b})" for a, b in recs))

hr("I. TOP-1 SIMILARITY BY CONFIG")
for tag in CONFIGS:
    M = L if tag == "L" else combine_features(features, weights=CONFIGS[tag])
    mod = NearestNeighbors(metric="cosine", algorithm="brute").fit(M)
    d, _ = mod.kneighbors(M, n_neighbors=11)
    s1 = 1 - d[:, 1]
    s10 = 1 - d[:, 10]
    print(f"  {tag}: top1 mean={s1.mean():.4f} median={np.median(s1):.4f} "
          f"min={s1.min():.4f} max={s1.max():.4f} | "
          f"mean sim@10={s10.mean():.4f} | "
          f"movies with top1<0.2: {int((s1 < 0.2).sum())}")

hr("J. SHORT / INCOMPLETE METADATA RECORDS")
shortish = movies.assign(ov=ov_len).query("ov > 0 and ov < 60")[["id", "title", "ov"]]
print(f"  overviews shorter than 60 chars: {len(shortish)}")
print(shortish.head(12).to_string(index=False))
print("\n  movies with runtime missing/zero:")
print(movies.loc[movies["runtime"].isna() | (movies["runtime"] <= 0), ["id", "title", "runtime"]]
      .head(10).to_string(index=False))
print("\n  movie with missing release_date:")
print(movies.loc[movies["release_date"].isna(), ["id", "title", "release_date"]]
      .to_string(index=False))

hr("K. DO THE 5 TEST MOVIES TOUCH ANY OF THESE PROBLEMS?")
for t in ["Avatar", "The Dark Knight", "Finding Nemo", "The Godfather", "Jurassic Park"]:
    i = movies.index[movies["title"].str.lower() == t.lower()][0]
    m = movies.iloc[i]
    d1, ix1 = model.kneighbors(L[i], n_neighbors=11)
    top10 = [int(j) for j in ix1[0] if int(j) != i][:10]
    hub_in_top10 = sum(1 for j in top10 if kw_e.iloc[j])
    print(f"  {t:<16} own metadata ok | of its top-10: {hub_in_top10} have zero keywords, "
          f"{sum(1 for j in top10 if len(movies.iloc[j]['keywords']) <= 1)} have <=1 keyword")
