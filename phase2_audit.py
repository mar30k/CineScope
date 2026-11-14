"""
Phase 2 - Data & feature quality audit for CineScope.

Read-only: loads, measures and reports. Does not modify source, data or the
recommendation algorithm, and does not drop/filter any movie.
"""

import ast
import hashlib
import sys
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from src.data.processor import load_datasets, process_movies
from src.features.feature_builder import build_features, combine_features

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

pd.set_option("display.width", 160)
pd.set_option("display.max_colwidth", 60)


def hr(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def pct(a, b):
    return f"{100.0 * a / b:6.2f}%" if b else "   n/a"


def row_norms(mat):
    sq = mat.multiply(mat).sum(axis=1)
    return np.sqrt(np.asarray(sq).ravel())


def row_hash(mat):
    out = []
    for i in range(mat.shape[0]):
        r = mat[i]
        h = hashlib.md5(r.indices.astype(np.int64).tobytes())
        h.update(np.round(r.data, 8).astype(np.float64).tobytes())
        out.append(h.hexdigest())
    return out


# ---------------------------------------------------------------- raw data
movies_raw, credits_raw = load_datasets()

hr("0. ENVIRONMENT")
print("pandas", pd.__version__, "| numpy", np.__version__)
import scipy
import sklearn
print("scipy", scipy.__version__, "| sklearn", sklearn.__version__)

hr("1. RAW DATASET INTEGRITY")
print(f"tmdb_5000_movies : {movies_raw.shape[0]} rows x {movies_raw.shape[1]} cols")
print(f"tmdb_5000_credits: {credits_raw.shape[0]} rows x {credits_raw.shape[1]} cols")

print("\n-- Missing values in raw movie columns --")
print(f"  {'column':<24}{'missing':>9}{'percent':>10}")
cands = [
    "id", "title", "genres", "keywords", "overview", "runtime", "release_date",
    "vote_average", "vote_count", "budget", "revenue", "tagline", "homepage",
    "production_companies", "production_countries", "spoken_languages", "status",
]
miss = [(c, int(movies_raw[c].isna().sum())) for c in cands if c in movies_raw.columns]
for c, n in sorted(miss, key=lambda t: -t[1]):
    print(f"  {c:<24}{n:>9}{pct(n, len(movies_raw)):>10}")

print("\n-- Missing values in raw credit columns --")
for c in credits_raw.columns:
    n = int(credits_raw[c].isna().sum())
    print(f"  {c:<24}{n:>9}{pct(n, len(credits_raw)):>10}")

print("\n-- Identifier integrity --")
print(f"  duplicate movie ids in movies.csv : {int(movies_raw['id'].duplicated().sum())}")
print(f"  duplicate movie_id in credits.csv : {int(credits_raw['movie_id'].duplicated().sum())}")
print(f"  duplicate exact titles            : {int(movies_raw['title'].duplicated().sum())}")
print(f"  movies not present in credits     : "
      f"{int((~movies_raw['id'].isin(credits_raw['movie_id'])).sum())}")
print(f"  credits not present in movies     : "
      f"{int((~credits_raw['movie_id'].isin(movies_raw['id'])).sum())}")

hm = movies_raw["title"].str.strip().str.lower()
dups = movies_raw.loc[hm.duplicated(keep=False), ["id", "title"]].copy()
print("\n-- Duplicate title groups (case-insensitive, whitespace-stripped) --")
if len(dups):
    for key, grp in dups.groupby(hm[dups.index]):
        print(f"  '{key}' -> " + ", ".join(f"id={r.id}" for r in grp.itertuples()))
else:
    print("  none")

n_ws = int((movies_raw["title"].str.strip() != movies_raw["title"]).sum())
print(f"\n  titles with leading/trailing whitespace: {n_ws}")

# ------------------------------------------------------- JSON parse health
hr("2. JSON COLUMN PARSING HEALTH")


def probe_parse(series, label):
    nan = empty = ok = fail = nonlist = 0
    for v in series:
        if pd.isna(v):
            nan += 1
            continue
        if not isinstance(v, str):
            nonlist += 1
            continue
        s = v.strip()
        if s == "[]":
            empty += 1
            continue
        if s == "":
            fail += 1
            continue
        try:
            val = ast.literal_eval(s)
        except (ValueError, SyntaxError):
            fail += 1
            continue
        if not isinstance(val, list):
            nonlist += 1
        elif len(val) == 0:
            empty += 1
        else:
            ok += 1
    print(f"  {label:<10} NaN={nan:<5} empty='[]'={empty:<5} parsed-nonempty={ok:<5} "
          f"parse-failure={fail:<3} non-list={nonlist}")


for col in ["genres", "keywords"]:
    probe_parse(movies_raw[col], col)
for col in ["cast", "crew"]:
    probe_parse(credits_raw[col], col)


def parsed_list(value):
    if not isinstance(value, str):
        return []
    try:
        val = ast.literal_eval(value.strip())
        return val if isinstance(val, list) else []
    except (ValueError, SyntaxError):
        return []


print("\n-- Dict key integrity inside parsed JSON arrays --")
for col in ["genres", "keywords"]:
    no_name = sum(1 for v in movies_raw[col] for d in parsed_list(v)
                  if not isinstance(d, dict) or "name" not in d)
    print(f"  movies.{col}: entries without a 'name' key = {no_name}")

cast_no_name = sum(1 for v in credits_raw["cast"] for d in parsed_list(v)
                   if not isinstance(d, dict) or "name" not in d)
crew_no_job = sum(1 for v in credits_raw["crew"] for d in parsed_list(v)
                  if not isinstance(d, dict) or "job" not in d)
print(f"  credits.cast: entries without a 'name' key = {cast_no_name}")
print(f"  credits.crew: entries without a 'job' key   = {crew_no_job}")

# ---------------------------------------------------- cast / crew extraction
hr("3. CAST TRUNCATION AND DIRECTOR EXTRACTION")

cast_lists = credits_raw["cast"].apply(parsed_list)
lens = cast_lists.apply(len)
print("-- Raw cast list length (pre-truncation) --")
print(f"  total credited cast entries : {int(lens.sum())}")
print(f"  mean / median / max         : {lens.mean():.2f} / {lens.median():.0f} / {lens.max()}")
print(f"  movies with exactly 0 cast  : {int((lens == 0).sum())}")
print(f"  movies with < 5 cast        : {int((lens < 5).sum())}")
print(f"  movies with > 5 cast        : {int((lens > 5).sum())}  "
      f"({pct(int((lens > 5).sum()), len(lens))})")
print(f"  cast entries dropped by [:5]: {int((lens - 5).clip(lower=0).sum())}"
      f"  ({pct(int((lens - 5).clip(lower=0).sum()), int(lens.sum()))} of all entries)")
print(f"  length quantiles 50/75/90/99 : "
      + " / ".join(str(int(lens.quantile(q))) for q in (0.5, 0.75, 0.9, 0.99)))

all_actors = {d.get("name") for v in credits_raw["cast"] for d in parsed_list(v)
              if isinstance(d, dict)}
top5_actors = set()
for v in cast_lists:
    top5_actors.update(d.get("name") for d in v[:5] if isinstance(d, dict))
print(f"  distinct actors in full cast : {len(all_actors)}")
print(f"  distinct actors in top-5     : {len(top5_actors)}")
print(f"  distinct actors lost entirely: {len(all_actors - top5_actors)}")

directors = credits_raw["crew"].apply(
    lambda v: [d.get("name") for d in parsed_list(v) if d.get("job") == "Director"]
)
dl = directors.apply(len)
print("\n-- Director extraction from crew --")
print(f"  movies with 0 directors  : {int((dl == 0).sum())}")
print(f"  movies with 1 director   : {int((dl == 1).sum())}")
print(f"  movies with >1 directors : {int((dl > 1).sum())} (co-directors)")
if (dl > 1).any():
    ex = credits_raw.loc[dl > 1, ["movie_id"]].head(5)
    for mid in ex["movie_id"]:
        names = directors[credits_raw["movie_id"] == mid].iloc[0]
        print(f"    id={mid}: {names}")

# ------------------------------------------------------- processed dataset
hr("4. PROCESSED DATASET")
movies = process_movies()
print(f"  shape: {movies.shape}")
print(f"  index is default RangeIndex: "
      f"{movies.index.equals(pd.RangeIndex(len(movies)))}")
print(f"  duplicate ids after merge  : {int(movies['id'].duplicated().sum())}")
print(f"  duplicate titles after merge: {int(movies['title'].duplicated().sum())}")

genres_e = movies["genres"].apply(len) == 0
kw_e = movies["keywords"].apply(len) == 0
ov_txt = movies["overview"].fillna("").str.strip()
ov_e = ov_txt.str.len() == 0
cast_e = movies["cast"].apply(len) == 0
dir_e = movies["director"].apply(len) == 0

print("\n-- Empty / missing feature groups (count and % of 4803) --")
print(f"  {'group':<12}{'empty':>8}{'percent':>10}{'mean size':>12}{'max size':>10}")
for name, flag, size in [
    ("genres", genres_e, movies["genres"].apply(len)),
    ("keywords", kw_e, movies["keywords"].apply(len)),
    ("overview", ov_e, ov_txt.str.len()),
    ("cast", cast_e, movies["cast"].apply(len)),
    ("director", dir_e, movies["director"].apply(len)),
]:
    print(f"  {name:<12}{int(flag.sum()):>8}{pct(int(flag.sum()), len(movies)):>10}"
          f"{size.mean():>12.2f}{size.max():>10}")

print("\n-- Feature-group size distribution --")
for name, size in [("genres", movies["genres"].apply(len)),
                   ("keywords", movies["keywords"].apply(len)),
                   ("cast", movies["cast"].apply(len)),
                   ("director", movies["director"].apply(len))]:
    vc = size.value_counts().sort_index()
    head = ", ".join(f"{k}:{v}" for k, v in vc.head(6).items())
    print(f"  {name:<9} {head}")

empty_flags = pd.DataFrame({
    "genres": genres_e, "keywords": kw_e, "overview": ov_e,
    "cast": cast_e, "director": dir_e,
})
n_empty = empty_flags.sum(axis=1)
print("\n-- Number of EMPTY feature groups per movie --")
for k, v in n_empty.value_counts().sort_index().items():
    print(f"  {k} empty group(s): {v:5d}  ({pct(v, len(movies))})")

print("\n-- Movies with >= 3 empty feature groups --")
bad = movies.loc[n_empty >= 3, ["id", "title", "vote_count"]].copy()
bad["n_empty"] = n_empty[n_empty >= 3]
print(bad.sort_values(["n_empty", "vote_count"], ascending=[False, False])
      .head(25).to_string(index=False))
print(f"  total: {len(bad)}")

print("\n-- Movies with an empty overview --")
print(movies.loc[ov_e, ["id", "title"]].head(20).to_string(index=False))
print(f"  total: {int(ov_e.sum())}")

print("\n-- Movies with an empty cast --")
print(movies.loc[cast_e, ["id", "title"]].head(20).to_string(index=False))
print(f"  total: {int(cast_e.sum())}")

print("\n-- Movies with an empty director --")
print(movies.loc[dir_e, ["id", "title"]].head(20).to_string(index=False))
print(f"  total: {int(dir_e.sum())}")

print("\n-- Movies with an empty keywords list --")
print(movies.loc[kw_e, ["id", "title"]].head(10).to_string(index=False))
print(f"  total: {int(kw_e.sum())}")

print("\n-- Overview length distribution (characters) --")
print("  " + "  ".join(f"{q}%={int(ov_txt.str.len().quantile(q/100))}"
                       for q in (0, 1, 5, 25, 50, 75, 95, 100)))
short = ov_txt.str.len()
print(f"  overviews < 20 chars : {int(((short < 20) & ~ov_e).sum())}")
print(f"  overviews < 50 chars : {int(((short < 50) & ~ov_e).sum())}")
print("\n  5 shortest non-empty overviews:")
srt = movies.loc[~ov_e, ["id", "title"]].copy()
srt["chars"] = short[~ov_e]
print(srt.nsmallest(5, "chars").to_string(index=False))

print("\n-- Non-text metadata completeness (raw columns, kept by processor) --")
for col, test, label in [
    ("runtime", lambda s: s.isna() | (s <= 0), "runtime missing/zero"),
    ("release_date", lambda s: s.isna(), "release_date missing"),
    ("vote_count", lambda s: s.isna() | (s == 0), "vote_count missing/zero"),
    ("budget", lambda s: s.isna() | (s == 0), "budget missing/zero"),
    ("revenue", lambda s: s.isna() | (s == 0), "revenue missing/zero"),
    ("tagline", lambda s: s.isna() | (s.astype(str).str.strip() == ""), "tagline empty"),
    ("homepage", lambda s: s.isna(), "homepage missing"),
]:
    if col in movies.columns:
        m = test(movies[col])
        print(f"  {label:<24}{int(m.sum()):>6}{pct(int(m.sum()), len(movies)):>10}")

# ------------------------------------------------------- feature matrices
hr("5. FEATURE MATRIX INTEGRITY")
features = build_features(movies)

print(f"  {'group':<10}{'shape':>16}{'nnz':>9}{'density':>11}{'empty rows':>12}"
      f"{'coverage':>11}{'mean nnz/row':>14}")
for name, mat in features.items():
    n_rows, n_cols = mat.shape
    nnz = mat.nnz
    dense = n_rows * n_cols
    per_row = np.asarray((mat != 0).sum(axis=1)).ravel()
    empty_rows = int((per_row == 0).sum())
    print(f"  {name:<10}{str(mat.shape):>16}{nnz:>9}{pct(nnz, dense):>11}"
          f"{empty_rows:>12}{pct(n_rows - empty_rows, n_rows):>11}"
          f"{per_row.mean():>14.2f}")

print("\n-- Value sanity per matrix (data array only) --")
print(f"  {'group':<10}{'dtype':>8}{'min':>12}{'max':>12}{'NaN':>6}{'inf':>5}"
      f"{'neg':>5}{'uniq vals':>11}")
for name, mat in features.items():
    d = mat.data
    print(f"  {name:<10}{str(mat.dtype):>8}{d.min():>12.6f}{d.max():>12.6f}"
          f"{int(np.isnan(d).sum()):>6}{int(np.isinf(d).sum()):>5}"
          f"{int((d < 0).sum()):>5}{len(np.unique(np.round(d, 10))):>11}")

print("\n-- Sparse-structure checks --")
for name, mat in features.items():
    print(f"  {name:<10} sorted_indices={mat.has_sorted_indices} "
          f"canonical={mat.has_canonical_format} "
          f"explicit_zeros={int((mat.data == 0).sum())}")

print("\n-- Vocabulary / column statistics --")
for name in ["genres", "cast", "director"]:
    mat = features[name]
    col_nnz = np.asarray((mat != 0).sum(axis=0)).ravel()
    singles = int((col_nnz == 1).sum())
    print(f"  {name:<10} columns={mat.shape[1]:<7} singletons(1 movie)={singles:<7}"
          f"({pct(singles, mat.shape[1])}) unused={int((col_nnz == 0).sum())}")

for name in ["keywords", "overview"]:
    mat = features[name]
    col_nnz = np.asarray((mat != 0).sum(axis=0)).ravel()
    singles = int((col_nnz == 1).sum())
    print(f"  {name:<10} columns={mat.shape[1]:<7} singletons(1 movie)={singles:<7}"
          f"({pct(singles, mat.shape[1])}) unused={int((col_nnz == 0).sum())}")

print("\n-- MultiLabelBinarizer vocabulary sizes --")
from sklearn.preprocessing import MultiLabelBinarizer
for name, col in [("genres", "genres"), ("cast", "cast"), ("director", "director")]:
    mb = MultiLabelBinarizer()
    mb.fit(movies[col])
    print(f"  {name:<10} classes={len(mb.classes_)}")

print("\n-- Genres vocabulary (complete) --")
mbg = MultiLabelBinarizer()
mbg.fit(movies["genres"])
print("  " + ", ".join(sorted(mbg.classes_)))

print("\n-- Text vectorizer diagnostics --")
from sklearn.feature_extraction.text import TfidfVectorizer

kw_text = movies["keywords"].apply(lambda items: " ".join(items))
uniq_kw = {k for items in movies["keywords"] for k in items}
print(f"  keywords: distinct keyword strings={len(uniq_kw)} "
      f"-> tfidf vocab={features['keywords'].shape[1]}")

# keywords that the default token_pattern (>=2 chars) would split or drop
multi = [k for k in uniq_kw if " " in k]
onechar = [k for k in uniq_kw if len(k) == 1]
print(f"  keywords containing spaces (split into parts): {len(multi)}")
print(f"  examples: {multi[:5]}")
print(f"  single-character keywords (dropped by token_pattern): {len(onechar)} {onechar}")

concatenated = [k for k in uniq_kw if "stinger" in k.lower() or
                (len(k) > 14 and len(k.split()) == 1)]
print(f"  concatenated/unspaced keywords (corpus artefacts): {len(concatenated)}")
print(f"  examples: {sorted(concatenated)[:8]}")

tv_full = TfidfVectorizer(stop_words="english")
full_ov = tv_full.fit_transform(movies["overview"].fillna(""))
print(f"\n  overview: full vocab without max_features={full_ov.shape[1]} "
      f"vs retained=5000 ({pct(5000, full_ov.shape[1])})")
ov_terms = np.asarray((full_ov != 0).sum(axis=1)).ravel()
print(f"  overview docs with 0 terms after stopword removal: {int((ov_terms == 0).sum())}")
print(f"  mean distinct terms per overview: {ov_terms.mean():.2f}")

# keyword columns that appear in exactly one movie are non-discriminative
kw_nnz = features["keywords"]
col_df = np.asarray((kw_nnz != 0).sum(axis=0)).ravel()
top_idx = np.argsort(col_df)[::-1][:10]
inv_kw = {v: k for k, v in tv_full.vocabulary_.items()}

# ------------------------------------------------------- combined matrices
hr("6. COMBINED MATRICES FOR EXPERIMENTS J / L / N")

CONFIGS = {
    "J": {"genres": 1.0, "keywords": 1.0, "overview": 1.0},
    "L": {"genres": 1.0, "keywords": 1.0, "overview": 1.0, "cast": 0.5, "director": 0.25},
    "N": {"genres": 1.0, "keywords": 1.0, "overview": 1.0, "cast": 0.5, "director": 1.0},
}

combined = {}
for tag, w in CONFIGS.items():
    mat = combine_features(features, weights=w)
    combined[tag] = mat
    norms = row_norms(mat)
    zero_rows = int((norms == 0).sum())
    print(f"\n  Config {tag}: shape={mat.shape} nnz={mat.nnz} "
          f"density={pct(mat.nnz, mat.shape[0]*mat.shape[1])}")
    print(f"    rows with zero norm (no usable features): {zero_rows} "
          f"({pct(zero_rows, mat.shape[0])})")
    print(f"    NaN={int(np.isnan(mat.data).sum())} inf={int(np.isinf(mat.data).sum())} "
          f"neg={int((mat.data < 0).sum())} max={mat.data.max():.6f}")
    nz = np.round(norms[norms > 0], 6)
    vc = Counter(nz)
    print("    row-norm distribution (non-zero rows): "
          + ", ".join(f"{v:.4f}x{c}" for v, c in sorted(vc.items())[:6]))

print("\n-- Movies with a zero-length combined vector, per config --")
for tag in CONFIGS:
    z = row_norms(combined[tag]) == 0
    print(f"\n  Config {tag}: {int(z.sum())} movie(s)")
    if z.any():
        cols = ["id", "title"]
        print(movies.loc[z, cols].to_string(index=False))

# ------------------------------------------------------- neighbourhood audit
hr("7. NEIGHBOURHOOD / SIMILARITY AUDIT (config L = production setup)")

L = combined["L"]
model = NearestNeighbors(metric="cosine", algorithm="brute").fit(L)
dist, ind = model.kneighbors(L, n_neighbors=11)

top1 = 1.0 - dist[:, 1]
print("-- Top-1 neighbour similarity over all 4803 movies --")
print("  " + "  ".join(f"{q}%={np.quantile(top1, q/100):.4f}"
                       for q in (0, 1, 5, 25, 50, 75, 95, 100)))
for thr in (0.95, 0.8, 0.5, 0.3, 0.2, 0.1, 0.05, 0.0):
    n = int((top1 <= thr).sum())
    print(f"  top-1 similarity <= {thr:<5}: {n:5d} ({pct(n, len(top1))})")

print("\n-- Exact duplicate row vectors in combined L --")
hashes = row_hash(L)
groups = {}
for i, h in enumerate(hashes):
    groups.setdefault(h, []).append(i)
dup_groups = {h: g for h, g in groups.items() if len(g) > 1}
zero_norm_rows = set(np.where(row_norms(L) == 0)[0])
real_dups = {h: g for h, g in dup_groups.items() if not set(g) <= zero_norm_rows}
print(f"  duplicate vector groups: {len(dup_groups)} "
      f"(of which non-trivial/all-zero-mixed: {len(real_dups)})")
for h, g in list(dup_groups.items())[:10]:
    titles = ", ".join(f"{movies.iloc[i]['title'][:28]} (id={movies.iloc[i]['id']})"
                       for i in g[:4])
    tag = "ALL-ZERO" if set(g) <= zero_norm_rows else "IDENTICAL"
    print(f"    [{tag}] {len(g)} movies: {titles}")

print("\n-- Hub analysis: how often a movie appears in another movie's top-10 --")
hub = Counter()
for row in ind[:, 1:]:
    hub.update(int(i) for i in row)
print(f"  distinct movies appearing in some top-10: {len(hub)} / {len(movies)}")
print(f"  movies never recommended to anyone      : {len(movies) - len(hub)}")
print("  top 10 hubs:")
for idx, c in hub.most_common(10):
    m = movies.iloc[idx]
    print(f"    {c:5d} lists  {m['title'][:50]:<50} (id={m['id']}) "
          f"genres={len(m['genres'])} kw={len(m['keywords'])} ov={len(str(m['overview']))}")

print("\n-- Demo: query whose own vector is all zeros (config L) --")
if zero_norm_rows:
    zi = sorted(zero_norm_rows)[0]
    d, i = model.kneighbors(L[zi], n_neighbors=11)
    print(f"  query = {movies.iloc[zi]['title']} (id={movies.iloc[zi]['id']})")
    print(f"  distances returned: {np.round(d[0], 6).tolist()}")
    print(f"  indices returned  : {i[0].tolist()}  (i.e. positional order 0..10)")
else:
    print("  none")

hr("8. TEST-MOVIE PROFILES ACROSS J / L / N")
TEST = ["Avatar", "The Dark Knight", "Finding Nemo", "The Godfather", "Jurassic Park"]
for title in TEST:
    m = movies.index[movies["title"].str.lower() == title.lower()]
    if len(m) == 0:
        print(f"\n  {title}: NOT FOUND")
        continue
    i = m[0]
    row = movies.iloc[i]
    print(f"\n  {title} (id={row['id']}, index={i}, n_matches={len(m)})")
    print(f"    genres={len(row['genres'])} keywords={len(row['keywords'])} "
          f"overview_chars={len(str(row['overview']))} cast={len(row['cast'])} "
          f"director={row['director']}")
    for tag in CONFIGS:
        mod = NearestNeighbors(metric="cosine", algorithm="brute").fit(combined[tag])
        d, ix = mod.kneighbors(combined[tag][i], n_neighbors=4)
        pairs = [(movies.iloc[int(j)]["title"][:34], round(1 - dd, 4))
                 for j, dd in zip(ix[0], d[0]) if int(j) != i][:3]
        print(f"    {tag}: " + " | ".join(f"{t} ({s})" for t, s in pairs))

hr("9. END-TO-END SPOT CHECKS")
print("-- Do movies with empty cast/director exist in the 3 duplicate-title groups? --")
for key, grp in dups.groupby(hm[dups.index]):
    idxs = [movies.index[movies["id"] == r][0] for r in grp["id"]]
    for i in idxs:
        r = movies.loc[i]
        print(f"  '{key}' id={r['id']} cast={len(r['cast'])} dir={len(r['director'])} "
              f"genres={r['genres']}")

print("\n-- Titles that would fail exact lookup due to whitespace/case --")
print(f"  whitespace-padded: {n_ws}")
print(f"  case-insensitive duplicates: {len(dups)} records in {dups.size and len(set(hm[dups.index]))} groups")

print("\nAudit complete.")
