import io
import re
import ast
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from difflib import get_close_matches

# --- ADDED: quick URL health checker for Secrets ---
def _head_status(url: str) -> str:
    import requests
    try:
        r = requests.head(url, allow_redirects=True, timeout=20)
        return f"{r.status_code} {r.headers.get('Content-Type', '')}"
    except Exception as e:
        return f"HEAD failed: {e}"
# --- END ADDED ---

st.set_page_config(page_title="Movie Recommender (TMDB) – Enhanced", layout="wide")

# -----------------------------
# Data Loading
# -----------------------------
def _fetch_csv(url: str) -> pd.DataFrame:
    """Robust HTTP fetch that works with Hugging Face 'resolve' links"""
    sess = requests.Session()
    sess.headers.update({"User-Agent": "streamlit-movie-recs/1.0"})
    trials = [url] + ([url + "?download=true"] if "?" not in url else [])
    last = None
    for u in trials:
        try:
            r = sess.get(u, timeout=60, allow_redirects=True)
            r.raise_for_status()
            try:
                text = r.content.decode("utf-8")
            except UnicodeDecodeError:
                text = r.content.decode("latin-1")
            df = pd.read_csv(io.StringIO(text))
            df.columns = [c.strip() for c in df.columns]
            return df
        except Exception as e:
            last = e
    raise RuntimeError(f"Failed to download CSV from {url} ({last})")

@st.cache_data(show_spinner=False)
def load_data():
    """Local files first; then Secrets fallback (Path B). Returns (df, source_str)."""
    # 1) Local files (original behavior)
    for p in ["movies.csv", "tmdb_5000_movies.csv"]:
        try:
            df = pd.read_csv(p)
            df.columns = [c.strip() for c in df.columns]
            return df, p
        except Exception:
            continue

    # 2) Secrets fallback (prefer 'movies', else 'tmdb_5000_movies') with diagnostics
    tried = []
    for key in ["movies", "tmdb_5000_movies"]:
        url = st.secrets.get(key, "")
        if isinstance(url, str) and url.startswith(("http://", "https://")):
            tried.append((key, url))
            try:
                df = _fetch_csv(url)
                src = f"secrets:{key}"
                return df, src
            except Exception as e:
                tried[-1] = (key, url, f"FETCH ERROR: {e}")

    # 3) Nothing worked: surface exactly what we tried
    def _fmt(t):
        k, u = t[0], t[1]
        hs = _head_status(u)
        extra = f" | {t[2]}" if len(t) == 3 else ""
        return f"{k} -> {hs}{extra}"

    detail = "; ".join(_fmt(t) for t in tried) if tried else "No valid URLs in Secrets."
    raise RuntimeError(
        "No data loaded. Either put CSVs in repo root or set Secrets.\n"
        "Checked keys: movies / tmdb_5000_movies.\n"
        f"Diagnostics: {detail}"
    )

# -----------------------------
# Helpers
# -----------------------------
def safe_title_column(df):
    for c in ["title", "original_title", "movie_title", "name"]:
        if c in df.columns:
            return c
    df = df.copy()
    df["title"] = [f"Movie #{i+1}" for i in range(len(df))]
    return "title"

def split_tokens(s):
    if pd.isna(s):
        return []
    s = str(s)
    if "[" in s and "]" in s and "name" in s:
        return re.findall(r"'name': '([^']+)'|\"name\": \"([^\"]+)\"", s)
    for sep in ["|", ",", ";", " / ", "/", " "]:
        if sep in s:
            out = [x for x in s.split(sep) if x]
            return out
    return [s]

def build_text_corpus(df):
    candidates = []
    for col in ["genres", "keywords", "tagline", "overview", "cast", "director"]:
        if col in df.columns:
            candidates.append(col)
    if not candidates:
        for alt in ["title", "original_title"]:
            if alt in df.columns:
                candidates.append(alt)
    fused = []
    for _, row in df.iterrows():
        parts = []
        for col in candidates:
            val = row.get(col, "")
            if pd.isna(val):
                val = ""
            parts.append(str(val))
        fused.append(" ".join(parts))
    return fused, candidates

@st.cache_resource(show_spinner=False)
def fit_vectorizer_and_knn(df, min_df=2):
    texts, used_cols = build_text_corpus(df)
    vectorizer = TfidfVectorizer(stop_words="english", min_df=min_df)
    X = vectorizer.fit_transform(texts)
    knn = NearestNeighbors(metric="cosine", algorithm="brute").fit(X)
    return X, vectorizer, knn, used_cols

def fuzzy_find_title(user_title, titles):
    if not isinstance(user_title, str) or not user_title.strip():
        return None
    matches = get_close_matches(user_title, titles, n=1, cutoff=0.5)
    return matches[0] if matches else None

def recommend_from_title(title, titles, X, knn, top_k=10, method="cosine", genre_filter=None, year_filter=None, df=None, title_col="title"):
    idx_series = pd.Series(range(len(titles)), index=titles)
    if title not in idx_series.index:
        return pd.DataFrame()
    i = idx_series[title]
    if method == "knn":
        distances, indices = knn.kneighbors(X[i], n_neighbors=min(top_k + 25, len(titles)))
        distances, indices = distances.flatten(), indices.flatten()
        mask = indices != i
        indices, distances = indices[mask], distances[mask]
        sims = 1 - distances
        recs = pd.DataFrame({"index": indices, "title": [titles[j] for j in indices], "similarity": sims})
    else:
        sims = cosine_similarity(X[i], X).flatten()
        order = np.argsort(-sims)
        order = order[order != i]
        recs = pd.DataFrame({"index": order, "title": [titles[j] for j in order], "similarity": sims[order]})
    # Optional filters
    if df is not None:
        tmp = df.iloc[recs["index"].values].copy()
        tmp["title"] = recs["title"].values
        tmp["similarity"] = recs["similarity"].values
        if genre_filter and "genres" in tmp.columns:
            tmp = tmp[tmp["genres"].astype(str).str.contains(genre_filter, case=False, na=False)]
        if year_filter:
            year_col = None
            for c in ["release_year", "year"]:
                if c in tmp.columns:
                    year_col = c
            if "release_date" in tmp.columns and year_col is None:
                tmp["year_parsed"] = tmp["release_date"].astype(str).str[:4].str.extract(r"(\\d{4})")
                year_col = "year_parsed"
            if year_col:
                tmp = tmp[tmp[year_col].astype(str) == str(year_filter)]
        recs = tmp
    return recs.head(top_k)[["title", "similarity"] + [c for c in ["genres","release_date","vote_average"] if c in recs.columns]]

def get_unique_years(df):
    years = set()
    if "release_year" in df.columns:
        years.update(df["release_year"].dropna().astype(int).tolist())
    elif "release_date" in df.columns:
        years.update(pd.to_datetime(df["release_date"], errors="coerce").dt.year.dropna().astype(int).tolist())
    return sorted([y for y in years if 1800 < y < 2100])

# -----------------------------
# Load Data
# -----------------------------
df, source = load_data()

# --- ADDED: visibility into secrets/URLs and the loaded DF ---
with st.expander("🔎 Data diagnostics"):
    st.write("Source:", source)
    for key in ["movies", "tmdb_5000_movies", "tmdb_5000_credits"]:
        val = st.secrets.get(key, "")
        st.write(f"Secret `{key}` present:", bool(val))
        if isinstance(val, str) and val.startswith(("http://", "https://")):
            st.write(f"HEAD {key} → {_head_status(val)}")
    st.write("Loaded shape:", df.shape)
    st.dataframe(df.head(5), use_container_width=True)
# --- END ADDED ---

st.title("🎬 Movie Recommendation System (Content-Based)")
if df is None:
    st.warning("Couldn't find data locally and no valid URLs in Secrets. Add:\n- `movies` OR `tmdb_5000_movies` to **Settings → Secrets** (Hugging Face 'resolve' URLs).")
    st.stop()

title_col = safe_title_column(df)

# Sidebar controls
with st.sidebar:
    st.header("Data & Vectorization")
    st.caption(f"Loaded: **{source}**  |  Rows: **{len(df)}**")
    show_cols = st.checkbox("Show raw columns", value=False)
    if show_cols:
        st.write(sorted(df.columns.tolist()))
    st.divider()
    min_df = st.slider("TF-IDF min_df", 1, 10, 2)
    X, vectorizer, knn, used_cols = fit_vectorizer_and_knn(df, min_df=min_df)

# Prepare lists
titles = df[title_col].fillna("").astype(str).tolist()
all_genres = []
if "genres" in df.columns:
    for g in df["genres"].tolist():
        toks = [t for t in split_tokens(g) if t]
        all_genres.extend(toks)
genre_counts = Counter(all_genres)
genre_options = ["(All)"] + [g for g, _ in genre_counts.most_common(50)]
year_options = ["(All)"] + [str(y) for y in get_unique_years(df)]

# Tabs
tab1, tab2, tab3 = st.tabs(["🔎 Recommender", "📊 EDA", "🧪 Model Performance"])

# -----------------------------
# Tab 1: Recommender
# -----------------------------
with tab1:
    st.subheader("Find similar movies")
    c1, c2, c3, c4 = st.columns([3,1.2,1.2,1.2])
    with c1:
        title_input = st.selectbox("Type or select a movie title", options=[""] + titles, index=0)
    with c2:
        top_k = st.number_input("Top-K", min_value=1, max_value=25, value=10, step=1)
    with c3:
        method = st.radio("Method", ["cosine", "knn"], horizontal=True, index=0)
    with c4:
        genre_pick = st.selectbox("Genre filter", options=genre_options, index=0)

    year_pick = st.selectbox("Year filter", options=year_options, index=0, key="yearpick_tab1")

    chosen = None
    if title_input:
        chosen = title_input
    else:
        typed = st.text_input("...or enter title for fuzzy match", "")
        if typed:
            chosen = fuzzy_find_title(typed, titles)

    if chosen:
        if genre_pick == "(All)":
            genre_pick_val = None
        else:
            genre_pick_val = genre_pick
        year_val = None if year_pick == "(All)" else year_pick
        recs = recommend_from_title(
            chosen, titles, X, knn, top_k=top_k, method=method,
            genre_filter=genre_pick_val, year_filter=year_val, df=df, title_col=title_col
        )
        st.info(f"Closest match: **{chosen}**")
        st.dataframe(recs, use_container_width=True)
    else:
        st.warning("Select or enter a movie title to see recommendations.")

# -----------------------------
# Tab 2: EDA
# -----------------------------
with tab2:
    st.subheader("Exploratory Data Analysis")
    st.write(f"**Text features used**: {', '.join(used_cols) if used_cols else '(auto)'}")
    st.write(f"**TF-IDF shape**: {X.shape[0]} docs × {X.shape[1]} terms")
    nnz = X.nnz if hasattr(X, 'nnz') else np.count_nonzero(X)
    sparsity = 1.0 - (nnz / (X.shape[0] * X.shape[1]))
    st.metric("Sparsity", f"{sparsity*100:.2f}%")

    n_sample = min(300, X.shape[0])
    idxs = np.random.default_rng(42).choice(X.shape[0], size=n_sample, replace=False)
    sims_sample = cosine_similarity(X[idxs], X[idxs])
    tri = sims_sample[np.triu_indices_from(sims_sample, k=1)]
    st.write(f"Sample pairwise cosine similarity — mean **{tri.mean():.3f}**, std **{tri.std():.3f}**, min **{tri.min():.3f}**, max **{tri.max():.3f}**")

    colA, colB = st.columns(2)
    with colA:
        st.caption("Distribution: Sample Pairwise Similarity")
        fig, ax = plt.subplots()
        ax.hist(tri, bins=30)
        ax.set_xlabel("Cosine similarity")
        ax.set_ylabel("Count")
        st.pyplot(fig, clear_figure=True)

    with colB:
        if "genres" in df.columns and len(genre_counts) > 0:
            st.caption("Top Genres")
            names, counts = zip(*genre_counts.most_common(15))
            fig2, ax2 = plt.subplots()
            ax2.bar(range(len(names)), counts)
            ax2.set_xticks(range(len(names)))
            ax2.set_xticklabels(names, rotation=45, ha="right")
            ax2.set_ylabel("Count")
            st.pyplot(fig2, clear_figure=True)
        else:
            st.info("No `genres` column found for genre distribution plot.")

    if "popularity" in df.columns and "vote_average" in df.columns:
        st.caption("Popularity vs Vote Average")
        fig3, ax3 = plt.subplots()
        ax3.scatter(df["popularity"], df["vote_average"], s=5)
        ax3.set_xlabel("Popularity")
        ax3.set_ylabel("Vote Average")
        st.pyplot(fig3, clear_figure=True)

# -----------------------------
# Tab 3: Model Performance
# -----------------------------
with tab3:
    st.subheader("Baseline Rating Prediction (Linear Regression)")
    candidate_feats = [c for c in ["popularity","budget","revenue","runtime","vote_count"] if c in df.columns]
    target_col = "vote_average" if "vote_average" in df.columns else None

    if target_col and len(candidate_feats) >= 1:
        Xnum = df[candidate_feats].fillna(0.0).astype(float)
        y = df[target_col].fillna(df[target_col].mean())
        X_train, X_test, y_train, y_test = train_test_split(Xnum, y, test_size=0.2, random_state=42)

        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = mse ** 0.5

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("R²", f"{r2:.3f}")
        c2.metric("MAE", f"{mae:.3f}")
        c3.metric("MSE", f"{mse:.3f}")
        c4.metric("RMSE", f"{rmse:.3f}")

        fig4, ax4 = plt.subplots()
        ax4.scatter(y_pred, y_test - y_pred, s=8)
        ax4.axhline(0, linestyle="--")
        ax4.set_xlabel("Predicted")
        ax4.set_ylabel("Residuals")
        st.pyplot(fig4, clear_figure=True)
    else:
        st.info("Not enough numeric columns to train a baseline regressor (need target `vote_average` and at least one of: popularity, budget, revenue, runtime, vote_count).")

st.caption("Data source: TMDB (for educational use). This app adapts to whichever columns are available in your dataset.")
