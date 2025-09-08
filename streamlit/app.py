import io
import ast
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import requests
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


st.set_page_config(page_title="🎬 Movie Recommendation System (Content-Based)", page_icon="🎬", layout="wide")

# =========================
# Helpers: fetching & parsing
# =========================

def _safe_eval_list(x):
    try:
        v = ast.literal_eval(x) if isinstance(x, str) else x
        if isinstance(v, list):
            return v
    except Exception:
        pass
    return []

def _names_from_list(lst, key="name", n=None):
    out = []
    for item in lst:
        if isinstance(item, dict) and key in item and item[key]:
            out.append(str(item[key]))
            if n and len(out) >= n:
                break
        elif isinstance(item, str):
            out.append(item)
            if n and len(out) >= n:
                break
    return out

def _director_from_crew(lst):
    for item in lst:
        if isinstance(item, dict) and item.get("job") == "Director":
            return item.get("name","")
    return ""

def _fetch_csv(url: str) -> pd.DataFrame:
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
    raise RuntimeError(f"Failed to download: {last}")

@st.cache_data(show_spinner=True)
def load_data_from_secrets():
    keys = ["movies", "tmdb_5000_movies", "tmdb_5000_credits"]
    loaded = {}
    for k in keys:
        url = st.secrets.get(k, "")
        if isinstance(url, str) and url.startswith(("http://", "https://")):
            loaded[k] = _fetch_csv(url)
    if not loaded:
        raise RuntimeError("No datasets loaded from Secrets. Add 'movies' OR both 'tmdb_5000_movies' & 'tmdb_5000_credits'.")
    return loaded

# Build a master table with text 'soup' and helper columns (title, year, genres, popularity, vote_average)
@st.cache_data(show_spinner=True)
def build_master_table(dsets: dict) -> pd.DataFrame:
    if "movies" in dsets:
        df = dsets["movies"].copy()
        # Attempt to parse standard tmdb-like columns
        for col in ["genres", "keywords", "cast", "crew"]:
            if col in df.columns:
                df[col] = df[col].fillna("[]").apply(_safe_eval_list)
        if "genres" in df.columns:
            df["genres_names"] = df["genres"].apply(lambda xs: _names_from_list(xs, "name"))
        else:
            df["genres_names"] = [[] for _ in range(len(df))]
        if "cast" in df.columns:
            df["cast_names"] = df["cast"].apply(lambda xs: _names_from_list(xs, "name", n=5))
        else:
            df["cast_names"] = [[] for _ in range(len(df))]
        if "crew" in df.columns:
            df["director"] = df["crew"].apply(_director_from_crew)
        else:
            df["director"] = ""

        parts = []
        for col in ["overview", "tagline"]:
            if col in df.columns:
                parts.append(df[col].fillna(""))
        parts.append(df.get("title", pd.Series([""]*len(df))).fillna(""))
        parts.append(df["cast_names"].apply(lambda xs: " ".join(xs)))
        parts.append(df["genres_names"].apply(lambda xs: " ".join(xs)))
        parts.append(df["director"].astype(str))
        df["soup"] = ""
        for p in parts:
            df["soup"] = (df["soup"] + " " + p.astype(str)).str.strip()

        # Year
        if "release_date" in df.columns:
            df["year"] = pd.to_datetime(df["release_date"], errors="coerce").dt.year
        elif "year" in df.columns:
            df["year"] = pd.to_numeric(df["year"], errors="coerce")
        else:
            df["year"] = np.nan

        df["vote_average"] = pd.to_numeric(df.get("vote_average", np.nan), errors="coerce")
        df["popularity"] = pd.to_numeric(df.get("popularity", np.nan), errors="coerce")

        keep = ["title", "soup", "genres_names", "year", "vote_average", "popularity"]
        for k in keep:
            if k not in df.columns:
                df[k] = np.nan if k in ["year","vote_average","popularity"] else ""
        return df[keep].copy()

    # Otherwise require tmdb 5000 pair
    if not {"tmdb_5000_movies", "tmdb_5000_credits"} <= set(dsets):
        raise RuntimeError("Provide 'movies' OR both 'tmdb_5000_movies' & 'tmdb_5000_credits'.")

    movies = dsets["tmdb_5000_movies"].copy()
    credits = dsets["tmdb_5000_credits"].copy()
    movies["id"] = pd.to_numeric(movies["id"], errors="coerce")
    credits["movie_id"] = pd.to_numeric(credits["movie_id"], errors="coerce")
    movies = movies.dropna(subset=["id"])
    credits = credits.dropna(subset=["movie_id"])
    df = movies.merge(credits, left_on="id", right_on="movie_id", how="left")

    for col in ["genres", "keywords", "cast"]:
        df[col] = df[col].fillna("[]").apply(_safe_eval_list)
    df["genres_names"] = df["genres"].apply(lambda xs: _names_from_list(xs, "name"))
    df["cast_names"] = df["cast"].apply(lambda xs: _names_from_list(xs, "name", n=5))
    df["director"] = df["crew"].fillna("[]").apply(_safe_eval_list).apply(_director_from_crew)

    def _join_no_space(xs):  # remove internal spaces for better signal, like 'ScienceFiction'
        return " ".join([str(x).replace(" ", "") for x in xs if x])

    df["soup"] = (
        df["keywords"].apply(_join_no_space) + " " +
        df["cast_names"].apply(_join_no_space) + " " +
        df["director"].astype(str).str.replace(" ", "", regex=False) + " " +
        df.get("overview", pd.Series([""]*len(df))).fillna("").astype(str)
    ).str.strip()

    df["title"] = df.get("title", df.get("original_title", ""))
    df["year"] = pd.to_datetime(df.get("release_date", np.nan), errors="coerce").dt.year
    df["vote_average"] = pd.to_numeric(df.get("vote_average", np.nan), errors="coerce")
    df["popularity"] = pd.to_numeric(df.get("popularity", np.nan), errors="coerce")

    keep = ["title", "soup", "genres_names", "year", "vote_average", "popularity"]
    return df[keep].copy()

# Vectorization (TF-IDF) & similarity
@st.cache_resource(show_spinner=True)
def build_vectorizer_matrix(df: pd.DataFrame, min_df: int):
    tfidf = TfidfVectorizer(stop_words="english", min_df=min_df, max_features=50000)
    X = tfidf.fit_transform(df["soup"].fillna(""))
    return tfidf, X

def get_cosine_neighbors(X, idx, topk):
    sims = cosine_similarity(X[idx], X).ravel()
    order = sims.argsort()[::-1]
    order = [i for i in order if i != idx][:topk]
    return order, sims[order]

def get_knn_neighbors(X, idx, topk):
    knn = NearestNeighbors(metric="cosine", algorithm="brute")
    knn.fit(X)
    dists, inds = knn.kneighbors(X[idx], n_neighbors=topk+1, return_distance=True)
    inds = inds.ravel().tolist()
    dists = dists.ravel().tolist()
    # remove self
    if inds and inds[0] == idx:
        inds, dists = inds[1:], dists[1:]
    sims = [1.0 - d for d in dists]
    return inds[:topk], sims[:topk]

# =========================
# UI
# =========================

st.title("🎬 Movie Recommendation System (Content-Based)")

# Load data
try:
    datasets = load_data_from_secrets()
except Exception as e:
    st.error(f"❌ Data load failed: {e}")
    st.stop()

df = build_master_table(datasets)

# Sidebar: vectorization controls
st.sidebar.header("Data & Vectorization")
src = "movies.csv" if "movies" in datasets else "tmdb_5000_movies.csv"
st.sidebar.caption(f"Loaded: {src} | Rows: {len(df):,}")
show_raw = st.sidebar.checkbox("Show raw columns", value=False)
min_df = int(st.sidebar.slider("TF-IDF min_df", min_value=1, max_value=10, value=2, step=1))

# Build vectorizer/matrix
tfidf, X = build_vectorizer_matrix(df, min_df=min_df)

tab_rec, tab_eda, tab_perf = st.tabs(["🔎 Recommender", "📊 EDA", "🧪 Model Performance"])

with tab_rec:
    cols = st.columns([2,1,1,1])
    with cols[0]:
        # Filters
        years = ["(All)"] + sorted([int(y) for y in df["year"].dropna().unique()])
        genres_flat = sorted({g for lst in df["genres_names"] for g in (lst or [])})
        genre_choice = st.selectbox("Genre filter", ["(All)"] + genres_flat, index=0)
        year_choice = st.selectbox("Year filter", years, index=0)
    with cols[1]:
        topk = st.number_input("Top-K", min_value=5, max_value=30, value=10, step=1)
    with cols[2]:
        method = st.radio("Method", options=["cosine", "knn"], index=0, horizontal=True)
    with cols[3]:
        pass

    # Candidate titles after filters
    mask = pd.Series([True]*len(df))
    if genre_choice != "(All)":
        mask &= df["genres_names"].apply(lambda lst: genre_choice in (lst or []))
    if year_choice != "(All)":
        mask &= df["year"] == int(year_choice)
    options = df.loc[mask, "title"].dropna().unique().tolist()
    options.sort(key=lambda s: s.lower())

    selected = st.selectbox("Type or select a movie title", options=options if options else df["title"].tolist())
    fuzzy = st.text_input("…or enter title for fuzzy match")

    go = st.button("Recommend")
    if go:
        query = fuzzy.strip() or selected
        if not query:
            st.warning("Select or enter a movie title to see recommendations.")
        else:
            # find index
            idx_candidates = df.index[df["title"].str.lower() == query.lower()].tolist()
            if not idx_candidates and fuzzy:
                idx_candidates = df.index[df["title"].str.lower().str.contains(query.lower())].tolist()
            if not idx_candidates:
                st.info("No matching title found.")
            else:
                idx = idx_candidates[0]
                if method == "cosine":
                    inds, sims = get_cosine_neighbors(X, idx, topk)
                else:
                    inds, sims = get_knn_neighbors(X, idx, topk)

                out = df.iloc[inds][["title","year","genres_names","vote_average","popularity"]].copy()
                out.insert(1, "similarity", np.round(sims, 3))
                st.write(f"**Because you watched:** {df.iloc[idx]['title']}")
                st.dataframe(out, use_container_width=True)

with tab_eda:
    st.subheader("Exploratory Data Analysis")
    used_cols = ["genres", "keywords", "tagline", "overview", "cast", "director"]
    st.caption("Text features used: genres, keywords, tagline, overview, cast, director")

    st.write(f"**TF-IDF shape:** {X.shape[0]} docs × {X.shape[1]} terms")
    sparsity = 1.0 - (X.nnz / (X.shape[0]*X.shape[1]))
    st.write(f"**Sparsity** {sparsity*100:.2f}%")

    # sample pairwise similarities
    try:
        sample = min(1000, X.shape[0])
        rng = np.random.default_rng(42)
        rows = rng.choice(X.shape[0], size=sample, replace=False)
        sims_sample = cosine_similarity(X[rows], dense_output=False)
        tril = sims_sample.toarray()
        vals = tril[np.triu_indices_from(tril, k=1)]
        st.write(f"Sample pairwise cosine similarity — mean **{vals.mean():.3f}**, std **{vals.std():.3f}**, min **{vals.min():.3f}**, max **{vals.max():.3f}**")
        fig1, ax1 = plt.subplots()
        ax1.hist(vals, bins=30)
        ax1.set_title("Distribution: Sample Pairwise Similarity")
        st.pyplot(fig1)
    except Exception as e:
        st.caption(f"Similarity sample plot skipped: {e}")

    # Top genres
    all_genres = [g for lst in df["genres_names"] for g in (lst or [])]
    cnt = Counter(all_genres)
    if cnt:
        topg = cnt.most_common(10)
        labels, values = zip(*topg)
        fig2, ax2 = plt.subplots()
        ax2.bar(labels, values)
        ax2.set_title("Top Genres")
        ax2.tick_params(axis='x', rotation=45)
        st.pyplot(fig2)

    # Popularity vs rating
    if "popularity" in df.columns and "vote_average" in df.columns:
        fig3, ax3 = plt.subplots()
        ax3.scatter(df["popularity"], df["vote_average"], s=8, alpha=0.6)
        ax3.set_xlabel("Popularity")
        ax3.set_ylabel("Vote Average")
        st.pyplot(fig3)

with tab_perf:
    st.subheader("Baseline Rating Prediction (Linear Regression)")
    y = df["vote_average"].fillna(df["vote_average"].median())
    Xdense = X  # using sparse in Ridge works
    X_tr, X_te, y_tr, y_te = train_test_split(Xdense, y, test_size=0.2, random_state=42)
    model = Ridge(alpha=1.0)
    model.fit(X_tr, y_tr)
    pred = model.predict(X_te)

    r2 = r2_score(y_te, pred)
    mae = mean_absolute_error(y_te, pred)
    mse = mean_squared_error(y_te, pred)
    rmse = np.sqrt(mse)

    cols = st.columns(4)
    cols[0].metric("R²", f"{r2:.3f}")
    cols[1].metric("MAE", f"{mae:.3f}")
    cols[2].metric("MSE", f"{mse:.3f}")
    cols[3].metric("RMSE", f"{rmse:.3f}")

    fig4, ax4 = plt.subplots()
    ax4.scatter(y_te, pred, s=8, alpha=0.6)
    ax4.axhline(y_te.mean(), linestyle="--")
    ax4.set_xlabel("True Ratings")
    ax4.set_ylabel("Predicted Ratings")
    st.pyplot(fig4)

if show_raw:
    st.write("Raw data preview")
    st.dataframe(df.head(20))
