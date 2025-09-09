
import os
import io
import re
import requests
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from difflib import get_close_matches
from collections import Counter


st.set_page_config(page_title="Movie Recommender – Remote Data", layout="wide")


# =======================
# Remote data loading
# =======================
def _get_secret_url():
    # Accept either flat or namespaced secrets
    # e.g. st.secrets["DATA_URL"] or st.secrets["dataset"]["url"]
    url = None
    token = None
    try:
        url = st.secrets.get("DATA_URL", None)
        token = st.secrets.get("HF_TOKEN", None)
    except Exception:
        pass
    try:
        if not url and "dataset" in st.secrets:
            url = st.secrets["dataset"].get("url", url)
            token = st.secrets["dataset"].get("token", token)
    except Exception:
        pass
    # Fallback to env vars for local runs
    if not url:
        url = os.environ.get("DATA_URL")
    if not token:
        token = os.environ.get("HF_TOKEN")
    return url, token


@st.cache_data(show_spinner=True)
def load_remote_csv(url: str, token: str | None = None) -> pd.DataFrame:
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    with requests.get(url, headers=headers, timeout=120) as r:
        r.raise_for_status()
        b = io.BytesIO(r.content)
    df = pd.read_csv(b)
    # Normalize columns
    df.columns = [c.strip() for c in df.columns]
    return df


@st.cache_data(show_spinner=False)
def detect_title_col(df: pd.DataFrame) -> str:
    for c in ["title", "original_title", "movie_title", "name"]:
        if c in df.columns:
            return c
    return None


# =======================
# Vectorization & Recs
# =======================
@st.cache_data(show_spinner=False)
def build_text_corpus(df: pd.DataFrame):
    cols = [c for c in ["genres", "keywords", "tagline", "overview", "cast", "director"] if c in df.columns]
    if not cols:
        cols = [c for c in ["title", "original_title"] if c in df.columns]
    texts = (df[cols].fillna("").astype(str).agg(" ".join, axis=1)).tolist()
    return texts, cols


@st.cache_resource(show_spinner=True)
def fit_models(df: pd.DataFrame, min_df: int):
    texts, used_cols = build_text_corpus(df)
    vec = TfidfVectorizer(stop_words="english", min_df=min_df)
    X = vec.fit_transform(texts)
    knn = NearestNeighbors(metric="cosine", algorithm="brute").fit(X)
    return X, vec, knn, used_cols


def fuzzy_match(q: str, options: list[str]):
    from difflib import get_close_matches
    m = get_close_matches(q, options, n=1, cutoff=0.5)
    return m[0] if m else None


def split_tokens(s: str):
    if pd.isna(s):
        return []
    s = str(s)
    if "[" in s and "]" in s and "name" in s:
        import re
        parts = re.findall(r"'name': '([^']+)'|\"name\": \"([^\"]+)\"", s)
        parts = [p[0] or p[1] for p in parts]
        return [p for p in parts if p]
    for sep in ["|", ",", ";", " / ", "/", " "]:
        if sep in s:
            return [x for x in s.split(sep) if x]
    return [s]


# =======================
# UI
# =======================
st.title("🎬 Movie Recommendation System")

with st.status("Fetching dataset URL from secrets...", expanded=False) as s:
    data_url, hf_token = _get_secret_url()
    if not data_url:
        st.error("No dataset URL found. Add it in **Settings → Secrets** as `DATA_URL` (or under `[dataset] url`).")
        st.stop()
    s.update(label=f"Dataset URL found", state="complete")

with st.status("Downloading dataset...", expanded=False):
    df = load_remote_csv(data_url, hf_token)
    st.success(f"Loaded rows: {len(df):,}")

title_col = detect_title_col(df)
if not title_col:
    title_col = "title"
    if "title" not in df.columns:
        df[title_col] = [f"Movie #{i+1}" for i in range(len(df))]

with st.sidebar:
    st.header("Controls")
    st.caption(f"Rows: **{len(df):,}**")
    min_df = st.slider("TF‑IDF min_df", 1, 10, 2)
    X, vec, knn, used_cols = fit_models(df, min_df=min_df)

titles = df[title_col].astype(str).tolist()

# genre/year filters
all_genres = []
if "genres" in df.columns:
    for g in df["genres"].tolist():
        all_genres.extend([t for t in split_tokens(g) if t])
from collections import Counter
genre_options = ["(All)"] + [g for g, _ in Counter(all_genres).most_common(50)]
year_options = ["(All)"]
if "release_date" in df.columns:
    yrs = pd.to_datetime(df["release_date"], errors="coerce").dt.year.dropna().astype(int)
    year_options += [str(y) for y in sorted(yrs.unique())]

tab1, tab2, tab3 = st.tabs(["🔎 Recommender", "📊 EDA", "🧪 Model"])

with tab1:
    c1, c2, c3, c4 = st.columns([3,1.2,1.2,1.2])
    with c1:
        pick = st.selectbox("Select a movie", [""] + titles, index=0)
    with c2:
        topk = st.number_input("Top‑K", 1, 25, 10)
    with c3:
        method = st.radio("Method", ["cosine", "knn"], horizontal=True)
    with c4:
        genre_pick = st.selectbox("Genre", genre_options, index=0)
    year_pick = st.selectbox("Year", year_options, index=0)

    if not pick:
        typed = st.text_input("...or type for fuzzy match", "")
        if typed:
            mm = fuzzy_match(typed, titles)
            if mm:
                pick = mm
                st.info(f"Using closest match: **{mm}**")

    if pick:
        import numpy as np
        idx = titles.index(pick)
        if method == "cosine":
            sims = cosine_similarity(X[idx], X).flatten()
            order = np.argsort(-sims)
            order = order[order != idx]
            sim_vals = sims[order]
            idxs = order
        else:
            dist, ind = knn.kneighbors(X[idx], n_neighbors=min(50, len(titles)))
            ind = ind.flatten()
            dist = dist.flatten()
            mask = ind != idx
            idxs = ind[mask]
            sim_vals = 1 - dist[mask]

        recs = pd.DataFrame({"idx": idxs, "title": [titles[i] for i in idxs], "similarity": sim_vals})
        out = df.iloc[recs["idx"]].copy()
        out["title"] = recs["title"].values
        out["similarity"] = recs["similarity"].values
        if genre_pick != "(All)" and "genres" in out.columns:
            out = out[out["genres"].astype(str).str.contains(genre_pick, case=False, na=False)]
        if year_pick != "(All)" and "release_date" in out.columns:
            out = out[pd.to_datetime(out["release_date"], errors="coerce").dt.year.astype("Int64").astype(str) == year_pick]

        keep_cols = ["title", "similarity"] + [c for c in ["genres", "release_date", "vote_average"] if c in out.columns]
        st.dataframe(out[keep_cols].head(topk), use_container_width=True)
    else:
        st.warning("Pick a movie to see recommendations.")

with tab2:
    st.write(f"**Text features**: {', '.join(used_cols)}")
    st.write(f"**TF‑IDF shape**: {X.shape[0]} × {X.shape[1]}")
    nnz = X.nnz if hasattr(X, 'nnz') else np.count_nonzero(X)
    st.metric("Sparsity", f"{(1 - nnz/(X.shape[0]*X.shape[1]))*100:.2f}%")

with tab3:
    feats = [c for c in ["popularity","budget","revenue","runtime","vote_count"] if c in df.columns]
    target = "vote_average" if "vote_average" in df.columns else None
    if target and feats:
        Xn = df[feats].fillna(0.0).astype(float)
        y = df[target].fillna(df[target].mean())
        Xtr, Xte, ytr, yte = train_test_split(Xn, y, test_size=0.2, random_state=42)
        model = LinearRegression().fit(Xtr, ytr)
        yp = model.predict(Xte)
        st.metric("R²", f"{r2_score(yte, yp):.3f}")
        st.metric("MAE", f"{mean_absolute_error(yte, yp):.3f}")
        st.metric("RMSE", f"{mean_squared_error(yte, yp) ** 0.5:.3f}")
    else:
        st.info("Need `vote_average` and at least one numeric feature (popularity/budget/revenue/runtime/vote_count).")
