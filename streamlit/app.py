import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
from difflib import get_close_matches
import matplotlib.pyplot as plt

st.set_page_config(page_title="Movie Recommender (TMDB)", layout="wide")

@st.cache_data(show_spinner=False)
def load_data():
    # Prefer a merged file if it exists
    for p in ["movies.csv", "tmdb_5000_movies.csv"]:
        try:
            df = pd.read_csv(p)
            # Normalize column names
            df.columns = [c.strip() for c in df.columns]
            return df, p
        except Exception:
            continue
    return None, None

def build_text_corpus(df):
    # Use whatever columns are available.
    candidates = []
    for col in ["genres", "keywords", "tagline", "overview", "cast", "director"]:
        if col in df.columns:
            candidates.append(col)

    if not candidates:
        # Fallback: try common TMDB field names
        for alt in ["title", "original_title"]:
            if alt in df.columns:
                candidates.append(alt)

    # Fill NA and fuse
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
def fit_models(df):
    texts, used_cols = build_text_corpus(df)
    vectorizer = TfidfVectorizer(stop_words="english", min_df=2)
    X = vectorizer.fit_transform(texts)

    # Cosine: precompute dense or use pairwise on demand (avoid huge memory by chunking via NearestNeighbors for top-k)
    # We'll keep KNN for efficient retrieval; cosine scores computed per query vector.
    knn = NearestNeighbors(metric="cosine", algorithm="brute")
    knn.fit(X)
    return X, vectorizer, knn, used_cols

def fuzzy_find_title(user_title, titles):
    if not isinstance(user_title, str) or not user_title.strip():
        return None
    matches = get_close_matches(user_title, titles, n=1, cutoff=0.5)
    return matches[0] if matches else None

def get_recommendations(title, titles, X, knn, vectorizer, top_k=10, method="cosine"):
    idx_series = pd.Series(range(len(titles)), index=titles)  # title -> idx
    if title not in idx_series.index:
        return pd.DataFrame()

    i = idx_series[title]
    if method == "knn":
        distances, indices = knn.kneighbors(X[i], n_neighbors=min(top_k + 1, len(titles)))
        distances = distances.flatten()
        indices = indices.flatten()
        # exclude the movie itself
        mask = indices != i
        indices, distances = indices[mask], distances[mask]
        sims = 1 - distances
        return pd.DataFrame({
            "title": [titles[j] for j in indices[:top_k]],
            "similarity": sims[:top_k]
        })
    else:
        # cosine vs all (vector-vector for stability)
        sims = cosine_similarity(X[i], X).flatten()
        order = np.argsort(-sims)
        order = order[order != i][:top_k]
        return pd.DataFrame({
            "title": [titles[j] for j in order],
            "similarity": sims[order]
        })

def safe_title_column(df):
    for c in ["title", "original_title", "movie_title", "name"]:
        if c in df.columns:
            return c
    # create synthetic title if missing
    df = df.copy()
    df["title"] = [f"Movie #{i+1}" for i in range(len(df))]
    return "title"

df, source = load_data()
st.title("🎬 Movie Recommendation System (Content-Based)")

if df is None:
    st.warning("Couldn't find `movies.csv` or `tmdb_5000_movies.csv` in the working directory. Upload one to continue.")
    st.stop()

title_col = safe_title_column(df)

with st.sidebar:
    st.header("Data")
    st.caption(f"Loaded: **{source}**  |  Rows: **{len(df)}**")
    show_cols = st.checkbox("Show raw columns", value=False)
    if show_cols:
        st.write(sorted(df.columns.tolist()))
    st.divider()
    st.header("Vectorization")
    min_df = st.slider("Min doc frequency (TF-IDF min_df)", 1, 10, 2)
    # Refit if min_df changes
    @st.cache_resource(show_spinner=False)
    def fit_models_with_min_df(df, min_df):
        texts, used_cols = build_text_corpus(df)
        vectorizer = TfidfVectorizer(stop_words="english", min_df=min_df)
        X = vectorizer.fit_transform(texts)
        knn = NearestNeighbors(metric="cosine", algorithm="brute").fit(X)
        return X, vectorizer, knn, used_cols
    X, vectorizer, knn, used_cols = fit_models_with_min_df(df, min_df)

tab1, tab2 = st.tabs(["🔎 Recommender", "📊 Insights & Metrics"])

with tab1:
    st.subheader("Find similar movies")
    c1, c2 = st.columns([3,1])
    with c1:
        user_title = st.text_input("Enter a movie title", "")
    with c2:
        top_k = st.number_input("Top-K", min_value=1, max_value=25, value=10, step=1)

    titles = df[title_col].fillna("").astype(str).tolist()
    chosen = fuzzy_find_title(user_title, titles) if user_title else None
    method = st.radio("Method", ["cosine", "knn"], horizontal=True, index=0)

    if user_title and not chosen:
        st.error("No close title match found. Try another title.")
    elif chosen:
        st.info(f"Closest match: **{chosen}**")
        recs = get_recommendations(chosen, titles, X, knn, vectorizer, top_k=top_k, method=method)
        st.dataframe(recs, use_container_width=True)

with tab2:
    st.subheader("Vectorizer & Similarity Stats")
    st.write(f"**Text features used**: {', '.join(used_cols) if used_cols else '(auto)'}")
    st.write(f"**TF-IDF shape**: {X.shape[0]} docs × {X.shape[1]} terms")
    # Sparsity
    nnz = X.nnz if hasattr(X, 'nnz') else np.count_nonzero(X)
    sparsity = 1.0 - (nnz / (X.shape[0] * X.shape[1]))
    st.metric("Sparsity", f"{sparsity*100:.2f}%")

    # Sample similarity distribution (on a small sample to stay snappy)
    n_sample = min(250, X.shape[0])
    idxs = np.random.default_rng(42).choice(X.shape[0], size=n_sample, replace=False)
    sims_sample = cosine_similarity(X[idxs], X[idxs], dense_output=False)
    # Extract upper triangle without diagonal
    if hasattr(sims_sample, "toarray"):
        sims_arr = sims_sample.toarray()
    else:
        sims_arr = sims_sample
    tri = sims_arr[np.triu_indices_from(sims_arr, k=1)]
    st.write(f"Sample pairwise cosine similarity — mean **{tri.mean():.3f}**, std **{tri.std():.3f}**, min **{tri.min():.3f}**, max **{tri.max():.3f}**")

    # Plots
    colA, colB = st.columns(2)
    with colA:
        st.caption("Distribution: Sample Pairwise Similarity")
        fig, ax = plt.subplots()
        ax.hist(tri, bins=30)
        ax.set_xlabel("Cosine similarity")
        ax.set_ylabel("Count")
        st.pyplot(fig, clear_figure=True)
    with colB:
        # Genre distribution if available
        if "genres" in df.columns:
            st.caption("Top Genres (parsed by token)")
            def split_tokens(s):
                if pd.isna(s):
                    return []
                # Try to handle both JSON-like or pipe/space separated strings
                s = str(s)
                if "[" in s and "]" in s and "name" in s:
                    # JSON-like; naive parse to names
                    import re
                    return re.findall(r"'name': '([^']+)'|\"name\": \"([^\"]+)\"", s)
                # otherwise split on separators
                for sep in ["|", ",", ";", " ", "/"]:
                    if sep in s:
                        return [x for x in s.split(sep) if x]
                return [s]

            from collections import Counter
            all_genres = []
            for g in df["genres"].head(5000).tolist():
                all_genres.extend(split_tokens(g))
            top = Counter(all_genres).most_common(15)
            if top:
                names, counts = zip(*top)
                fig2, ax2 = plt.subplots()
                ax2.bar(range(len(names)), counts)
                ax2.set_xticks(range(len(names)))
                ax2.set_xticklabels(names, rotation=45, ha="right")
                ax2.set_ylabel("Count")
                st.pyplot(fig2, clear_figure=True)
        else:
            st.info("No `genres` column found for genre distribution plot.")
