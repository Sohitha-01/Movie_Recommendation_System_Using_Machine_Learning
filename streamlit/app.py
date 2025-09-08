import io
import ast
from pathlib import Path

import requests
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


st.set_page_config(page_title="🎬 Movie Recommendation System", page_icon="🎬", layout="wide")

# ======================================================================================
# Data loading (Path B): download from URLs stored in Streamlit Secrets
# ======================================================================================

def _fetch_csv(url: str) -> pd.DataFrame:
    """
    Robust HTTP fetch with fallback to ?download=true and encoding handling.
    """
    session = requests.Session()
    session.headers.update({"User-Agent": "streamlit-movie-recs/1.0"})
    trials = [url]
    if "?" not in url:
        trials.append(url + "?download=true")
    last_err = None
    for u in trials:
        try:
            r = session.get(u, timeout=60, allow_redirects=True)
            r.raise_for_status()
            # try utf-8, then latin-1
            try:
                text = r.content.decode("utf-8")
            except UnicodeDecodeError:
                text = r.content.decode("latin-1")
            df = pd.read_csv(io.StringIO(text))
            df.columns = [c.strip() for c in df.columns]
            return df
        except Exception as e:
            last_err = e
    raise RuntimeError(f"Failed to download CSV from {url} ({last_err})")


@st.cache_data(show_spinner="Downloading datasets…")
def load_datasets_from_secrets() -> dict:
    """
    Reads URLs from Streamlit Secrets:
        movies
        tmdb_5000_movies
        tmdb_5000_credits
    and returns whichever of them successfully load.
    """
    required_keys = ["movies", "tmdb_5000_movies", "tmdb_5000_credits"]
    loaded = {}
    for key in required_keys:
        url = st.secrets.get(key, "")
        if isinstance(url, str) and url.startswith(("http://", "https://")):
            loaded[key] = _fetch_csv(url)
    if not loaded:
        raise RuntimeError(
            "No datasets loaded. Add URLs in Settings → Secrets with keys: "
            "movies, tmdb_5000_movies, tmdb_5000_credits"
        )
    return loaded


# ======================================================================================
# Preprocessing for content-based recommendations
# ======================================================================================

def _safe_literal_eval(val):
    try:
        return ast.literal_eval(val)
    except Exception:
        return []


def _get_director(crew_list):
    for member in crew_list:
        if isinstance(member, dict) and member.get("job") == "Director":
            return member.get("name", "")
    return ""


def _top_n_names(items, n=5, key="name"):
    out = []
    for item in items:
        if isinstance(item, dict):
            name = item.get(key)
            if name:
                out.append(name)
        if len(out) >= n:
            break
    return out


@st.cache_data(show_spinner="Preparing data…")
def prepare_master_table(dsets: dict) -> pd.DataFrame:
    """
    Build a master movie table with a 'soup' text field for vectorization.
    Supports either the 'movies' dataset or the tmdb_5000_* pair.
    """
    if "movies" in dsets:
        # If a simple 'movies.csv' is provided, use its text columns directly when available
        df = dsets["movies"].copy()
        # Try to build a rich soup if typical columns exist
        candidates = []
        for col in ["genres", "keywords", "cast", "crew", "overview", "tagline", "title"]:
            if col in df.columns:
                candidates.append(col)
        if not candidates:
            # fallback to title+overview
            df["soup"] = (df.get("title", pd.Series([""] * len(df))).fillna("") + " " +
                          df.get("overview", pd.Series([""] * len(df))).fillna(""))
        else:
            # try to parse list-like string columns
            parts = []
            for col in ["genres", "keywords", "cast"]:
                if col in df.columns:
                    parts.append(
                        df[col].fillna("[]").apply(_safe_literal_eval).apply(
                            lambda xs: " ".join(x if isinstance(x, str) else x.get("name", "") for x in xs)  # name or raw
                        )
                    )
            # director if crew exists
            if "crew" in df.columns:
                parts.append(
                    df["crew"].fillna("[]").apply(_safe_literal_eval).apply(_get_director)
                )
            # overview/tagline/title if exist
            for col in ["overview", "tagline", "title"]:
                if col in df.columns:
                    parts.append(df[col].fillna(""))

            # combine
            df["soup"] = ""
            for p in parts:
                df["soup"] = (df["soup"] + " " + p.astype(str)).str.strip()

        # Normalize id/title columns
        if "title" not in df.columns:
            df["title"] = df.get("original_title", df.index.astype(str))
        df["movie_id"] = df.get("id", pd.RangeIndex(start=0, stop=len(df)))
        return df[["movie_id", "title", "soup"]].fillna("")

    # Otherwise use tmdb_5000_movies + tmdb_5000_credits
    if not {"tmdb_5000_movies", "tmdb_5000_credits"} <= set(dsets):
        raise RuntimeError("Need either 'movies' OR both 'tmdb_5000_movies' and 'tmdb_5000_credits' datasets.")

    movies = dsets["tmdb_5000_movies"].copy()
    credits = dsets["tmdb_5000_credits"].copy()

    # Ensure id types match for merge
    movies["id"] = pd.to_numeric(movies["id"], errors="coerce")
    credits["movie_id"] = pd.to_numeric(credits["movie_id"], errors="coerce")
    movies = movies.dropna(subset=["id"])
    credits = credits.dropna(subset=["movie_id"])

    merged = movies.merge(credits, left_on="id", right_on="movie_id", how="left")

    # Parse JSON-like strings
    for col in ["genres", "keywords"]:
        if col in merged.columns:
            merged[col] = merged[col].fillna("[]").apply(_safe_literal_eval).apply(
                lambda xs: [x.get("name", "") for x in xs if isinstance(x, dict)]
            )
        else:
            merged[col] = [[]] * len(merged)

    if "cast" in merged.columns:
        merged["cast"] = merged["cast"].fillna("[]").apply(_safe_literal_eval).apply(lambda xs: _top_n_names(xs, 5))
    else:
        merged["cast"] = [[]] * len(merged)

    if "crew" in merged.columns:
        merged["director"] = merged["crew"].fillna("[]").apply(_safe_literal_eval).apply(_get_director)
    else:
        merged["director"] = ""

    # Build the soup
    def _join(lst):
        return " ".join([str(x).replace(" ", "") for x in lst if x])

    merged["soup"] = (
        merged["keywords"].apply(_join) + " " +
        merged["cast"].apply(_join) + " " +
        merged["director"].astype(str).str.replace(" ", "", regex=False) + " " +
        merged.get("overview", pd.Series([""] * len(merged))).fillna("").astype(str)
    ).str.strip()

    merged["title"] = merged.get("title", merged.get("original_title", pd.Series([""] * len(merged)))).fillna("")
    merged["movie_id"] = merged["id"].astype(int)
    return merged[["movie_id", "title", "soup"]]


@st.cache_resource(show_spinner="Training similarity model…")
def build_vectorizer_and_matrix(master: pd.DataFrame):
    vectorizer = CountVectorizer(stop_words="english", max_features=5000)
    matrix = vectorizer.fit_transform(master["soup"].fillna(""))
    sims = cosine_similarity(matrix, dense_output=False)
    return vectorizer, sims


def recommend(title: str, master: pd.DataFrame, sims, top_k: int = 10):
    title_lower = title.strip().lower()
    matches = master[master["title"].str.lower() == title_lower]
    if matches.empty:
        # try partial match
        matches = master[master["title"].str.lower().str.contains(title_lower)]
        if matches.empty:
            return []

    idx = matches.index[0]
    row = sims[idx].toarray().ravel()
    # highest scores (excluding self)
    top_idx = row.argsort()[::-1]
    top_idx = [i for i in top_idx if i != idx][:top_k]
    result = master.iloc[top_idx][["title"]].copy()
    result["similarity"] = row[top_idx]
    return result.values.tolist()


# ======================================================================================
# UI
# ======================================================================================

st.title("🎬 Movie Recommendation System (Content-Based)")
st.write(
    "Data is fetched at runtime from URLs you store in **Settings → Secrets**. "
    "Make sure you added keys: `movies`, `tmdb_5000_movies`, `tmdb_5000_credits`."
)

# Load data
try:
    datasets = load_datasets_from_secrets()
    st.success(
        "Loaded datasets: "
        + ", ".join(f"{k} ({len(v):,} rows)" for k, v in datasets.items())
    )
except Exception as e:
    st.error(f"❌ Data load failed: {e}")
    st.stop()

# Build master table & model
master = prepare_master_table(datasets)
vec, sim = build_vectorizer_and_matrix(master)

# Controls
left, right = st.columns([2, 1])
with left:
    default_title = master["title"].iloc[0] if not master.empty else ""
    user_title = st.text_input("Type a movie title:", value=default_title, placeholder="e.g., Avatar")
with right:
    top_k = st.number_input("How many recommendations?", min_value=5, max_value=20, value=10, step=1)

# Recommend
if st.button("Recommend"):
    if not user_title.strip():
        st.warning("Please enter a movie title.")
    else:
        recs = recommend(user_title, master, sim, top_k=top_k)
        if not recs:
            st.info("No exact match found. Try another title or check spelling.")
        else:
            st.subheader(f"Because you watched: {user_title}")
            df = pd.DataFrame(recs, columns=["Title", "Similarity"])
            # nicer view
            st.dataframe(df.style.format({"Similarity": "{:.3f}"}), use_container_width=True)

# Debug info (toggleable)
with st.expander("Debug info"):
    st.write("CWD:", Path.cwd())
    st.write("Master table shape:", master.shape)
    st.write(master.head())
