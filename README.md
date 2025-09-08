# 🎬 Movie Recommendation System (Content-Based ML)

> A collaborative academic project building a content-based movie recommender using TF‑IDF + cosine similarity and KNN over merged TMDB metadata.

---

## 🧭 Index
- [Overview](#overview)
- [Project Structure](#project-structure)
- [Datasets](#datasets)
- [Setup](#setup)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [Modeling Details](#modeling-details)
- [Evaluation & Results](#evaluation--results)
- [Reproducibility](#reproducibility)
- [License & Attribution](#license--attribution)

---

## Overview
This repository contains a **content-based movie recommendation system** developed for a Machine Learning course.  
It merges the **TMDB 5000 Movies** and **TMDB 5000 Credits** datasets into a single dataset (`movies.csv`) and builds recommendations using **TF‑IDF embeddings** and **cosine similarity** (with a **KNN** variant). The notebook also includes **rating prediction baselines** (Random Forest & Linear Regression) and lightweight EDA.

**Core ideas**
- Fuse textual metadata: `genres`, `keywords`, `tagline`, `cast`, `director`  
- Vectorize with **TfidfVectorizer** (~40k+ terms)  
- Retrieve similar titles using **cosine similarity** or **NearestNeighbors (KNN)**  
- Optional rating prediction for `vote_average` (regression baselines)

---

## Project Structure
```
.
├── Movie_Recommendation_System_using_Machine_Learning.ipynb   # Full implementation (EDA + recommender + baselines)
├── tmdb_5000_movies.csv                                       # Raw TMDB movies
├── tmdb_5000_credits.csv                                      # Raw TMDB credits
├── movies.csv                                                 # Merged dataset used by the pipeline
└── README.md                                                  # You are here
```

---

## Datasets
- **TMDB 5000 Movies** (`tmdb_5000_movies.csv`)
- **TMDB 5000 Credits** (`tmdb_5000_credits.csv`)
- **Merged** → `movies.csv` (primary input to the notebook)
  
> Note: Data is used **for educational purposes**. The TMDB name and logo are trademarks of TMDB. This project uses the data under TMDB’s terms; it is not endorsed by or affiliated with TMDB.

---

## Setup
1. **Clone repo**
   ```bash
   git clone https://github.com/<your-username>/<your-repo>.git
   cd <your-repo>
   ```

2. **Create environment**
   ```bash
   python -m venv .venv
   # Windows
   .venv\Scripts\activate
   # macOS / Linux
   source .venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

> If `requirements.txt` is not present, typical libs are: `pandas numpy scikit-learn matplotlib seaborn`

---

## Quick Start
Open the notebook and run all cells:
```bash
jupyter notebook Movie_Recommendation_System_using_Machine_Learning.ipynb
```

Or execute as a script (optional, if you later export code):
```bash
python src/recommender.py --data movies.csv --title "Avatar" --top_k 10
```

---

## Usage
### 1) Recommend similar movies
Inside the notebook:
```python
title = "Avatar"
top_k = 10  # number of recommendations
recommendations = get_recommendations(title, top_k=top_k, method="cosine")  # or method="knn"
for rank, rec in enumerate(recommendations, start=1):
    print(f"{rank}. {rec}")
```

### 2) Predict ratings (baseline)
```python
rf_metrics, lin_metrics = train_and_evaluate_regressors(df)  # returns dicts with R2, MAE, MSE
print("RandomForest:", rf_metrics)
print("LinearRegression:", lin_metrics)
```

### 3) Handle typos / fuzzy matches
```python
approx_title = fuzzy_match("Avatr", titles_index)  # → "Avatar"
print(approx_title)
```

---

## Modeling Details
- **Feature engineering**: Combine `genres`, `keywords`, `tagline`, `cast`, `director` into one text field.
- **Vectorization**: `TfidfVectorizer` with basic text normalization.
- **Similarity search**:
  - **Cosine similarity** on TF‑IDF vectors (top‑N retrieval).
  - **KNN (NearestNeighbors)** with cosine metric (alt. retrieval path).
- **Baselines**: `RandomForestRegressor`, `LinearRegression` on available numeric/text-derived features.
- **Metrics**:
  - Regression: **R²**, **MAE**, **MSE**.
  - Optional classification view for “high-rated” (thresholded): **accuracy**, **precision**, **recall**, **F1**.
- **EDA**: Distributions (genres, ratings), correlation heatmaps, simple trend charts.

---

## Evaluation & Results
- Content-based retrieval yields coherent **top‑N similar titles**.
- KNN retrieval closely mirrors cosine results and provides a handy alternative.
- Regression baselines serve as a sanity check for rating predictability.
  
- **Dataset scale**: 4,803 movies · **TF‑IDF feature space**: 41,698 terms.
- **TF‑IDF stats**: Sparsity **99.86%** · Feature diversity **100%** · Document similarity (mean **0.013**, min **0.000**, max **0.830**).
- **KNN recommender (k=10)**:
  - Recommendation coverage: **99.73%**
  - Genre diversity score: **0.024**
  - Similarity distribution — mean **0.104**, std **0.046**, min **0.000**, max **0.830**
- **Rating prediction (baseline)**:
  - R²: **0.253**
  - MAE: **0.687**
  - MSE: **1.082**
  - RMSE: **1.040**

**Example: Top‑10 recommendations for “Avatar” (cosine/KNN):**  
1) Star Trek Into Darkness · 2) Idiocracy · 3) The Dark Knight · 4) Charlie's Angels · 5) Alien ·
6) The Wendell Baker Story · 7) Step Up 3D · 8) The Dark Knight Rises · 9) Enemy of the State · 10) The SpongeBob Movie: Sponge Out of Water

---

## Reproducibility
- Fixed random seeds where applicable.
- Clear preprocessing steps in the notebook.
- Deterministic TF‑IDF + cosine pipeline.
- Instructions included to recreate the merged dataset and rerun models.

---

## License & Attribution
- Code: MIT License (see `LICENSE` if provided)
- Data: © TMDB — used for educational purposes only. This project is not endorsed by or affiliated with TMDB.
