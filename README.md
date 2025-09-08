# 🎬 Movie Recommendation System (Content-Based ML)

> A collaborative academic project building a content-based movie recommender using TF-IDF + cosine similarity and KNN over merged TMDB metadata.  

---

## 🧭 Index
- [Overview](#overview)
- [Project Structure](#project-structure)
- [Datasets](#datasets)
- [Setup](#setup)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [Streamlit App](#streamlit-app)
- [Modeling Details](#modeling-details)
- [Evaluation & Results](#evaluation--results)
- [Reproducibility](#reproducibility)
- [License & Attribution](#license--attribution)

---

## Overview
This repository contains a **content-based movie recommendation system** developed for a Machine Learning course.  
It merges the **TMDB 5000 Movies** and **TMDB 5000 Credits** datasets into a single dataset (`movies.csv`) and builds recommendations using **TF-IDF embeddings** and **cosine similarity** (with a **KNN** variant). The notebook also includes **rating prediction baselines** (Random Forest & Linear Regression) and lightweight EDA.

---

## Project Structure
```
.
├── Movie_Recommendation_System_using_Machine_Learning.ipynb   # Full implementation (EDA + recommender + baselines)
├── tmdb_5000_movies.csv                                       # Raw TMDB movies
├── tmdb_5000_credits.csv                                      # Raw TMDB credits
├── movies.csv                                                 # Merged dataset used by the pipeline
├── app.py                                                     # Streamlit app
├── requirements.txt                                           # Dependencies for deployment
└── README.md
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
   .venv\Scripts\activate   # Windows
   source .venv/bin/activate  # macOS / Linux
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

---

## Quick Start
### Notebook
Open the notebook and run all cells:
```bash
jupyter notebook Movie_Recommendation_System_using_Machine_Learning.ipynb
```

### Script (optional)
```bash
python src/recommender.py --data movies.csv --title "Avatar" --top_k 10
```

---

## Usage
### 1) Recommend similar movies
```python
title = "Avatar"
top_k = 10
recommendations = get_recommendations(title, top_k=top_k, method="cosine")
```

### 2) Predict ratings
```python
rf_metrics, lin_metrics = train_and_evaluate_regressors(df)
print("RandomForest:", rf_metrics)
print("LinearRegression:", lin_metrics)
```

### 3) Handle fuzzy matches
```python
approx_title = fuzzy_match("Avatr", titles_index)
```

---

## Streamlit App
We provide a **Streamlit web app** for interactive exploration and recommendations:  

👉 [Check it Out](https://movie-recommendations-ysytem.streamlit.app/)  

Features:
- Search movie titles with autocomplete + fuzzy matching  
- Get top-N recommendations (Cosine/KNN)  
- Explore EDA charts (genres, ratings, popularity)  
- Check baseline model metrics & visualizations  

---

## Modeling Details
- Text features: genres, keywords, tagline, overview, cast, director  
- Vectorization: TF-IDF with stopword removal  
- Similarity: Cosine similarity + KNN  
- Baselines: Random Forest Regressor, Linear Regression  
- Metrics: R², MAE, MSE, RMSE, classification metrics for “high-rated” detection  

---

## Evaluation & Results
- **TF-IDF features**: 41,698 terms, sparsity ~99.9%  
- **Cosine/KNN similarity**: Coherent top-N retrieval  
- **Baseline regression**: R² ≈ 0.25, MAE ≈ 0.69, RMSE ≈ 1.04  

**Example: Top-10 recommendations for “Avatar”**  
1) Star Trek Into Darkness  
2) Idiocracy  
3) The Dark Knight  
4) Charlie's Angels  
5) Alien  
6) The Wendell Baker Story  
7) Step Up 3D  
8) The Dark Knight Rises  
9) Enemy of the State  
10) The SpongeBob Movie: Sponge Out of Water

---

## Reproducibility
- Deterministic preprocessing & seeds  
- Clear merging pipeline (`movies.csv`)  
- Streamlit app + notebook both reproducible  

---

## License & Attribution
- Code: MIT License (see `LICENSE` if provided)
- Data: © TMDB — used for educational purposes only. This project is not endorsed by or affiliated with TMDB.
