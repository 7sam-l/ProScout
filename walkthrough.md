# ProScout Interview Walkthrough Guide

This document is designed to serve as a comprehensive study guide for your MSc Data Science interview. It explains the project's architecture, data science methodologies, technical tradeoffs, and provides a clear narrative for presenting your work.

---

## 1. Overall Architecture

The project follows a classic, modular machine learning architecture with a strong emphasis on separation of concerns.

> [!NOTE] 
> **Architectural Paradigm**: The codebase is split into an independent **Data/ML Backend layer** (`src/`) and an interactive **Frontend/Visualization layer** (`app/`).

- **Data Generation & Ingestion**: `src/generate_data.py` synthesizes the dataset.
- **Data Engineering**: `src/preprocessing.py` cleans data and applies standardization.
- **Machine Learning Core**: `src/recommendation.py` and `src/clustering.py` hold the predictive/analytical logic.
- **Presentation Layer**: `src/visualization.py` generates the Plotly charts, and `app/streamlit_app.py` serves as the unifying interactive dashboard.

## 2. Data Flow from Dataset to Dashboard

1. **Raw Data Extraction**: `data/players.csv` is loaded into a Pandas DataFrame.
2. **Preprocessing Pipeline**: The raw dataset is pushed through `src/preprocessing.py` where null values are imputed and ML features are scaled. The pipeline outputs `data/players_clean.csv` and a serialized `scaler.pkl`.
3. **Model Initialization**: The Streamlit application (`app/streamlit_app.py`) loads the clean data and initializes the `PlayerRecommender` and K-Means models.
4. **Interactive Dashboarding**: User inputs from the Streamlit UI trigger calls to the ML modules (e.g., retrieving Similar Players or calculating PCA reductions). The results are passed to `src/visualization.py` which dynamically generates Plotly charts rendered by Streamlit.

## 3. Data Cleaning Process

The data cleaning process was designed explicitly to **preserve domain-specific distributions**.

- **Position-Wise Imputation**: Instead of using global means for missing values, numerical columns like `tackles` or `passing_accuracy` are imputed using the **median value for that specific player's position**. 
  > *Interview Talking Point*: "I chose position-wise median imputation because replacing a missing tackle stat for a Center Forward with the global mean (heavily skewed by Center Backs) would distort the player's tactical profile and ruin the clustering algorithms."
- **Standardization**: The 7 core tactical features (goals, assists, tackles, etc.) are standardized using `scikit-learn`'s `StandardScaler` (Z-score normalization). This ensures features with large ranges (e.g., passing accuracy percentage) don't numerically dominate features with small ranges (e.g., goals).

## 4. Recommendation Engine Implementation

The Recommendation Engine (`src/recommendation.py`) relies on **Cosine Similarity**.

- It computes the pairwise cosine similarity across the standardized feature matrix of all players.
- When a target player is queried, the engine retrieves their similarity vector and sorts the dataset to find the nearest neighbors (values closest to 1.0).
- **Explainability**: The engine doesn't just return a score; it calculates the percentage difference between the target player and the recommended player on a feature-by-feature basis, ensuring the scout understands *why* the recommendation was made.

## 5. Clustering Implementation

The system utilizes **K-Means Clustering** (`src/clustering.py`) for unsupervised tactical archetype discovery.

- The model partitions the dataset into `K=5` distinct clusters based on their standardized performance metrics.
- **Semantic Auto-Labeling**: Rather than presenting unexplainable "Cluster 0", "Cluster 1" labels, the code inspects the mathematical centroids (average feature values) of each cluster and uses a greedy heuristic scoring algorithm to map them to recognizable tactical roles: *Clinical Finishers, Creative Playmakers, Dynamic Wingers, Ball Winners, Defensive Anchors*.
- **Dimensionality Reduction**: To visualize 7-dimensional clusters, the system applies **Principal Component Analysis (PCA)** to reduce the feature space to 2D (or 3D), retaining the maximum possible variance.

## 6. Apple-Style Minimalist UI Architecture

The entire platform recently underwent a visual redesign (shifting away from dark mode themes to a strict, pure white Apple-style analytics dashboard). 

- **Strict Light Mode Enforcement**: The theme uses Streamlit's native `config.toml` to enforce a strict Light Mode (`base="light"`). It completely ignores OS-level dark mode settings to ensure the stark black-on-white contrast is never compromised.
- **Apple UI Elements**:
  - *Typography*: Utilizes `-apple-system`, `SF Pro Display`, and standard sans-serif stacks with tight kerning.
  - *Containers*: Flat `#F5F5F7` background cards (no borders) mimicking Apple's interface design.
- **Apple Minimalist Palette**: 
  - *Primary Typography*: `#1D1D1F` (Apple Dark Grey)
  - *Accents & Data Points*: `#0066CC` (System Blue), `#000000` (Black), `#FF9500` (Orange), `#FF3B30` (Red), `#8E8E93` (Silver).
- **Dynamic Charts**: All Plotly charts (Radar, PCA, Heatmaps) use stark minimalist colors with very subtle `#E5E5EA` gridlines.

## 7. Key Files & Responsibilities

| File | Primary Responsibility |
|------|------------------------|
| `generate_data.py` | Generates the synthetic raw dataset with logical statistical distributions. |
| `preprocessing.py` | Handles missing values, performs position-wise median imputation, and standardizes features via `StandardScaler`. |
| `recommendation.py` | Calculates Cosine Similarity between player vectors and applies dynamic scout filtering (age, value). |
| `clustering.py` | Computes K-Means clusters and dynamically labels them based on their mathematical centroids. |
| `visualization.py` | Houses all complex Plotly visual logic (Radar charts, PCA scatter matrices, Heatmaps) using the minimalist color palette. |
| `streamlit_app.py` | The main controller. Handles the web application layout, custom flat CSS injections, UI state, and routing. |
| `.streamlit/config.toml` | Establishes the native strict light theme. |

## 8. Design Decisions & Tradeoffs

> [!TIP]
> Use these points to demonstrate maturity and pragmatic engineering during your interview.

*   **Tradeoff 1: K-Means vs DBSCAN/Hierarchical**
    *   *Decision*: Chose K-Means over density-based algorithms.
    *   *Reason*: K-Means is computationally fast and highly explainable. Since we have a fixed conceptual idea of football tactical roles (roughly 4-6 archetypes), selecting a discrete `K=5` makes the output easier to interpret for a football scout compared to arbitrary density clusters.
*   **Tradeoff 2: Cosine Similarity vs Euclidean Distance**
    *   *Decision*: Used Cosine Similarity for the recommendation engine.
    *   *Reason*: Cosine similarity measures the *angle* between feature vectors rather than magnitude. This means a young player with lower overall output but an identical *proportion* of tackles to passes can still be recommended as a "similar tactical profile" to an elite veteran, which is exactly what scouts want.
*   **Tradeoff 3: Synthetic Data vs Web Scraping**
    *   *Decision*: Synthesized data using `numpy.random` instead of scraping FBRef or Transfermarkt.
    *   *Reason*: Allowed for strict control over the schema and distributions, ensuring the ML models worked perfectly without getting bogged down in weeks of HTML parsing and legal scraping considerations.

## 9. How to Present This in an MSc Interview

When explaining this project, use the **STAR Method** (Situation, Task, Action, Result) and emphasize your focus on **explainability over complexity**.

**1. The Hook (The Problem):**
> "In modern football scouting, subjective bias often leads to poor recruitment. My objective was to build an end-to-end, data-driven system that identifies statistically similar players to replace departing stars, and uncovers underlying tactical archetypes."

**2. The Approach (The Action):**
> "I built the system using a modular architecture. I prioritized data engineering—specifically using position-wise median imputation to preserve tactical profiles. I then implemented unsupervised learning (K-Means) to map the dataset into 5 distinct tactical archetypes, and used a Cosine Similarity engine to find player replacements."

**3. The Engineering Pragmatism (The "Why"):**
> "While I could have used deep learning autoencoders for similarity, I explicitly chose Cosine Similarity and K-Means. Why? Because in a sports analytics context, *explainability is king*. A scout needs to understand *why* a player is recommended. That's why I also built a dynamic Plotly radar chart overlay, so the statistical differences are immediately visually apparent."

**4. The Result:**
> "The final product is an interactive Streamlit application that handles the full data pipeline in real-time, allowing scouts to filter targets, compare radar profiles, and visualize high-dimensional clusters using PCA."
