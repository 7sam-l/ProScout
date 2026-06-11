# Football Scouting & Player Recommendation System

This repository implements an interactive, interview-ready football scouting and player recommendation platform designed to demonstrate key data science concepts. The project was built using Python, Scikit-Learn, Streamlit, and Plotly.

---

## Project Folder Structure

```text
football-scouting-system/
│
├── data/
│   ├── players.csv             # Raw generated player dataset with missing values
│   └── players_clean.csv       # Preprocessed and clustered dataset
│
├── notebooks/
│   └── eda.ipynb               # Walkthrough notebook detailing cleaning, EDA, and models
│
├── src/
│   ├── generate_data.py        # Dataset generator (creates 600 realistic players)
│   ├── preprocessing.py        # Cleans and standardizes player data
│   ├── recommendation.py       # Computes Cosine Similarity and filters results
│   ├── clustering.py           # K-Means clustering and centroid auto-labeling
│   └── visualization.py        # Visual layouts (Radar, PCA scatter, heatmaps)
│
├── app/
│   └── streamlit_app.py        # Interactive Multipage Streamlit Dashboard
│
├── models/
│   └── scaler.pkl              # Saved StandardScaler object
│
├── outputs/
│   └── charts/                 # Visual output folder
│
├── requirements.txt            # Project requirements file
└── README.md                   # This project documentation
```

---

## Installation & Setup

Follow these steps to run the scouting dashboard locally:

### 1. Initialize Python Virtual Environment
Using macOS terminal, create and activate a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate
```

### 2. Install Project Dependencies
Install all library requirements:
```bash
pip install -r requirements.txt
```

### 3. Generate Player Dataset
Generate the synthetic football statistics dataset:
```bash
python src/generate_data.py
```

### 4. Run the Data Preprocessing Pipeline
Clean raw dataset features and fit the scaler:
```bash
python src/preprocessing.py
```

### 5. Launch the Streamlit Dashboard
Launch the interactive web portal locally:
```bash
streamlit run app/streamlit_app.py
```
The application will open automatically in your browser at `http://localhost:8501`.

---

## Core Data Science Implementations

### 1. Position-Wise Median Imputation
Instead of performing a global median or mean imputation, missing performance metrics are filled using the median of the player's specific playing position.
* **Why?**: A central midfielder's tackles and a forward's tackles are on entirely different scales. Imputing a defender's missing tackles with a global average would severely dilute their defensive profile.

### 2. Feature Standardization
Attributes have highly variable units (e.g. age: 18-36, minutes: 400-3400, passing accuracy: 70-93%). We fit a `StandardScaler` to remove unit scale effects, converting all attributes to standard normal distributions ($\mu=0, \sigma=1$).

### 3. Tactical K-Means Clustering ($K=5$)
Unsupervised learning splits players into 5 distinct clusters based on standardized stats. Clusters are assigned descriptive tactical archetypes dynamically by evaluating centroid averages:
* **Clinical Finishers**: High goals, low defensive involvements.
* **Creative Playmakers**: High assists and passing accuracy.
* **Dynamic Wingers**: High dribbles, moderate goals/assists.
* **Ball Winners**: High tackles and interceptions in midfield blocks.
* **Defensive Anchors**: Low attacking output, high tackles/interceptions, high passing accuracy (center backs).

### 4. Cosine Similarity Recommendations
Cosine similarity calculates the cosine angle between two scaled attribute vectors:
$$\text{similarity} = \cos(\theta) = \frac{\mathbf{A} \cdot \mathbf{B}}{\|\mathbf{A}\| \|\mathbf{B}\|}$$
* It measures profile shape and attribute balance rather than size magnitude alone, making it ideal for scouting.
* The search is enriched with filters (budget, league, age, position) and includes attribute-level percent difference tables explaining the matching results.

---

## Interview-Readiness Study Guide

Prepare to discuss these key questions for your MSc Data Science interview:

### Q1: Why did you use position-wise imputation instead of global mean imputation?
> **Answer**: Football metrics are highly position-dependent. A forward's standard tackling numbers are naturally very low. If we impute a defender's missing tackles using a global average, it will artificially drag down their score, causing similarity models to recommend less defensive players. Aggregating groups by position before calculating medians preserves the tactical integrity of individual player profiles.

### Q2: Why is Cosine Similarity preferred over Euclidean Distance for recommending players?
> **Answer**: Euclidean distance measures the absolute distance in n-dimensional space. If Player A and Player B have the exact same ratio of attributes but one played twice as many minutes (yielding higher totals), Euclidean distance would classify them as far apart. Cosine similarity calculates the angle between vector directions, focusing on the *proportionate balance* of their stats (e.g., playmaker ratio of passing-to-dribbling) rather than raw magnitudes.

### Q3: How did you decide on $K=5$ for clustering, and how are cluster names assigned?
> **Answer**: We targeted the 5 standard outfield archetypes requested in our scouting brief (Finishers, Playmakers, Wingers, Midfield Ball Winners, and Center Backs). Centroid coordinates representing average stats for each cluster were graded programmatically (e.g., the cluster with highest average dribbles maps to "Dynamic Wingers").

### Q4: Why is it important to save the StandardScaler (`scaler.pkl`) object?
> **Answer**: To avoid "data leakage" and keep scaling configurations consistent between offline training and real-time inference. When predicting or comparing a new player added to the streamlit dashboard, we must transform their statistics using the exact mean ($\mu$) and standard deviation ($\sigma$) parameters fit during preprocessing, rather than refitting on a single player.
