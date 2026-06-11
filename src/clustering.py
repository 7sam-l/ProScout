import os
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from src.preprocessing import ML_FEATURES

def perform_clustering(df_clean, scaled_df, n_clusters=5, random_state=42):
    """
    Fits K-Means clustering on the scaled performance features.
    Assigns archetypes based on centroid features.
    """
    # Initialize and fit KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    cluster_labels = kmeans.fit_predict(scaled_df)
    
    df_result = df_clean.copy()
    df_result["cluster_id"] = cluster_labels
    
    # Label the clusters programmatically based on centroids
    centroids = kmeans.cluster_centers_
    archetypes = assign_archetypes(centroids, ML_FEATURES)
    
    # Map cluster IDs to archetypes
    df_result["cluster_label"] = df_result["cluster_id"].map(archetypes)
    
    return df_result, kmeans, archetypes

def assign_archetypes(centroids, feature_names):
    """
    Maps cluster centroids to football tactical archetypes using simple score heuristics.
    Archetypes:
      - 'Clinical Finishers'
      - 'Creative Playmakers'
      - 'Dynamic Wingers'
      - 'Ball Winners'
      - 'Defensive Anchors'
    """
    # Create mapping of feature name to index for centroids array
    feat_to_idx = {name: idx for idx, name in enumerate(feature_names)}
    
    # We want to match each centroid (cluster index) to one of the 5 archetypes.
    # We will score each cluster for each profile.
    n_clusters = len(centroids)
    scores = {
        "Clinical Finishers": [],
        "Creative Playmakers": [],
        "Dynamic Wingers": [],
        "Ball Winners": [],
        "Defensive Anchors": []
    }
    
    for c_idx in range(n_clusters):
        c = centroids[c_idx]
        
        # Extract centroid values for features
        goals = c[feat_to_idx["goals"]]
        assists = c[feat_to_idx["assists"]]
        passing = c[feat_to_idx["passing_accuracy"]]
        tackles = c[feat_to_idx["tackles"]]
        interceptions = c[feat_to_idx["interceptions"]]
        dribbles = c[feat_to_idx["dribbles"]]
        
        # Calculate archetype scores
        scores["Clinical Finishers"].append(goals * 3.0 + dribbles * 0.5 - tackles - interceptions)
        scores["Dynamic Wingers"].append(dribbles * 2.5 + assists * 1.5 + goals * 1.0 - tackles)
        scores["Creative Playmakers"].append(passing * 2.0 + assists * 3.0 - tackles - interceptions)
        scores["Ball Winners"].append(tackles * 3.0 + interceptions * 2.0 - goals * 2.0 - dribbles)
        scores["Defensive Anchors"].append(interceptions * 3.0 + tackles * 1.5 - dribbles * 2.0 - goals * 2.0)

    # Resolve matchups so each cluster gets a unique archetype
    # We will match greedily by finding the maximum score in the scores grid.
    cluster_mapping = {}
    available_clusters = list(range(n_clusters))
    available_archetypes = list(scores.keys())
    
    while len(available_clusters) > 0 and len(available_archetypes) > 0:
        best_score = -9999
        best_cluster = None
        best_arch = None
        
        for arch in available_archetypes:
            for c_idx in available_clusters:
                score = scores[arch][c_idx]
                if score > best_score:
                    best_score = score
                    best_cluster = c_idx
                    best_arch = arch
                    
        cluster_mapping[best_cluster] = best_arch
        available_clusters.remove(best_cluster)
        available_archetypes.remove(best_arch)
        
    # Fallback in case anything is not mapped (unlikely)
    for c_idx in range(n_clusters):
        if c_idx not in cluster_mapping:
            cluster_mapping[c_idx] = f"Archetype {c_idx}"
            
    return cluster_mapping

def get_cluster_stats(df_clustered):
    """
    Returns average metrics per cluster to describe cluster characteristics.
    """
    cols_to_groupby = ["cluster_label"] + ML_FEATURES + ["market_value", "age"]
    # Group by label and compute mean
    stats = df_clustered[cols_to_groupby].groupby("cluster_label").mean()
    # Also get the size of each cluster
    sizes = df_clustered.groupby("cluster_label").size()
    stats["count"] = sizes
    return stats
