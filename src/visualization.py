import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from src.preprocessing import ML_FEATURES

def plot_radar_comparison(df_clean, player1_name, player2_name=None):
    """
    Generates a Plotly radar chart comparing one or two players on key performance metrics.
    Metrics are min-max scaled dynamically for visual clarity (so high passing and low goals don't distort).
    """
    # Filter players
    p1_idx = df_clean[df_clean["player_name"].str.lower() == player1_name.lower()].index
    if len(p1_idx) == 0:
        raise ValueError(f"Player '{player1_name}' not found.")
    p1_data = df_clean.loc[p1_idx[0]]
    
    p2_data = None
    if player2_name:
        p2_idx = df_clean[df_clean["player_name"].str.lower() == player2_name.lower()].index
        if len(p2_idx) == 0:
            raise ValueError(f"Player '{player2_name}' not found.")
        p2_data = df_clean.loc[p2_idx[0]]
        
    # Scale ML features locally between 0 and 1 using min-max of the dataset for proper visualization
    # This prevents high range features (like minutes_played or passing_accuracy) from dominating low range ones (like goals or assists)
    scaled_p1 = {}
    scaled_p2 = {}
    
    for feat in ML_FEATURES:
        feat_min = df_clean[feat].min()
        feat_max = df_clean[feat].max()
        feat_range = feat_max - feat_min if feat_max - feat_min > 0 else 1.0
        
        scaled_p1[feat] = (p1_data[feat] - feat_min) / feat_range
        if p2_data is not None:
            scaled_p2[feat] = (p2_data[feat] - feat_min) / feat_range

    # Build radar chart
    fig = go.Figure()
    
    # Format labels for the chart
    labels = [feat.replace("_", " ").title() for feat in ML_FEATURES]
    # Add first element to end to close the radar loop
    labels_closed = labels + [labels[0]]
    
    # Player 1 values
    p1_vals = [scaled_p1[f] for f in ML_FEATURES]
    p1_vals_closed = p1_vals + [p1_vals[0]]
    
    p1_hover = [f"{feat.replace('_', ' ').title()}: {p1_data[feat]:.1f}" for feat in ML_FEATURES]
    p1_hover_closed = p1_hover + [p1_hover[0]]
    
    # Dark modern colors
    fig.add_trace(go.Scatterpolar(
        r=p1_vals_closed,
        theta=labels_closed,
        fill="toself",
        name=p1_data["player_name"],
        text=p1_hover_closed,
        hoverinfo="name+text",
        line_color="#000000",
        fillcolor="rgba(0, 0, 0, 0.15)"
    ))
    
    if p2_data is not None:
        p2_vals = [scaled_p2[f] for f in ML_FEATURES]
        p2_vals_closed = p2_vals + [p2_vals[0]]
        
        p2_hover = [f"{feat.replace('_', ' ').title()}: {p2_data[feat]:.1f}" for feat in ML_FEATURES]
        p2_hover_closed = p2_hover + [p2_hover[0]]
        
        fig.add_trace(go.Scatterpolar(
            r=p2_vals_closed,
            theta=labels_closed,
            fill="toself",
            name=p2_data["player_name"],
            text=p2_hover_closed,
            hoverinfo="name+text",
            line_color="#0066CC",
            fillcolor="rgba(0, 102, 204, 0.15)"
        ))
        
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1],
                showticklabels=False,
                gridcolor="rgba(0, 0, 0, 0.1)",
                linecolor="rgba(0, 0, 0, 0.1)"
            ),
            angularaxis=dict(
                gridcolor="rgba(0, 0, 0, 0.1)",
                linecolor="rgba(0, 0, 0, 0.1)"
            ),
            gridshape="linear",
            bgcolor="rgba(0,0,0,0)"
        ),
        showlegend=True,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(size=12)
    )
    
    return fig

def plot_clusters_pca(df_clustered, scaled_df, dimensions=2):
    """
    Applies PCA to reduce scaled features and plots them in a 2D or 3D interactive scatter plot.
    """
    n_components = 2 if (dimensions == "2D Scatter" or dimensions == 2) else 3
    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(scaled_df)
    
    plot_df = df_clustered.copy()
    plot_df["PC1"] = pca_result[:, 0]
    plot_df["PC2"] = pca_result[:, 1]
    
    # Calculate explained variance ratio
    explained_var = pca.explained_variance_ratio_ * 100
    
    # Custom color palette for cluster labels
    color_map = {
        "Clinical Finishers": "#000000",      
        "Creative Playmakers": "#0066CC",     
        "Dynamic Wingers": "#FF9500",         
        "Ball Winners": "#FF3B30",            
        "Defensive Anchors": "#8E8E93"        
    }
    
    is_2d = (dimensions == "2D Scatter" or dimensions == 2)
    
    if is_2d:
        fig = px.scatter(
            plot_df,
            x="PC1",
            y="PC2",
            color="cluster_label",
            color_discrete_map=color_map,
            hover_name="player_name",
            hover_data={
                "age": True,
                "club": True,
                "league": True,
                "position": True,
                "market_value": True,
                "goals": True,
                "assists": True,
                "passing_accuracy": True,
                "tackles": True,
                "interceptions": True,
                "dribbles": True,
                "cluster_label": False,
                "PC1": False,
                "PC2": False
            },
            title="Player Clusters (PCA Dimension Reduction)",
            labels={
                "PC1": f"PC1 ({explained_var[0]:.1f}% Variance Explained)",
                "PC2": f"PC2 ({explained_var[1]:.1f}% Variance Explained)",
                "cluster_label": "Player Archetype"
            }
        )
        fig.update_traces(marker=dict(size=8, opacity=0.85, line=dict(width=1, color="DarkSlateGrey")))
    else: # 3D Fallback (2D Bubble mapping PC3 to size)
        plot_df["PC3"] = pca_result[:, 2]
        
        # Normalize PC3 for size (must be strictly positive)
        min_pc3 = plot_df["PC3"].min()
        plot_df["PC3_size"] = plot_df["PC3"] - min_pc3 + 1.0 
        
        fig = px.scatter(
            plot_df,
            x="PC1",
            y="PC2",
            size="PC3_size",
            color="cluster_label",
            color_discrete_map=color_map,
            hover_name="player_name",
            hover_data={
                "age": True,
                "club": True,
                "league": True,
                "market_value": True,
                "goals": True,
                "assists": True,
                "passing_accuracy": True,
                "tackles": True,
                "interceptions": True,
                "cluster_label": False,
                "PC1": False,
                "PC2": False,
                "PC3": True,
                "PC3_size": False
            },
            title="Player Clusters (Feature Matrix Fallback: Bubble Size = PC3)",
            labels={
                "PC1": f"PC1 ({explained_var[0]:.1f}% var)",
                "PC2": f"PC2 ({explained_var[1]:.1f}% var)",
                "PC3": f"PC3 ({explained_var[2]:.1f}% var)",
                "cluster_label": "Player Archetype"
            }
        )
        fig.update_traces(marker=dict(opacity=0.75, line=dict(width=1, color="DarkSlateGrey")))
        
    fig.update_layout(
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, title=""),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )
    
    return fig

def plot_age_distribution(df):
    """Plots player age distribution using Plotly."""
    fig = px.histogram(
        df,
        x="age",
        color="position",
        nbins=20,
        title="Player Age Distribution by Position",
        labels={"age": "Age", "count": "Player Count", "position": "Position"},
        marginal="box",
        color_discrete_sequence=["#000000", "#0066CC", "#FF9500", "#FF3B30", "#8E8E93"]
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        bargap=0.05
    )
    return fig

def plot_market_value_distribution(df):
    """Plots market value distribution using Plotly."""
    fig = px.box(
        df,
        x="league",
        y="market_value",
        color="position",
        title="Market Value Distribution by League and Position",
        labels={"market_value": "Market Value (€ Millions)", "league": "League", "position": "Position"},
        color_discrete_sequence=["#000000", "#0066CC", "#FF9500", "#FF3B30", "#8E8E93"]
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )
    return fig

def plot_correlation_heatmap(df):
    """Generates correlation heatmap between numerical statistics."""
    corr = df[ML_FEATURES + ["age", "market_value"]].corr()
    
    fig = px.imshow(
        corr,
        text_auto=".2f",
        aspect="auto",
        color_continuous_scale=[[0, "#FFFFFF"], [0.5, "#8E8E93"], [1, "#0066CC"]],
        title="Player Performance Feature Correlation Matrix",
        labels=dict(color="Correlation Coefficient")
    )
    
    # Format labels to be human-readable
    labels = [f.replace("_", " ").title() for f in ML_FEATURES + ["age", "market value"]]
    fig.update_layout(
        xaxis=dict(tickmode="array", tickvals=list(range(len(labels))), ticktext=labels),
        yaxis=dict(tickmode="array", tickvals=list(range(len(labels))), ticktext=labels),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )
    return fig
