import os
import sys
# Add parent directory to path to resolve local imports when running streamlit directly
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pickle
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from src.preprocessing import ML_FEATURES, run_preprocessing_pipeline
from src.clustering import perform_clustering, get_cluster_stats
from src.recommendation import PlayerRecommender
from src.visualization import (
    plot_radar_comparison,
    plot_clusters_pca,
    plot_age_distribution,
    plot_market_value_distribution,
    plot_correlation_heatmap
)

# Page configuration
st.set_page_config(
    page_title="ProScout | Football Scouting & Recommendation",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inject modern styling with glassmorphism and custom styles
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=SF+Pro+Display:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: -apple-system, BlinkMacSystemFont, 'SF Pro Display', 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
    }
    
    /* Elegant stat cards */
    .stat-card {
        background: #F5F5F7;
        border: none;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        transition: transform 0.2s ease;
    }
    .stat-card:hover {
        transform: scale(1.02);
    }
    .stat-val {
        font-size: 1.8rem;
        font-weight: 700;
        color: #1D1D1F;
        margin-bottom: 5px;
        letter-spacing: -0.02em;
    }
    .stat-label {
        font-size: 0.85rem;
        color: #86868B;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        font-weight: 600;
    }
    
    /* Header styling */
    .main-title {
        color: #1D1D1F;
        font-size: 2.8rem;
        font-weight: 700;
        margin-bottom: 5px;
        letter-spacing: -0.03em;
    }
    
    .section-title {
        color: #1D1D1F;
        border-bottom: 1px solid #E5E5EA;
        padding-bottom: 10px;
        margin-bottom: 20px;
        font-weight: 600;
    }
    
    .page-subtitle {
        color: #86868B;
        font-size: 1.1rem;
        margin-bottom: 25px;
        font-weight: 400;
    }
    
    /* Info banners */
    .info-banner {
        background: #F5F5F7;
        border-left: 4px solid #0066CC;
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 25px;
    }
    
    /* Custom button states */
    div.stButton > button {
        background-color: #000000;
        color: #ffffff;
        border: none;
        border-radius: 20px;
        font-weight: 600;
        padding: 10px 24px;
        transition: background-color 0.2s ease;
    }
    div.stButton > button:hover {
        background-color: #333333;
        color: #ffffff;
    }
</style>
""", unsafe_allow_html=True)

# Helper function to load dataset and prepare ML structures
@st.cache_data
def get_dataset():
    # If the clean dataset doesn't exist, build it
    if not os.path.exists("data/players_clean.csv") or not os.path.exists("models/scaler.pkl"):
        run_preprocessing_pipeline()
        
    df_clean = pd.read_csv("data/players_clean.csv")
    
    # Check if cluster assignments exist in the clean CSV, if not perform clustering
    if "cluster_label" not in df_clean.columns:
        with open("models/scaler.pkl", "rb") as f:
            scaler = pickle.load(f)
        scaled_data = scaler.transform(df_clean[ML_FEATURES])
        scaled_df = pd.DataFrame(scaled_data, columns=ML_FEATURES, index=df_clean.index)
        
        df_clustered, _, _ = perform_clustering(df_clean, scaled_df)
        df_clustered.to_csv("data/players_clean.csv", index=False)
        df_clean = df_clustered
        
    # Recreate scaled dataframe
    with open("models/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    scaled_data = scaler.transform(df_clean[ML_FEATURES])
    scaled_df = pd.DataFrame(scaled_data, columns=ML_FEATURES, index=df_clean.index)
    
    return df_clean, scaled_df

# Load data once
try:
    df_clean, scaled_df = get_dataset()
    recommender = PlayerRecommender(df_clean, scaled_df)
except Exception as e:
    st.error(f"Error loading or preparing dataset: {e}")
    st.info("Make sure you generated the dataset first: python src/generate_data.py")
    st.stop()

# Persistent global branding
st.markdown("""
<div class="brand-header" style='display: flex; align-items: center; padding: 10px 0 20px 0; border-bottom: 1px solid #E5E5EA; margin-bottom: 20px;'>
    <h2 style='color:#1D1D1F; margin:0; font-weight: 700; letter-spacing: 1px;'>⚽ PROSCOUT</h2>
    <span style='color:#86868B; font-size:1.1rem; margin-left: 15px; font-weight: 400;'>| Player Intelligence System</span>
</div>
""", unsafe_allow_html=True)

# Sidebar navigation
page = st.sidebar.radio(
    "Navigation Menu",
    ["Dashboard Home", "Player Explorer", "Similar Player Finder", "Player Comparison", "Cluster Analysis"]
)

# ----------------- PAGE 1: HOME -----------------
if page == "Dashboard Home":
    st.markdown("<h1 class='main-title'>Football Scouting & Player Recommendation System</h1>", unsafe_allow_html=True)
    st.markdown("<p class='page-subtitle' style='font-size: 1.15rem;'>An end-to-end data science application leveraging machine learning (K-Means, Cosine Similarity) to analyze player attributes, identify scout archetypes, and locate statistical targets.</p>", unsafe_allow_html=True)
    
    # KPIs Row
    st.markdown("### Key Statistics")
    col1, col2, col3, col4, col5 = st.columns(5)
    
    num_players = len(df_clean)
    num_clubs = df_clean["club"].nunique()
    num_leagues = df_clean["league"].nunique()
    avg_age = df_clean["age"].mean()
    avg_val = df_clean["market_value"].mean()
    
    with col1:
        st.markdown(f"<div class='stat-card'><div class='stat-val'>{num_players}</div><div class='stat-label'>Players Profiled</div></div>", unsafe_allow_html=True)
    with col2:
        st.markdown(f"<div class='stat-card'><div class='stat-val'>{num_clubs}</div><div class='stat-label'>Clubs Represented</div></div>", unsafe_allow_html=True)
    with col3:
        st.markdown(f"<div class='stat-card'><div class='stat-val'>{num_leagues}</div><div class='stat-label'>Leagues covered</div></div>", unsafe_allow_html=True)
    with col4:
        st.markdown(f"<div class='stat-card'><div class='stat-val'>{avg_age:.1f}</div><div class='stat-label'>Average Age</div></div>", unsafe_allow_html=True)
    with col5:
        st.markdown(f"<div class='stat-card'><div class='stat-val'>€{avg_val:.1f}M</div><div class='stat-label'>Avg Market Value</div></div>", unsafe_allow_html=True)
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Main sections: Project Description and Methodology
    col_left, col_right = st.columns([1, 1])
    
    with col_left:
        st.markdown("<h3 class='section-title'>Overview & Features</h3>", unsafe_allow_html=True)
        st.markdown("""
        ProScout addresses modern football recruitment challenges by mapping on-field actions directly to statistical profiles.
        
        **Available Workflows:**
        * **Player Explorer**: Filter and search through players across 5 top leagues using tactical metrics.
        * **Similar Player Finder**: Provide a benchmark player, and the similarity engine returns the top 5 statistical matches.
        * **Player Comparison**: Side-by-side player visual profiling with dynamic difference metrics and overlays.
        * **Cluster Analysis**: Explore player groupings representing tactical roles (e.g. creative playmakers, clinical finishers).
        """)
        
        # Display sample players table
        st.markdown("#### Sample Player Profiles")
        sample_cols = ["player_name", "position", "club", "league", "market_value", "goals", "assists", "passing_accuracy"]
        st.dataframe(df_clean[sample_cols].head(5), use_container_width=True)
        
    with col_right:
        st.markdown("<h3 class='section-title'>Data Science Methodology</h3>", unsafe_allow_html=True)
        st.markdown("""
        This dashboard integrates several core data science concepts:
        
        1. **Data Imputation & Cleaning**:
           * Injected missing metrics are cleaned using **position-wise median imputation** rather than global medians, maintaining tactical statistics integrity.
        2. **Standardization**:
           * Performance variables (goals, tackles, passing, etc.) are standardized using `StandardScaler` to remove unit scale effects.
        3. **Cosine Similarity**:
           * Similarity engine computes the dot product of scaled vectors to calculate the cosine angle, providing distance metrics bounded between -1 and 1.
        4. **K-Means Clustering**:
           * Unsupervised K-Means ($K=5$) classifies players into tactical archetypes. Archetypes are auto-labeled using cluster centroid averages.
        5. **PCA (Principal Component Analysis)**:
           * Dimensionality reduction scales down the 7D feature space to 2D/3D for visualization.
        """)

# ----------------- PAGE 2: EXPLORER -----------------
elif page == "Player Explorer":
    st.markdown("<h1 class='main-title'>Player Explorer</h1>", unsafe_allow_html=True)
    st.markdown("<p class='page-subtitle'>Search and filter player profiles using demographic and performance metrics.</p>", unsafe_allow_html=True)
    
    # Sidebar-like panel inside page for filters
    col_filters, col_table = st.columns([1, 3])
    
    with col_filters:
        st.markdown("#### Search & Filters")
        search_query = st.text_input("Search by Name", "")
        
        leagues = st.multiselect("Leagues", options=sorted(df_clean["league"].unique()), default=sorted(df_clean["league"].unique()))
        positions = st.multiselect("Positions", options=sorted(df_clean["position"].unique()), default=sorted(df_clean["position"].unique()))
        
        age_range = st.slider("Age Range", int(df_clean["age"].min()), int(df_clean["age"].max()), (int(df_clean["age"].min()), int(df_clean["age"].max())))
        val_range = st.slider("Market Value (€M)", float(df_clean["market_value"].min()), float(df_clean["market_value"].max()), (float(df_clean["market_value"].min()), float(df_clean["market_value"].max())))
        
        min_goals = st.number_input("Minimum Goals", min_value=0, max_value=int(df_clean["goals"].max()), value=0)
        min_assists = st.number_input("Minimum Assists", min_value=0, max_value=int(df_clean["assists"].max()), value=0)

    # Filter data
    filtered_df = df_clean.copy()
    if search_query:
        filtered_df = filtered_df[filtered_df["player_name"].str.contains(search_query, case=False)]
    if leagues:
        filtered_df = filtered_df[filtered_df["league"].isin(leagues)]
    if positions:
        filtered_df = filtered_df[filtered_df["position"].isin(positions)]
        
    filtered_df = filtered_df[
        (filtered_df["age"] >= age_range[0]) & (filtered_df["age"] <= age_range[1]) &
        (filtered_df["market_value"] >= val_range[0]) & (filtered_df["market_value"] <= val_range[1]) &
        (filtered_df["goals"] >= min_goals) & (filtered_df["assists"] >= min_assists)
    ]
    
    with col_table:
        st.markdown(f"##### Results ({len(filtered_df)} players found)")
        # Show table
        display_df = filtered_df.drop(columns=["cluster_id"])
        # Format columns for display
        st.dataframe(
            display_df.style.format({
                "market_value": "€{:.1f}M",
                "minutes_played": "{:.0f}",
                "passing_accuracy": "{:.1f}%",
                "tackles": "{:.1f}",
                "interceptions": "{:.1f}",
                "dribbles": "{:.1f}"
            }),
            use_container_width=True,
            height=600
        )

# ----------------- PAGE 3: SIMILAR PLAYER FINDER -----------------
elif page == "Similar Player Finder":
    st.markdown("<h1 class='main-title'>Similar Player Finder</h1>", unsafe_allow_html=True)
    st.markdown("<p class='page-subtitle'>Input a benchmark player to identify targets with the closest statistical attributes.</p>", unsafe_allow_html=True)
    
    col_input, col_results = st.columns([1, 2])
    
    with col_input:
        st.markdown("#### Benchmark Config")
        # Player selector
        player_list = sorted(df_clean["player_name"].tolist())
        target_player = st.selectbox("Select Benchmark Player", player_list)
        
        # Display selected player profile details
        p_row = df_clean[df_clean["player_name"] == target_player].iloc[0]
        st.markdown("---")
        st.markdown(f"### {p_row['player_name']}")
        st.markdown(f"**Club**: {p_row['club']} ({p_row['league']})")
        st.markdown(f"**Position**: {p_row['position']} | **Age**: {p_row['age']}")
        st.markdown(f"**Market Value**: €{p_row['market_value']:.1f}M")
        st.markdown(f"**Archetype Label**: *{p_row['cluster_label']}*")
        
        # Filters for recommendations
        st.markdown("---")
        st.markdown("#### Scouting Target Filters")
        apply_pos_filter = st.checkbox("Match Benchmark Position")
        restrict_age = st.checkbox("Apply Maximum Age Filter")
        max_age_val = st.slider("Max Age Limit", 18, 36, 25, disabled=not restrict_age)
        
        restrict_budget = st.checkbox("Apply Maximum Budget (Market Value)")
        max_val_val = st.slider("Max Value Limit (€M)", 1.0, 150.0, float(p_row["market_value"]), disabled=not restrict_budget)
        
        limit_league = st.multiselect("Restrict to Leagues", options=df_clean["league"].unique().tolist(), default=[])
        
    with col_results:
        # Run recommendations
        pos_filter = p_row["position"] if apply_pos_filter else None
        age_filter = max_age_val if restrict_age else None
        budget_filter = max_val_val if restrict_budget else None
        league_filter = limit_league if len(limit_league) > 0 else None
        
        try:
            recs = recommender.get_recommendations(
                player_name=target_player,
                top_n=5,
                position_filter=pos_filter,
                max_age=age_filter,
                max_value=budget_filter,
                league_filter=league_filter
            )
            
            st.markdown(f"### Top 5 Statistical Matches")
            
            # Show cards for the recommendations
            for idx, rec in enumerate(recs):
                rec_name = rec["recommended_player"]
                rec_score = rec["similarity_score"]
                details = rec["details"]
                
                # Expandable details
                with st.expander(f"{idx+1}. {rec_name} - Similarity: {rec_score*100:.1f}%"):
                    # Split into stats and info
                    sub_col1, sub_col2 = st.columns([1, 1])
                    with sub_col1:
                        st.markdown(f"**Club**: {details['club']} ({details['league']})")
                        st.markdown(f"**Position**: {details['position']} | **Age**: {details['age']}")
                        st.markdown(f"**Market Value**: €{details['market_value']:.1f}M")
                        st.markdown(f"**Tactical Archetype**: *{details['cluster_label']}*")
                        
                    with sub_col2:
                        # Draw a mini visual comparison table
                        expl = recommender.explain_similarity(target_player, rec_name)
                        expl_df = pd.DataFrame(expl).T
                        # Format explaining df
                        expl_df_disp = expl_df[["source_value", "recommended_value", "difference", "percent_difference"]]
                        expl_df_disp.columns = [f"{target_player}", f"{rec_name}", "Diff", "% Diff"]
                        st.dataframe(expl_df_disp.style.format(precision=1), use_container_width=True)
            
            # Radar comparison against the top match
            if len(recs) > 0:
                top_rec = recs[0]["recommended_player"]
                st.markdown(f"### Profile Overlay: {target_player} vs. {top_rec} (Top Match)")
                radar_fig = plot_radar_comparison(df_clean, target_player, top_rec)
                st.plotly_chart(radar_fig, use_container_width=True)
                
        except Exception as e:
            st.error(f"Error producing recommendations: {e}")

# ----------------- PAGE 4: COMPARISON -----------------
elif page == "Player Comparison":
    st.markdown("<h1 class='main-title'>Player Comparison</h1>", unsafe_allow_html=True)
    st.markdown("<p class='page-subtitle'>Compare two players side-by-side to understand relative strengths and differences.</p>", unsafe_allow_html=True)
    
    col_p1, col_p2 = st.columns(2)
    player_list = sorted(df_clean["player_name"].tolist())
    
    with col_p1:
        p1 = st.selectbox("Select Player A", player_list, index=0)
        p1_row = df_clean[df_clean["player_name"] == p1].iloc[0]
        st.markdown(f"### {p1_row['player_name']}")
        st.markdown(f"**Club**: {p1_row['club']} ({p1_row['league']}) | **Age**: {p1_row['age']}")
        st.markdown(f"**Position**: {p1_row['position']} | **Value**: €{p1_row['market_value']:.1f}M")
        
    with col_p2:
        # Default to a different player
        p2_default_idx = min(1, len(player_list) - 1)
        p2 = st.selectbox("Select Player B", player_list, index=p2_default_idx)
        p2_row = df_clean[df_clean["player_name"] == p2].iloc[0]
        st.markdown(f"### {p2_row['player_name']}")
        st.markdown(f"**Club**: {p2_row['club']} ({p2_row['league']}) | **Age**: {p2_row['age']}")
        st.markdown(f"**Position**: {p2_row['position']} | **Value**: €{p2_row['market_value']:.1f}M")
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    col_radar, col_table_comp = st.columns([1, 1])
    
    with col_radar:
        st.markdown("#### Radar Profile Overlay")
        radar_comp = plot_radar_comparison(df_clean, p1, p2)
        st.plotly_chart(radar_comp, use_container_width=True)
        
    with col_table_comp:
        st.markdown("#### Attribute Comparison Details")
        
        # Calculate comparison dataframe
        comp_data = []
        for feat in ML_FEATURES:
            v1 = float(p1_row[feat])
            v2 = float(p2_row[feat])
            diff = v2 - v1
            
            if v1 > 0:
                pct = (diff / v1) * 100
            else:
                pct = 100.0 if v2 > 0 else 0.0
                
            comp_data.append({
                "Attribute": feat.replace("_", " ").title(),
                f"{p1} Stats": v1,
                f"{p2} Stats": v2,
                "Difference": diff,
                "Percent Difference": f"{pct:+.1f}%"
            })
            
        comp_df = pd.DataFrame(comp_data)
        st.dataframe(
            comp_df.style.format({
                f"{p1} Stats": "{:.1f}",
                f"{p2} Stats": "{:.1f}",
                "Difference": "{:+.1f}"
            }),
            use_container_width=True,
            height=400
        )

# ----------------- PAGE 5: CLUSTERING -----------------
elif page == "Cluster Analysis":
    st.markdown("<h1 class='main-title'>Player Cluster & Archetype Analysis</h1>", unsafe_allow_html=True)
    st.markdown("<p class='page-subtitle'>Analyze clusters to understand tactical playing styles generated via K-Means ($K=5$).</p>", unsafe_allow_html=True)
    
    col_desc, col_plot = st.columns([1, 2])
    
    with col_desc:
        st.markdown("### Tactical Archetypes Overview")
        st.markdown("""
        Unsupervised K-Means clustering aggregates players sharing similar attribute profiles, mapping them to 5 tactical roles:
        
        * <span style='color:#000000;'>●</span> **Clinical Finishers**: Characterized by extremely high goal scoring outputs, lower assist rates, and minimal defensive actions.
        * <span style='color:#0066CC;'>●</span> **Creative Playmakers**: Highlighted by high passing accuracy, high assists output, and moderate goal contributions.
        * <span style='color:#FF9500;'>●</span> **Dynamic Wingers**: Possess exceptionally high dribble statistics combined with moderate-to-high goals and assists.
        * <span style='color:#FF3B30;'>●</span> **Ball Winners**: Feature high volume tackles and interceptions, primarily mid-block ball winners or fullbacks.
        * <span style='color:#8E8E93;'>●</span> **Defensive Anchors**: Low goals and assists output with highly concentrated tackles and interceptions, primarily center backs.
        """, unsafe_allow_html=True)
        
        # Cluster sizes table
        cluster_sizes = df_clean["cluster_label"].value_counts().reset_index()
        cluster_sizes.columns = ["Archetype", "Player Count"]
        st.markdown("#### Player Count per Archetype")
        st.table(cluster_sizes)
        
    with col_plot:
        # Toggle 2D vs 3D Bubble PCA
        pca_dim = st.radio("PCA Visualization Dimensions", ["2D Scatter", "3D Bubble (WebGL Fallback)"], horizontal=True)
        pca_fig = plot_clusters_pca(df_clean, scaled_df, dimensions=pca_dim)
        st.plotly_chart(pca_fig, use_container_width=True)
        
    # Heatmaps or Cluster Characteristics
    st.markdown("<br><h3 class='section-title'>Statistical Traits of Archetypes</h3>", unsafe_allow_html=True)
    cluster_stats = get_cluster_stats(df_clean)
    
    # Clean column names for output
    cluster_stats_disp = cluster_stats.rename(columns={col: col.replace("_", " ").title() for col in cluster_stats.columns})
    st.dataframe(
        cluster_stats_disp.style.format("{:.2f}").background_gradient(cmap="coolwarm", axis=0),
        use_container_width=True
    )
    
    st.markdown("<br><h3 class='section-title'>Exploratory Data Analysis Plots</h3>", unsafe_allow_html=True)
    col_dist1, col_dist2 = st.columns(2)
    
    with col_dist1:
        st.plotly_chart(plot_age_distribution(df_clean), use_container_width=True)
    with col_dist2:
        st.plotly_chart(plot_market_value_distribution(df_clean), use_container_width=True)
        
    st.markdown("<br>", unsafe_allow_html=True)
    st.plotly_chart(plot_correlation_heatmap(df_clean), use_container_width=True)
