import pandas as pd
# import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from src.preprocessing import ML_FEATURES

class PlayerRecommender:
    def __init__(self, df_clean, scaled_df):
        """
        Initializes the recommender with clean data and scaled features.
        df_clean: Pandas DataFrame containing the clean dataset.
        scaled_df: Pandas DataFrame containing standardized ML features.
        """
        self.df_clean = df_clean.copy()
        self.scaled_df = scaled_df.copy()
        
        # Calculate full cosine similarity matrix
        self.similarity_matrix = cosine_similarity(self.scaled_df)
        
        # Create a mapping from player name to index
        self.name_to_idx = {name.lower(): idx for idx, name in enumerate(self.df_clean["player_name"])}

    def get_recommendations(self, player_name, top_n=5, position_filter=None, max_age=None, max_value=None, league_filter=None):
        """
        Generates top_n player recommendations based on cosine similarity.
        Can apply filters on the candidate pool (e.g. age, market value, position, league).
        """
        player_name_lower = player_name.lower()
        if player_name_lower not in self.name_to_idx:
            raise ValueError(f"Player '{player_name}' not found in the dataset.")
            
        player_idx = self.name_to_idx[player_name_lower]
        player_row = self.df_clean.iloc[player_idx]
        
        # Get similarities for the target player
        similarities = self.similarity_matrix[player_idx]
        
        # Create a DataFrame of candidate players with their similarity score
        candidates = self.df_clean.copy()
        candidates["similarity_score"] = similarities
        
        # Exclude the target player from recommendations
        candidates = candidates.drop(player_idx)
        
        # Apply filters
        if position_filter:
            if isinstance(position_filter, list):
                candidates = candidates[candidates["position"].isin(position_filter)]
            else:
                candidates = candidates[candidates["position"] == position_filter]
                
        if max_age:
            candidates = candidates[candidates["age"] <= max_age]
            
        if max_value:
            candidates = candidates[candidates["market_value"] <= max_value]
            
        if league_filter:
            if isinstance(league_filter, list):
                candidates = candidates[candidates["league"].isin(league_filter)]
            else:
                candidates = candidates[candidates["league"] == league_filter]
                
        # Sort by similarity score descending and pick top N
        recommendations = candidates.sort_values(by="similarity_score", ascending=False).head(top_n)
        
        # Format the output matching our data model:
        # source_player, recommended_player, similarity_score
        results = []
        for _, rec_row in recommendations.iterrows():
            results.append({
                "source_player": player_row["player_name"],
                "recommended_player": rec_row["player_name"],
                "similarity_score": float(rec_row["similarity_score"]),
                "details": rec_row.to_dict() # include full details for UI displays
            })
            
        return results

    def explain_similarity(self, source_player_name, recommended_player_name):
        """
        Provides a feature-by-feature comparison explaining the similarity score.
        Calculates the percentage difference in performance metrics.
        """
        src_lower = source_player_name.lower()
        rec_lower = recommended_player_name.lower()
        
        if src_lower not in self.name_to_idx or rec_lower not in self.name_to_idx:
            raise ValueError("One or both players not found in the dataset.")
            
        src_idx = self.name_to_idx[src_lower]
        rec_idx = self.name_to_idx[rec_lower]
        
        src_stats = self.df_clean.iloc[src_idx][ML_FEATURES]
        rec_stats = self.df_clean.iloc[rec_idx][ML_FEATURES]
        
        explanation = {}
        for feature in ML_FEATURES:
            src_val = float(src_stats[feature])
            rec_val = float(rec_stats[feature])
            
            # Percent difference relative to source player
            if src_val > 0:
                pct_diff = round(((rec_val - src_val) / src_val) * 100, 1)
            else:
                pct_diff = 100.0 if rec_val > 0 else 0.0
                
            explanation[feature] = {
                "source_value": src_val,
                "recommended_value": rec_val,
                "difference": round(rec_val - src_val, 2),
                "percent_difference": pct_diff
            }
            
        return explanation
