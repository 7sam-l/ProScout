import os
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Features that represent on-field performance for scaling
ML_FEATURES = ["goals", "assists", "minutes_played", "passing_accuracy", "tackles", "interceptions", "dribbles"]

def load_data(filepath):
    """Loads the raw players CSV file."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found at {filepath}")
    return pd.read_csv(filepath)

def clean_data(df):
    """
    Cleans the raw dataset.
    Imputes missing values using position-wise medians for numerical columns
    to ensure tactical profiles are preserved (e.g., defenders get defender-like tackles).
    """
    df_clean = df.copy()
    
    # 1. Impute missing numerical columns using the median of their respective positions
    numeric_cols = ["passing_accuracy", "tackles", "interceptions"]
    for col in numeric_cols:
        if col in df_clean.columns:
            # Group by position and fill NaNs with the group median
            df_clean[col] = df_clean.groupby("position")[col].transform(lambda x: x.fillna(x.median()))
            
            # If any NaNs remain (e.g., if a position is completely empty, which is unlikely), fill with global median
            df_clean[col] = df_clean[col].fillna(df_clean[col].median())
            
    # 2. Impute missing categorical columns if any (e.g., nationality, club, league) with mode or 'Unknown'
    categorical_cols = ["nationality", "club", "league", "position"]
    for col in categorical_cols:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].fillna("Unknown")
            
    return df_clean

def fit_scale_features(df_clean, scaler_path="models/scaler.pkl"):
    """
    Standardizes performance metrics and saves the fitted scaler to disk.
    Returns the scaled features as a DataFrame.
    """
    # Create models directory if it doesn't exist
    os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
    
    # Extract numerical features for ML
    features_data = df_clean[ML_FEATURES]
    
    # Initialize and fit StandardScaler
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(features_data)
    
    # Save the scaler to disk
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
        
    # Create a DataFrame of scaled features
    scaled_df = pd.DataFrame(scaled_data, columns=ML_FEATURES, index=df_clean.index)
    
    return scaled_df, scaler

def run_preprocessing_pipeline(raw_path="data/players.csv", clean_path="data/players_clean.csv", scaler_path="models/scaler.pkl"):
    """Runs the full preprocessing pipeline from end to end."""
    print("Starting data preprocessing pipeline...")
    
    # Load
    df = load_data(raw_path)
    
    # Clean
    df_clean = clean_data(df)
    
    # Save clean dataset
    df_clean.to_csv(clean_path, index=False)
    print(f"Cleaned dataset saved to {clean_path}")
    
    # Scale
    scaled_df, scaler = fit_scale_features(df_clean, scaler_path)
    print(f"StandardScaler fitted and saved to {scaler_path}")
    
    return df_clean, scaled_df

if __name__ == "__main__":
    run_preprocessing_pipeline()
