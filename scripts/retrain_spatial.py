"""
Retrain spatial-aware Random Forest model with weather features.

This script trains the enhanced congestion prediction model that includes
spatial (grid location) and weather features in addition to temporal features.

Usage:
    python scripts/retrain_spatial.py
"""

import sys
from pathlib import Path
from time import time

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import (
    MODELING_DATASET_FILE,
    FEATURES_FILE,
    WEATHER_DATA_FILE,
    ENHANCED_MODEL_FILE,
    ALL_MODEL_FEATURES,
    RANDOM_FOREST_PARAMS,
    RAIN_THRESHOLD_INCHES,
    COLD_THRESHOLD_F,
    HOT_THRESHOLD_F
)
from src.utils import setup_logging, save_model, print_header, print_separator

# Setup logging
logger = setup_logging("model_training")


def prepare_weather_stats(
    complaints_path: Path,
    weather_path: Path
) -> pd.DataFrame:
    """
    Prepare aggregated weather statistics by hour and day of week.
    
    Args:
        complaints_path: Path to complaints with features CSV.
        weather_path: Path to weather data CSV.
        
    Returns:
        DataFrame with weather statistics aggregated by (hour, day_of_week).
    """
    logger.info("Loading complaints and weather data...")
    
    complaints = pd.read_csv(complaints_path)
    complaints["date"] = pd.to_datetime(
        pd.to_datetime(complaints["created_date"]).dt.date
    )
    
    weather = pd.read_csv(weather_path)
    weather["date"] = pd.to_datetime(weather["date"])
    
    # Merge complaints with weather
    merged = complaints.merge(weather, on="date", how="left")
    
    # Create weather flags
    merged["is_rainy"] = (merged["precipitation"] > RAIN_THRESHOLD_INCHES).astype(int)
    merged["is_cold"] = (merged["temp_high"] < COLD_THRESHOLD_F).astype(int)
    merged["is_hot"] = (merged["temp_high"] > HOT_THRESHOLD_F).astype(int)
    
    # Aggregate by hour and day of week
    weather_stats = merged.groupby(["hour", "day_of_week"]).agg({
        "temp_high": "mean",
        "precipitation": "mean",
        "is_rainy": "mean",
        "is_cold": "mean",
        "is_hot": "mean"
    }).reset_index()
    
    weather_stats.rename(columns={
        "temp_high": "avg_temp",
        "precipitation": "avg_precip",
        "is_rainy": "pct_rainy",
        "is_cold": "pct_cold",
        "is_hot": "pct_hot"
    }, inplace=True)
    
    return weather_stats


def train_model(X: pd.DataFrame, y: pd.Series) -> RandomForestClassifier:
    """
    Train a Random Forest classifier with balanced class weights.
    
    Args:
        X: Feature DataFrame.
        y: Target Series.
        
    Returns:
        Trained RandomForestClassifier.
    """
    logger.info("Training Random Forest model...")
    logger.info(f"Features: {list(X.columns)}")
    logger.info(f"Training samples: {len(X):,}")
    
    start_time = time()
    
    model = RandomForestClassifier(**RANDOM_FOREST_PARAMS)
    model.fit(X, y)
    
    elapsed = time() - start_time
    logger.info(f"Training completed in {elapsed:.1f} seconds")
    
    return model


def evaluate_model(
    model: RandomForestClassifier,
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> dict:
    """
    Evaluate model performance on test set.
    
    Args:
        model: Trained classifier.
        X_test: Test features.
        y_test: Test labels.
        
    Returns:
        Dictionary of metric scores.
    """
    y_pred = model.predict(X_test)
    
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred)
    }
    
    return metrics


def main() -> None:
    """Main entry point for model training."""
    print_header("SPATIAL + WEATHER MODEL TRAINING")
    
    # Load main modeling dataset
    logger.info(f"Loading modeling dataset from {MODELING_DATASET_FILE}")
    df = pd.read_csv(MODELING_DATASET_FILE)
    logger.info(f"Loaded {len(df):,} samples")
    
    # Prepare weather statistics
    weather_stats = prepare_weather_stats(FEATURES_FILE, WEATHER_DATA_FILE)
    
    # Merge weather stats into dataset
    final_df = df.merge(weather_stats, on=["hour", "day_of_week"], how="left")
    
    # Prepare features and target
    X = final_df[ALL_MODEL_FEATURES]
    y = final_df["high_congestion"]
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train model
    model = train_model(X_train, y_train)
    
    # Evaluate
    print_separator("-", 50)
    print("Model Evaluation:")
    metrics = evaluate_model(model, X_test, y_test)
    
    for metric_name, score in metrics.items():
        print(f"  {metric_name.capitalize():>10}: {score:.3f}")
    
    # Feature importance
    print_separator("-", 50)
    print("Top 5 Feature Importances:")
    importances = pd.Series(
        model.feature_importances_,
        index=ALL_MODEL_FEATURES
    ).sort_values(ascending=False)
    
    for feature, importance in importances.head(5).items():
        print(f"  {feature:>15}: {importance:.3f}")
    
    # Save model
    save_model(model, ENHANCED_MODEL_FILE.name)
    logger.info(f"Model saved to {ENHANCED_MODEL_FILE}")
    
    print_header("TRAINING COMPLETE")


if __name__ == "__main__":
    main()
