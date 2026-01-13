"""
Train enhanced XGBoost model with holiday features.

This script retrains the XGBoost model including holiday and special day
features for improved prediction accuracy.

Usage:
    python scripts/train_enhanced_model.py
"""

import sys
from pathlib import Path
from time import time

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib

try:
    from xgboost import XGBClassifier
except ImportError:
    print("XGBoost not installed. Run: pip install xgboost")
    sys.exit(1)

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import (
    MODELING_DATASET_FILE,
    FEATURES_FILE,
    WEATHER_DATA_FILE,
    MODELS_DIR,
    ALL_MODEL_FEATURES,
    LEGACY_MODEL_FEATURES,
    RAIN_THRESHOLD_INCHES,
    COLD_THRESHOLD_F,
    HOT_THRESHOLD_F
)
from src.holidays import add_holiday_features
from src.utils import setup_logging

# Setup logging
logger = setup_logging("train_enhanced")


def prepare_data_with_holidays():
    """Prepare training data including holiday features."""
    logger.info("Loading data...")
    
    # Load main dataset
    df = pd.read_csv(MODELING_DATASET_FILE)
    
    # Load complaints with dates for holiday features
    complaints = pd.read_csv(FEATURES_FILE)
    complaints['created_date'] = pd.to_datetime(complaints['created_date'])
    complaints['date'] = complaints['created_date'].dt.date
    
    # Load weather data
    weather = pd.read_csv(WEATHER_DATA_FILE)
    weather['date'] = pd.to_datetime(weather['date'])
    
    # Merge complaints with weather
    complaints['date'] = pd.to_datetime(complaints['date'])
    cw = complaints.merge(weather, on='date', how='left')
    
    # Add weather flags
    cw['is_rainy'] = (cw['precipitation'] > RAIN_THRESHOLD_INCHES).astype(int)
    cw['is_cold'] = (cw['temp_high'] < COLD_THRESHOLD_F).astype(int)
    cw['is_hot'] = (cw['temp_high'] > HOT_THRESHOLD_F).astype(int)
    
    # Add holiday features
    logger.info("Adding holiday features...")
    cw = add_holiday_features(cw, date_column='date')
    
    # Aggregate weather and holiday stats by hour and day
    agg_stats = cw.groupby(['hour', 'day_of_week']).agg({
        'temp_high': 'mean',
        'precipitation': 'mean',
        'is_rainy': 'mean',
        'is_cold': 'mean',
        'is_hot': 'mean',
        'is_holiday': 'mean',
        'is_holiday_week': 'mean',
        'is_month_end': 'mean',
        'is_month_start': 'mean'
    }).reset_index()
    
    agg_stats.rename(columns={
        'temp_high': 'avg_temp',
        'precipitation': 'avg_precip',
        'is_rainy': 'pct_rainy',
        'is_cold': 'pct_cold',
        'is_hot': 'pct_hot'
    }, inplace=True)
    
    # Merge with main dataset
    final_df = df.merge(agg_stats, on=['hour', 'day_of_week'], how='left')
    
    # Fill any missing values
    for col in ALL_MODEL_FEATURES:
        if col in final_df.columns:
            final_df[col] = final_df[col].fillna(0)
    
    logger.info(f"Dataset: {len(final_df):,} samples, {len(ALL_MODEL_FEATURES)} features")
    
    X = final_df[ALL_MODEL_FEATURES]
    y = final_df['high_congestion']
    
    return X, y


def train_model(X, y):
    """Train XGBoost model with optimized parameters."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    logger.info(f"Train size: {len(X_train):,}, Test size: {len(X_test):,}")
    
    # Calculate scale_pos_weight for imbalanced classes
    scale_weight = len(y_train[y_train==0]) / len(y_train[y_train==1])
    
    model = XGBClassifier(
        n_estimators=150,
        max_depth=8,
        learning_rate=0.1,
        random_state=42,
        n_jobs=-1,
        scale_pos_weight=scale_weight,
        eval_metric='logloss',
        verbosity=0
    )
    
    logger.info("Training XGBoost with holiday features...")
    start_time = time()
    model.fit(X_train, y_train)
    train_time = time() - start_time
    logger.info(f"Training completed in {train_time:.2f}s")
    
    # Evaluate
    y_pred = model.predict(X_test)
    
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred)
    }
    
    return model, metrics


def main():
    print("=" * 60)
    print("TRAINING ENHANCED MODEL WITH HOLIDAY FEATURES")
    print("=" * 60)
    print()
    
    # Prepare data
    X, y = prepare_data_with_holidays()
    
    # Train model
    model, metrics = train_model(X, y)
    
    # Display results
    print("\n" + "=" * 60)
    print("MODEL PERFORMANCE")
    print("=" * 60)
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1 Score:  {metrics['f1']:.4f}")
    
    # Feature importance
    print("\n" + "=" * 60)
    print("FEATURE IMPORTANCE (Top 10)")
    print("=" * 60)
    
    importance = pd.Series(model.feature_importances_, index=ALL_MODEL_FEATURES)
    for feat, imp in importance.nlargest(10).items():
        print(f"  {feat:>18}: {imp:.4f}")
    
    # Save model
    model_path = MODELS_DIR / "xgboost_enhanced.pkl"
    joblib.dump(model, model_path)
    print(f"\n✅ Model saved to: {model_path}")
    
    # Also update the main model
    main_model_path = MODELS_DIR / "xgboost_model.pkl"
    joblib.dump(model, main_model_path)
    print(f"✅ Updated main model: {main_model_path}")
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    
    return metrics


if __name__ == "__main__":
    main()
