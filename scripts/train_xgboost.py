"""
Model comparison script: Random Forest vs XGBoost.

This script trains both models on the same data and compares their performance
to determine which one is better for congestion prediction.

Usage:
    pip install xgboost
    python scripts/train_xgboost.py
"""

import sys
from pathlib import Path
from time import time

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("XGBoost not installed. Run: pip install xgboost")

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import (
    MODELING_DATASET_FILE,
    FEATURES_FILE,
    WEATHER_DATA_FILE,
    MODELS_DIR,
    ALL_MODEL_FEATURES,
    RANDOM_FOREST_PARAMS,
    RAIN_THRESHOLD_INCHES,
    COLD_THRESHOLD_F,
    HOT_THRESHOLD_F
)
from src.utils import setup_logging, print_header, print_separator

# Setup logging
logger = setup_logging("model_comparison")


def prepare_data():
    """Prepare the training data with all features."""
    logger.info("Loading data...")
    
    # Load main dataset
    df = pd.read_csv(MODELING_DATASET_FILE)
    
    # Load and prepare weather stats (same logic as retrain_spatial.py)
    complaints = pd.read_csv(FEATURES_FILE)
    complaints['date'] = pd.to_datetime(pd.to_datetime(complaints['created_date']).dt.date)
    
    weather = pd.read_csv(WEATHER_DATA_FILE)
    weather['date'] = pd.to_datetime(weather['date'])
    
    # Merge and create weather flags
    cw = complaints.merge(weather, on='date', how='left')
    cw['is_rainy'] = (cw['precipitation'] > RAIN_THRESHOLD_INCHES).astype(int)
    cw['is_cold'] = (cw['temp_high'] < COLD_THRESHOLD_F).astype(int)
    cw['is_hot'] = (cw['temp_high'] > HOT_THRESHOLD_F).astype(int)
    
    # Aggregate weather stats by hour and day
    weather_stats = cw.groupby(['hour', 'day_of_week']).agg({
        'temp_high': 'mean',
        'precipitation': 'mean',
        'is_rainy': 'mean',
        'is_cold': 'mean',
        'is_hot': 'mean'
    }).reset_index()
    
    weather_stats.rename(columns={
        'temp_high': 'avg_temp',
        'precipitation': 'avg_precip',
        'is_rainy': 'pct_rainy',
        'is_cold': 'pct_cold',
        'is_hot': 'pct_hot'
    }, inplace=True)
    
    # Merge with main dataset
    final_df = df.merge(weather_stats, on=['hour', 'day_of_week'], how='left')
    
    X = final_df[ALL_MODEL_FEATURES]
    y = final_df['high_congestion']
    
    logger.info(f"Dataset: {len(X):,} samples, {len(ALL_MODEL_FEATURES)} features")
    logger.info(f"Class distribution: {y.value_counts().to_dict()}")
    
    return X, y


def train_and_evaluate(model, X_train, X_test, y_train, y_test, model_name):
    """Train a model and return evaluation metrics."""
    logger.info(f"Training {model_name}...")
    
    start_time = time()
    model.fit(X_train, y_train)
    train_time = time() - start_time
    
    # Predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    metrics = {
        "model": model_name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "train_time": train_time
    }
    
    return model, metrics


def main():
    """Main comparison function."""
    print_header("MODEL COMPARISON: Random Forest vs XGBoost")
    
    if not XGBOOST_AVAILABLE:
        logger.error("XGBoost is required. Install with: pip install xgboost")
        return
    
    # Prepare data
    X, y = prepare_data()
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    logger.info(f"Train size: {len(X_train):,}, Test size: {len(X_test):,}")
    
    results = []
    
    # Train Random Forest
    print_separator("-", 50)
    rf = RandomForestClassifier(**RANDOM_FOREST_PARAMS)
    rf_model, rf_metrics = train_and_evaluate(
        rf, X_train, X_test, y_train, y_test, "Random Forest"
    )
    results.append(rf_metrics)
    
    # Train XGBoost
    print_separator("-", 50)
    xgb = XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        n_jobs=-1,
        scale_pos_weight=len(y_train[y_train==0]) / len(y_train[y_train==1]),  # Handle imbalance
        eval_metric='logloss',
        verbosity=0
    )
    xgb_model, xgb_metrics = train_and_evaluate(
        xgb, X_train, X_test, y_train, y_test, "XGBoost"
    )
    results.append(xgb_metrics)
    
    # Display comparison
    print_separator("=", 50)
    print("\n📊 MODEL COMPARISON RESULTS\n")
    
    results_df = pd.DataFrame(results)
    results_df = results_df.set_index("model")
    
    # Format for display
    display_df = results_df.copy()
    for col in ["accuracy", "precision", "recall", "f1"]:
        display_df[col] = display_df[col].apply(lambda x: f"{x:.3f}")
    display_df["train_time"] = display_df["train_time"].apply(lambda x: f"{x:.2f}s")
    
    print(display_df.to_string())
    
    # Determine winner
    rf_f1 = rf_metrics["f1"]
    xgb_f1 = xgb_metrics["f1"]
    
    print("\n" + "="*50)
    if xgb_f1 > rf_f1:
        improvement = ((xgb_f1 - rf_f1) / rf_f1) * 100
        print(f"🏆 WINNER: XGBoost (+{improvement:.1f}% F1 improvement)")
        
        # Save XGBoost model
        import joblib
        model_path = MODELS_DIR / "xgboost_model.pkl"
        joblib.dump(xgb_model, model_path)
        logger.info(f"Saved XGBoost model to {model_path}")
        print(f"\n✅ XGBoost model saved to: {model_path}")
    else:
        print("🏆 WINNER: Random Forest (XGBoost did not improve)")
        print("\nKeeping existing Random Forest model.")
    
    print("="*50)
    
    # Feature importance comparison
    print("\n📈 TOP 5 FEATURE IMPORTANCES\n")
    
    print("Random Forest:")
    rf_importance = pd.Series(rf_model.feature_importances_, index=ALL_MODEL_FEATURES)
    for feat, imp in rf_importance.nlargest(5).items():
        print(f"  {feat:>15}: {imp:.3f}")
    
    print("\nXGBoost:")
    xgb_importance = pd.Series(xgb_model.feature_importances_, index=ALL_MODEL_FEATURES)
    for feat, imp in xgb_importance.nlargest(5).items():
        print(f"  {feat:>15}: {imp:.3f}")
    
    return results_df


if __name__ == "__main__":
    main()
