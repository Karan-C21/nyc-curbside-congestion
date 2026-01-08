"""
Quick validation of the Step 5 modeling pipeline.

Trains a basic Random Forest model on temporal features only
to verify the modeling dataset is correctly structured.

Usage:
    python scripts/check_step5.py
"""

import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import MODELING_DATASET_FILE, TEMPORAL_FEATURES
from src.utils import setup_logging, print_header, print_separator

# Setup logging
logger = setup_logging("step5_check")


def main() -> None:
    """Main entry point for Step 5 validation."""
    print_header("STEP 5 VALIDATION")
    
    # Load data
    df = pd.read_csv(MODELING_DATASET_FILE)
    logger.info(f"Dataset: {len(df):,} rows")
    
    # Show target distribution
    print("\nTarget Distribution:")
    print_separator("-", 40)
    target_counts = df["high_congestion"].value_counts()
    for label, count in target_counts.items():
        pct = count / len(df) * 100
        label_str = "High Congestion" if label == 1 else "Normal"
        print(f"  {label_str:>15}: {count:>6,} ({pct:.1f}%)")
    
    # Prepare features
    X = df[TEMPORAL_FEATURES]
    y = df["high_congestion"]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train model
    logger.info("Training basic Random Forest model...")
    rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    rf.fit(X_train, y_train)
    
    # Evaluate
    y_pred = rf.predict(X_test)
    
    print("\nModel Performance:")
    print_separator("-", 40)
    print(f"  Accuracy:  {accuracy_score(y_test, y_pred):.3f}")
    print(f"  Precision: {precision_score(y_test, y_pred):.3f}")
    print(f"  Recall:    {recall_score(y_test, y_pred):.3f}")
    
    print_header("VALIDATION COMPLETE")


if __name__ == "__main__":
    main()
