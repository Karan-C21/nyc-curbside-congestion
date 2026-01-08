"""
Feature engineering validation script.

Validates and processes temporal features from 311 truck complaint data.
Creates the complaints_with_features.csv dataset for downstream analysis.

Usage:
    python scripts/validate_features.py
"""

import sys
from pathlib import Path

import pandas as pd

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import FILTERED_DATA_FILE, FEATURES_FILE
from src.utils import (
    setup_logging,
    add_temporal_features,
    get_day_name,
    print_header,
    print_separator
)

# Setup logging
logger = setup_logging("feature_validation")


def validate_feature_ranges(df: pd.DataFrame) -> bool:
    """
    Validate that all extracted features have expected ranges.
    
    Args:
        df: DataFrame with temporal features.
        
    Returns:
        True if all validations pass.
        
    Raises:
        AssertionError: If any validation fails.
    """
    assert df["hour"].min() >= 0 and df["hour"].max() <= 23, "Hour out of range!"
    assert df["day_of_week"].min() >= 0 and df["day_of_week"].max() <= 6, "Day out of range!"
    assert set(df["is_weekend"].unique()) <= {0, 1}, "Weekend flag invalid!"
    assert set(df["is_rush_hour"].unique()) <= {0, 1}, "Rush hour flag invalid!"
    assert df["month"].min() >= 1 and df["month"].max() <= 12, "Month out of range!"
    
    return True


def print_day_breakdown(df: pd.DataFrame) -> None:
    """Print complaint counts by day of week."""
    print("\nDay of Week Breakdown:")
    print_separator("-", 50)
    
    day_counts = df["day_of_week"].value_counts().sort_index()
    
    for day_num, count in day_counts.items():
        day_name = get_day_name(day_num)
        pct = (count / len(df)) * 100
        weekend_marker = " (WEEKEND)" if day_num >= 5 else ""
        print(f"  {day_name:>9}: {count:>6,} ({pct:>4.1f}%){weekend_marker}")
    
    weekday_avg = day_counts[:5].mean()
    weekend_avg = day_counts[5:].mean()
    weekend_drop = (1 - weekend_avg / weekday_avg) * 100
    
    print(f"\n  Weekday average: {weekday_avg:,.0f}")
    print(f"  Weekend average: {weekend_avg:,.0f}")
    print(f"  Weekend drop:    {weekend_drop:.1f}%")


def print_hour_breakdown(df: pd.DataFrame) -> None:
    """Print top 5 busiest hours for complaints."""
    print("\nBusiest Hours:")
    print_separator("-", 50)
    
    hour_counts = df["hour"].value_counts().sort_index()
    top_hours = hour_counts.nlargest(5)
    
    for hour, count in top_hours.items():
        if hour == 0:
            time_str = "12:00 AM"
        elif hour < 12:
            time_str = f"{hour}:00 AM"
        elif hour == 12:
            time_str = "12:00 PM"
        else:
            time_str = f"{hour-12}:00 PM"
        print(f"  {time_str:>8} (hour {hour:>2}): {count:>6,} complaints")


def main() -> None:
    """Main entry point for feature validation."""
    print_header("FEATURE ENGINEERING VALIDATION")
    
    # Load data
    logger.info(f"Loading data from {FILTERED_DATA_FILE}")
    df = pd.read_csv(FILTERED_DATA_FILE)
    logger.info(f"Loaded {len(df):,} truck-related complaints")
    
    # Add temporal features
    logger.info("Extracting temporal features...")
    df = add_temporal_features(df, "created_date")
    df["day_name"] = df["day_of_week"].apply(get_day_name)
    
    # Validate features
    logger.info("Validating feature ranges...")
    validate_feature_ranges(df)
    logger.info("All features have valid ranges")
    
    # Print statistics
    print("\nFeature Statistics:")
    print_separator("-", 50)
    print(f"  Hour range:          {df['hour'].min()}-{df['hour'].max()}")
    print(f"  Day of week range:   {df['day_of_week'].min()}-{df['day_of_week'].max()}")
    print(f"  Weekend complaints:  {df['is_weekend'].sum():,} ({df['is_weekend'].mean()*100:.1f}%)")
    print(f"  Rush hour complaints: {df['is_rush_hour'].sum():,} ({df['is_rush_hour'].mean()*100:.1f}%)")
    
    print_day_breakdown(df)
    print_hour_breakdown(df)
    
    # Save enhanced dataset
    output_cols = [
        "created_date", "complaint_type", "descriptor", "borough",
        "latitude", "longitude", "street_name",
        "hour", "day_of_week", "day_name", "is_weekend", "is_rush_hour", "month"
    ]
    output_df = df[output_cols].copy()
    output_df.to_csv(FEATURES_FILE, index=False)
    
    logger.info(f"Saved {len(output_df):,} rows to {FEATURES_FILE}")
    
    print_header("VALIDATION COMPLETE")
    print("\nNew features added:")
    print("  • hour (0-23)")
    print("  • day_of_week (0=Mon, 6=Sun)")
    print("  • day_name (readable day names)")
    print("  • is_weekend (0/1)")
    print("  • is_rush_hour (0/1)")
    print("  • month (1-12)")


if __name__ == "__main__":
    main()
