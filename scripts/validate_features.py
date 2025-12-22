"""
Validation script for Step 3 - Feature Engineering
Runs the feature engineering process and validates outputs
"""

import pandas as pd
import numpy as np

print("="*60)
print("STEP 3 VALIDATION - Feature Engineering")
print("="*60)

# Load the filtered truck data
print("\n1. Loading data...")
df = pd.read_csv('data/311_truck_broad_filtered.csv')
print(f"   ✓ Loaded {len(df):,} truck-related complaints")

# Convert to datetime
print("\n2. Converting timestamps...")
df['created_date'] = pd.to_datetime(df['created_date'])
print(f"   ✓ Converted to datetime (type: {df['created_date'].dtype})")

# Extract temporal features
print("\n3. Extracting temporal features...")
df['hour'] = df['created_date'].dt.hour
df['day_of_week'] = df['created_date'].dt.dayofweek
df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
df['is_rush_hour'] = df['hour'].isin([7, 8, 9, 16, 17, 18]).astype(int)
df['month'] = df['created_date'].dt.month

print("   ✓ Created 5 temporal features")

# Validate ranges
print("\n4. Validating feature ranges...")
assert df['hour'].min() >= 0 and df['hour'].max() <= 23, "Hour out of range!"
assert df['day_of_week'].min() >= 0 and df['day_of_week'].max() <= 6, "Day out of range!"
assert set(df['is_weekend'].unique()) <= {0, 1}, "Weekend flag has invalid values!"
assert set(df['is_rush_hour'].unique()) <= {0, 1}, "Rush hour flag has invalid values!"
assert df['month'].min() >= 1 and df['month'].max() <= 12, "Month out of range!"
print("   ✓ All features have valid ranges")

# Show key statistics
print("\n5. Feature Statistics:")
print(f"   • Hour range: {df['hour'].min()}-{df['hour'].max()}")
print(f"   • Day of week range: {df['day_of_week'].min()}-{df['day_of_week'].max()}")
print(f"   • Weekend complaints: {df['is_weekend'].sum():,} ({df['is_weekend'].mean()*100:.1f}%)")
print(f"   • Rush hour complaints: {df['is_rush_hour'].sum():,} ({df['is_rush_hour'].mean()*100:.1f}%)")

# Day of week breakdown
print("\n6. Day of Week Breakdown:")
day_names = {0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday', 
             4: 'Friday', 5: 'Saturday', 6: 'Sunday'}
df['day_name'] = df['day_of_week'].map(day_names)
day_counts = df['day_of_week'].value_counts().sort_index()

for day_num, count in day_counts.items():
    day = day_names[day_num]
    pct = (count / len(df)) * 100
    weekend_marker = "  (WEEKEND)" if day_num >= 5 else ""
    print(f"   {day:>9}: {count:>6,} ({pct:>4.1f}%){weekend_marker}")

weekday_avg = day_counts[:5].mean()
weekend_avg = day_counts[5:].mean()
weekend_drop = (1 - weekend_avg/weekday_avg) * 100
print(f"\n   Weekday avg: {weekday_avg:,.0f}")
print(f"   Weekend avg: {weekend_avg:,.0f}")
print(f"   Weekend drop: {weekend_drop:.1f}%")

# Hour breakdown
print("\n7. Busiest Hours:")
hour_counts = df['hour'].value_counts().sort_index()
top_hours = hour_counts.nlargest(5)
for hour, count in top_hours.items():
    time_str = f"{hour}:00 AM" if hour < 12 else f"{hour-12 if hour > 12 else 12}:00 PM"
    print(f"   {time_str:>8} (hour {hour:>2}): {count:>6,} complaints")

# Save the enhanced dataset
print("\n8. Saving enhanced dataset...")
output_cols = [
    'created_date', 'complaint_type', 'descriptor', 'borough',
    'latitude', 'longitude', 'street_name',
    'hour', 'day_of_week', 'day_name', 'is_weekend', 'is_rush_hour', 'month'
]
output_df = df[output_cols].copy()
output_df.to_csv('data/complaints_with_features.csv', index=False)
print(f"   ✓ Saved {len(output_df):,} rows with {len(output_df.columns)} columns")
print(f"   ✓ File: data/complaints_with_features.csv")

print("\n" + "="*60)
print("✅ VALIDATION COMPLETE - All features extracted successfully!")
print("="*60)
print("\nNew features added:")
print("  • hour (0-23)")
print("  • day_of_week (0=Mon, 6=Sun)")
print("  • day_name (readable day names)")
print("  • is_weekend (0/1)")
print("  • is_rush_hour (0/1)")
print("  • month (1-12)")
