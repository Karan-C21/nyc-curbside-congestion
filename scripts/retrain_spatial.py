import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

def retrain_and_save():
    print("Loading data...")
    # Load Main Data
    df = pd.read_csv('data/modeling_dataset.csv')
    
    # Load Weather Stats (Re-creating local logic for speed)
    complaints = pd.read_csv('data/complaints_with_features.csv')
    complaints['date'] = pd.to_datetime(pd.to_datetime(complaints['created_date']).dt.date)
    
    weather = pd.read_csv('data/nyc_weather_2023_present.csv')
    weather['date'] = pd.to_datetime(weather['date'])
    
    cw = complaints.merge(weather, on='date', how='left')
    cw['is_rainy'] = (cw['precipitation'] > 0.1).astype(int)
    cw['is_cold'] = (cw['temp_high'] < 40).astype(int)
    cw['is_hot'] = (cw['temp_high'] > 85).astype(int)
    
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
    
    # Merge
    final_df = df.merge(weather_stats, on=['hour', 'day_of_week'], how='left')
    y = final_df['high_congestion']
    
    # --- KEY CHANGE: Adding grid_lat and grid_lon ---
    feature_cols = [
        'hour', 'day_of_week', 'is_weekend', 'is_rush_hour', 'month', 
        'grid_lat', 'grid_lon',  # <--- SPATIAL AWARENESS
        'avg_temp', 'avg_precip', 'pct_rainy', 'pct_cold', 'pct_hot'
    ]
    
    print("Training Spatial+Weather Model...")
    rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1, class_weight='balanced')
    rf.fit(final_df[feature_cols], y)
    
    # Save
    path = 'models/random_forest_weather_enhanced.pkl'
    joblib.dump(rf, path)
    print(f"✅ Saved smarter model to {path}")

if __name__ == "__main__":
    retrain_and_save()
