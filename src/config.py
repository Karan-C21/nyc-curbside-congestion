"""
Centralized configuration for NYC Curbside Congestion project.

This module contains all project-wide constants, paths, and settings
to ensure consistency across scripts and the dashboard.
"""

from pathlib import Path
from typing import List, Dict, Any

# =============================================================================
# Project Paths
# =============================================================================

# Get the project root directory (parent of src/)
PROJECT_ROOT = Path(__file__).parent.parent.resolve()

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"

# Key data files
RAW_DATA_FILE = DATA_DIR / "311_manhattan_clean_2023_present.csv"
FILTERED_DATA_FILE = DATA_DIR / "311_truck_broad_filtered.csv"
FEATURES_FILE = DATA_DIR / "complaints_with_features.csv"
SPATIAL_FEATURES_FILE = DATA_DIR / "complaints_with_spatial_features.csv"
MODELING_DATASET_FILE = DATA_DIR / "modeling_dataset.csv"
WEATHER_DATA_FILE = DATA_DIR / "nyc_weather_2023_present.csv"

# Model files
LOGISTIC_MODEL_FILE = MODELS_DIR / "logistic_regression_model.pkl"
RANDOM_FOREST_MODEL_FILE = MODELS_DIR / "random_forest_model.pkl"
ENHANCED_MODEL_FILE = MODELS_DIR / "random_forest_weather_enhanced.pkl"
XGBOOST_MODEL_FILE = MODELS_DIR / "xgboost_model.pkl"

# =============================================================================
# NYC Geographic Constants
# =============================================================================

# Central Park coordinates (used as NYC reference point)
NYC_LATITUDE = 40.7829
NYC_LONGITUDE = -73.9654

# Manhattan bounds (approximate)
MANHATTAN_BOUNDS = {
    "lat_min": 40.70,
    "lat_max": 40.88,
    "lon_min": -74.02,
    "lon_max": -73.90
}

# Default map view settings
MAP_VIEW = {
    "latitude": 40.76,
    "longitude": -73.97,
    "zoom": 11.5,
    "pitch": 0,
    "bearing": 0
}

# =============================================================================
# Feature Engineering Settings
# =============================================================================

# Rush hour definitions (7-10 AM, 4-7 PM)
RUSH_HOUR_MORNING = range(7, 11)  # 7, 8, 9, 10
RUSH_HOUR_EVENING = range(16, 20)  # 16, 17, 18, 19

# Weather thresholds
RAIN_THRESHOLD_INCHES = 0.1
COLD_THRESHOLD_F = 40
HOT_THRESHOLD_F = 85

# Feature columns for modeling
TEMPORAL_FEATURES: List[str] = [
    "hour",
    "day_of_week", 
    "is_weekend",
    "is_rush_hour",
    "month"
]

SPATIAL_FEATURES: List[str] = [
    "grid_lat",
    "grid_lon"
]

WEATHER_FEATURES: List[str] = [
    "avg_temp",
    "avg_precip",
    "pct_rainy",
    "pct_cold",
    "pct_hot"
]

# All features used by the enhanced model
ALL_MODEL_FEATURES: List[str] = TEMPORAL_FEATURES + SPATIAL_FEATURES + WEATHER_FEATURES

# =============================================================================
# Weather API Settings
# =============================================================================

WEATHER_API_URL = "https://archive-api.open-meteo.com/v1/archive"
WEATHER_API_PARAMS = {
    "latitude": NYC_LATITUDE,
    "longitude": NYC_LONGITUDE,
    "daily": [
        "temperature_2m_max",
        "temperature_2m_min",
        "precipitation_sum",
        "snowfall_sum",
        "precipitation_hours"
    ],
    "temperature_unit": "fahrenheit",
    "precipitation_unit": "inch",
    "timezone": "America/New_York"
}

# =============================================================================
# Visualization Settings
# =============================================================================

# Risk level thresholds (percentage)
RISK_THRESHOLDS = {
    "low": 25,
    "moderate": 40
}

# Color palette for risk levels (RGBA)
RISK_COLORS: Dict[str, List[int]] = {
    "low": [46, 204, 113, 200],      # Emerald Green
    "medium_low": [241, 196, 15, 200], # Sunflower Yellow  
    "medium_high": [230, 126, 34, 200], # Carrot Orange
    "high": [231, 76, 60, 200]        # Alizarin Red
}

# Hex colors for text display
RISK_HEX_COLORS = {
    "low": "#2ecc71",
    "moderate": "#f1c40f", 
    "high": "#e74c3c"
}

# =============================================================================
# Model Settings
# =============================================================================

RANDOM_FOREST_PARAMS = {
    "n_estimators": 100,
    "max_depth": 10,
    "random_state": 42,
    "n_jobs": -1,
    "class_weight": "balanced"
}

LOGISTIC_REGRESSION_PARAMS = {
    "max_iter": 1000,
    "random_state": 42,
    "class_weight": "balanced"
}
