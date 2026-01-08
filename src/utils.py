"""
Utility functions for NYC Curbside Congestion project.

This module provides shared helper functions used across scripts
and the dashboard application.
"""

import logging
import sys
from pathlib import Path
from typing import Optional, Tuple, List

import pandas as pd
import numpy as np
import joblib

from src.config import (
    PROJECT_ROOT,
    DATA_DIR,
    MODELS_DIR,
    RUSH_HOUR_MORNING,
    RUSH_HOUR_EVENING,
    RAIN_THRESHOLD_INCHES,
    COLD_THRESHOLD_F,
    HOT_THRESHOLD_F,
    RISK_COLORS
)


def setup_logging(
    name: str = "nyc_congestion",
    level: int = logging.INFO
) -> logging.Logger:
    """
    Set up a configured logger for the project.
    
    Args:
        name: Name for the logger instance.
        level: Logging level (default: INFO).
        
    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Avoid duplicate handlers
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    return logger


def load_csv(filename: str, data_dir: Optional[Path] = None) -> pd.DataFrame:
    """
    Load a CSV file from the data directory.
    
    Args:
        filename: Name of the CSV file (with or without .csv extension).
        data_dir: Optional custom data directory path.
        
    Returns:
        Loaded DataFrame.
        
    Raises:
        FileNotFoundError: If the file doesn't exist.
    """
    if data_dir is None:
        data_dir = DATA_DIR
    
    # Add .csv extension if not present
    if not filename.endswith(".csv"):
        filename = f"{filename}.csv"
    
    filepath = data_dir / filename
    
    if not filepath.exists():
        raise FileNotFoundError(f"Data file not found: {filepath}")
    
    return pd.read_csv(filepath)


def save_csv(
    df: pd.DataFrame,
    filename: str,
    data_dir: Optional[Path] = None,
    index: bool = False
) -> Path:
    """
    Save a DataFrame to CSV in the data directory.
    
    Args:
        df: DataFrame to save.
        filename: Name for the output file.
        data_dir: Optional custom data directory path.
        index: Whether to include the index column.
        
    Returns:
        Path to the saved file.
    """
    if data_dir is None:
        data_dir = DATA_DIR
    
    if not filename.endswith(".csv"):
        filename = f"{filename}.csv"
    
    filepath = data_dir / filename
    df.to_csv(filepath, index=index)
    
    return filepath


def load_model(model_name: str) -> object:
    """
    Load a trained model from the models directory.
    
    Args:
        model_name: Name of the model file (with or without .pkl extension).
        
    Returns:
        Loaded model object.
        
    Raises:
        FileNotFoundError: If the model file doesn't exist.
    """
    if not model_name.endswith(".pkl"):
        model_name = f"{model_name}.pkl"
    
    filepath = MODELS_DIR / model_name
    
    if not filepath.exists():
        raise FileNotFoundError(f"Model file not found: {filepath}")
    
    return joblib.load(filepath)


def save_model(model: object, model_name: str) -> Path:
    """
    Save a trained model to the models directory.
    
    Args:
        model: Model object to save.
        model_name: Name for the model file.
        
    Returns:
        Path to the saved model file.
    """
    if not model_name.endswith(".pkl"):
        model_name = f"{model_name}.pkl"
    
    filepath = MODELS_DIR / model_name
    joblib.dump(model, filepath)
    
    return filepath


def add_temporal_features(df: pd.DataFrame, datetime_col: str = "created_date") -> pd.DataFrame:
    """
    Extract temporal features from a datetime column.
    
    Args:
        df: DataFrame with a datetime column.
        datetime_col: Name of the datetime column.
        
    Returns:
        DataFrame with added temporal features.
    """
    df = df.copy()
    
    # Ensure datetime type
    df[datetime_col] = pd.to_datetime(df[datetime_col])
    
    # Extract features
    df["hour"] = df[datetime_col].dt.hour
    df["day_of_week"] = df[datetime_col].dt.dayofweek
    df["month"] = df[datetime_col].dt.month
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    
    # Rush hour: 7-10 AM or 4-7 PM
    df["is_rush_hour"] = df["hour"].apply(
        lambda h: 1 if h in RUSH_HOUR_MORNING or h in RUSH_HOUR_EVENING else 0
    )
    
    return df


def add_weather_flags(
    df: pd.DataFrame,
    temp_col: str = "temp_high",
    precip_col: str = "precipitation"
) -> pd.DataFrame:
    """
    Add weather-based binary flags.
    
    Args:
        df: DataFrame with weather columns.
        temp_col: Name of temperature column.
        precip_col: Name of precipitation column.
        
    Returns:
        DataFrame with added weather flags.
    """
    df = df.copy()
    
    df["is_rainy"] = (df[precip_col] > RAIN_THRESHOLD_INCHES).astype(int)
    df["is_cold"] = (df[temp_col] < COLD_THRESHOLD_F).astype(int)
    df["is_hot"] = (df[temp_col] > HOT_THRESHOLD_F).astype(int)
    
    return df


def get_risk_color(probability: float) -> List[int]:
    """
    Get RGBA color based on congestion probability.
    
    Args:
        probability: Congestion probability (0.0 to 1.0).
        
    Returns:
        RGBA color as list of 4 integers.
    """
    if probability < 0.2:
        return RISK_COLORS["low"]
    elif probability < 0.4:
        return RISK_COLORS["medium_low"]
    elif probability < 0.6:
        return RISK_COLORS["medium_high"]
    else:
        return RISK_COLORS["high"]


def format_hour(hour: int) -> str:
    """
    Format hour as human-readable time string.
    
    Args:
        hour: Hour in 24-hour format (0-23).
        
    Returns:
        Formatted time string (e.g., "9:00 AM").
    """
    if hour == 0:
        return "12:00 AM"
    elif hour < 12:
        return f"{hour}:00 AM"
    elif hour == 12:
        return "12:00 PM"
    else:
        return f"{hour - 12}:00 PM"


def get_day_name(day_of_week: int) -> str:
    """
    Convert day of week number to name.
    
    Args:
        day_of_week: Day number (0=Monday, 6=Sunday).
        
    Returns:
        Day name string.
    """
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", 
            "Friday", "Saturday", "Sunday"]
    return days[day_of_week]


def print_separator(char: str = "=", length: int = 60) -> None:
    """Print a visual separator line."""
    print(char * length)


def print_header(title: str, char: str = "=", length: int = 60) -> None:
    """Print a formatted header."""
    print_separator(char, length)
    print(title)
    print_separator(char, length)
