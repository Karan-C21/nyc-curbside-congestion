"""
Fetch historical weather data for NYC from Open-Meteo API.

This script retrieves daily weather data including temperature, precipitation,
and snowfall for New York City to use in congestion prediction modeling.

Usage:
    python scripts/fetch_weather_data.py
"""

import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import (
    NYC_LATITUDE,
    NYC_LONGITUDE,
    WEATHER_API_URL,
    WEATHER_DATA_FILE,
)
from src.utils import setup_logging, print_header

# Setup logging
logger = setup_logging("weather_fetch")


def fetch_weather_data(
    start_date: str = "2023-01-01",
    end_date: Optional[str] = None
) -> pd.DataFrame:
    """
    Fetch historical weather data from Open-Meteo API.
    
    Args:
        start_date: Start date in YYYY-MM-DD format.
        end_date: End date in YYYY-MM-DD format. Defaults to today.
        
    Returns:
        DataFrame with daily weather data.
        
    Raises:
        requests.RequestException: If API request fails.
        ValueError: If API returns invalid data.
    """
    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")
    
    logger.info(f"Fetching weather data from {start_date} to {end_date}")
    logger.info(f"Location: NYC ({NYC_LATITUDE}, {NYC_LONGITUDE})")
    
    params = {
        "latitude": NYC_LATITUDE,
        "longitude": NYC_LONGITUDE,
        "start_date": start_date,
        "end_date": end_date,
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
    
    try:
        response = requests.get(WEATHER_API_URL, params=params, timeout=30)
        response.raise_for_status()
    except requests.Timeout:
        logger.error("API request timed out after 30 seconds")
        raise
    except requests.RequestException as e:
        logger.error(f"API request failed: {e}")
        raise
    
    data = response.json()
    
    if "daily" not in data:
        raise ValueError("API response missing 'daily' data")
    
    daily = data["daily"]
    
    weather_df = pd.DataFrame({
        "date": pd.to_datetime(daily["time"]),
        "temp_high": daily["temperature_2m_max"],
        "temp_low": daily["temperature_2m_min"],
        "precipitation": daily["precipitation_sum"],
        "snowfall": daily["snowfall_sum"],
        "precip_hours": daily["precipitation_hours"]
    })
    
    logger.info(f"Successfully fetched {len(weather_df):,} days of weather data")
    
    return weather_df


def main() -> None:
    """Main entry point for weather data fetching."""
    print_header("NYC WEATHER DATA FETCH")
    
    try:
        # Fetch data
        weather_df = fetch_weather_data()
        
        # Display summary
        logger.info("Weather data summary:")
        print(weather_df.describe().round(2))
        
        # Save to file
        weather_df.to_csv(WEATHER_DATA_FILE, index=False)
        logger.info(f"Saved weather data to: {WEATHER_DATA_FILE}")
        
        # Show sample
        print("\nSample data (first 5 rows):")
        print(weather_df.head())
        
        print_header("FETCH COMPLETE")
        
    except Exception as e:
        logger.error(f"Weather fetch failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
