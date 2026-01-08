"""
NYC 311 API Client for fetching live complaint data.

This module provides functions to fetch real-time 311 complaint data
from the NYC Open Data API (Socrata).

API Endpoint: https://data.cityofnewyork.us/resource/erm2-nwe9.json
Documentation: https://dev.socrata.com/foundry/data.cityofnewyork.us/erm2-nwe9
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any

import pandas as pd
import requests

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import (
    DATA_DIR,
    NYC_LATITUDE,
    NYC_LONGITUDE,
    MANHATTAN_BOUNDS
)
from src.utils import setup_logging, add_temporal_features

# Setup logging
logger = setup_logging("api_311")

# NYC 311 Open Data API endpoint (Socrata)
API_ENDPOINT = "https://data.cityofnewyork.us/resource/erm2-nwe9.json"

# Truck-related complaint types to filter
TRUCK_COMPLAINT_TYPES = [
    "Blocked Driveway",
    "Illegal Parking", 
    "Derelict Vehicle",
    "Posted Parking Sign Violation"
]

# Truck-related keywords in descriptors
TRUCK_KEYWORDS = [
    "truck", "commercial", "delivery", "tractor trailer",
    "semi", "van", "large vehicle"
]


def fetch_recent_complaints(
    days_back: int = 30,
    limit: int = 10000,
    borough: str = "MANHATTAN"
) -> pd.DataFrame:
    """
    Fetch recent 311 complaints from the NYC Open Data API.
    
    Args:
        days_back: Number of days of history to fetch.
        limit: Maximum number of records to fetch.
        borough: Borough to filter (default: MANHATTAN).
        
    Returns:
        DataFrame with complaint data.
        
    Raises:
        requests.RequestException: If API request fails.
    """
    # Calculate date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_back)
    start_str = start_date.strftime("%Y-%m-%dT00:00:00")
    
    logger.info(f"Fetching 311 complaints from {start_date.date()} to {end_date.date()}")
    
    # Build SoQL query for truck-related complaints
    complaint_filter = " OR ".join([f"complaint_type='{ct}'" for ct in TRUCK_COMPLAINT_TYPES])
    
    params = {
        "$where": f"created_date >= '{start_str}' AND borough = '{borough}' AND ({complaint_filter})",
        "$limit": limit,
        "$order": "created_date DESC",
        "$select": "unique_key,created_date,complaint_type,descriptor,borough,latitude,longitude,street_name"
    }
    
    try:
        response = requests.get(API_ENDPOINT, params=params, timeout=30)
        response.raise_for_status()
    except requests.Timeout:
        logger.error("API request timed out")
        raise
    except requests.RequestException as e:
        logger.error(f"API request failed: {e}")
        raise
    
    data = response.json()
    
    if not data:
        logger.warning("No data returned from API")
        return pd.DataFrame()
    
    df = pd.DataFrame(data)
    logger.info(f"Fetched {len(df):,} raw complaints from API")
    
    return df


def process_live_complaints(df: pd.DataFrame) -> pd.DataFrame:
    """
    Process raw API data into the format needed for the dashboard.
    
    Args:
        df: Raw DataFrame from API.
        
    Returns:
        Processed DataFrame with temporal features.
    """
    if df.empty:
        return df
    
    # Convert types
    df["created_date"] = pd.to_datetime(df["created_date"])
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    
    # Filter to valid coordinates within Manhattan bounds
    df = df[
        (df["latitude"].notna()) &
        (df["longitude"].notna()) &
        (df["latitude"] >= MANHATTAN_BOUNDS["lat_min"]) &
        (df["latitude"] <= MANHATTAN_BOUNDS["lat_max"]) &
        (df["longitude"] >= MANHATTAN_BOUNDS["lon_min"]) &
        (df["longitude"] <= MANHATTAN_BOUNDS["lon_max"])
    ].copy()
    
    # Filter for truck-related descriptors (case-insensitive)
    def is_truck_related(descriptor):
        if pd.isna(descriptor):
            return False
        descriptor_lower = str(descriptor).lower()
        return any(keyword in descriptor_lower for keyword in TRUCK_KEYWORDS)
    
    # Keep complaints that are either truck-related by descriptor OR by complaint type
    df = df[
        df["descriptor"].apply(is_truck_related) | 
        df["complaint_type"].isin(TRUCK_COMPLAINT_TYPES)
    ].copy()
    
    if df.empty:
        logger.warning("No truck-related complaints found after filtering")
        return df
    
    # Add temporal features
    df = add_temporal_features(df, "created_date")
    
    logger.info(f"Processed {len(df):,} truck-related complaints")
    
    return df


def get_live_stats() -> Dict[str, Any]:
    """
    Get live statistics from the 311 API for display.
    
    Returns:
        Dictionary with live stats including complaint counts and last update time.
    """
    try:
        df = fetch_recent_complaints(days_back=7, limit=5000)
        df = process_live_complaints(df)
        
        if df.empty:
            return {
                "total_complaints": 0,
                "last_update": datetime.now(),
                "is_live": True,
                "error": None
            }
        
        return {
            "total_complaints": len(df),
            "last_update": datetime.now(),
            "is_live": True,
            "complaints_today": len(df[df["created_date"].dt.date == datetime.now().date()]),
            "peak_hour": df["hour"].mode().iloc[0] if not df.empty else None,
            "error": None
        }
    except Exception as e:
        logger.error(f"Failed to get live stats: {e}")
        return {
            "total_complaints": 0,
            "last_update": None,
            "is_live": False,
            "error": str(e)
        }


def get_current_weather() -> Dict[str, Any]:
    """
    Fetch current weather for NYC from Open-Meteo API.
    
    Returns:
        Dictionary with current temperature and precipitation.
    """
    url = "https://api.open-meteo.com/v1/forecast"
    
    params = {
        "latitude": NYC_LATITUDE,
        "longitude": NYC_LONGITUDE,
        "current": ["temperature_2m", "precipitation"],
        "temperature_unit": "fahrenheit",
        "precipitation_unit": "inch",
        "timezone": "America/New_York"
    }
    
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        current = data.get("current", {})
        
        return {
            "temperature": current.get("temperature_2m", 65),
            "precipitation": current.get("precipitation", 0.0),
            "is_live": True,
            "last_update": datetime.now(),
            "error": None
        }
    except Exception as e:
        logger.warning(f"Failed to fetch weather: {e}")
        return {
            "temperature": 65,  # Default fallback
            "precipitation": 0.0,
            "is_live": False,
            "last_update": None,
            "error": str(e)
        }



def get_weather_forecast(target_date, target_hour: int = 12) -> Dict[str, Any]:
    """
    Fetch weather forecast for a specific date and hour (up to 7 days ahead).
    
    Args:
        target_date: The date to get forecast for.
        target_hour: The hour (0-23) to get forecast for.
        
    Returns:
        Dictionary with temperature and precipitation for that date/hour,
        or None if date is outside forecast window.
    """
    today = datetime.now().date()
    target = target_date.date() if isinstance(target_date, datetime) else target_date
    days_ahead = (target - today).days
    
    # Check if within forecast window
    if days_ahead < 0 or days_ahead > 7:
        return {
            "temperature": None,
            "precipitation": None,
            "is_forecast": False,
            "in_window": False,
            "error": "Date outside 7-day forecast window"
        }
    
    # Fetch hourly forecast
    url = "https://api.open-meteo.com/v1/forecast"
    
    params = {
        "latitude": NYC_LATITUDE,
        "longitude": NYC_LONGITUDE,
        "hourly": ["temperature_2m", "precipitation"],
        "temperature_unit": "fahrenheit",
        "precipitation_unit": "inch",
        "timezone": "America/New_York",
        "forecast_days": 8
    }
    
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        hourly = data.get("hourly", {})
        times = hourly.get("time", [])
        temps = hourly.get("temperature_2m", [])
        precip = hourly.get("precipitation", [])
        
        # Build target datetime string
        target_datetime_str = f"{target.strftime('%Y-%m-%d')}T{target_hour:02d}:00"
        
        if target_datetime_str in times:
            idx = times.index(target_datetime_str)
            
            # Determine if this is future (forecast) or current/past
            is_forecast = days_ahead > 0 or (days_ahead == 0 and target_hour > datetime.now().hour)
            
            return {
                "temperature": round(temps[idx], 1),
                "precipitation": round(precip[idx], 2),
                "is_forecast": is_forecast,
                "in_window": True,
                "error": None
            }
        else:
            # Fallback: find closest hour
            target_prefix = target.strftime('%Y-%m-%d')
            matching = [(i, t) for i, t in enumerate(times) if t.startswith(target_prefix)]
            
            if matching:
                # Get the hour closest to target_hour
                closest_idx = min(matching, key=lambda x: abs(int(x[1][11:13]) - target_hour))[0]
                
                return {
                    "temperature": round(temps[closest_idx], 1),
                    "precipitation": round(precip[closest_idx], 2),
                    "is_forecast": True,
                    "in_window": True,
                    "error": None
                }
            
            return {
                "temperature": None,
                "precipitation": None,
                "is_forecast": False,
                "in_window": False,
                "error": "DateTime not found in forecast"
            }
            
    except Exception as e:
        logger.warning(f"Failed to fetch forecast: {e}")
        return {
            "temperature": None,
            "precipitation": None,
            "is_forecast": False,
            "in_window": False,
            "error": str(e)
        }


def fetch_and_save_live_data(output_file: str = "live_complaints.csv") -> Path:
    """
    Fetch live data and save to CSV for offline use.
    
    Args:
        output_file: Name of output CSV file.
        
    Returns:
        Path to saved file.
    """
    df = fetch_recent_complaints(days_back=90, limit=50000)
    df = process_live_complaints(df)
    
    output_path = DATA_DIR / output_file
    df.to_csv(output_path, index=False)
    
    logger.info(f"Saved {len(df):,} complaints to {output_path}")
    
    return output_path


# For command-line usage
if __name__ == "__main__":
    from src.utils import print_header
    
    print_header("NYC 311 LIVE DATA FETCH")
    
    # Test the API
    stats = get_live_stats()
    
    if stats["error"]:
        print(f"❌ Error: {stats['error']}")
    else:
        print(f"✅ Live data connection successful!")
        print(f"   Complaints (last 7 days): {stats['total_complaints']:,}")
        print(f"   Complaints today: {stats.get('complaints_today', 'N/A')}")
        print(f"   Peak hour: {stats.get('peak_hour', 'N/A')}:00")
        print(f"   Last update: {stats['last_update']}")
