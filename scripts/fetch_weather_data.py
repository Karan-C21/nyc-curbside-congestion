"""
Fetch historical weather data for NYC from Open-Meteo API
Date range: 2023-01-01 to present
Location: NYC (Central Park coordinates)
"""

import requests
import pandas as pd
from datetime import datetime, timedelta

# NYC coordinates (Central Park)
LATITUDE = 40.7829
LONGITUDE = -73.9654

# Date range (your data is 2023-present)
START_DATE = "2023-01-01"
END_DATE = datetime.now().strftime("%Y-%m-%d")

print("="*60)
print("FETCHING NYC WEATHER DATA")
print("="*60)
print(f"Location: NYC (Central Park)")
print(f"Coordinates: {LATITUDE}, {LONGITUDE}")
print(f"Date range: {START_DATE} to {END_DATE}")
print()

# Open-Meteo API endpoint
url = "https://archive-api.open-meteo.com/v1/archive"

# Parameters
params = {
    "latitude": LATITUDE,
    "longitude": LONGITUDE,
    "start_date": START_DATE,
    "end_date": END_DATE,
    "daily": [
        "temperature_2m_max",      # Daily high temp
        "temperature_2m_min",      # Daily low temp  
        "precipitation_sum",       # Total precipitation
        "snowfall_sum",           # Total snowfall
        "precipitation_hours"      # Hours of precipitation
    ],
    "temperature_unit": "fahrenheit",
    "precipitation_unit": "inch",
    "timezone": "America/New_York"
}

print("Fetching weather data from Open-Meteo API...")
response = requests.get(url, params=params)

if response.status_code == 200:
    print("✓ Successfully fetched weather data!")
    
    # Parse JSON response
    data = response.json()
    
    # Extract daily data
    daily = data['daily']
    
    # Create DataFrame
    weather_df = pd.DataFrame({
        'date': daily['time'],
        'temp_high': daily['temperature_2m_max'],
        'temp_low': daily['temperature_2m_min'],
        'precipitation': daily['precipitation_sum'],
        'snowfall': daily['snowfall_sum'],
        'precip_hours': daily['precipitation_hours']
    })
    
    # Convert date to datetime
    weather_df['date'] = pd.to_datetime(weather_df['date'])
    
    print(f"\nFetched {len(weather_df):,} days of weather data")
    print(f"Date range: {weather_df['date'].min()} to {weather_df['date'].max()}")
    
    # Show summary stats
    print("\n" + "="*60)
    print("WEATHER DATA SUMMARY")
    print("="*60)
    print(weather_df.describe())
    
    # Save to CSV
    output_path = 'data/nyc_weather_2023_present.csv'
    weather_df.to_csv(output_path, index=False)
    print(f"\n✓ Saved weather data to: {output_path}")
    
    # Show sample
    print("\nSample data:")
    print(weather_df.head(10))
    
else:
    print(f"✗ Failed to fetch data. Status code: {response.status_code}")
    print(f"Error: {response.text}")
