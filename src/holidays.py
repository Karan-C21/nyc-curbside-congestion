"""
Holiday and Special Events module for NYC Curbside Congestion.

This module provides functions to identify holidays, special days,
and NYC events that affect delivery congestion patterns.
"""

from datetime import datetime, date
from typing import Dict, List, Optional, Any
import pandas as pd
import requests

# =============================================================================
# US Federal Holidays (Fixed dates - approximate for some)
# =============================================================================

def get_us_holidays(year: int) -> Dict[date, str]:
    """
    Get US federal holidays for a given year.
    
    Args:
        year: The year to get holidays for.
        
    Returns:
        Dictionary mapping date to holiday name.
    """
    holidays = {}
    
    # Fixed-date holidays
    holidays[date(year, 1, 1)] = "New Year's Day"
    holidays[date(year, 7, 4)] = "Independence Day"
    holidays[date(year, 11, 11)] = "Veterans Day"
    holidays[date(year, 12, 25)] = "Christmas Day"
    holidays[date(year, 12, 31)] = "New Year's Eve"
    
    # Valentine's Day (high delivery volume)
    holidays[date(year, 2, 14)] = "Valentine's Day"
    
    # Calculate floating holidays
    
    # MLK Day: Third Monday of January
    jan1 = date(year, 1, 1)
    days_until_monday = (7 - jan1.weekday()) % 7
    if days_until_monday == 0:
        days_until_monday = 7
    mlk_day = date(year, 1, 1 + days_until_monday + 14)
    holidays[mlk_day] = "MLK Day"
    
    # Presidents Day: Third Monday of February
    feb1 = date(year, 2, 1)
    days_until_monday = (7 - feb1.weekday()) % 7
    if days_until_monday == 0:
        days_until_monday = 7
    pres_day = date(year, 2, 1 + days_until_monday + 14)
    holidays[pres_day] = "Presidents Day"
    
    # Memorial Day: Last Monday of May
    may31 = date(year, 5, 31)
    days_since_monday = may31.weekday()
    memorial_day = date(year, 5, 31 - days_since_monday)
    holidays[memorial_day] = "Memorial Day"
    
    # Labor Day: First Monday of September
    sep1 = date(year, 9, 1)
    days_until_monday = (7 - sep1.weekday()) % 7
    labor_day = date(year, 9, 1 + days_until_monday)
    holidays[labor_day] = "Labor Day"
    
    # Columbus Day: Second Monday of October
    oct1 = date(year, 10, 1)
    days_until_monday = (7 - oct1.weekday()) % 7
    if days_until_monday == 0:
        days_until_monday = 7
    columbus_day = date(year, 10, 1 + days_until_monday + 7)
    holidays[columbus_day] = "Columbus Day"
    
    # Thanksgiving: Fourth Thursday of November
    nov1 = date(year, 11, 1)
    days_until_thursday = (3 - nov1.weekday()) % 7
    thanksgiving = date(year, 11, 1 + days_until_thursday + 21)
    holidays[thanksgiving] = "Thanksgiving"
    
    # Black Friday (major delivery day)
    black_friday = date(year, 11, thanksgiving.day + 1)
    holidays[black_friday] = "Black Friday"
    
    return holidays


def is_holiday(check_date: date) -> tuple[bool, Optional[str]]:
    """
    Check if a date is a US holiday.
    
    Args:
        check_date: Date to check.
        
    Returns:
        Tuple of (is_holiday, holiday_name or None).
    """
    if isinstance(check_date, datetime):
        check_date = check_date.date()
    
    holidays = get_us_holidays(check_date.year)
    
    if check_date in holidays:
        return True, holidays[check_date]
    return False, None


def is_holiday_week(check_date: date) -> bool:
    """
    Check if date is within a week of a major holiday.
    High delivery volume around holidays.
    """
    if isinstance(check_date, datetime):
        check_date = check_date.date()
    
    holidays = get_us_holidays(check_date.year)
    
    for holiday_date in holidays:
        days_diff = abs((check_date - holiday_date).days)
        if days_diff <= 3:  # Within 3 days of holiday
            return True
    return False


# =============================================================================
# Special Day Flags
# =============================================================================

def get_special_day_flags(check_date: date) -> Dict[str, int]:
    """
    Get various special day flags that affect delivery patterns.
    
    Args:
        check_date: Date to check.
        
    Returns:
        Dictionary of flag names to values (0 or 1).
    """
    if isinstance(check_date, datetime):
        check_date = check_date.date()
    
    flags = {
        "is_holiday": 0,
        "is_holiday_week": 0,
        "is_month_end": 0,
        "is_month_start": 0,
        "is_first_of_month": 0,
    }
    
    # Holiday flags
    is_hol, _ = is_holiday(check_date)
    flags["is_holiday"] = 1 if is_hol else 0
    flags["is_holiday_week"] = 1 if is_holiday_week(check_date) else 0
    
    # Month boundaries (moving days, rent due, etc.)
    flags["is_month_end"] = 1 if check_date.day >= 28 else 0
    flags["is_month_start"] = 1 if check_date.day <= 3 else 0
    flags["is_first_of_month"] = 1 if check_date.day == 1 else 0
    
    return flags


def add_holiday_features(df: pd.DataFrame, date_column: str = "date") -> pd.DataFrame:
    """
    Add holiday and special day features to a DataFrame.
    
    Args:
        df: DataFrame with a date column.
        date_column: Name of the date column.
        
    Returns:
        DataFrame with added holiday features.
    """
    df = df.copy()
    
    # Ensure date column is datetime
    if df[date_column].dtype == 'object':
        df[date_column] = pd.to_datetime(df[date_column])
    
    # Add holiday flags
    df["is_holiday"] = df[date_column].apply(lambda d: 1 if is_holiday(d.date())[0] else 0)
    df["is_holiday_week"] = df[date_column].apply(lambda d: 1 if is_holiday_week(d.date()) else 0)
    df["is_month_end"] = df[date_column].apply(lambda d: 1 if d.day >= 28 else 0)
    df["is_month_start"] = df[date_column].apply(lambda d: 1 if d.day <= 3 else 0)
    
    return df


# =============================================================================
# NYC Events API (Street Closures)
# =============================================================================

NYC_EVENTS_API = "https://data.cityofnewyork.us/resource/3eay-6yrq.json"

def fetch_nyc_events(
    start_date: date,
    end_date: date,
    borough: str = "Manhattan"
) -> pd.DataFrame:
    """
    Fetch NYC permitted events (street closures) from Open Data.
    
    Args:
        start_date: Start date for events.
        end_date: End date for events.
        borough: Borough to filter (default: Manhattan).
        
    Returns:
        DataFrame of events.
    """
    start_str = start_date.strftime("%Y-%m-%dT00:00:00")
    end_str = end_date.strftime("%Y-%m-%dT23:59:59")
    
    params = {
        "$where": f"eventenddate >= '{start_str}' AND eventstartdate <= '{end_str}'",
        "$limit": 5000,
        "$select": "eventid,eventname,eventstartdate,eventenddate,eventlocation,eventborough,eventtype"
    }
    
    try:
        response = requests.get(NYC_EVENTS_API, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        if not data:
            return pd.DataFrame()
        
        df = pd.DataFrame(data)
        
        # Filter to Manhattan if specified
        if borough and "eventborough" in df.columns:
            df = df[df["eventborough"].str.contains(borough, case=False, na=False)]
        
        # Parse dates
        if "eventstartdate" in df.columns:
            df["eventstartdate"] = pd.to_datetime(df["eventstartdate"])
        if "eventenddate" in df.columns:
            df["eventenddate"] = pd.to_datetime(df["eventenddate"])
        
        return df
        
    except Exception as e:
        print(f"Warning: Could not fetch NYC events: {e}")
        return pd.DataFrame()


def count_events_on_date(check_date: date, events_df: Optional[pd.DataFrame] = None) -> int:
    """
    Count number of permitted events on a specific date.
    
    Args:
        check_date: Date to check.
        events_df: Pre-fetched events DataFrame (optional).
        
    Returns:
        Number of events on that date.
    """
    if isinstance(check_date, datetime):
        check_date = check_date.date()
    
    if events_df is None or events_df.empty:
        # Fetch events for just that date
        events_df = fetch_nyc_events(check_date, check_date)
    
    if events_df.empty:
        return 0
    
    # Count events where check_date falls between start and end
    check_dt = datetime.combine(check_date, datetime.min.time())
    
    count = 0
    for _, event in events_df.iterrows():
        start = event.get("eventstartdate")
        end = event.get("eventenddate")
        
        if pd.notna(start) and pd.notna(end):
            if start <= check_dt <= end:
                count += 1
    
    return count


# =============================================================================
# Main Test
# =============================================================================

if __name__ == "__main__":
    from datetime import date, timedelta
    
    print("=== Holiday Module Test ===\n")
    
    # Test holidays
    print("2026 US Holidays:")
    holidays = get_us_holidays(2026)
    for d, name in sorted(holidays.items()):
        print(f"  {d.strftime('%Y-%m-%d')} ({d.strftime('%A')}): {name}")
    
    # Test today
    today = date.today()
    is_hol, name = is_holiday(today)
    print(f"\nToday ({today}): {'Holiday' if is_hol else 'Not a holiday'}")
    
    # Test special flags
    print(f"\nSpecial day flags for {today}:")
    flags = get_special_day_flags(today)
    for flag, value in flags.items():
        print(f"  {flag}: {value}")
    
    # Test NYC events
    print("\n=== NYC Events Test ===\n")
    events = fetch_nyc_events(today, today + timedelta(days=7))
    if not events.empty:
        print(f"Found {len(events)} events in next 7 days:")
        for _, event in events.head(5).iterrows():
            print(f"  - {event.get('eventname', 'Unknown')}")
    else:
        print("No events found (or API unavailable)")
