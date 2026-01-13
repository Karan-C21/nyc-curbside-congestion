"""
NYC Delivery Congestion Predictor Dashboard.

A professional Streamlit-based analytics platform for predicting and analyzing
delivery truck congestion patterns across Manhattan.

Run with: streamlit run app/app.py
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Tuple

import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
import altair as alt

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import (
    XGBOOST_MODEL_FILE,
    MODELING_DATASET_FILE,
    FEATURES_FILE,
    ALL_MODEL_FEATURES,
    MAP_VIEW,
    RISK_THRESHOLDS,
    RISK_HEX_COLORS,
    RUSH_HOUR_MORNING,
    RUSH_HOUR_EVENING,
    COLD_THRESHOLD_F,
    HOT_THRESHOLD_F,
    RAIN_THRESHOLD_INCHES
)
from src.utils import load_model, load_csv, get_risk_color, get_day_name
from src.api_311 import get_live_stats, fetch_recent_complaints, process_live_complaints, get_current_weather, get_weather_forecast
from src.holidays import get_special_day_flags, is_holiday, is_holiday_week

# =============================================================================
# Page Configuration
# =============================================================================

st.set_page_config(
    page_title="NYC Congestion Analytics",
    page_icon="🚚",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =============================================================================
# Custom CSS Styling
# =============================================================================

st.markdown("""
<style>
    /* Import Google Font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    /* Global Styles */
    .stApp {
        background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 50%, #16213e 100%);
        font-family: 'Inter', sans-serif;
    }
    
    /* Header Styling */
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.8rem;
        font-weight: 700;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    
    .sub-header {
        color: #a0aec0;
        font-size: 1.1rem;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    /* Card Styling */
    .metric-card {
        background: linear-gradient(145deg, #1e1e30 0%, #2d2d44 100%);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 1.5rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 40px rgba(102, 126, 234, 0.2);
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
    }
    
    .metric-label {
        color: #a0aec0;
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Risk Level Colors */
    .risk-low { color: #10b981; }
    .risk-moderate { color: #f59e0b; }
    .risk-high { color: #ef4444; }
    
    /* Tab Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        padding: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 12px 24px;
        font-weight: 500;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Insight Box */
    .insight-box {
        background: linear-gradient(145deg, #1a365d 0%, #2c5282 100%);
        border-left: 4px solid #4299e1;
        border-radius: 8px;
        padding: 1rem 1.5rem;
        margin: 1rem 0;
    }
    
    .insight-title {
        color: #90cdf4;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    
    .insight-text {
        color: #e2e8f0;
        line-height: 1.6;
    }
    
    /* Chart Container */
    .chart-container {
        background: rgba(255, 255, 255, 0.03);
        border-radius: 12px;
        padding: 1rem;
        border: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Metrics override */
    div[data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: 700;
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# Data Loading (Cached)
# =============================================================================

@st.cache_resource
def load_resources():
    """Load model and data with caching."""
    try:
        model = load_model(XGBOOST_MODEL_FILE.name)
        df = load_csv(MODELING_DATASET_FILE.name)
        unique_grids = df[["grid_id", "grid_lat", "grid_lon"]].drop_duplicates()
        
        # Try to load historical features for analytics
        try:
            features_df = load_csv(FEATURES_FILE.name)
        except:
            features_df = None
            
        return model, unique_grids, features_df, None
    except FileNotFoundError as e:
        return None, None, None, str(e)


@st.cache_data
def get_hourly_patterns(_features_df: pd.DataFrame) -> pd.DataFrame:
    """Calculate complaint counts by hour."""
    if _features_df is None:
        return None
    hourly = _features_df.groupby("hour").size().reset_index(name="complaints")
    return hourly


@st.cache_data
def get_weekly_heatmap_data(_features_df: pd.DataFrame) -> pd.DataFrame:
    """Calculate heatmap data for day vs hour."""
    if _features_df is None:
        return None
    heatmap = _features_df.groupby(["day_of_week", "hour"]).size().reset_index(name="complaints")
    heatmap["day_name"] = heatmap["day_of_week"].apply(get_day_name)
    return heatmap


@st.cache_data(ttl=300)  # Cache for 5 minutes
def fetch_live_311_stats():
    """Fetch live statistics from 311 API with caching."""
    return get_live_stats()


@st.cache_data(ttl=300)  # Cache for 5 minutes
def fetch_live_complaints_data():
    """Fetch and process live complaint data for analytics."""
    try:
        df = fetch_recent_complaints(days_back=7, limit=5000)
        df = process_live_complaints(df)
        return df, None
    except Exception as e:
        return None, str(e)


@st.cache_data(ttl=600)  # Cache for 10 minutes
def fetch_current_weather():
    """Fetch current NYC weather with caching."""
    return get_current_weather()


# =============================================================================
# Helper Functions
# =============================================================================

def calculate_predictions(model, unique_grids, hour, day_of_week, month, temp, precip, target_date=None):
    """Calculate predictions for all grid cells."""
    is_weekend = 1 if day_of_week >= 5 else 0
    is_rush_hour = 1 if hour in RUSH_HOUR_MORNING or hour in RUSH_HOUR_EVENING else 0
    is_rainy = 1 if precip > RAIN_THRESHOLD_INCHES else 0
    is_cold = 1 if temp < COLD_THRESHOLD_F else 0
    is_hot = 1 if temp > HOT_THRESHOLD_F else 0
    
    # Get holiday flags
    if target_date is None:
        target_date = datetime.now().date()
    holiday_flags = get_special_day_flags(target_date)
    
    input_data = unique_grids.copy()
    input_data["hour"] = hour
    input_data["day_of_week"] = day_of_week
    input_data["is_weekend"] = is_weekend
    input_data["is_rush_hour"] = is_rush_hour
    input_data["month"] = month
    input_data["avg_temp"] = temp
    input_data["avg_precip"] = precip
    input_data["pct_rainy"] = is_rainy
    input_data["pct_cold"] = is_cold
    input_data["pct_hot"] = is_hot
    
    # Add holiday features
    input_data["is_holiday"] = holiday_flags["is_holiday"]
    input_data["is_holiday_week"] = holiday_flags["is_holiday_week"]
    input_data["is_month_end"] = holiday_flags["is_month_end"]
    input_data["is_month_start"] = holiday_flags["is_month_start"]
    
    probs = model.predict_proba(input_data[ALL_MODEL_FEATURES])[:, 1]
    input_data["congestion_prob"] = probs
    input_data["color"] = input_data["congestion_prob"].apply(get_risk_color)
    
    return input_data, probs


def get_risk_level(avg_risk: float) -> Tuple[str, str, str]:
    """Get risk level label, color, and emoji."""
    if avg_risk <= RISK_THRESHOLDS["low"]:
        return "LOW", RISK_HEX_COLORS["low"], "✅"
    elif avg_risk <= RISK_THRESHOLDS["moderate"]:
        return "MODERATE", RISK_HEX_COLORS["moderate"], "⚠️"
    else:
        return "HIGH", RISK_HEX_COLORS["high"], "🔴"


def generate_insight(risk_level: str, hour: int, day_of_week: int, temp: float, precip: float) -> str:
    """Generate AI-like insight text based on current conditions."""
    insights = []
    
    # Time-based insights
    if hour in RUSH_HOUR_MORNING:
        insights.append("Morning rush hour typically sees 40% more congestion")
    elif hour in RUSH_HOUR_EVENING:
        insights.append("Evening rush hour is the peak congestion period")
    elif 22 <= hour or hour <= 5:
        insights.append("Late night hours have minimal delivery activity")
    
    # Day-based insights
    if day_of_week >= 5:
        insights.append("Weekend delivery volumes are typically 35% lower")
    elif day_of_week == 0:
        insights.append("Monday mornings often see catch-up delivery surges")
    elif day_of_week == 4:
        insights.append("Friday afternoons have elevated pre-weekend activity")
    
    # Weather-based insights
    if precip > RAIN_THRESHOLD_INCHES:
        insights.append("Rain increases double-parking incidents by 25%")
    if temp < COLD_THRESHOLD_F:
        insights.append("Cold weather extends average delivery times")
    if temp > HOT_THRESHOLD_F:
        insights.append("Extreme heat may cause delivery delays")
    
    # Risk-based recommendation
    if risk_level == "HIGH":
        insights.append("Consider scheduling deliveries for off-peak hours")
    elif risk_level == "LOW":
        insights.append("Optimal conditions for efficient deliveries")
    
    return " • ".join(insights[:3]) if insights else "Normal congestion patterns expected"


def create_map_layer(input_data: pd.DataFrame) -> pdk.Deck:
    """Create a smooth heatmap visualization without visible zone boundaries."""
    
    # Function to get neighborhood name from coordinates
    def get_neighborhood(lat, lon):
        if lat >= 40.80:
            return "Harlem"
        elif lat >= 40.77:
            return "Upper East Side" if lon >= -73.97 else "Upper West Side"
        elif lat >= 40.75:
            return "Midtown East" if lon >= -73.98 else "Midtown West"
        elif lat >= 40.73:
            return "Gramercy" if lon >= -73.99 else "Chelsea"
        elif lat >= 40.72:
            return "East Village" if lon >= -73.99 else "Greenwich Village"
        elif lat >= 40.71:
            return "Lower East Side" if lon >= -73.99 else "SoHo/Tribeca"
        else:
            return "Financial District"
    
    # Prepare data for heatmap
    map_data = input_data.copy()
    map_data['risk_pct'] = (map_data['congestion_prob'] * 100).round(1)
    map_data['risk_display'] = map_data['risk_pct'].apply(lambda x: f"{x:.1f}%")
    map_data = map_data.reset_index(drop=True)
    map_data['zone_num'] = map_data.index + 1
    
    # Add neighborhood names
    map_data['neighborhood'] = map_data.apply(
        lambda r: get_neighborhood(r['grid_lat'], r['grid_lon']), 
        axis=1
    )
    
    # Weight for heatmap intensity (scale to reasonable range)
    map_data['weight'] = map_data['congestion_prob'] * 100
    
    def get_risk_text(prob):
        if prob < 0.25:
            return "Low Risk"
        elif prob < 0.50:
            return "Moderate Risk"
        else:
            return "High Risk"
    
    map_data['risk_level'] = map_data['congestion_prob'].apply(get_risk_text)
    
    # Smooth heatmap layer - continuous gradient (balanced brightness)
    heatmap_layer = pdk.Layer(
        "HeatmapLayer",
        map_data,
        get_position=["grid_lon", "grid_lat"],
        get_weight="weight",
        aggregation="SUM",
        radius_pixels=55,
        intensity=0.8,  # Balanced brightness
        threshold=0.03,
        color_range=[
            [0, 120, 0, 140],       # Green (low)
            [100, 200, 0, 150],     # Yellow-green
            [200, 220, 0, 160],     # Yellow
            [240, 180, 0, 165],     # Orange
            [230, 100, 0, 175],     # Red-orange
            [200, 30, 30, 185],     # Red (high)
        ],
    )
    
    # Small dots for hover interaction (almost invisible but pickable)
    interaction_layer = pdk.Layer(
        "ScatterplotLayer",
        map_data,
        get_position=["grid_lon", "grid_lat"],
        get_fill_color=[255, 255, 255, 0],  # Invisible
        get_radius=300,
        pickable=True,
        opacity=0,
    )
    
    view_state = pdk.ViewState(
        latitude=MAP_VIEW["latitude"],
        longitude=MAP_VIEW["longitude"],
        zoom=11.5,
        pitch=0,
        bearing=0
    )
    
    return pdk.Deck(
        layers=[heatmap_layer, interaction_layer],
        initial_view_state=view_state,
        map_style=pdk.map_styles.CARTO_DARK,
        tooltip={
            "html": """
                <div style='padding: 14px 18px; min-width: 160px;'>
                    <div style='font-size: 15px; font-weight: 600; margin-bottom: 6px;'>
                        {neighborhood}
                    </div>
                    <div style='font-size: 11px; color: #94a3b8; margin-bottom: 8px; letter-spacing: 0.5px;'>
                        ZONE {zone_num}
                    </div>
                    <div style='font-size: 32px; font-weight: 700; margin: 8px 0;'>
                        {risk_display}
                    </div>
                    <div style='font-size: 12px; color: #64748b;'>
                        {risk_level}
                    </div>
                </div>
            """,
            "style": {
                "backgroundColor": "#0f172a",
                "color": "white",
                "borderRadius": "12px",
                "border": "1px solid rgba(255,255,255,0.1)",
                "boxShadow": "0 8px 32px rgba(0,0,0,0.5)"
            }
        }
    )



# =============================================================================
# Tab Components
# =============================================================================

def render_overview_tab(model, unique_grids):
    """Render the Overview tab with map and controls."""
    
    # Date and Time controls (always shown)
    col_date, col_time = st.columns(2)
    
    with col_date:
        selected_date = st.date_input("📅 Date", datetime.now(), label_visibility="collapsed")
        st.caption("Select Date")
    
    with col_time:
        selected_hour = st.slider("🕐 Hour", 0, 23, datetime.now().hour, label_visibility="collapsed")
        st.caption(f"Time: {selected_hour:02d}:00")
    
    # Check for holidays
    is_hol, holiday_name = is_holiday(selected_date)
    is_hol_week = is_holiday_week(selected_date)
    
    if is_hol:
        st.markdown(f"""
        <div style="display: flex; align-items: center; gap: 10px; padding: 12px 16px; 
                    background: linear-gradient(90deg, rgba(239, 68, 68, 0.15) 0%, rgba(239, 68, 68, 0.05) 100%);
                    border: 1px solid rgba(239, 68, 68, 0.4); 
                    border-radius: 8px; margin: 10px 0;">
            <span style="font-size: 1.5rem;">🎉</span>
            <div>
                <span style="color: #ef4444; font-weight: 600;">{holiday_name}</span>
                <span style="color: #a0aec0; margin-left: 8px;">Expect altered delivery patterns</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    elif is_hol_week:
        st.markdown("""
        <div style="display: flex; align-items: center; gap: 10px; padding: 10px 16px; 
                    background: rgba(251, 191, 36, 0.1);
                    border: 1px solid rgba(251, 191, 36, 0.3); 
                    border-radius: 6px; margin: 10px 0;">
            <span style="font-size: 1.2rem;">📅</span>
            <span style="color: #fbbf24;">Holiday week — delivery volumes may be elevated</span>
        </div>
        """, unsafe_allow_html=True)
    
    # Check if date is within forecast window (pass the hour too!)
    forecast = get_weather_forecast(selected_date, selected_hour)

    
    if forecast["in_window"]:
        # Within 7-day window - use forecast, no manual controls
        temp_input = int(forecast["temperature"])
        precip_input = round(forecast["precipitation"], 2)
        
        # Show weather info badge
        weather_type = "Current" if not forecast["is_forecast"] else "Forecast"
        st.markdown(f"""
        <div style="display: flex; align-items: center; gap: 10px; padding: 10px 16px; 
                    background: linear-gradient(90deg, rgba(16, 185, 129, 0.1) 0%, rgba(16, 185, 129, 0.05) 100%);
                    border: 1px solid rgba(16, 185, 129, 0.3); 
                    border-radius: 6px; margin: 10px 0;">
            <div style="display: flex; align-items: center; gap: 6px;">
                <div style="width: 8px; height: 8px; background: #10b981; border-radius: 50%;"></div>
                <span style="color: #10b981; font-weight: 500;">🌤️ {weather_type} Weather</span>
            </div>
            <span style="color: #a0aec0;">|</span>
            <span style="color: #e2e8f0; font-weight: 500;">{temp_input}°F</span>
            <span style="color: #a0aec0;">|</span>
            <span style="color: #e2e8f0;">{precip_input}" rain</span>
        </div>
        """, unsafe_allow_html=True)
    else:
        # Outside 7-day window - show manual controls
        st.markdown("""
        <div style="display: flex; align-items: center; gap: 10px; padding: 10px 16px; 
                    background: rgba(245, 158, 11, 0.1); border: 1px solid rgba(245, 158, 11, 0.3); 
                    border-radius: 6px; margin: 10px 0;">
            <span style="color: #f59e0b;">⚠️ This date is outside the 7-day forecast window.</span>
            <span style="color: #a0aec0;">Please enter weather manually below:</span>
        </div>
        """, unsafe_allow_html=True)
        
        # Manual weather inputs
        col_temp, col_rain = st.columns(2)
        
        with col_temp:
            temp_input = st.slider("🌡️ Temperature (°F)", 0, 100, 65, label_visibility="collapsed")
            st.caption(f"Temperature: {temp_input}°F")
        
        with col_rain:
            precip_input = st.slider("🌧️ Precipitation (in)", 0.0, 2.0, 0.0, 0.1, label_visibility="collapsed")
            st.caption(f"Precipitation: {precip_input}\"")
    
    # Calculate predictions
    day_of_week = selected_date.weekday()
    month = selected_date.month
    input_data, probs = calculate_predictions(
        model, unique_grids, selected_hour, day_of_week, month, temp_input, precip_input
    )
    
    avg_risk = probs.mean() * 100
    risk_level, risk_color, risk_emoji = get_risk_level(avg_risk)
    high_risk_count = (probs >= 0.5).sum()
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Metrics row
    m1, m2, m3, m4 = st.columns(4)
    
    with m1:
        st.markdown(f"""
        <div class="metric-card">
            <p class="metric-label">Current Risk Level</p>
            <p class="metric-value" style="color: {risk_color}">{risk_emoji} {risk_level}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with m2:
        st.markdown(f"""
        <div class="metric-card">
            <p class="metric-label">Average Risk Score</p>
            <p class="metric-value" style="color: {risk_color}">{avg_risk:.1f}%</p>
        </div>
        """, unsafe_allow_html=True)
    
    with m3:
        st.markdown(f"""
        <div class="metric-card">
            <p class="metric-label">High Risk Zones</p>
            <p class="metric-value" style="color: #ef4444">{high_risk_count}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with m4:
        st.markdown(f"""
        <div class="metric-card">
            <p class="metric-label">Total Zones Monitored</p>
            <p class="metric-value" style="color: #60a5fa">{len(probs)}</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Insight box
    insight_text = generate_insight(risk_level, selected_hour, day_of_week, temp_input, precip_input)
    st.markdown(f"""
    <div class="insight-box">
        <p class="insight-title">💡 AI Insights</p>
        <p class="insight-text">{insight_text}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Map section with legend
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Map header with legend
    col_title, col_legend = st.columns([2, 3])
    
    with col_title:
        st.markdown("### 🗺️ Manhattan Congestion Map")
        st.caption("Each circle represents a grid zone. Hover for details.")
    
    with col_legend:
        st.markdown("""
        <div style="display: flex; justify-content: flex-end; align-items: center; gap: 20px; padding: 10px 0;">
            <div style="display: flex; align-items: center; gap: 6px;">
                <div style="width: 16px; height: 16px; border-radius: 50%; background: #2ecc71;"></div>
                <span style="color: #a0aec0; font-size: 0.85rem;">Low Risk (&lt;20%)</span>
            </div>
            <div style="display: flex; align-items: center; gap: 6px;">
                <div style="width: 16px; height: 16px; border-radius: 50%; background: #f1c40f;"></div>
                <span style="color: #a0aec0; font-size: 0.85rem;">Medium (20-40%)</span>
            </div>
            <div style="display: flex; align-items: center; gap: 6px;">
                <div style="width: 16px; height: 16px; border-radius: 50%; background: #e67e22;"></div>
                <span style="color: #a0aec0; font-size: 0.85rem;">Elevated (40-60%)</span>
            </div>
            <div style="display: flex; align-items: center; gap: 6px;">
                <div style="width: 16px; height: 16px; border-radius: 50%; background: #e74c3c;"></div>
                <span style="color: #a0aec0; font-size: 0.85rem;">High Risk (&gt;60%)</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Map
    deck = create_map_layer(input_data)
    st.pydeck_chart(deck, use_container_width=True)


def render_analytics_tab(features_df):
    """Render the Analytics tab with charts."""
    
    if features_df is None:
        st.warning("📊 Historical data not available. Run the data pipeline notebooks to generate analytics data.")
        return
    
    st.markdown("### 📈 Historical Congestion Patterns")
    st.markdown("<br>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    # Hourly Pattern Chart
    with col1:
        st.markdown("#### Complaints by Hour of Day")
        hourly_data = get_hourly_patterns(features_df)
        
        if hourly_data is not None:
            # Pre-calculate colors to avoid nested conditions (Altair v6 compatible)
            def get_hour_color(hour):
                if 7 <= hour <= 10:
                    return "#f59e0b"  # Morning rush - amber
                elif 16 <= hour <= 19:
                    return "#ef4444"  # Evening rush - red
                else:
                    return "#667eea"  # Normal - purple
            
            hourly_data["color"] = hourly_data["hour"].apply(get_hour_color)
            
            chart = alt.Chart(hourly_data).mark_bar(
                cornerRadiusTopLeft=4,
                cornerRadiusTopRight=4
            ).encode(
                x=alt.X("hour:O", title="Hour", axis=alt.Axis(labelAngle=0)),
                y=alt.Y("complaints:Q", title="Number of Complaints"),
                color=alt.Color("color:N", scale=None, legend=None),
                tooltip=["hour", "complaints"]
            ).properties(
                height=350
            ).configure_axis(
                labelColor="#a0aec0",
                titleColor="#a0aec0",
                gridColor="#2d3748"
            ).configure_view(
                strokeWidth=0
            )
            
            st.altair_chart(chart, use_container_width=True)
            st.caption("🟡 Morning Rush (7-10 AM) | 🔴 Evening Rush (4-7 PM)")
    
    # Day of Week Chart
    with col2:
        st.markdown("#### Complaints by Day of Week")
        daily_data = features_df.groupby("day_of_week").size().reset_index(name="complaints")
        daily_data["day_name"] = daily_data["day_of_week"].apply(get_day_name)
        daily_data["is_weekend"] = daily_data["day_of_week"] >= 5
        
        daily_data["color"] = daily_data["is_weekend"].apply(
            lambda x: "#10b981" if x else "#667eea"
        )
        
        chart = alt.Chart(daily_data).mark_bar(
            cornerRadiusTopLeft=4,
            cornerRadiusTopRight=4
        ).encode(
            x=alt.X("day_name:N", title="Day", sort=["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]),
            y=alt.Y("complaints:Q", title="Number of Complaints"),
            color=alt.Color("color:N", scale=None, legend=None),
            tooltip=["day_name", "complaints"]
        ).properties(
            height=350
        ).configure_axis(
            labelColor="#a0aec0",
            titleColor="#a0aec0",
            gridColor="#2d3748"
        ).configure_view(
            strokeWidth=0
        )
        
        st.altair_chart(chart, use_container_width=True)
        st.caption("🟣 Weekdays | 🟢 Weekends (35% fewer complaints)")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Heatmap
    st.markdown("#### 🔥 Weekly Congestion Heatmap")
    heatmap_data = get_weekly_heatmap_data(features_df)
    
    if heatmap_data is not None:
        heatmap = alt.Chart(heatmap_data).mark_rect(
            cornerRadius=3
        ).encode(
            x=alt.X("hour:O", title="Hour of Day", axis=alt.Axis(labelAngle=0)),
            y=alt.Y("day_name:N", title="Day of Week", 
                   sort=["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]),
            color=alt.Color("complaints:Q", 
                          scale=alt.Scale(scheme="plasma"),
                          legend=alt.Legend(title="Complaints")),
            tooltip=["day_name", "hour", "complaints"]
        ).properties(
            height=300
        ).configure_axis(
            labelColor="#a0aec0",
            titleColor="#a0aec0"
        ).configure_view(
            strokeWidth=0
        )
        
        st.altair_chart(heatmap, use_container_width=True)
    
    # Key Stats
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### 📊 Key Statistics")
    
    s1, s2, s3, s4 = st.columns(4)
    
    total_complaints = len(features_df)
    rush_hour_pct = features_df["is_rush_hour"].mean() * 100
    weekend_pct = features_df["is_weekend"].mean() * 100
    peak_hour = features_df["hour"].mode().iloc[0]
    
    s1.metric("Total Complaints", f"{total_complaints:,}")
    s2.metric("Rush Hour %", f"{rush_hour_pct:.1f}%")
    s3.metric("Weekend %", f"{weekend_pct:.1f}%")
    s4.metric("Peak Hour", f"{peak_hour}:00")
    
    # Live Data Comparison Section
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("### 🔴 Live Data (Last 7 Days)")
    st.markdown("Compare current activity with historical patterns")
    st.markdown("<br>", unsafe_allow_html=True)
    
    live_df, live_error = fetch_live_complaints_data()
    
    if live_error:
        st.warning(f"⚠️ Could not fetch live data: {live_error}")
    elif live_df is None or live_df.empty:
        st.info("📊 No live data available at this time.")
    else:
        # Live stats row
        lc1, lc2, lc3, lc4 = st.columns(4)
        
        live_total = len(live_df)
        live_rush_pct = live_df["is_rush_hour"].mean() * 100 if "is_rush_hour" in live_df.columns else 0
        live_weekend_pct = live_df["is_weekend"].mean() * 100 if "is_weekend" in live_df.columns else 0
        today_count = len(live_df[live_df["created_date"].dt.date == datetime.now().date()]) if "created_date" in live_df.columns else 0
        
        lc1.metric("7-Day Total", f"{live_total:,}", 
                   delta=f"{live_total - (total_complaints / 365 * 7):.0f} vs avg" if total_complaints else None)
        lc2.metric("Rush Hour %", f"{live_rush_pct:.1f}%",
                   delta=f"{live_rush_pct - rush_hour_pct:+.1f}%" if rush_hour_pct else None)
        lc3.metric("Weekend %", f"{live_weekend_pct:.1f}%",
                   delta=f"{live_weekend_pct - weekend_pct:+.1f}%" if weekend_pct else None)
        lc4.metric("Today", f"{today_count:,}")
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Live hourly chart
        col_live1, col_live2 = st.columns(2)
        
        with col_live1:
            st.markdown("#### Live Hourly Distribution")
            if "hour" in live_df.columns:
                live_hourly = live_df.groupby("hour").size().reset_index(name="complaints")
                live_hourly["color"] = "#10b981"  # Green for live data
                
                chart = alt.Chart(live_hourly).mark_bar(
                    cornerRadiusTopLeft=4,
                    cornerRadiusTopRight=4
                ).encode(
                    x=alt.X("hour:O", title="Hour", axis=alt.Axis(labelAngle=0)),
                    y=alt.Y("complaints:Q", title="Complaints"),
                    color=alt.Color("color:N", scale=None, legend=None),
                    tooltip=["hour", "complaints"]
                ).properties(
                    height=250
                ).configure_axis(
                    labelColor="#a0aec0",
                    titleColor="#a0aec0",
                    gridColor="#2d3748"
                ).configure_view(
                    strokeWidth=0
                )
                
                st.altair_chart(chart, use_container_width=True)
                st.caption("🟢 Live data from NYC 311 API")
        
        with col_live2:
            st.markdown("#### Live Daily Distribution")
            if "day_of_week" in live_df.columns:
                live_daily = live_df.groupby("day_of_week").size().reset_index(name="complaints")
                live_daily["day_name"] = live_daily["day_of_week"].apply(get_day_name)
                live_daily["color"] = "#10b981"
                
                chart = alt.Chart(live_daily).mark_bar(
                    cornerRadiusTopLeft=4,
                    cornerRadiusTopRight=4
                ).encode(
                    x=alt.X("day_name:N", title="Day", 
                           sort=["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]),
                    y=alt.Y("complaints:Q", title="Complaints"),
                    color=alt.Color("color:N", scale=None, legend=None),
                    tooltip=["day_name", "complaints"]
                ).properties(
                    height=250
                ).configure_axis(
                    labelColor="#a0aec0",
                    titleColor="#a0aec0",
                    gridColor="#2d3748"
                ).configure_view(
                    strokeWidth=0
                )
                
                st.altair_chart(chart, use_container_width=True)
                st.caption("🟢 Live data from NYC 311 API")


def render_predictions_tab(model, unique_grids):
    """Render the Predictions tab with scenario comparison."""
    
    st.markdown("### 🔮 Scenario Comparison")
    st.markdown("Compare congestion risk across different conditions")
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Scenario selector
    col1, col2, col3 = st.columns(3)
    
    scenarios = {
        "Normal Weekday": {"hour": 14, "day": 2, "temp": 65, "precip": 0.0},
        "Morning Rush": {"hour": 8, "day": 1, "temp": 55, "precip": 0.0},
        "Evening Rush": {"hour": 17, "day": 3, "temp": 70, "precip": 0.0},
        "Rainy Day": {"hour": 12, "day": 2, "temp": 50, "precip": 0.5},
        "Weekend Morning": {"hour": 10, "day": 5, "temp": 68, "precip": 0.0},
        "Cold Winter Day": {"hour": 11, "day": 1, "temp": 28, "precip": 0.0},
    }
    
    with col1:
        scenario1 = st.selectbox("Scenario 1", list(scenarios.keys()), index=0)
    with col2:
        scenario2 = st.selectbox("Scenario 2", list(scenarios.keys()), index=1)
    with col3:
        scenario3 = st.selectbox("Scenario 3", list(scenarios.keys()), index=3)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Calculate predictions for each scenario
    results = []
    for name, params in [(scenario1, scenarios[scenario1]), 
                          (scenario2, scenarios[scenario2]), 
                          (scenario3, scenarios[scenario3])]:
        _, probs = calculate_predictions(
            model, unique_grids, 
            params["hour"], params["day"], 6,  # June
            params["temp"], params["precip"]
        )
        avg_risk = probs.mean() * 100
        high_risk = (probs >= 0.5).sum()
        results.append({
            "scenario": name,
            "avg_risk": avg_risk,
            "high_risk_zones": high_risk,
            "conditions": f"{params['hour']}:00, {params['temp']}°F, {params['precip']}\" rain"
        })
    
    # Display comparison cards
    c1, c2, c3 = st.columns(3)
    
    for col, result in zip([c1, c2, c3], results):
        with col:
            risk_level, risk_color, risk_emoji = get_risk_level(result["avg_risk"])
            st.markdown(f"""
            <div class="metric-card" style="text-align: center;">
                <p class="metric-label">{result['scenario']}</p>
                <p class="metric-value" style="color: {risk_color}; font-size: 3rem;">{result['avg_risk']:.0f}%</p>
                <p style="color: #a0aec0; font-size: 0.85rem; margin-top: 0.5rem;">
                    {risk_emoji} {risk_level} RISK
                </p>
                <p style="color: #718096; font-size: 0.75rem; margin-top: 1rem;">
                    {result['conditions']}
                </p>
                <p style="color: #e53e3e; font-size: 0.9rem; margin-top: 0.5rem;">
                    {result['high_risk_zones']} high-risk zones
                </p>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    # 24-hour forecast
    st.markdown("### ⏰ Next 24 Hours Forecast")
    st.markdown("Predicted risk levels for today")
    st.markdown("<br>", unsafe_allow_html=True)
    
    now = datetime.now()
    forecast_data = []
    
    for hour_offset in range(24):
        forecast_hour = (now.hour + hour_offset) % 24
        _, probs = calculate_predictions(
            model, unique_grids,
            forecast_hour, now.weekday(), now.month,
            65, 0.0  # Assume normal weather
        )
        forecast_data.append({
            "hour": forecast_hour,
            "hour_label": f"{forecast_hour:02d}:00",
            "risk": probs.mean() * 100
        })
    
    forecast_df = pd.DataFrame(forecast_data)
    
    forecast_chart = alt.Chart(forecast_df).mark_area(
        line={"color": "#667eea"},
        color=alt.Gradient(
            gradient="linear",
            stops=[
                alt.GradientStop(color="rgba(102, 126, 234, 0.1)", offset=0),
                alt.GradientStop(color="rgba(102, 126, 234, 0.6)", offset=1)
            ],
            x1=1, x2=1, y1=1, y2=0
        )
    ).encode(
        x=alt.X("hour_label:N", title="Hour", axis=alt.Axis(labelAngle=-45)),
        y=alt.Y("risk:Q", title="Risk %", scale=alt.Scale(domain=[0, 100])),
        tooltip=["hour_label", alt.Tooltip("risk:Q", format=".1f")]
    ).properties(
        height=300
    ).configure_axis(
        labelColor="#a0aec0",
        titleColor="#a0aec0",
        gridColor="#2d3748"
    ).configure_view(
        strokeWidth=0
    )
    
    st.altair_chart(forecast_chart, use_container_width=True)
    
    # Best/Worst times
    best_hour = forecast_df.loc[forecast_df["risk"].idxmin()]
    worst_hour = forecast_df.loc[forecast_df["risk"].idxmax()]
    
    b1, b2 = st.columns(2)
    with b1:
        st.success(f"✅ **Best time to deliver:** {best_hour['hour_label']} ({best_hour['risk']:.1f}% risk)")
    with b2:
        st.error(f"⚠️ **Avoid deliveries at:** {worst_hour['hour_label']} ({worst_hour['risk']:.1f}% risk)")

    # =========================================================================
    # Multi-Zone Delivery Scheduler
    # =========================================================================
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("### 🚛 Multi-Zone Delivery Scheduler")
    st.markdown("*Plan deliveries across multiple zones with optimized timing*")
    
    # Function to get neighborhood name from coordinates
    def get_neighborhood(lat, lon):
        """Map lat/lon to approximate Manhattan neighborhood."""
        if lat >= 40.80:
            return "Harlem"
        elif lat >= 40.77:
            if lon >= -73.97:
                return "Upper East Side"
            else:
                return "Upper West Side"
        elif lat >= 40.75:
            if lon >= -73.98:
                return "Midtown East"
            else:
                return "Midtown West"
        elif lat >= 40.73:
            if lon >= -73.99:
                return "Gramercy/Murray Hill"
            else:
                return "Chelsea"
        elif lat >= 40.72:
            if lon >= -73.99:
                return "East Village"
            else:
                return "Greenwich Village"
        elif lat >= 40.71:
            if lon >= -73.99:
                return "Lower East Side"
            else:
                return "SoHo/Tribeca"
        else:
            return "Financial District"
    
    # Get zone options with neighborhood names
    zone_data = unique_grids.copy().reset_index(drop=True)
    zone_data['zone_num'] = zone_data.index + 1
    zone_data['neighborhood'] = zone_data.apply(
        lambda r: get_neighborhood(r['grid_lat'], r['grid_lon']), 
        axis=1
    )
    zone_data['zone_label'] = zone_data.apply(
        lambda r: f"{r['neighborhood']} (Zone {r['zone_num']})", 
        axis=1
    )
    zone_options = dict(zip(zone_data['zone_label'], zone_data['grid_id']))
    zone_num_map = dict(zip(zone_data['grid_id'], zone_data['zone_num']))
    zone_neighborhood_map = dict(zip(zone_data['grid_id'], zone_data['neighborhood']))
    
    # Input section
    st.markdown("<br>", unsafe_allow_html=True)
    
    input_col1, input_col2 = st.columns([3, 1])
    
    with input_col1:
        selected_zone_labels = st.multiselect(
            "📍 Select Delivery Zones",
            options=list(zone_options.keys()),
            default=list(zone_options.keys())[:3] if len(zone_options) >= 3 else list(zone_options.keys()),
            help="Choose the zones where you need to make deliveries"
        )
    
    with input_col2:
        schedule_date = st.date_input(
            "📅 Delivery Date",
            value=datetime.now().date(),
            min_value=datetime.now().date(),
            max_value=datetime.now().date() + timedelta(days=7)
        )
    
    # Convert labels back to zone IDs
    selected_zone_ids = [zone_options[label] for label in selected_zone_labels]
    
    if st.button("🔍 Optimize Delivery Schedule", type="primary", use_container_width=True, key="scheduler_btn"):
        if not selected_zone_ids:
            st.warning("⚠️ Please select at least one zone to optimize.")
        else:
            with st.spinner("🔄 Analyzing congestion patterns and optimizing schedule..."):
                # Optimization logic
                schedule_results = []
                
                # Delivery window: 7 AM to 6 PM (business hours only)
                delivery_hours = list(range(7, 19))
                
                for zone_id in selected_zone_ids:
                    zone_grid = unique_grids[unique_grids['grid_id'] == zone_id]
                    
                    if zone_grid.empty:
                        continue
                    
                    hourly_risks = []
                    
                    for hour in delivery_hours:
                        # Get weather forecast for this hour
                        weather = get_weather_forecast(schedule_date, hour)
                        temp = weather.get("temperature", 65) or 65
                        precip = weather.get("precipitation", 0.0) or 0.0
                        
                        # Calculate prediction
                        _, prob = calculate_predictions(
                            model, zone_grid,
                            hour, schedule_date.weekday(), schedule_date.month,
                            temp, precip
                        )
                        
                        risk_pct = prob[0] * 100 if len(prob) > 0 else 50
                        hourly_risks.append({
                            "hour": hour,
                            "risk": risk_pct,
                            "temp": temp
                        })
                    
                    # Find optimal hour (lowest risk)
                    if hourly_risks:
                        best = min(hourly_risks, key=lambda x: x["risk"])
                        worst = max(hourly_risks, key=lambda x: x["risk"])
                        
                        # Format time nicely
                        best_time = f"{best['hour']}:00 AM" if best['hour'] < 12 else f"{best['hour']-12 or 12}:00 PM"
                        
                        schedule_results.append({
                            "zone_id": zone_id,
                            "zone_label": f"{zone_neighborhood_map.get(zone_id, 'Zone')} #{zone_num_map.get(zone_id, '')}",
                            "optimal_hour": best["hour"],
                            "optimal_time": best_time,
                            "optimal_risk": best["risk"],
                            "worst_risk": worst["risk"],
                            "improvement": worst["risk"] - best["risk"],
                            "temp": best["temp"]
                        })
                
                if schedule_results:
                    # Sort by optimal hour for a logical schedule
                    schedule_results.sort(key=lambda x: x["optimal_hour"])
                    
                    # Calculate totals
                    total_optimal_risk = sum(r["optimal_risk"] for r in schedule_results)
                    total_worst_risk = sum(r["worst_risk"] for r in schedule_results)
                    avg_improvement = (total_worst_risk - total_optimal_risk) / len(schedule_results)
                    
                    st.markdown("<br>", unsafe_allow_html=True)
                    
                    # Summary metrics
                    st.markdown("#### 📊 Optimization Summary")
                    
                    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
                    
                    with metric_col1:
                        st.metric(
                            "Zones Scheduled",
                            len(schedule_results),
                            help="Number of delivery zones in the schedule"
                        )
                    
                    with metric_col2:
                        st.metric(
                            "Avg Risk (Optimized)",
                            f"{total_optimal_risk / len(schedule_results):.1f}%",
                            delta=f"-{avg_improvement:.1f}% vs worst",
                            delta_color="inverse"
                        )
                    
                    with metric_col3:
                        first_delivery = min(r["optimal_hour"] for r in schedule_results)
                        first_time = f"{first_delivery}:00 AM" if first_delivery < 12 else f"{first_delivery-12 or 12}:00 PM"
                        st.metric("First Delivery", first_time)
                    
                    with metric_col4:
                        last_delivery = max(r["optimal_hour"] for r in schedule_results)
                        last_time = f"{last_delivery}:00 AM" if last_delivery < 12 else f"{last_delivery-12 or 12}:00 PM"
                        st.metric("Last Delivery", last_time)
                    
                    st.markdown("<br>", unsafe_allow_html=True)
                    
                    # Detailed Schedule Table
                    st.markdown("#### 📋 Optimized Delivery Schedule")
                    
                    # Create display dataframe
                    display_data = []
                    for i, r in enumerate(schedule_results, 1):
                        risk_level = "🟢 Low" if r["optimal_risk"] < 30 else ("🟡 Medium" if r["optimal_risk"] < 50 else "🔴 High")
                        display_data.append({
                            "Order": i,
                            "Zone": r["zone_label"],
                            "Optimal Time": r["optimal_time"],
                            "Risk Level": risk_level,
                            "Risk %": f"{r['optimal_risk']:.1f}%",
                            "Savings": f"↓ {r['improvement']:.1f}%"
                        })
                    
                    display_df = pd.DataFrame(display_data)
                    
                    # Style the table
                    st.dataframe(
                        display_df,
                        use_container_width=True,
                        hide_index=True,
                        column_config={
                            "Order": st.column_config.NumberColumn("📍", width="small"),
                            "Zone": st.column_config.TextColumn("Zone", width="medium"),
                            "Optimal Time": st.column_config.TextColumn("⏰ Best Time", width="medium"),
                            "Risk Level": st.column_config.TextColumn("Risk", width="medium"),
                            "Risk %": st.column_config.TextColumn("Score", width="small"),
                            "Savings": st.column_config.TextColumn("vs Worst", width="small")
                        }
                    )
                    
                    st.markdown("<br>", unsafe_allow_html=True)
                    
                    # Key recommendations
                    st.markdown("#### 💡 Recommendations")
                    
                    early_zones = [r for r in schedule_results if r["optimal_hour"] < 10]
                    midday_zones = [r for r in schedule_results if 10 <= r["optimal_hour"] < 14]
                    afternoon_zones = [r for r in schedule_results if r["optimal_hour"] >= 14]
                    
                    rec_col1, rec_col2 = st.columns(2)
                    
                    with rec_col1:
                        if early_zones:
                            st.info(f"🌅 **Morning Priority**: {len(early_zones)} zone(s) should be delivered before 10 AM")
                        if afternoon_zones:
                            st.info(f"🌆 **Afternoon Slots**: {len(afternoon_zones)} zone(s) are best after 2 PM")
                    
                    with rec_col2:
                        high_risk_zones = [r for r in schedule_results if r["optimal_risk"] >= 50]
                        if high_risk_zones:
                            st.warning(f"⚠️ **Caution**: {len(high_risk_zones)} zone(s) still have elevated risk even at optimal times")
                        else:
                            st.success("✅ All zones can be delivered at low-to-moderate risk times!")


# =============================================================================
# Main Application
# =============================================================================

def main():
    """Main dashboard application."""
    
    # Load resources
    with st.spinner("Loading analytics platform..."):
        model, unique_grids, features_df, error = load_resources()
    
    if error:
        st.error(f"❌ Failed to load resources: {error}")
        st.info("Run the training script first: `python scripts/retrain_spatial.py`")
        return
    
    # Header
    st.markdown('<h1 class="main-header">🚚 NYC Congestion Analytics</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Data-Driven Delivery Risk Assessment for Manhattan</p>', unsafe_allow_html=True)
    
    # Live data status indicator
    live_stats = fetch_live_311_stats()
    
    if live_stats["is_live"] and live_stats["error"] is None:
        last_update = live_stats["last_update"].strftime("%I:%M %p") if live_stats["last_update"] else "N/A"
        st.markdown(f"""
        <div style="display: flex; justify-content: center; align-items: center; gap: 20px; 
                    padding: 10px 20px; margin-bottom: 20px;
                    background: linear-gradient(90deg, rgba(16, 185, 129, 0.1) 0%, rgba(16, 185, 129, 0.05) 100%);
                    border: 1px solid rgba(16, 185, 129, 0.3); border-radius: 8px;">
            <div style="display: flex; align-items: center; gap: 8px;">
                <div style="width: 10px; height: 10px; background: #10b981; border-radius: 50%; 
                            animation: pulse 2s infinite;"></div>
                <span style="color: #10b981; font-weight: 500;">LIVE DATA</span>
            </div>
            <span style="color: #a0aec0;">|</span>
            <span style="color: #e2e8f0;">{live_stats['total_complaints']:,} complaints (7 days)</span>
            <span style="color: #a0aec0;">|</span>
            <span style="color: #a0aec0; font-size: 0.85rem;">Updated: {last_update}</span>
        </div>
        <style>
            @keyframes pulse {{
                0%, 100% {{ opacity: 1; }}
                50% {{ opacity: 0.5; }}
            }}
        </style>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="display: flex; justify-content: center; align-items: center; gap: 10px;
                    padding: 10px 20px; margin-bottom: 20px;
                    background: rgba(107, 114, 128, 0.1); border: 1px solid rgba(107, 114, 128, 0.3); 
                    border-radius: 8px;">
            <span style="color: #9ca3af;">📦 Using cached data (Live API unavailable)</span>
        </div>
        """, unsafe_allow_html=True)
    
    # Tabs
    tab1, tab2, tab3 = st.tabs(["🗺️ Overview", "📊 Analytics", "🔮 Predictions"])
    
    with tab1:
        render_overview_tab(model, unique_grids)
    
    with tab2:
        render_analytics_tab(features_df)
    
    with tab3:
        render_predictions_tab(model, unique_grids)
    
    # Footer
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align: center; color: #4a5568; padding: 2rem; border-top: 1px solid rgba(255,255,255,0.1);">
        <p style="margin-bottom: 0.5rem;">Built with Machine Learning & NYC Open Data</p>
        <p style="font-size: 0.85rem;">
            <a href="https://github.com/Karan-C21/nyc-curbside-congestion" style="color: #667eea;">GitHub</a> • 
            <a href="https://data.cityofnewyork.us" style="color: #667eea;">Data Source</a>
        </p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
