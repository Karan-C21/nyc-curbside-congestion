import streamlit as st
import pandas as pd
import joblib
import pydeck as pdk
from datetime import datetime
import numpy as np
import os

# -- Page Configuration --
st.set_page_config(
    page_title="NYC Delivery Congestion",
    page_icon="🚚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -- Styles --
st.markdown("""
<style>
    /* Sleek Dark Mode */
    .stApp {
        background-color: #0e1117;
        color: #fafafa;
    }
    .main_title {
        font-size: 2.5rem;
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 5px;
    }
    .subtitle {
        font-size: 1.1rem;
        color: #a0a0a0;
        margin-bottom: 25px;
    }
    /* Metrics */
    div[data-testid="stMetricValue"] {
        color: #ffffff;
    }
</style>
""", unsafe_allow_html=True)

# -- 1. Load Resources --
# @st.cache_resource
def load_resources():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.join(current_dir, '..')
    
    model_path = os.path.join(root_dir, 'models', 'random_forest_weather_enhanced.pkl')
    data_path = os.path.join(root_dir, 'data', 'modeling_dataset.csv')
    
    model = joblib.load(model_path)
    df = pd.read_csv(data_path)
    unique_grids = df[['grid_id', 'grid_lat', 'grid_lon']].drop_duplicates()
    
    return model, unique_grids

# Load data with a spinner for UX
with st.spinner("Loading intelligent traffic model..."):
    model, unique_grids = load_resources()

# -- 2. Layout & Header --
st.markdown('<div class="main_title">🚚 NYC Congestion Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Real-time AI risk assessment for delivery logistics</div>', unsafe_allow_html=True)

col_map, col_inputs = st.columns([3, 1]) # Wider map column

# -- 3. Sidebar / Input Panel (Right Column) --
with col_inputs:
    st.markdown("### ⚙️ Simulation Settings")
    with st.container():
        # Date & Time
        st.subheader("📅 Date & Time")
        selected_date = st.date_input("Select Date", datetime.now())
        selected_hour = st.slider("Hour of Day", 0, 23, 9, format="%d:00")
        
        # Weather
        st.subheader("🌦️ Weather Conditions")
        col_w1, col_w2 = st.columns(2)
        with col_w1:
            temp_input = st.number_input("Temp (°F)", 0, 100, 65, step=5)
        with col_w2:
            precip_input = st.number_input("Rain (mm)", 0.0, 50.0, 0.0, step=0.5)

        # Derived Logic
        day_of_week = selected_date.weekday()
        month = selected_date.month
        is_weekend = 1 if day_of_week >= 5 else 0
        is_rush_hour = 1 if (7 <= selected_hour <= 10) or (16 <= selected_hour <= 19) else 0
        is_rainy = 1 if precip_input > 0.1 else 0
        is_cold = 1 if temp_input < 40 else 0
        is_hot = 1 if temp_input > 85 else 0

# -- 4. Prediction Logic --
input_data = unique_grids.copy()
input_data['hour'] = selected_hour
input_data['day_of_week'] = day_of_week
input_data['is_weekend'] = is_weekend
input_data['is_rush_hour'] = is_rush_hour
input_data['month'] = month
input_data['grid_lat'] = unique_grids['grid_lat']
input_data['grid_lon'] = unique_grids['grid_lon']
input_data['avg_temp'] = temp_input
input_data['avg_precip'] = precip_input
input_data['pct_rainy'] = is_rainy
input_data['pct_cold'] = is_cold
input_data['pct_hot'] = is_hot

feature_cols = ['hour', 'day_of_week', 'is_weekend', 'is_rush_hour', 'month', 
                'grid_lat', 'grid_lon', 
                'avg_temp', 'avg_precip', 'pct_rainy', 'pct_cold', 'pct_hot']

# Get probabilities (0.0 to 1.0)
probs = model.predict_proba(input_data[feature_cols])[:, 1]
input_data['congestion_prob'] = probs

# Color Logic: Red (Congestion) vs Blue/Green (Clear)
def get_color(prob):
    # prob 0.0 -> Green/Blue (Clear)
    # prob 0.5 -> Yellow (Busy)
    # prob 1.0 -> Red (Gridlock)
    if prob < 0.2:
        return [46, 204, 113, 200]  # Emerald Green
    elif prob < 0.4:
        return [241, 196, 15, 200]  # Sunflower Yellow
    elif prob < 0.6:
        return [230, 126, 34, 200]  # Carrot Orange
    else:
        return [231, 76, 60, 200]   # Alizarin Red

input_data['color'] = input_data['congestion_prob'].apply(get_color)

# -- 5. Map Visualization --
with col_map:
    avg_risk = probs.mean() * 100
    risk_level = "LOW"
    color_hex = "#2ecc71"
    if avg_risk > 25: 
        risk_level = "MODERATE"
        color_hex = "#f1c40f"
    if avg_risk > 40: 
        risk_level = "HIGH"
        color_hex = "#e74c3c"

    st.markdown(f"<h3 style='color: {color_hex}; margin-bottom:0'>Current Risk Level: {risk_level}</h3>", unsafe_allow_html=True)
    st.caption(f"Average Congestion Probability: {avg_risk:.1f}%")

    # ColumnLayer (Cylinders/Hex-Pills)
    layer = pdk.Layer(
        "ColumnLayer",
        input_data,
        get_position=['grid_lon', 'grid_lat'],
        get_fill_color='color',
        get_elevation=0,       # Flat 2D discs
        elevation_scale=0,
        radius=300,            # 300m radius discs
        pickable=True,
        opacity=0.8,
        filled=True,
        stroked=True,
        get_line_color=[30, 30, 30, 255], # Dark borders for segmentation
        line_width_min_pixels=1,
    )

    view_state = pdk.ViewState(
        latitude=40.76,
        longitude=-73.97,
        zoom=11.5,
        pitch=0,
        bearing=0
    )

    r = pdk.Deck(
        layers=[layer],
        initial_view_state=view_state,
        map_style=pdk.map_styles.CARTO_DARK, # <--- Fix: Uses public dark map (No API Token needed)
        tooltip={"text": "Grid: {grid_id}\nRisk: {congestion_prob:.1%}", 
                 "style": {"backgroundColor": "#1f2937", "color": "white"}}
    )

    st.pydeck_chart(r)

# -- Footer --
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #888;">
    Built with Random Forest & Open Data • Refreshes automatically on input change
</div>
""", unsafe_allow_html=True)
