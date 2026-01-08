<p align="center">
  <h1 align="center">🚚 NYC Curbside Congestion Predictor</h1>
  <p align="center">
    <strong>Predicting delivery truck congestion patterns across Manhattan using machine learning</strong>
  </p>
  <p align="center">
    <img src="https://img.shields.io/badge/Python-3.9+-blue?logo=python&logoColor=white" alt="Python">
    <img src="https://img.shields.io/badge/Streamlit-1.28+-FF4B4B?logo=streamlit&logoColor=white" alt="Streamlit">
    <img src="https://img.shields.io/badge/scikit--learn-1.3+-F7931E?logo=scikit-learn&logoColor=white" alt="scikit-learn">
    <img src="https://img.shields.io/badge/License-MIT-green" alt="License">
  </p>
</p>

---

## 📋 Overview

This project analyzes **NYC 311 complaint data** to predict where and when delivery truck congestion is most likely to occur across Manhattan. The interactive dashboard allows logistics planners, city officials, and researchers to explore congestion risk under different conditions.

### Key Features

- 🗺️ **Interactive Map** — Visualize congestion risk across Manhattan grid zones
- 🌦️ **Weather Integration** — Factor in temperature and precipitation impacts
- ⏰ **Temporal Analysis** — Understand rush hour and weekend patterns
- 🤖 **ML Predictions** — Random Forest model with balanced class handling

---

## 🏗️ Project Structure

```
nyc-curbside-congestion/
├── app/
│   └── app.py                 # Streamlit dashboard
├── data/                      # Data files (gitignored)
│   ├── 311_truck_broad_filtered.csv
│   ├── complaints_with_features.csv
│   ├── modeling_dataset.csv
│   └── nyc_weather_2023_present.csv
├── models/                    # Trained models (gitignored)
│   └── random_forest_weather_enhanced.pkl
├── notebooks/
│   ├── 01_data_loading_and_exploration.ipynb
│   ├── 02_fixed_exploration.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_spatial_and_aggregation.ipynb
│   ├── 05_modeling.ipynb
│   └── 06_external_data_integration.ipynb
├── scripts/
│   ├── fetch_weather_data.py  # Weather API integration
│   ├── validate_features.py   # Feature engineering validation
│   ├── retrain_spatial.py     # Model training script
│   ├── fix_class_imbalance.py # Notebook patcher utility
│   └── check_step5.py         # Quick model validation
├── src/
│   ├── __init__.py
│   ├── config.py              # Centralized configuration
│   └── utils.py               # Shared utility functions
├── requirements.txt
├── LICENSE
└── README.md
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.9 or higher
- pip package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Karan-C21/nyc-curbside-congestion.git
   cd nyc-curbside-congestion
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   
   # Windows
   venv\Scripts\activate
   
   # macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the dashboard**
   ```bash
   streamlit run app/app.py
   ```

---

## 📊 Data Pipeline

The project follows a structured notebook pipeline:

| Step | Notebook | Description |
|------|----------|-------------|
| 1 | `01_data_loading_and_exploration` | Load and explore raw 311 complaint data |
| 2 | `02_fixed_exploration` | Clean and filter truck-related complaints |
| 3 | `03_feature_engineering` | Extract temporal features (hour, day, rush hour) |
| 4 | `04_spatial_and_aggregation` | Create Manhattan grid zones and aggregate |
| 5 | `05_modeling` | Train baseline ML models |
| 6 | `06_external_data_integration` | Add weather data and train enhanced model |

---

## 🧠 Model Performance

The enhanced Random Forest model includes:
- **Temporal features**: hour, day of week, weekend flag, rush hour flag, month
- **Spatial features**: grid latitude/longitude
- **Weather features**: temperature, precipitation, weather condition flags

| Metric | Score |
|--------|-------|
| Accuracy | ~0.75 |
| Precision | ~0.65 |
| Recall | ~0.70 |
| F1 Score | ~0.67 |

*Note: Scores may vary based on data updates*

---

## 🛠️ Tech Stack

| Category | Technologies |
|----------|-------------|
| **Language** | Python 3.9+ |
| **ML Framework** | scikit-learn |
| **Dashboard** | Streamlit, PyDeck |
| **Data Processing** | Pandas, NumPy |
| **Visualization** | Matplotlib, Seaborn, Folium |
| **External Data** | Open-Meteo Weather API |

---

## 📁 Data Sources

- **NYC 311 Complaints**: [NYC Open Data Portal](https://data.cityofnewyork.us/Social-Services/311-Service-Requests-from-2010-to-Present/erm2-nwe9)
- **Weather Data**: [Open-Meteo Historical API](https://open-meteo.com/)

---

## 🔮 Future Improvements

- [ ] Add real-time 311 data streaming
- [ ] Incorporate traffic camera data
- [ ] Deploy to Streamlit Cloud
- [ ] Add time-series forecasting
- [ ] Expand to all NYC boroughs

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

<p align="center">
  <sub>Built with ❤️ using NYC Open Data</sub>
</p>
