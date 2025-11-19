"""
Interactive AQI Prediction Dashboard
Save as: dashboard.py
Run with: streamlit run dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
from datetime import datetime
import os

st.set_page_config(page_title="AQI Predictor", page_icon="🌍", layout="wide")

st.markdown("""
    <style>
    .main-header {
        font-size: 48px;
        color: #1E88E5;
        text-align: center;
        padding: 20px;
        background: linear-gradient(90deg, #E3F2FD 0%, #BBDEFB 100%);
        border-radius: 10px;
        margin-bottom: 30px;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models():
    try:
        rf_model = joblib.load('models/random_forest_model.pkl')
        xgb_model = joblib.load('models/xgboost_model.pkl')
        feature_names = joblib.load('models/feature_names.pkl')
        return rf_model, xgb_model, feature_names
    except:
        return None, None, None

@st.cache_data
def load_data():
    try:
        data = pd.read_csv('data/aqi_data.csv')
        data['timestamp'] = pd.to_datetime(data['timestamp'])
        return data
    except:
        return None

def get_aqi_category(aqi_value):
    if aqi_value <= 50:
        return "Good", "#00E400", "🟢"
    elif aqi_value <= 100:
        return "Moderate", "#FFFF00", "🟡"
    elif aqi_value <= 150:
        return "Unhealthy for Sensitive", "#FF7E00", "🟠"
    elif aqi_value <= 200:
        return "Unhealthy", "#FF0000", "🔴"
    elif aqi_value <= 300:
        return "Very Unhealthy", "#8F3F97", "🟣"
    else:
        return "Hazardous", "#7E0023", "🟤"

def create_features(inputs):
    features = inputs.copy()
    features['pm25_rolling_3h'] = inputs['pm25']
    features['pm25_rolling_24h'] = inputs['pm25']
    features['traffic_rolling_3h'] = inputs['vehicle_count']
    features['traffic_wind_interaction'] = inputs['vehicle_count'] * (1 / (inputs['wind_speed'] + 1))
    features['temp_humidity_interaction'] = inputs['temperature'] * inputs['humidity'] / 100
    features['pm25_lag1'] = inputs['pm25']
    features['traffic_lag1'] = inputs['vehicle_count']
    return pd.DataFrame([features])

# Main App
st.markdown('<div class="main-header">🌍 Air Quality Index Prediction System</div>', unsafe_allow_html=True)

rf_model, xgb_model, feature_names = load_models()
data = load_data()

if rf_model is None:
    st.error("⚠️ Models not found! Run 'python aqi_prediction.py' first.")
    st.stop()

tab1, tab2, tab3 = st.tabs(["🔮 Predict AQI", "📈 Data Analysis", "ℹ️ About"])

with tab1:
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🚗 Traffic Parameters")
        vehicle_count = st.slider("Vehicle Count", 0, 400, 150)
        congestion_level = st.slider("Congestion Level (0-10)", 0.0, 10.0, 5.0, 0.1)
        avg_speed = st.slider("Average Speed (km/h)", 10, 80, 40)
        
        st.subheader("🌤️ Weather Parameters")
        temperature = st.slider("Temperature (°C)", 0, 45, 25)
        humidity = st.slider("Humidity (%)", 20, 100, 60)
        wind_speed = st.slider("Wind Speed (m/s)", 0.0, 15.0, 3.0, 0.1)
        rainfall = st.slider("Rainfall (mm)", 0.0, 10.0, 0.0, 0.1)
    
    with col2:
        st.subheader("💨 Pollution Parameters")
        pm25 = st.slider("PM2.5 (µg/m³)", 0, 250, 50)
        pm10 = st.slider("PM10 (µg/m³)", 0, 400, 75)
        no2 = st.slider("NO₂ (ppb)", 0, 150, 30)
        co = st.slider("CO (ppm)", 0.0, 5.0, 1.0, 0.1)
        o3 = st.slider("O₃ (ppb)", 0, 120, 50)
        so2 = st.slider("SO₂ (ppb)", 0, 50, 10)
        
        st.subheader("🕐 Time Parameters")
        hour = st.slider("Hour of Day", 0, 23, 12)
        day_of_week = st.selectbox("Day of Week", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
        month = st.slider("Month", 1, 12, 6)
    
    day_map = {"Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3, "Friday": 4, "Saturday": 5, "Sunday": 6}
    day_num = day_map[day_of_week]
    is_weekend = 1 if day_num >= 5 else 0
    is_peak_hour = 1 if (7 <= hour <= 9) or (17 <= hour <= 19) else 0
    
    inputs = {
        'pm25': pm25, 'pm10': pm10, 'no2': no2, 'co': co, 'o3': o3, 'so2': so2,
        'vehicle_count': vehicle_count, 'congestion_level': congestion_level,
        'avg_speed': avg_speed, 'temperature': temperature, 'humidity': humidity,
        'wind_speed': wind_speed, 'rainfall': rainfall, 'hour': hour,
        'day_of_week': day_num, 'month': month, 'is_weekend': is_weekend,
        'is_peak_hour': is_peak_hour
    }
    
    if st.button("🔮 Predict AQI", type="primary", use_container_width=True):
        features_df = create_features(inputs)
        
        rf_pred = rf_model.predict(features_df)[0]
        xgb_pred = xgb_model.predict(features_df)[0]
        avg_pred = (rf_pred + xgb_pred) / 2
        
        st.markdown("---")
        st.subheader("📊 Prediction Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            cat, color, emoji = get_aqi_category(rf_pred)
            st.markdown(f"<div style='background-color: {color}; padding: 20px; border-radius: 10px; text-align: center;'><h3 style='color: black;'>Random Forest</h3><h1 style='color: black;'>{rf_pred:.1f}</h1><p style='color: black;'>{emoji} {cat}</p></div>", unsafe_allow_html=True)
        
        with col2:
            cat, color, emoji = get_aqi_category(xgb_pred)
            st.markdown(f"<div style='background-color: {color}; padding: 20px; border-radius: 10px; text-align: center;'><h3 style='color: black;'>XGBoost</h3><h1 style='color: black;'>{xgb_pred:.1f}</h1><p style='color: black;'>{emoji} {cat}</p></div>", unsafe_allow_html=True)
        
        with col3:
            cat, color, emoji = get_aqi_category(avg_pred)
            st.markdown(f"<div style='background-color: {color}; padding: 20px; border-radius: 10px; text-align: center;'><h3 style='color: black;'>Average</h3><h1 style='color: black;'>{avg_pred:.1f}</h1><p style='color: black;'>{emoji} {cat}</p></div>", unsafe_allow_html=True)

with tab2:
    if data is not None:
        st.subheader("📊 Data Overview")
        st.dataframe(data.head(100), use_container_width=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.scatter(data.sample(1000), x='vehicle_count', y='aqi', 
                            color='aqi', title='Traffic vs AQI')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.scatter(data.sample(1000), x='pm25', y='aqi', 
                            color='aqi', title='PM2.5 vs AQI')
            st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.markdown("""
    ## About This Project
    
    This Air Quality Index (AQI) prediction system uses machine learning to forecast air quality based on:
    - **Traffic Data**: Vehicle count, congestion, speed
    - **Weather Data**: Temperature, humidity, wind speed, rainfall
    - **Pollution Data**: PM2.5, PM10, NO₂, CO, O₃, SO₂
    
    ### Models Used
    - **Random Forest Regressor**
    - **XGBoost Regressor**
    
    ### Key Features
    ✅ Real-time AQI prediction  
    ✅ Interactive parameter adjustment  
    ✅ Visual data analysis  
    ✅ AQI categorization (Good to Hazardous)
    """)