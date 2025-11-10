# dashboard.py

"""
dashboard.py (v3 - Refined UI with Clearer AQI Meter)

A lightweight Streamlit dashboard that:
1. Runs the AQI prediction pipeline (fetches data, preprocesses, runs inference).
2. Displays tomorrow's predicted AQI on a Plotly Gauge Chart with improved clarity.
3. Shows a color-coded AQI category, emoji, and actionable advice.
4. Plots a bar chart of the last 14 days of historical daily max AQI.

This script re-uses the helper functions from the prediction pipeline.
"""

import os
import requests
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
import streamlit as st
import plotly.graph_objects as go
from datetime import datetime, timedelta

# --- Constants ---
GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
AIR_QUALITY_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
MODEL_DIR = "model"
TFLITE_PATH = os.path.join(MODEL_DIR, "aqi_model.tflite")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.save")

# --- Model & Data Parameters ---
WINDOW_SIZE = 7
SMOOTHING_WINDOW = 3
TARGET_VARIABLE = "us_aqi"
FETCH_PAST_DAYS = 14

# --- Prediction & Data Fetching Logic ---
# (Re-implemented here to be self-contained for Streamlit)

@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_delhi_coords():
    """Fetches latitude and longitude for Delhi."""
    try:
        params = {"name": "Delhi", "count": 1}
        response = requests.get(GEOCODING_URL, params=params)
        response.raise_for_status()
        data = response.json()
        if "results" in data and len(data["results"]) > 0:
            location = data["results"][0]
            return location["latitude"], location["longitude"]
    except Exception as e:
        st.error(f"Error fetching coordinates: {e}")
        return None, None
    raise ValueError("Could not find coordinates for Delhi.")


@st.cache_data(ttl=3600)  # Cache for 1 hour
def fetch_latest_data(latitude, longitude):
    """Fetches latest hourly AQI data."""
    try:
        params = {
            "latitude": latitude,
            "longitude": longitude,
            "hourly": TARGET_VARIABLE,
            "past_days": FETCH_PAST_DAYS,
            "forecast_days": 1,
            "timezone": "auto"
        }
        response = requests.get(AIR_QUALITY_URL, params=params)
        response.raise_for_status()
        data = response.json()
        if "hourly" in data and "time" in data["hourly"]:
            return data["hourly"]
    except Exception as e:
        st.error(f"Error fetching latest AQI data: {e}")
        return None
    raise ValueError("Invalid data received from Open-Meteo API.")

def run_inference(tflite_path, input_data):
    """Runs inference using the TFLite interpreter."""
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    interpreter.set_tensor(input_details['index'], input_data)
    interpreter.invoke()
    prediction = interpreter.get_tensor(output_details['index'])
    return prediction[0][0]

@st.cache_data(ttl=3600)  # Cache the entire pipeline
def get_prediction_and_history():
    """
    Runs the full prediction pipeline and returns the
    predicted AQI and the historical data for plotting.
    """
    # 1. Load Artifacts
    if not os.path.exists(SCALER_PATH) or not os.path.exists(TFLITE_PATH):
        raise FileNotFoundError("Model artifacts (scaler/model) not found. Run `train_model.py`.")
        
    scaler = joblib.load(SCALER_PATH)
    
    # 2. Fetch Data
    lat, lon = get_delhi_coords()
    if lat is None:
        return None, None
        
    hourly_data = fetch_latest_data(lat, lon)
    if hourly_data is None:
        return None, None

    # 3. Preprocess for BOTH history and prediction
    df = pd.DataFrame(hourly_data)
    df['time'] = pd.to_datetime(df['time'])
    df.set_index('time', inplace=True)
    
    # For history plot: simple daily max
    daily_max_aqi = df[TARGET_VARIABLE].resample('D').max().dropna()
    history_df = daily_max_aqi.tail(FETCH_PAST_DAYS).reset_index()
    history_df.columns = ['date', 'aqi']
    # Format date for cleaner plotting
    history_df['date'] = history_df['date'].dt.strftime('%b %d')
    
    # For prediction: apply full pipeline
    smoothed_aqi = daily_max_aqi.rolling(
        window=SMOOTHING_WINDOW, min_periods=1
    ).mean().dropna()
    
    latest_data_window = smoothed_aqi.values[-WINDOW_SIZE:]
    
    if len(latest_data_window) < WINDOW_SIZE:
        raise ValueError("Not enough processed data to make a prediction.")
    
    scaled_data = scaler.transform(latest_data_window.reshape(-1, 1))
    input_tensor = np.array(scaled_data).reshape(1, WINDOW_SIZE, 1).astype(np.float32)
    
    # 4. Run Inference
    scaled_prediction = run_inference(TFLITE_PATH, input_tensor)
    
    # 5. Unscale
    unscaled_prediction = scaler.inverse_transform(np.array([[scaled_prediction]]))
    final_aqi = int(round(unscaled_prediction[0][0]))
    
    return final_aqi, history_df

# --- UI Helper Functions ---

def get_aqi_details(aqi):
    """
    Returns a comprehensive tuple of category, color, emoji,
    and actionable advice based on the US AQI index.
    """
    if aqi <= 50:
        return (
            "Good",
            "#00E400",  # Green
            "😊",
            "It's a great day to be outside! Air quality is excellent."
        )
    if aqi <= 100:
        return (
            "Moderate",
            "#FFFF00",  # Yellow
            "😐",
            "Air quality is acceptable. Unusually sensitive people should consider reducing prolonged or heavy exertion outdoors."
        )
    if aqi <= 150:
        return (
            "Unhealthy for Sensitive",
            "#FF7E00",  # Orange
            "😷",
            "Sensitive groups (like people with lung disease, children, and older adults) should reduce outdoor exertion. Everyone else, it's okay to be outside."
        )
    if aqi <= 200:
        return (
            "Unhealthy",
            "#FF0000",  # Red
            "🤢",
            "Everyone may begin to experience health effects. Sensitive groups should avoid outdoor activity. Others should reduce prolonged exertion."
        )
    if aqi <= 300:
        return (
            "Very Unhealthy",
            "#8F3F97",  # Purple
            "😵",
            "**Health Alert:** Everyone may experience more serious health effects. Avoid all outdoor physical activity. Keep windows closed."
        )
    return (
        "Hazardous",
        "#7E0023",  # Maroon
        "☠️",
        "**DANGER:** This is an emergency condition. The entire population is likely to be affected. Stay indoors, keep windows closed, and use an air purifier if possible."
    )

def create_aqi_gauge(aqi_value, category, color):
    """Creates a Plotly gauge chart for the AQI value with improved clarity."""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta", # Added delta mode for aesthetic
        value = aqi_value,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {
            'text': f"<b>{category}</b>",
            'font': {'size': 24, 'color': color}
        },
        number = {'font': {'size': 48, 'color': "white"}},
        # Placeholder delta for visualization, could be difference from previous day
        delta = {'reference': 75, 'position': "bottom", 'relative': False, 'font': {'color': "lightgray"}}, 
        gauge = {
            'axis': {
                'range': [0, 500],
                'tickwidth': 1,
                'tickcolor': "darkgray",
                'tickvals': [0, 50, 100, 150, 200, 300, 500], # Explicit tick values
                'ticktext': ['0', '50', '100', '150', '200', '300', '500+'],
                'dtick': 50 # Primary ticks every 50
            },
            'bar': {'color': color, 'thickness': 0.3},
            'bgcolor': "#262730",
            'borderwidth': 0, # Remove border for cleaner look
            'steps': [
                {'range': [0, 50], 'color': '#00E400'},
                {'range': [50, 100], 'color': '#FFFF00'},
                {'range': [100, 150], 'color': '#FF7E00'},
                {'range': [150, 200], 'color': '#FF0000'},
                {'range': [200, 300], 'color': '#8F3F97'},
                {'range': [300, 500], 'color': '#7E0023'}
            ],
            'threshold': { # Add a small threshold indicator for the current value
                'line': {'color': "white", 'width': 4},
                'thickness': 0.75,
                'value': aqi_value
            }
        }
    ))
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': "white", 'family': "Arial"},
        height=300,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    return fig

def create_aqi_bar_chart(history_df):
    """Creates a Plotly bar chart for historical AQI."""
    fig = go.Figure()
    
    # Add bars
    fig.add_trace(go.Bar(
        x=history_df['date'],
        y=history_df['aqi'],
        name='AQI',
        marker_color='#1E90FF'
    ))
    
    fig.update_layout(
        title="Historical Daily Max AQI (US)",
        xaxis_title="Date",
        yaxis_title="AQI Value",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color="#EAEAEA")
    )
    return fig

# --- Streamlit App ---

def main():
    st.set_page_config(
        page_title="Delhi AQI Forecast",
        page_icon="💨",
        layout="wide"
    )
    
    st.title("💨 Delhi Air Quality (AQI) Forecast")
    
    try:
        # Run the pipeline
        predicted_aqi, history_df = get_prediction_and_history()
        
        if predicted_aqi is None or history_df is None:
            st.error("Could not fetch or process data. Please try again later.")
            return
            
        category, color, emoji, suggestion = get_aqi_details(predicted_aqi)
        
        # --- Display Prediction ---
        tomorrow_date = (datetime.now() + timedelta(days=1)).strftime('%A, %b %d')
        st.subheader(f"Tomorrow's Forecast: {tomorrow_date}")
        
        col1, col2 = st.columns([2, 1]) # Give gauge more space
        
        with col1:
            gauge_fig = create_aqi_gauge(predicted_aqi, category, color)
            st.plotly_chart(gauge_fig, use_container_width=True)

        with col2:
            st.markdown(
                f"""
                <div style="
                    background-color: #262730; 
                    padding: 20px 20px 20px 20px; 
                    border-radius: 10px; 
                    border-left: 10px solid {color};
                    height: 280px;
                    display: flex; 
                    flex-direction: column; 
                    justify-content: center;
                ">
                    <h3 style="color: #FAFAFA; margin-bottom: 10px;">
                        {emoji} Tomorrow's Outlook
                    </h3>
                    <p style="color: #EAEAEA; font-size: 1.1rem; line-height: 1.6;">
                        {suggestion}
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )

        st.markdown("---")

        # --- Display Historical Chart ---
        st.subheader(f"Historical Daily Max AQI (Last {FETCH_PAST_DAYS} Days)")
        
        if history_df.empty:
            st.warning("No historical data available to display.")
        else:
            fig = create_aqi_bar_chart(history_df)
            st.plotly_chart(fig, use_container_width=True)
            
        # --- Methodology Note ---
        with st.expander("ℹ️ About this Forecast"):
            st.markdown(
                """
                This forecast is generated by a machine learning model
                (a 1D Convolutional Neural Network, or Conv1D)
                running locally via TensorFlow Lite.
                
                **Pipeline:**
                1.  **Data:** Fetches historical daily max AQI from Open-Meteo.
                2.  **Smoothing:** Applies a 3-day rolling average to the data.
                3.  **Input:** Uses the past 7 days of smoothed data...
                4.  **Output:** ...to predict the 8th day's AQI.
                
                *Disclaimer: This is a demonstration project and should not be
                used for making critical health decisions.*
                """
            )

    except FileNotFoundError as e:
        st.error(
            f"❌ **Model artifacts not found!**\n\n"
            f"Please run the training script first from your terminal:\n\n"
            f"`python train_model.py`"
        )
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
        import traceback
        st.code(traceback.format_exc())

if __name__ == "__main__":
    main()