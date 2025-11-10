# predict_api.py

"""
predict_api.py

This script performs the following steps:
1. Loads the pre-trained TFLite model (aqi_model.tflite) and
   the scaler (scaler.save) from the 'model/' directory.
2. Fetches the latest 14 days of hourly AQI data for Delhi.
3. Preprocesses the data in the *exact same way* as the training script:
    - Converts hourly to daily max.
    - Applies 3-day smoothing.
    - Takes the last 7 days of the smoothed series.
    - Scales the 7-day window using the loaded scaler.
4. Runs inference using the TFLite interpreter.
5. Inverse-transforms the scaled prediction to get the real AQI value.
6. Prints the final predicted AQI for tomorrow.
"""

import os
import requests
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf

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
FETCH_PAST_DAYS = 14  # Need enough data for smoothing and windowing

# --- Helper Functions ---

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
        else:
            raise ValueError("Could not find coordinates for Delhi.")
    except requests.exceptions.RequestException as e:
        print(f"Error fetching coordinates: {e}")
        raise

def fetch_latest_data(latitude, longitude):
    """Fetches latest hourly AQI data for the past N days."""
    try:
        params = {
            "latitude": latitude,
            "longitude": longitude,
            "hourly": TARGET_VARIABLE,
            "past_days": FETCH_PAST_DAYS,
            "forecast_days": 1, # Include today's forecast to get full data
            "timezone": "auto"
        }
        response = requests.get(AIR_QUALITY_URL, params=params)
        response.raise_for_status()
        data = response.json()
        
        if "hourly" not in data or "time" not in data["hourly"]:
            raise ValueError("Invalid data received from Open-Meteo API.")
            
        return data["hourly"]
    except requests.exceptions.RequestException as e:
        print(f"Error fetching latest AQI data: {e}")
        raise

def preprocess_for_prediction(hourly_data, scaler, window_size):
    """
    Applies the full preprocessing pipeline to new data
    and returns a single, scaled input tensor for the model.
    """
    df = pd.DataFrame(hourly_data)
    df['time'] = pd.to_datetime(df['time'])
    df.set_index('time', inplace=True)
    
    # 1. Resample to daily maximum
    daily_max_aqi = df[TARGET_VARIABLE].resample('D').max()
    
    # 2. Apply smoothing
    smoothed_aqi = daily_max_aqi.rolling(
        window=SMOOTHING_WINDOW, min_periods=1
    ).mean()
    
    # 3. Drop NaNs
    smoothed_aqi.dropna(inplace=True)
    
    # 4. Get the last N days for the window
    latest_data_window = smoothed_aqi.values[-window_size:]
    
    if len(latest_data_window) < window_size:
        raise ValueError(
            f"Not enough data to form a {window_size}-day window. "
            f"Only have {len(latest_data_window)} days."
        )
    
    # 5. Scale the data
    # Reshape for scaler: [window_size] -> [window_size, 1]
    scaled_data = scaler.transform(latest_data_window.reshape(-1, 1))
    
    # 6. Reshape for TFLite model: [window_size, 1] -> [1, window_size, 1]
    input_tensor = np.array(scaled_data).reshape(1, window_size, 1)
    
    # Ensure input is float32, as required by TFLite
    return input_tensor.astype(np.float32)

def run_inference(tflite_path, input_data):
    """Runs inference using the TFLite interpreter."""
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    
    # Get input and output tensor details
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    
    # Set the input tensor
    interpreter.set_tensor(input_details['index'], input_data)
    
    # Run inference
    interpreter.invoke()
    
    # Get the output tensor
    # Output shape is [1, 1], so we get [0][0] for the scalar value
    prediction = interpreter.get_tensor(output_details['index'])
    return prediction[0][0]

# --- Main Prediction Pipeline ---

def main():
    """Main function to run the prediction pipeline."""
    print("🚀 Starting AQI prediction...")
    
    try:
        # 1. Load Artifacts
        if not os.path.exists(SCALER_PATH) or not os.path.exists(TFLITE_PATH):
            print(f"❌ Error: Model artifacts not found.")
            print(f"Please run `train_model.py` first.")
            return

        scaler = joblib.load(SCALER_PATH)
        print("✅ Loaded scaler and TFLite model path.")
        
        # 2. Fetch Latest Data
        lat, lon = get_delhi_coords()
        print(f"✅ Fetched coordinates for Delhi.")
        
        hourly_data = fetch_latest_data(lat, lon)
        print(f"✅ Fetched latest {len(hourly_data['time'])} hourly data points.")
        
        # 3. Preprocess Data
        input_tensor = preprocess_for_prediction(hourly_data, scaler, WINDOW_SIZE)
        print(f"✅ Preprocessed data into input tensor. Shape: {input_tensor.shape}")
        
        # 4. Run Inference
        scaled_prediction = run_inference(TFLITE_PATH, input_tensor)
        print(f"✅ Model inference complete (scaled result: {scaled_prediction:.4f}).")
        
        # 5. Unscale Prediction
        # Reshape for scaler: scalar -> [[scalar]]
        unscaled_prediction = scaler.inverse_transform(
            np.array([[scaled_prediction]])
        )
        
        # Get the final scalar value and round to integer
        final_aqi = int(round(unscaled_prediction[0][0]))
        
        # 6. Print Result
        print("\n" + "="*40)
        print(f"💨 Predicted AQI for tomorrow in Delhi: {final_aqi}")
        print("="*40 + "\n")

    except Exception as e:
        print(f"\n❌ An error occurred during prediction: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()