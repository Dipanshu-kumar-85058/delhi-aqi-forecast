# train_model.py

"""
train_model.py

This script performs the following steps:
1. Fetches historical air quality data for Delhi from the Open-Meteo API.
2. Preprocesses the data:
    - Converts hourly AQI to daily maximum AQI.
    - Applies a 3-day rolling average to smooth the data.
3. Scales the data using sklearn's MinMaxScaler.
4. Creates sequences of 7 past days to predict the next day.
5. Builds a TFLite-compatible Conv1D model.
6. Trains the model using MAE loss, Adam optimizer, and Early Stopping.
7. Saves the trained Keras model (.h5), the TFLite model (.tflite),
   and the data scaler (.save) to the 'model/' directory.
"""

import os
import requests
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# --- Constants ---
GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
AIR_QUALITY_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
MODEL_DIR = "model"
H5_PATH = os.path.join(MODEL_DIR, "aqi_model.h5")
TFLITE_PATH = os.path.join(MODEL_DIR, "aqi_model.tflite")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.save")

# --- Model & Data Parameters ---
WINDOW_SIZE = 7       # Use past 7 days
SMOOTHING_WINDOW = 3  # 3-day rolling average
TARGET_VARIABLE = "us_aqi"
HISTORICAL_START_DATE = "2022-07-01"  # Open-Meteo AQI history start

# --- Helper Functions ---

def get_delhi_coords():
    """Fetches latitude and longitude for Delhi."""
    try:
        params = {"name": "Delhi", "count": 1}
        response = requests.get(GEOCODING_URL, params=params)
        response.raise_for_status()  # Raise HTTPError for bad responses
        data = response.json()
        
        if "results" in data and len(data["results"]) > 0:
            location = data["results"][0]
            return location["latitude"], location["longitude"]
        else:
            raise ValueError("Could not find coordinates for Delhi.")
    except requests.exceptions.RequestException as e:
        print(f"Error fetching coordinates: {e}")
        raise

def fetch_historical_data(latitude, longitude):
    """Fetches historical hourly AQI data."""
    try:
        end_date = pd.Timestamp.now().strftime('%Y-%m-%d')
        params = {
            "latitude": latitude,
            "longitude": longitude,
            "hourly": TARGET_VARIABLE,
            "start_date": HISTORICAL_START_DATE,
            "end_date": end_date,
            "timezone": "auto"
        }
        response = requests.get(AIR_QUALITY_URL, params=params)
        response.raise_for_status()
        data = response.json()
        
        if "hourly" not in data or "time" not in data["hourly"]:
            raise ValueError("Invalid data received from Open-Meteo API.")
            
        return data["hourly"]
    except requests.exceptions.RequestException as e:
        print(f"Error fetching historical AQI data: {e}")
        raise

def preprocess_data(hourly_data):
    """Converts hourly data to smoothed daily max AQI."""
    df = pd.DataFrame(hourly_data)
    df['time'] = pd.to_datetime(df['time'])
    df.set_index('time', inplace=True)
    
    # Resample to daily maximum
    daily_max_aqi = df[TARGET_VARIABLE].resample('D').max()
    
    # Apply smoothing
    smoothed_aqi = daily_max_aqi.rolling(
        window=SMOOTHING_WINDOW, min_periods=1
    ).mean()
    
    # Drop any NaNs that might have resulted
    smoothed_aqi.dropna(inplace=True)
    
    return smoothed_aqi.values

def create_sequences(data, window_size):
    """Creates sliding window sequences (X) and corresponding targets (y)."""
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:(i + window_size)])
        y.append(data[i + window_size])
    return np.array(X), np.array(y)

def build_model(input_shape):
    """Builds a TFLite-compatible Conv1D model."""
    model = keras.Sequential([
        keras.layers.Input(shape=input_shape),
        keras.layers.Conv1D(
            filters=32, kernel_size=3, activation='relu', padding="causal"
        ),
        keras.layers.Flatten(),
        keras.layers.Dense(16, activation='relu'),
        keras.layers.Dense(1)  # Linear activation for regression
    ])
    
    model.compile(
        optimizer='adam',
        loss='mean_absolute_error',
        metrics=['mean_absolute_error']
    )
    return model

def convert_to_tflite(keras_model):
    """Converts a Keras model to a TFLite model."""
    converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    # Apply default optimizations (e.g., weight quantization)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    return tflite_model

# --- Main Training Pipeline ---

def main():
    """Main function to run the entire training pipeline."""
    print("🚀 Starting model training process...")
    
    try:
        # Ensure model directory exists
        os.makedirs(MODEL_DIR, exist_ok=True)
        
        # 1. Fetch Data
        lat, lon = get_delhi_coords()
        print(f"✅ Fetched coordinates for Delhi: Lat={lat:.4f}, Lon={lon:.4f}")
        
        hourly_data = fetch_historical_data(lat, lon)
        print(f"✅ Fetched {len(hourly_data['time'])} hourly data points.")
        
        # 2. Preprocess Data
        data = preprocess_data(hourly_data)
        print(f"✅ Preprocessed data. Shape: {data.shape}")
        
        # 3. Scale Data
        scaler = MinMaxScaler(feature_range=(0, 1))
        # Reshape for scaler: [n_samples] -> [n_samples, 1]
        data_scaled = scaler.fit_transform(data.reshape(-1, 1))
        
        # 4. Create Sequences
        X, y = create_sequences(data_scaled, WINDOW_SIZE)
        # Final X shape is (samples, window_size, 1)
        print(f"✅ Created sequences. X shape: {X.shape}, y shape: {y.shape}")

        if X.shape[0] < 100:
             print(f"⚠️ Warning: Very few data samples ({X.shape[0]}). Model performance may be poor.")
             
        # 5. Split Data (No shuffling for time series)
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, shuffle=False
        )
        print(f"✅ Data split: {len(X_train)} train, {len(X_val)} validation.")
        
        # 6. Build & Train Model
        model = build_model(input_shape=(WINDOW_SIZE, 1))
        model.summary()
        
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        print("\n🏋️ Training model...")
        history = model.fit(
            X_train,
            y_train,
            epochs=100,
            batch_size=32,
            validation_data=(X_val, y_val),
            callbacks=[early_stopping],
            verbose=2
        )
        print("✅ Model training complete.")
        
        # 7. Save Artifacts
        # Save Keras model
        model.save(H5_PATH)
        print(f"📦 Saved Keras model to {H5_PATH}")
        
        # Save Scaler
        joblib.dump(scaler, SCALER_PATH)
        print(f"📦 Saved scaler to {SCALER_PATH}")
        
        # 8. Convert and Save TFLite Model
        tflite_model = convert_to_tflite(model)
        with open(TFLITE_PATH, 'wb') as f:
            f.write(tflite_model)
        print(f"📦 Saved TFLite model to {TFLITE_PATH}")
        
        print("\n🎉 Training pipeline finished successfully!")

    except Exception as e:
        print(f"\n❌ An error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()