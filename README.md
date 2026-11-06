# delhi-aqi-forecast

# 💨 Delhi AQI Forecaster

A complete, end-to-end machine learning project that forecasts the daily Air Quality Index (AQI) for Delhi, India. The system uses a TFLite-compatible model and includes an interactive Streamlit dashboard for visualization.

![]([https://ibb.co/v6bGyLcx]) 

## ✨ Features

* **End-to-End Pipeline:** From data fetching to a usable web dashboard.
* **Real-Time Data:** Fetches the latest air quality data from the [Open-Meteo API](https://open-meteo.com/).
* **Optimized Inference:** Uses a lightweight **TensorFlow Lite (`.tflite`)** model for fast and efficient predictions.
* **Interactive Dashboard:** A clean UI built with **Streamlit** to visualize the forecast, historical data, and actionable health advice.
* **Modular Code:** The project is split into three clean scripts:
    1.  `train_model.py`: Trains and saves the model.
    2.  `predict_api.py`: Runs a prediction in the terminal.
    3.  `dashboard.py`: Launches the web application.

## 🛠️ Tech Stack

* **Python 3.10+**
* **TensorFlow & Keras:** For model creation and training.
* **TensorFlow Lite:** For the optimized inference engine.
* **Streamlit:** For the interactive web dashboard.
* **Scikit-learn:** For data scaling (`MinMaxScaler`).
* **Pandas & NumPy:** For data manipulation and preprocessing.
* **Plotly:** For creating the gauge and bar charts.
* **Requests:** For fetching data from the API.
* **Joblib:** For saving and loading the `sklearn` scaler.

## ⚙️ How It Works

The system is built on a time-series forecasting pipeline:

1.  **Training (`train_model.py`):**
    * Fetches historical hourly AQI data for Delhi.
    * Converts data to **Daily Maximum AQI**.
    * Applies a **3-day rolling average** to smooth the target variable.
    * Scales the data using `MinMaxScaler`.
    * Prepares sequences: **Past 7 days** are used to predict the **next 1 day**.
    * Trains a TFLite-compatible `Conv1D` model.
    * Saves three artifacts to the `model/` directory:
        * `aqi_model.h5` (Full Keras model)
        * `aqi_model.tflite` (Optimized inference model)
        * `scaler.save` (The `MinMaxScaler` object)

2.  **Prediction (`dashboard.py` & `predict_api.py`):**
    * Loads `aqi_model.tflite` and `scaler.save`.
    * Fetches the latest 14 days of data (required for smoothing and sequencing).
    * Applies the **exact same preprocessing pipeline** as in training.
    * Uses the TFLite interpreter to run inference on the last 7-day sequence.
    * Inverse-transforms the scaled prediction to get the final AQI value.
    * Displays the result in the Streamlit dashboard.

## 🚀 Getting Started

Follow these steps to run the project locally.

### 1. Clone the Repository

```bash
git clone [https://github.com/your-username/delhi-aqi-forecast.git](https://github.com/your-username/delhi-aqi-forecast.git)
cd delhi-aqi-forecast
```

### 2. Create and Activate a Virtual Environment

It's highly recommended to use a virtual environment.

```bash
# For macOS/Linux
python3 -m venv venv
source venv/bin/activate

# For Windows
python -m venv venv
.\venv\Scripts\activate
```

### 3. Install Dependencies

You can install the required packages using `pip`.

```bash
pip install -r requirements.txt
```

### 4. Run the Pipeline

You must run the scripts in this order:

**Step 1: Train the Model**
Run the training script first. This will create the `model/` directory and save the necessary `aqi_model.tflite` and `scaler.save` files.

```bash
python train_model.py
```
*Output:*
```
🚀 Starting model training process...
✅ Fetched coordinates for Delhi: Lat=28.6519, Lon=77.2315
✅ Fetched 5000+ hourly data points.
...
🏋️ Training model...
✅ Model training complete.
📦 Saved Keras model to model/aqi_model.h5
📦 Saved scaler to model/scaler.save
📦 Saved TFLite model to model/aqi_model.tflite
🎉 Training pipeline finished successfully!
```

**Step 2: Launch the Dashboard**
Once the model is trained, you can launch the Streamlit dashboard.

```bash
streamlit run dashboard.py
```
This will automatically open the application in your default web browser (usually at `http://localhost:8501`).

### 5. (Optional) Run CLI Prediction

To test the prediction directly in your terminal, you can run `predict_api.py`.

```bash
python predict_api.py
```
*Output:*
```
🚀 Starting AQI prediction...
✅ Loaded scaler and TFLite model path.
...
========================================
💨 Predicted AQI for tomorrow in Delhi: 178
========================================
```

## 📂 File Structure

```
.
├── 📜 predict_api.py     # Script to run inference in the terminal
├── 📁 app/
|   ├── 📜 dashboard.py       # The Streamlit web application
├── 📁 model/
|   ├── 📜 train_model.py     # Script to train and save the model/scaler
│   ├── 🤖 aqi_model.h5
│   ├── 🤖 aqi_model.tflite
│   └── 📦 scaler.save
├── 📜 requirements.txt   # Project dependencies
└── 📜 README.md          # This file
```

## 📄 License

This project is licensed under the MIT License. See the `LICENSE` file for details.
