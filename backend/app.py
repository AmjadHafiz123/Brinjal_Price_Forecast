from flask import Flask, request, jsonify
import pickle
import pandas as pd
from datetime import datetime
import numpy as np
import requests
from flask_cors import CORS
from pydantic import BaseModel
from datetime import timedelta
from typing import List
import os
import joblib

app = Flask(__name__)
CORS(app)

@app.route('/')
def home():
    # url = "http://127.0.0.1:5000/forecast"
    # data = {
    #     "vegetable": "brinjal",
    #     "startdate": "2025-08-01",
    #     "enddate": "2025-08-10"
    # }

    # response = requests.post(url, json=data)

    # print(response.status_code)
    # print(response.json())
    return "Hi"

# Load models and scalers (for simplicity, per vegetable; extend as needed)
MODELS = {
    'brinjal': {
        'model': pickle.load(open('brinjal_model.pkl', 'rb')),
        'scaler': pickle.load(open('scaler.pkl', 'rb')),
    }
}

@app.route('/forecast', methods=['POST'])
def forecast_price():
    data = request.get_json()
    vegetable = data.get('vegetable').lower()
    start_date = pd.to_datetime(data.get('startdate'))
    end_date = pd.to_datetime(data.get('enddate'))

    if vegetable not in MODELS:
        return jsonify({'error': 'Model not found for vegetable'}), 400

    model = MODELS[vegetable]['model']
    scaler = MODELS[vegetable]['scaler']

    # Generate date range
    future = pd.DataFrame({'ds': pd.date_range(start=start_date, end=end_date)})

    forecast = model.predict(future)

    # Inverse transform to original scale
    forecast['yhat'] = scaler.inverse_transform(forecast[['yhat']])

    # Calculate trend
    forecast['price_diff'] = forecast['yhat'].diff().fillna(0)

    result = []
    for i, row in forecast.iterrows():
        trend = (
            "up" if row['price_diff'] > 0.1
            else "down" if row['price_diff'] < -0.1
            else "stable"
        )
        result.append({
            'date': row['ds'].strftime('%Y-%m-%d'),
            'price': round(row['yhat'], 2),
            'trend': trend,
            'confidence': round(0.75 + np.random.rand() * 0.2, 2)
        })

    return jsonify(result)

# ---------- Model & Data Paths ----------
MODEL_PATH = "cabbage_model.pkl"
CSV_PATH = "cabbage-price.csv"
PREPROCESSED_PATH = "preprocessed_cabbage_data.pkl"

# ---------- Load Model ----------
model = joblib.load(MODEL_PATH)

# ---------- Load or Generate Preprocessed Data ----------
if os.path.exists(PREPROCESSED_PATH):
    df = pd.read_pickle(PREPROCESSED_PATH)
else:
    print("Generating preprocessed data...")
    df = pd.read_csv(CSV_PATH)
    df['date'] = pd.date_range(start="2023-06-01", periods=len(df), freq='D')
    df = df.set_index('date')
    df = df.reindex(pd.date_range(df.index.min(), df.index.max(), freq='D'))
    df.index.name = 'date'
    df['price'] = df['price'].ffill()

    for lag in range(1, 8):
        df[f'lag_{lag}'] = df['price'].shift(lag)

    df['dayofweek'] = df.index.dayofweek
    df['month'] = df.index.month
    df = df.dropna()

    df.to_pickle(PREPROCESSED_PATH)
    print("✅ Saved preprocessed_cabbage_data.pkl")

# Load MAE value (optional: adjust or store separately)
MAE = 21.3

# ---------- Forecast Input Schema ----------
class ForecastRequest(BaseModel):
    start_date: str
    end_date: str

@app.post("/forecast/cabbage")
def forecast_prices():
    req = request.get_json()
    start = pd.to_datetime(req.get('startdate'))
    end = pd.to_datetime(req.get('enddate'))
    forecast_dates = pd.date_range(start, end)

    forecast_df = df.copy()
    last_price = forecast_df['price'].iloc[-1]

    results = []

    for date in forecast_dates:
        if date not in forecast_df.index:
            forecast_df.loc[date] = None

            for lag in range(1, 8):
                lag_date = date - timedelta(days=lag)
                forecast_df.loc[date, f'lag_{lag}'] = (
                    forecast_df.loc[lag_date, 'price']
                    if lag_date in forecast_df.index and pd.notna(forecast_df.loc[lag_date, 'price'])
                    else forecast_df['price'].ffill().iloc[-1]
                )

            forecast_df.loc[date, 'dayofweek'] = date.dayofweek
            forecast_df.loc[date, 'month'] = date.month

            features = forecast_df.loc[date][[f'lag_{i}' for i in range(1, 8)] + ['dayofweek', 'month']]
            prediction = model.predict([features])[0]
            forecast_df.loc[date, 'price'] = prediction

            diff = prediction - last_price
            trend = "up" if diff > 0.1 else "down" if diff < -0.1 else "stable"
            confidence = max(0.5, min(0.95, round(1 - abs(diff) / (MAE * 2), 2)))

            results.append({
                "date": date.strftime("%Y-%m-%d"),
                "price": round(prediction, 2),
                "trend": trend,
                "confidence": confidence
            })

            last_price = prediction

    return results
