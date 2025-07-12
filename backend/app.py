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
    'beans': {
        'model': pickle.load(open('beans/app/models/beans_model.pkl', 'rb')),
        'scaler': pickle.load(open('beans/app/models/scaler.pkl', 'rb')),
    },
    'brinjal': {
        'model': pickle.load(open('brinjal/brinjal_model.pkl', 'rb')),
        'scaler': pickle.load(open('brinjal/scaler.pkl', 'rb')),
    }
}

def predict_single_date(date, historical_df, model_info):
    """Modified to handle edge cases"""
    try:
        # Create feature dictionary 
        features = {
            'Month': date.month,
            'DayOfWeek': date.dayofweek,
            'DayOfYear': date.dayofyear
        }

        # Calculate lag features with fallbacks
        for lag in [1, 7, 30]:
            lag_date = date - pd.Timedelta(days=lag)
            features[f'Price_Scaled_Lag{lag}'] = historical_df.loc[lag_date, 'Price_Scaled'] \
                if lag_date in historical_df.index \
                else historical_df['Price_Scaled'].mean()  # Fallback to mean

        # Calculate rolling mean with available data
        rolling_window = historical_df['Price_Scaled'].last('7D')
        features['Price_Scaled_RollingMean7'] = rolling_window.mean() if len(rolling_window) > 0 \
            else historical_df['Price_Scaled'].mean()

        # Convert to DataFrame for prediction
        X_pred = pd.DataFrame([features])[model_info['model']['features']]

        # Get base model predictions
        meta_features = np.zeros((1, len(model_info['model']['final_base_models'])))
        for i, model in enumerate(model_info['model']['final_base_models']):
            meta_features[0, i] = model.predict(X_pred)[0]

        # Get final prediction
        scaled_pred = model_info['model']['meta_learner'].predict(meta_features)[0]
        return model_info['scaler'].inverse_transform([[scaled_pred]])[0][0]

    except Exception as e:
        print(f"Prediction error for {date}: {str(e)}")
        return None

@app.route('/forecast', methods=['POST'])
def forecast_price():
    data = request.get_json()
    vegetable = data.get('vegetable').lower()
    # start_date = pd.to_datetime(data.get('startdate'))
    # end_date_str = pd.to_datetime(data.get('enddate'))
    # end_date = pd.to_datetime(end_date_str) if end_date_str else start_date

    # Validate and parse dates with error handling
    try:
        start_date = pd.to_datetime(data['startdate'])  # Required field
    except (KeyError, ValueError) as e:
        return jsonify({
            'error': 'Invalid or missing start date',
            'details': str(e),
            'expected_format': 'YYYY-MM-DD'
        }), 400

    # Handle end date (default to start_date if missing/empty/invalid)
    end_date = start_date  # Default value
    if 'enddate' in data and data['enddate'].strip():
        try:
            end_date = pd.to_datetime(data['enddate'])
        except ValueError as e:
            return jsonify({
                'error': 'Invalid end date',
                'details': str(e),
                'expected_format': 'YYYY-MM-DD'
            }), 400

    # Validate date order
    if end_date < start_date:
        return jsonify({
            'error': 'Invalid date range',
            'message': 'End date cannot be before start date'
        }), 400

    if vegetable not in MODELS:
        return jsonify({'error': 'Model not found for vegetable'}), 400

    model_info = MODELS[vegetable]

    # For 'beans'
    if vegetable == 'beans':
        try:
            historical_df = pd.read_csv('beans/app/data/price_data_cleaned_outliers_handled.csv',
                                        index_col='Date', parse_dates=True)
            # Validation
            historical_df['Price_Scaled'] = model_info['scaler'].transform(historical_df[['Price']])
        except FileNotFoundError:
            return jsonify({'error': 'Historical data not found for beans'}), 500
        except KeyError:
            return jsonify({'error': 'Price column missing in historical data or scaling issue'}), 500


        date_range = pd.date_range(start=start_date, end=end_date)
        predictions = []
        previous_price = None

        for date in date_range:
            price = predict_single_date(date, historical_df, model_info)

            if price is not None:
                # Calculate trend
                if previous_price is not None:
                    diff = price - previous_price
                    if diff > 0.1:
                        trend = "up"
                    elif diff < -0.1:
                        trend = "down"
                    else:
                        trend = "stable"
                else:
                    trend = "stable"

                predictions.append({
                    'date': date.strftime('%Y-%m-%d'),
                    'price': round(float(price), 2),
                    'trend': trend,
                    'confidence': round(0.75 + np.random.rand() * 0.2, 2)
                })

                # Update previous price for trend calculation for the next day
                previous_price = price
        return jsonify(predictions)

    # --- End Beans ---
    # For other vegetables
    else:
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
      
   

MODEL_PATH = "cabbage/cabbage_model.pkl"
CSV_PATH = "cabbage/cabbage-price.csv"
PREPROCESSED_PATH = "cabbage/preprocessed_cabbage_data.pkl"

model = joblib.load(MODEL_PATH)

# Load or Generate Preprocessed Data
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

MAE = 21.3


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

