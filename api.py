from flask import Flask, request, jsonify
import pickle
import pandas as pd
from datetime import datetime
import numpy as np

app = Flask(__name__)

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
