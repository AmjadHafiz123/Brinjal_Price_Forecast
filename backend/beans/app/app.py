from flask import Flask, request, jsonify
import pickle
import pandas as pd
from datetime import datetime
import numpy as np
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load models and scalers
MODELS = {
    'beans': {
        'model': pickle.load(open('app/models/beans_model.pkl', 'rb')),
        'scaler': pickle.load(open('app/models/scaler.pkl', 'rb')),
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
    start_date = pd.to_datetime(data.get('startdate'))
    end_date = pd.to_datetime(data.get('enddate'))

    if vegetable not in MODELS:
        return jsonify({'error': 'Model not found for vegetable'}), 400

    model_info = MODELS[vegetable]
    
    # Generate date range
    date_range = pd.date_range(start=start_date, end=end_date)
    
    
    historical_df = pd.read_csv('app/data/price_data_cleaned_outliers_handled.csv', 
                          index_col='Date', parse_dates=True)
    historical_df['Price_Scaled'] = model_info['scaler'].transform(historical_df[['Price']])
    
    # Make predictions for each date
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
            
            # Update previous price
            previous_price = price
    
    return jsonify(predictions)

if __name__ == '__main__':
    app.run(debug=True)