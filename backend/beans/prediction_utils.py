import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def predict_future_price(future_date_str, historical_df, features_list, scaler_obj, final_base_models_list, meta_model):
    
    future_date = pd.to_datetime(future_date_str)
    print(f"\n--- Predicting price for {future_date.strftime('%Y-%m-%d')} ---")

    # Create a DataFrame to hold features for the single future date
    future_features_df = pd.DataFrame(index=[future_date])

    # Populate time-based features for the future date
    future_features_df['Month'] = future_date.month
    future_features_df['DayOfWeek'] = future_date.dayofweek
    future_features_df['DayOfYear'] = future_date.dayofyear

    # Populate lagged and rolling features from historical_df
    try:
        lag1_date = future_date - pd.Timedelta(days=1)
        future_features_df['Price_Scaled_Lag1'] = historical_df.loc[lag1_date, 'Price_Scaled']

        lag7_date = future_date - pd.Timedelta(days=7)
        future_features_df['Price_Scaled_Lag7'] = historical_df.loc[lag7_date, 'Price_Scaled']

        lag30_date = future_date - pd.Timedelta(days=30)
        future_features_df['Price_Scaled_Lag30'] = historical_df.loc[lag30_date, 'Price_Scaled']

        rolling_mean_end_date = future_date - pd.Timedelta(days=1)
        rolling_mean_start_date = rolling_mean_end_date - pd.Timedelta(days=6)

        if rolling_mean_start_date < historical_df.index.min():
            print(f"Warning: Not enough historical data for full 7-day rolling mean ending {rolling_mean_end_date.strftime('%Y-%m-%d')}. Using available data.")
            rolling_data = historical_df['Price_Scaled'].loc[historical_df.index.min():rolling_mean_end_date]
        else:
            rolling_data = historical_df['Price_Scaled'].loc[rolling_mean_start_date:rolling_mean_end_date]

        future_features_df['Price_Scaled_RollingMean7'] = rolling_data.mean()

    except KeyError as e:
        print(f"Error: Not enough historical data to compute all required features for {future_date_str}.")
        print(f"Missing historical data for date: {e}. Cannot predict for this date.")
        print("Please ensure your historical data covers at least 30 days prior to your prediction date.")
        return None, None
    except Exception as e:
        print(f"An unexpected error occurred during feature creation: {e}")
        return None, None

    # Ensure the feature columns are in the exact same order as used during training
    X_future = future_features_df[features_list]
    print("\nFeatures prepared for future prediction:")
    print(X_future)

    # Generate base model predictions for the future date
    meta_features_future = np.zeros((1, len(final_base_models_list)))
    for i, model in enumerate(final_base_models_list):
        meta_features_future[0, i] = model.predict(X_future)[0]

    # Predict using the meta-learner on the base model predictions
    scaled_prediction = meta_model.predict(meta_features_future)[0]

    # Inverse transform the prediction back to original price scale
    original_price_prediction = scaler_obj.inverse_transform(np.array([[scaled_prediction]]))[0][0]

    return scaled_prediction, original_price_prediction

def predict_future_prices_range(start_date_str, end_date_str, historical_df, features_list, scaler_obj, final_base_models_list, meta_model):
    
    print(f"\n--- Predicting prices for range: {start_date_str} to {end_date_str} ---")
    start_date = pd.to_datetime(start_date_str)
    end_date = pd.to_datetime(end_date_str)
    date_range = pd.date_range(start=start_date, end=end_date)

    predictions = {}
    current_historical_df = historical_df.copy() # Work on a copy

    for date in date_range:
        
        scaled_pred, original_pred = predict_future_price(
            date.strftime('%Y-%m-%d'),
            historical_df=current_historical_df, # Pass the dynamically growing historical_df
            features_list=features_list,
            scaler_obj=scaler_obj,
            final_base_models_list=final_base_models_list,
            meta_model=meta_model
        )

        if original_pred is not None:
            predictions[date] = original_pred
            # Append the new prediction  to the current historical data
            new_row_scaled = pd.DataFrame({'Price_Scaled': scaled_pred}, index=[date])
            current_historical_df = pd.concat([current_historical_df, new_row_scaled])
            
    # Plot the predicted prices
    if predictions:
        plt.figure(figsize=(12, 6))
        plt.plot(list(predictions.keys()), list(predictions.values()), marker='o', linestyle='-', color='green', label='Predicted Price')
        
        
        
        plt.title(f"Predicted Vegetable Prices ({start_date_str} to {end_date_str})")
        plt.xlabel("Date")
        plt.ylabel("Predicted Price (Rupees/Kg)")
        plt.legend()
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
    else:
        print(" No predictions were generated for the range. Possibly due to insufficient historical data.")