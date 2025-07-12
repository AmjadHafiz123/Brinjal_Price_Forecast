import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta


# --- Data Loading and Initial Preprocessing 
original_cleaned_file = 'price_data_cleaned_outliers_handled.csv'
df_original_price = pd.read_csv(original_cleaned_file, index_col='Date', parse_dates=True)

# Initialize and fit the MinMaxScaler on the 'Price' column of your cleaned data.
scaler = MinMaxScaler()
df_original_price['Price_Scaled'] = scaler.fit_transform(df_original_price[['Price']])


# ---  Feature Engineering for Training Data ---
df = df_original_price.copy()
target_column = 'Price_Scaled'

# Create lagged features: these are historical values of the target variable.
df['Price_Scaled_Lag1'] = df[target_column].shift(1)   # Price from 1 day ago
df['Price_Scaled_Lag7'] = df[target_column].shift(7)   # Price from 7 days ago 
df['Price_Scaled_Lag30'] = df[target_column].shift(30) # Price from 30 days ago 

# Create rolling statistics: These capture trends or averages over a recent period.
df['Price_Scaled_RollingMean7'] = df[target_column].rolling(window=7).mean().shift(1)

# Create time-based features: These can capture seasonality or cyclical patterns based on date.
df['Month'] = df.index.month       # Month of the year (1-12)
df['DayOfWeek'] = df.index.dayofweek # Day of the week (0=Monday, 6=Sunday)
df['DayOfYear'] = df.index.dayofyear # Day of the year (1-366)

# Drop rows with NaN values. 
df.dropna(inplace=True)

print("DataFrame head after feature engineering (for training):")
print(df.head())
print(f"\nDataFrame shape after feature engineering: {df.shape}")

# --- Train-Test Split (Time Series Split) ---
train_size = int(len(df) * 0.8)
train_df, test_df = df[0:train_size], df[train_size:len(df)]

# Define features (X) and target (y) for both training and testing sets.
features = [col for col in train_df.columns if col not in ['Price', target_column]]
X_train, y_train = train_df[features], train_df[target_column]
X_test, y_test = test_df[features], test_df[target_column]

print(f"\nTraining set size: {len(X_train)} samples")
print(f"Testing set size: {len(X_test)} samples")
print(f"Features used for training/prediction: {features}")

# ---  Initialize and Train Base Learners ---
print("\n--- Training Individual Base Learners for Comparison ---")

# Individual models 

# Random Forest Regressor 
rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)
rf_predictions = rf_model.predict(X_test)
print("Random Forest Regressor trained.")

# Gradient Boosting Regressor 
gb_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
gb_model.fit(X_train, y_train)
gb_predictions = gb_model.predict(X_test)
print("Gradient Boosting Regressor trained.")

# Ridge Regression 
ridge_model = Ridge(alpha=1.0)
ridge_model.fit(X_train, y_train)
ridge_predictions = ridge_model.predict(X_test)
print("Ridge Regression trained.")

# Implement Stacking to Combine Predictions (and get the final stacked model) ---
print("\n--- Implementing Stacking Ensemble ---")


base_models = [
    ('rf', RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)),
    ('gb', GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)),
    ('ridge', Ridge(alpha=1.0))
]


meta_learner = LinearRegression()

oof_predictions = np.zeros((X_train.shape[0], len(base_models)))


test_predictions_base = np.zeros((X_test.shape[0], len(base_models)))


kf = KFold(n_splits=5, shuffle=False) # 5-fold cross-validation

# Iterate through each base model to train and generate OOF and test predictions
for i, (name, model) in enumerate(base_models):
    print(f"Generating out-of-fold and test predictions for base model: {name}...")
    # Initialize array to store OOF predictions for the current base model
    oof_preds_model = np.zeros(X_train.shape[0])
    # Initialize array to store average test predictions for the current base model
    test_preds_model_avg = np.zeros(X_test.shape[0])

    # Perform KFold cross-validation on the training data
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train, y_train)):
        # Split data into training and validation sets for the current fold
        X_fold_train, X_fold_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

        # Train the base model on the current fold's training data
        model.fit(X_fold_train, y_fold_train)

        # Make predictions on the current fold's validation set (OOF predictions)
        oof_preds_model[val_idx] = model.predict(X_fold_val)

        # Make predictions on the entire test set. These will be averaged later across all folds.
        test_preds_model_avg += model.predict(X_test) / kf.n_splits

    # Store the generated OOF predictions for this base model
    oof_predictions[:, i] = oof_preds_model
    # Store the averaged test predictions for this base model
    test_predictions_base[:, i] = test_preds_model_avg

# Train the Meta-Learner: The meta-learner learns from the OOF predictions of the base models.
print("Training Meta-Learner...")
meta_learner.fit(oof_predictions, y_train)

# For actual prediction on new data (like future dates), we need base models trained on the full X_train.
final_base_models = []
for name, model in base_models:
    model.fit(X_train, y_train) # Train base models on full training data
    final_base_models.append(model)
print("Final base models trained for single-point prediction.")

# The meta_learner is already trained as `meta_learner` from the OOF predictions.
print("Stacking ensemble is now ready for future predictions.")

# --- Function to Predict Future Price for a Specific Date ---
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
        # Lag 1: Price from the day before the future_date
        lag1_date = future_date - pd.Timedelta(days=1)
        future_features_df['Price_Scaled_Lag1'] = historical_df.loc[lag1_date, 'Price_Scaled']

        # Lag 7: Price from 7 days before the future_date
        lag7_date = future_date - pd.Timedelta(days=7)
        future_features_df['Price_Scaled_Lag7'] = historical_df.loc[lag7_date, 'Price_Scaled']

        # Lag 30: Price from 30 days before the future_date
        lag30_date = future_date - pd.Timedelta(days=30)
        future_features_df['Price_Scaled_Lag30'] = historical_df.loc[lag30_date, 'Price_Scaled']

        # Rolling Mean 7: Mean of the 7 days *ending the day before* the future_date
        rolling_mean_end_date = future_date - pd.Timedelta(days=1)
        rolling_mean_start_date = rolling_mean_end_date - pd.Timedelta(days=6) # 7 days window

        
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
    # The scaler expects a 2D array, even for a single value.
    original_price_prediction = scaler_obj.inverse_transform(np.array([[scaled_prediction]]))[0][0]

    return scaled_prediction, original_price_prediction


future_date_to_predict = '2025-07-08' #  Change this date as needed

scaled_pred, original_pred = predict_future_price(
    future_date_to_predict,
    historical_df=df, 
    features_list=features, # The list of features used in training
    scaler_obj=scaler, # The MinMaxScaler object
    final_base_models_list=final_base_models, # The list of trained base models
    meta_model=meta_learner # The trained meta-learner
)

if original_pred is not None:
    print(f"\nPredicted Scaled Price for {future_date_to_predict}: {scaled_pred:.4f}")
    print(f"Predicted Original Price for {future_date_to_predict}: {original_pred:.2f} Rupees/Kg")


print("\n--- Model Evaluation (on Test Set) ---")
def evaluate_model(name, y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    print(f"{name}:")
    print(f"  MAE (Mean Absolute Error): {mae:.4f}")
    print(f"  RMSE (Root Mean Squared Error): {rmse:.4f}")
    print(f"  R2 (R-squared): {r2:.4f}")

evaluate_model("Random Forest Regressor", y_test, rf_predictions)
evaluate_model("Gradient Boosting Regressor", y_test, gb_predictions)
evaluate_model("Ridge Regression", y_test, ridge_predictions)

# For the stacked model, re-run prediction on X_test using the final_base_models and meta_learner
meta_features_test_final = np.zeros((X_test.shape[0], len(final_base_models)))
for i, model in enumerate(final_base_models):
    meta_features_test_final[:, i] = model.predict(X_test)
stacked_predictions_final = meta_learner.predict(meta_features_test_final)

evaluate_model("Stacked Ensemble", y_test, stacked_predictions_final)

# --- Visualization of Test Set Predictions ---
plt.figure(figsize=(18, 8))
plt.plot(y_test.index, y_test, label='Actual Price (Scaled)', color='black', linewidth=2)
plt.plot(y_test.index, rf_predictions, label='RF Predictions', linestyle='--', alpha=0.7)
plt.plot(y_test.index, gb_predictions, label='GB Predictions', linestyle=':', alpha=0.7)
plt.plot(y_test.index, ridge_predictions, label='Ridge Predictions', linestyle='-.', alpha=0.7)
plt.plot(y_test.index, stacked_predictions_final, label='Stacked Ensemble Predictions', color='red', linewidth=2, alpha=0.8)

plt.title('Vegetable Price Prediction: Actual vs. Model Predictions (Scaled) - Test Set')
plt.xlabel('Date')
plt.ylabel('Price (Scaled)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

def predict_future_prices_range(start_date_str, end_date_str, historical_df, features_list, scaler_obj, final_base_models_list, meta_model):
    

    # Create list of dates
    start_date = pd.to_datetime(start_date_str)
    end_date = pd.to_datetime(end_date_str)
    date_range = pd.date_range(start=start_date, end=end_date)

    predictions = {}

    for date in date_range:
        scaled_pred, original_pred = predict_future_price(
            date.strftime('%Y-%m-%d'),
            historical_df=historical_df,
            features_list=features_list,
            scaler_obj=scaler_obj,
            final_base_models_list=final_base_models_list,
            meta_model=meta_model
        )

        if original_pred is not None:
            predictions[date] = original_pred
            # Append prediction to historical data (to allow future rolling/lags)
            new_row = pd.DataFrame({
                'Price_Scaled': scaled_pred
            }, index=[date])
            historical_df = pd.concat([historical_df, new_row])

    # Plot the predicted prices
    if predictions:
        plt.figure(figsize=(12, 6))
        plt.plot(list(predictions.keys()), list(predictions.values()), marker='o', color='green')
        plt.title(f"Predicted Vegetable Prices ({start_date_str} to {end_date_str})")
        plt.xlabel("Date")
        plt.ylabel("Predicted Price (Rupees/Kg)")
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
    else:
        print(" No predictions were generated. Possibly due to insufficient historical data.")

predict_future_prices_range(
    start_date_str='2025-07-08',
    end_date_str='2025-07-14',
    historical_df=df.copy(),  
    features_list=features,
    scaler_obj=scaler,
    final_base_models_list=final_base_models,
    meta_model=meta_learner
)

# --- Save predictions for further analysis (Optional) ---
# predictions_df = pd.DataFrame({
#     'Actual_Price_Scaled': y_test,
#     'RF_Predicted_Price_Scaled': rf_predictions,
#     'GB_Predicted_Price_Scaled': gb_predictions,
#     'Ridge_Predicted_Price_Scaled': ridge_predictions,
#     'Stacked_Predicted_Price_Scaled': stacked_predictions_final
# }, index=y_test.index)

# output_predictions_file = 'price_predictions_stacked_ensemble_full.csv'
# predictions_df.to_csv(output_predictions_file)
# print(f"\nTest set predictions saved to '{output_predictions_file}'")

# --- Print only the final predicted price for the specified date ---
if original_pred is not None:
    print(f"\n Predicted Vegetable Price for {future_date_to_predict}: {original_pred:.2f} Rupees/Kg")
else:
    print(" Prediction failed due to insufficient historical data.")