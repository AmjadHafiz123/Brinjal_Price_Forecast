import pandas as pd
import numpy as np
import pickle

# Import functions from your custom modules
from data_preprocessing import load_and_preprocess_data
from model_training import train_ensemble_models
from prediction_utils import predict_future_price, predict_future_prices_range
from evaluation_metrics import evaluate_model, plot_test_predictions

# --- Main Execution Flow ---
if __name__ == "__main__":
    # --- Configuration ---
    ORIGINAL_CLEANED_FILE = 'price_data_cleaned_outliers_handled.csv'
    TARGET_COLUMN = 'Price_Scaled'
    
    # Define a specific date for single prediction
    FUTURE_DATE_FOR_SINGLE_PREDICTION = '2025-07-08' 

    # Define a range of dates for range prediction and plotting
    PREDICTION_RANGE_START_DATE = '2025-07-08'
    PREDICTION_RANGE_END_DATE = '2025-07-14'

    #  Data Preprocessing ---
    # `df_processed` contains all features and scaled prices, `scaler` is for inverse transform
    df_processed, scaler, features = load_and_preprocess_data(ORIGINAL_CLEANED_FILE)

    # Model Training ---
    X_train, y_train, X_test, y_test, \
    final_base_models, meta_learner, \
    rf_predictions, gb_predictions, ridge_predictions = \
        train_ensemble_models(df_processed, features, TARGET_COLUMN)

    # Evaluate Models on Test Set ---
    print("\n--- Model Evaluation (on Test Set) ---")
    evaluate_model("Random Forest Regressor", y_test, rf_predictions)
    evaluate_model("Gradient Boosting Regressor", y_test, gb_predictions)
    evaluate_model("Ridge Regression", y_test, ridge_predictions)

    # Calculate stacked predictions on test set for evaluation
    meta_features_test_final = np.zeros((X_test.shape[0], len(final_base_models)))
    for i, model in enumerate(final_base_models):
        meta_features_test_final[:, i] = model.predict(X_test)
    stacked_predictions_final = meta_learner.predict(meta_features_test_final)
    
    evaluate_model("Stacked Ensemble", y_test, stacked_predictions_final)

    # Plot Test Set Predictions ---
    plot_test_predictions(y_test, rf_predictions, gb_predictions, ridge_predictions, stacked_predictions_final)

    # Make a Single Future Prediction ---
    scaled_pred, original_pred = predict_future_price(
        FUTURE_DATE_FOR_SINGLE_PREDICTION,
        historical_df=df_processed, 
        features_list=features,
        scaler_obj=scaler,
        final_base_models_list=final_base_models,
        meta_model=meta_learner
    )

    if original_pred is not None:
        print(f"\n Predicted Vegetable Price for {FUTURE_DATE_FOR_SINGLE_PREDICTION}: {original_pred:.2f} Rupees/Kg")
    else:
        print(f" Prediction for {FUTURE_DATE_FOR_SINGLE_PREDICTION} failed due to insufficient historical data.")
        
    # Make and Plot Future Predictions for a Range ---
    predict_future_prices_range(
        start_date_str=PREDICTION_RANGE_START_DATE,
        end_date_str=PREDICTION_RANGE_END_DATE,
        historical_df=df_processed.copy(), 
        features_list=features,
        scaler_obj=scaler,
        final_base_models_list=final_base_models,
        meta_model=meta_learner
    )

    # Save Test Set Predictions ---
    predictions_df = pd.DataFrame({
        'Actual_Price_Scaled': y_test,
        'RF_Predicted_Price_Scaled': rf_predictions,
        'GB_Predicted_Price_Scaled': gb_predictions,
        'Ridge_Predicted_Price_Scaled': ridge_predictions,
        'Stacked_Predicted_Price_Scaled': stacked_predictions_final
    }, index=y_test.index)

    output_predictions_file = 'price_predictions_stacked_ensemble_full.csv'
    predictions_df.to_csv(output_predictions_file)
    print(f"\nTest set predictions saved to '{output_predictions_file}'")

    # Save the ensemble model components 
brinjal_model = {
    'final_base_models': final_base_models,
    'meta_learner': meta_learner,
    'features': features
}

# Save the models
with open('beans_model.pkl', 'wb') as f:
    pickle.dump(beans_model, f)

# Save the scaler
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print("\nModels and scaler saved as pickle files.")