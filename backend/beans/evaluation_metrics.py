import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def evaluate_model(name, y_true, y_pred):
    
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    print(f"{name}:")
    print(f"  MAE (Mean Absolute Error): {mae:.4f}")
    print(f"  RMSE (Root Mean Squared Error): {rmse:.4f}")
    print(f"  R2 (R-squared): {r2:.4f}")

def plot_test_predictions(y_test, rf_predictions, gb_predictions, ridge_predictions, stacked_predictions_final):
    """
    Plots actual vs. predicted prices for the test set from various models.

    """
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