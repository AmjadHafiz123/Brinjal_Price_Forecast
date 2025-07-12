# model_training.py

import numpy as np
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.model_selection import train_test_split # Explicitly import if needed for clarity

def train_ensemble_models(df, features, target_column):
   
    print("\n--- Model Training ---")

    # Train-Test Split (Time Series Split)
    train_size = int(len(df) * 0.8)
    train_df, test_df = df[0:train_size], df[train_size:len(df)]

    X_train, y_train = train_df[features], train_df[target_column]
    X_test, y_test = test_df[features], test_df[target_column]

    print(f"\nTraining set size: {len(X_train)} samples")
    print(f"Testing set size: {len(X_test)} samples")

    # Initialize and Train Individual Base Learners for Comparison
    print("\n--- Training Individual Base Learners for Comparison ---")
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf_model.fit(X_train, y_train)
    rf_predictions = rf_model.predict(X_test)
    print("Random Forest Regressor trained.")

    gb_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    gb_model.fit(X_train, y_train)
    gb_predictions = gb_model.predict(X_test)
    print("Gradient Boosting Regressor trained.")

    ridge_model = Ridge(alpha=1.0)
    ridge_model.fit(X_train, y_train)
    ridge_predictions = ridge_model.predict(X_test)
    print("Ridge Regression trained.")


    # Implement Stacking Ensemble
    print("\n--- Implementing Stacking Ensemble ---")

    base_models_definitions = [
        ('rf', RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)),
        ('gb', GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)),
        ('ridge', Ridge(alpha=1.0))
    ]
    meta_learner = LinearRegression()

    oof_predictions = np.zeros((X_train.shape[0], len(base_models_definitions)))
    kf = KFold(n_splits=5, shuffle=False)

    for i, (name, model) in enumerate(base_models_definitions):
        print(f"Generating out-of-fold predictions for base model: {name}...")
        oof_preds_model = np.zeros(X_train.shape[0])

        for fold, (train_idx, val_idx) in enumerate(kf.split(X_train, y_train)):
            X_fold_train, X_fold_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

            model.fit(X_fold_train, y_fold_train)
            oof_preds_model[val_idx] = model.predict(X_fold_val)

        oof_predictions[:, i] = oof_preds_model

    # Train the Meta-Learner
    print("Training Meta-Learner...")
    meta_learner.fit(oof_predictions, y_train)

    # Train final base models on the full X_train for future predictions
    final_base_models = []
    for name, model_def in base_models_definitions:
        model = model_def # Create a new instance or reuse the one that was just trained on full data if kfold was skipped for that purpose
        model.fit(X_train, y_train)
        final_base_models.append(model)
    print("Final base models trained for prediction.")
    print("Stacking ensemble is now ready.")

    return X_train, y_train, X_test, y_test, final_base_models, meta_learner, \
           rf_predictions, gb_predictions, ridge_predictions