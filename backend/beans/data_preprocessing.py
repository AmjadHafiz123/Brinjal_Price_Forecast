
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def load_and_preprocess_data(file_path):
    
    print("--- Data Loading and Preprocessing ---")

    # Load the original cleaned data
    df_original_price = pd.read_csv(file_path, index_col='Date', parse_dates=True)

    # Initialize and fit the MinMaxScaler on the 'Price' column
    scaler = MinMaxScaler()
    df_original_price['Price_Scaled'] = scaler.fit_transform(df_original_price[['Price']])

    # Create a copy of the DataFrame for feature engineering
    df = df_original_price.copy()
    target_column = 'Price_Scaled'

    # Create lagged features
    df['Price_Scaled_Lag1'] = df[target_column].shift(1)
    df['Price_Scaled_Lag7'] = df[target_column].shift(7)
    df['Price_Scaled_Lag30'] = df[target_column].shift(30)

    # Create rolling statistics 
    df['Price_Scaled_RollingMean7'] = df[target_column].rolling(window=7).mean().shift(1)

    # Create time-based features
    df['Month'] = df.index.month
    df['DayOfWeek'] = df.index.dayofweek
    df['DayOfYear'] = df.index.dayofyear

    # Drop rows with NaN values introduced by lagging and rolling operations
    df.dropna(inplace=True)

    print("DataFrame head after feature engineering:")
    print(df.head())
    print(f"\nDataFrame shape after feature engineering: {df.shape}")

    # Define features (X) for the model
    features = [col for col in df.columns if col not in ['Price', target_column]]

    return df, scaler, features