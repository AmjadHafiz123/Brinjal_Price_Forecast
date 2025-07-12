import pandas as pd
import os

# Load price-only CSV
df = pd.read_csv("cabbage-price.csv")

# Make sure only one column exists
if 'price' not in df.columns:
    raise ValueError("CSV must contain a 'price' column.")

# Generate date index from 2023-06-01 onward
df['date'] = pd.date_range(start="2023-06-01", periods=len(df), freq='D')
df = df.set_index('date')

# Add weekends and fill missing prices
full_index = pd.date_range(df.index.min(), df.index.max(), freq='D')
df = df.reindex(full_index)
df.index.name = 'date'
df['price'] = df['price'].ffill()

# Add lag features
for lag in range(1, 8):
    df[f'lag_{lag}'] = df['price'].shift(lag)

# Add time-based features
df['dayofweek'] = df.index.dayofweek
df['month'] = df.index.month

# Drop rows with NaN (from lag shifts)
df = df.dropna()

# Save as pickle
df.to_pickle("preprocessed_cabbage_data.pkl")

print("✅ preprocessed_cabbage_data.pkl saved.")
