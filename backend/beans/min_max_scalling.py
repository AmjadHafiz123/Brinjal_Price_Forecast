import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# Load the cleaned data ---
file_path = 'price_data_cleaned_outliers_handled.csv'
df = pd.read_csv(file_path, index_col='Date', parse_dates=True)

print("Original DataFrame head (from cleaned data):")
print(df.head())
print("\nOriginal DataFrame info:")
print(df.info())
print("\nNumber of missing values before scaling:")
print(df.isnull().sum())

# Initialize the MinMaxScaler ---
scaler = MinMaxScaler()

# Apply Min-Max Scaling ---
df['Price_Scaled'] = scaler.fit_transform(df[['Price']])

print("\nDataFrame head after Min-Max Scaling:")
print(df.head())

print("\nMin-Max Scaling Parameters (values used to scale):")
print(f"Minimum value found in 'Price': {scaler.data_min_[0]:.2f}")
print(f"Maximum value found in 'Price': {scaler.data_max_[0]:.2f}")

#  Visualization  ---
plt.figure(figsize=(15, 6))
plt.plot(df.index, df['Price'], label='Original Price (Outliers Handled)', color='blue', alpha=0.7)
plt.plot(df.index, df['Price_Scaled'], label='Min-Max Scaled Price', color='red', alpha=0.9, linestyle='--')
plt.title('Vegetable Price: Original vs. Min-Max Scaled')
plt.xlabel('Date')
plt.ylabel('Price / Scaled Value') # Y-axis label changes as values are scaled
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show() # Display the plot

# Save the data with the scaled column ---
output_file_name = 'price_data_minmax_scaled.csv'
df.to_csv(output_file_name)

print(f"\nData with Min-Max Scaled prices saved to '{output_file_name}'")