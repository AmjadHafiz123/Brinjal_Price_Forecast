import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load and Prepare the Data ---
file_path = 'Data-set.csv'

# Load the data. Assuming clean headers: No, Date, Price
df = pd.read_csv(file_path)

# Drop the 'No' column as it's not needed for analysis
df_cleaned = df.drop(columns=['No']).copy()

# Convert 'Date' column to datetime objects
df_cleaned['Date'] = pd.to_datetime(df_cleaned['Date'], format='%d/%m/%Y')

# Set 'Date' as the DataFrame index
df_cleaned.set_index('Date', inplace=True)

# Sort the DataFrame by index (Date) to ensure chronological order
df_cleaned.sort_index(inplace=True)

# Apply Forward Fill to the 'Price' column as a prerequisite
df_cleaned['Price_ffilled'] = df_cleaned['Price'].ffill()

# Drop any remaining NaNs that ffill couldn't handle (e.g., if series started with NaNs)
df_cleaned.dropna(subset=['Price_ffilled'], inplace=True)


#Outlier Detection---
Q1 = df_cleaned['Price_ffilled'].quantile(0.25)
Q3 = df_cleaned['Price_ffilled'].quantile(0.75)
IQR = Q3 - Q1

# Define outlier bounds
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Create a boolean Series where True indicates an outlier
outliers = (df_cleaned['Price_ffilled'] < lower_bound) | (df_cleaned['Price_ffilled'] > upper_bound)

print(f"Detected {outliers.sum()} outliers using IQR method.")
if outliers.sum() > 0:
    print("\nExamples of detected outliers by IQR (first 10):")
    print(df_cleaned[outliers].head(10))

# Calculate Rolling Median ---
window_size = 7 # 7-day rolling median

# Calculate the rolling median of the 'Price_ffilled' column
df_cleaned['Rolling_Median'] = df_cleaned['Price_ffilled'].rolling(
    window=window_size, center=True, min_periods=1
).median()

# Conditional Replacement ---
df_cleaned['Price_outliers_handled'] = df_cleaned['Price_ffilled'].copy()

# Replace the values at the outlier positions with their corresponding Rolling_Median
df_cleaned.loc[outliers, 'Price_outliers_handled'] = df_cleaned.loc[outliers, 'Rolling_Median']

# --- Handle any potential NaNs introduced by rolling calculations at edges ---
df_cleaned['Price_outliers_handled'].fillna(method='ffill', inplace=True)
df_cleaned['Price_outliers_handled'].fillna(method='bfill', inplace=True)

print("\nDataFrame head with original, ffilled, and outlier-handled prices (first 15 rows):")
print(df_cleaned.head(15))

#  Visualization ---
plt.figure(figsize=(18, 8))
plt.plot(df_cleaned.index, df_cleaned['Price_ffilled'], label='Price (Forward-Filled)', color='blue', alpha=0.7)

# Highlight the detected outliers before replacement
plt.scatter(df_cleaned[outliers].index, df_cleaned[outliers]['Price_ffilled'],
            color='green', s=70, label='Detected Outliers (before replacement)', zorder=5, marker='o', edgecolors='black')

plt.plot(df_cleaned.index, df_cleaned['Price_outliers_handled'], label='Price after Outlier Handling (Rolling Median Replacement)', color='red', linestyle='--', alpha=0.9)

plt.title(f'Vegetable Price: Outlier Handling with Rolling Median Replacement (Window={window_size})')
plt.xlabel('Date')
plt.ylabel('Price (Rupees/Kg)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Save the Cleaned Data ---
output_file_name_cleaned = 'price_data_cleaned_outliers_handled.csv'
df_cleaned[['Price_outliers_handled']].rename(columns={'Price_outliers_handled': 'Price'}).to_csv(output_file_name_cleaned)
print(f"\nCleaned data (with outliers handled) saved to '{output_file_name_cleaned}'")