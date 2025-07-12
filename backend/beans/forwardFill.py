import pandas as pd

# Define the file path
file_path = 'Data-set.csv'

# Load the data
df = pd.read_csv(file_path)

# --- Prepare the DataFrame for ffill ---
# Drop the 'No' column as it's not needed for price data
df_cleaned = df.drop(columns=['No']).copy()

#Convert 'Date' column to datetime objects
df_cleaned['Date'] = pd.to_datetime(df_cleaned['Date'], format='%d/%m/%Y')

#  Set 'Date' as the DataFrame index
df_cleaned.set_index('Date', inplace=True)

# Sort the DataFrame by index (Date) to ensure chronological order
df_cleaned.sort_index(inplace=True)

# --- Apply Forward Fill (ffill) ---
df_cleaned['Price_ffilled'] = df_cleaned['Price'].ffill()

# Display the head of the DataFrame to show the original and ffilled prices
print("DataFrame head with original and forward-filled prices:")
print(df_cleaned.head(15))

# Display the number of missing values before and after ffill
print("\nNumber of missing values in 'Price' before ffill:")
print(df_cleaned['Price'].isnull().sum())

print("\nNumber of missing values in 'Price_ffilled' after ffill:")
print(df_cleaned['Price_ffilled'].isnull().sum())

# --- Save the DataFrame with forward-filled data to a CSV file ---
output_file_name = 'forward_filled_price_data.csv'
df_cleaned.to_csv(output_file_name)

print(f"\nForward-filled data saved to '{output_file_name}'")