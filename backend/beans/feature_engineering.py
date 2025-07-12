
def create_features(df, target_column='Price_Scaled'):
    df['Price_Scaled_Lag1'] = df[target_column].shift(1)
    df['Price_Scaled_Lag7'] = df[target_column].shift(7)
    df['Price_Scaled_Lag30'] = df[target_column].shift(30)
    df['Price_Scaled_RollingMean7'] = df[target_column].rolling(window=7).mean().shift(1)
    df['Month'] = df.index.month
    df['DayOfWeek'] = df.index.dayofweek
    df['DayOfYear'] = df.index.dayofyear
    df.dropna(inplace=True)
    return df
