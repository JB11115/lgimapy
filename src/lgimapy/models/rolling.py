"""
Models for rolling calculations on DataFrames.
"""

def rolling_zscore(df, window, min_periods=0):
    roll = df.rolling(window=window, min_periods=min_periods)
    mean = roll.mean().shift(1)
    std = roll.std(ddof=0).shift(1)
    z_score_df = (df - mean) / std
    # Remove days with no values.
    start = max(2, min_periods)
    return z_score_df.iloc[start:, :]  # remove days with no values
