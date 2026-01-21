import pandas as pd
import numpy as np
from datetime import datetime


def load_xbtusd_data(path, start_date=None):
    """
    Load hourly BTC/USD OHLCV data from CSV.

    Expected columns (no header):
        timestamp, open, high, low, close, volume, trades

    Args:
        path: Path to CSV file
        start_date: Optional start date for filtering (YYYY-MM-DD)

    Notes:
        - Assumes UNIX timestamps in seconds
        - Handles data sorted in either chronological order
    """
    df = pd.read_csv(
        path,
        header=None,
        names=["timestamp", "open", "high", "low", "close", "volume", "trades"]
    )

    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
    df = df.set_index("timestamp").sort_index()

    if start_date:
        original_len = len(df)
        df = df[df.index >= start_date]
        filtered = original_len - len(df)
        print(f"Filtered {filtered} rows prior to {start_date}")

    df = df[["open", "high", "low", "close", "volume"]]

    print(f"Loaded {len(df)} hourly rows")
    print(f"Date range: {df.index[0]} to {df.index[-1]}")

    return df


def load_spy_data(path, start_date=None):
    """
    Load SPY historical daily data.

    Expected columns:
        Date, Open, High, Low, Close/Last, Volume
    """
    df = pd.read_csv(path)

    df["Date"] = pd.to_datetime(df["Date"], format="%m/%d/%Y")

    df = df.rename(columns={
        "Date": "timestamp",
        "Open": "spy_open",
        "High": "spy_high",
        "Low": "spy_low",
        "Close/Last": "spy_close",
        "Volume": "spy_volume"
    })

    df = df.set_index("timestamp").sort_index()
    df.index = df.index.tz_localize("UTC")

    if start_date:
        df = df[df.index >= start_date]

    return df


def load_fear_greed_data(path, start_date=None):
    """
    Load Fear & Greed Index data.

    Expected columns:
        date, fng_value, fng_classification
    """
    df = pd.read_csv(path)

    df["date"] = pd.to_datetime(df["date"], format="%d-%m-%Y")

    df = df.rename(columns={
        "date": "timestamp",
        "fng_value": "fear_greed_index"
    })

    df = df.set_index("timestamp").sort_index()
    df.index = df.index.tz_localize("UTC")

    if start_date:
        df = df[df.index >= start_date]

    return df


def merge_all_data(xbtusd_df, spy_df, fng_df):
    """
    Merge hourly BTC/USD data with daily macro indicators.
    Daily data is forward-filled to match hourly frequency.
    """
    merged = xbtusd_df.copy()

    print(f"BTC/USD rows: {len(merged)}")
    print(f"SPY rows: {len(spy_df)}")
    print(f"Fear & Greed rows: {len(fng_df)}")

    merged = merged.rename(columns={
        "open": "open_open",
        "high": "high_high",
        "low": "low_low",
        "close": "close_close"
    })

    merged = merged.join(spy_df, how="left")
    merged = merged.join(fng_df, how="left")

    spy_cols = ["spy_open", "spy_high", "spy_low", "spy_close", "spy_volume"]
    fng_cols = ["fear_greed_index"]

    merged[spy_cols] = merged[spy_cols].ffill()
    merged[fng_cols] = merged[fng_cols].ffill()

    initial_len = len(merged)
    merged = merged.dropna()
    dropped = initial_len - len(merged)

    if dropped > 0:
        print(f"Dropped {dropped} initial rows due to missing macro data")

    print(f"Merged dataset size: {len(merged)} rows")
    print(f"Final date range: {merged.index[0]} to {merged.index[-1]}")

    return merged


def load_and_prepare_data(xbtusd_path, spy_path, fng_path, start_date=None):
    """
    Load and merge all datasets into a single DataFrame suitable
    for feature engineering and modeling.
    """
    print("Loading datasets...")

    xbtusd = load_xbtusd_data(xbtusd_path, start_date=start_date)
    spy = load_spy_data(spy_path, start_date=start_date)
    fng = load_fear_greed_data(fng_path, start_date=start_date)

    print("Merging datasets...")
    merged = merge_all_data(xbtusd, spy, fng)

    print("Data preparation complete")

    return merged


def train_test_split_time_series(df, test_size=0.2):
    """
    Chronological train/test split (no shuffling).
    """
    split_idx = int(len(df) * (1 - test_size))

    train = df.iloc[:split_idx]
    test = df.iloc[split_idx:]

    print("Train/Test split")
    print(f"Train: {len(train)} rows")
    print(f"Test: {len(test)} rows")

    return train, test


def calculate_sharpe_ratio(returns, periods_per_year=365 * 24):
    """
    Calculate annualized Sharpe ratio.
    """
    if len(returns) == 0 or returns.std() == 0:
        return 0.0

    return (returns.mean() / returns.std()) * np.sqrt(periods_per_year)


def calculate_max_drawdown(cumulative_returns):
    """
    Calculate maximum drawdown from cumulative log returns.
    """
    if len(cumulative_returns) == 0:
        return 0.0

    cum_returns = np.exp(cumulative_returns)
    running_max = cum_returns.cummax()
    drawdown = (cum_returns - running_max) / running_max

    return drawdown.min()


def calculate_win_rate(returns):
    """
    Calculate fraction of positive non-zero returns.
    """
    if len(returns) == 0:
        return 0.0

    non_zero = returns[returns != 0]
    if len(non_zero) == 0:
        return 0.0

    return (non_zero > 0).mean()


def winsorize(series, lower=0.01, upper=0.99):
    """
    Cap extreme values using percentile thresholds.
    """
    lower_val = series.quantile(lower)
    upper_val = series.quantile(upper)

    return series.clip(lower=lower_val, upper=upper_val)


def remove_highly_correlated_features(df, threshold=0.95):
    """
    Remove features with absolute correlation above threshold.
    """
    corr_matrix = df.corr().abs()
    upper = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )

    to_drop = [
        column for column in upper.columns
        if any(upper[column] > threshold)
    ]

    print(f"Removing {len(to_drop)} highly correlated features")

    return df.drop(columns=to_drop), to_drop
