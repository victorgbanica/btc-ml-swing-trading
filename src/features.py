import numpy as np
import pandas as pd


def make_features(df):
    """
    Feature engineering pipeline for BTC/USD swing trading.

    This public version intentionally omits proprietary feature construction.
    It preserves interface, dimensionality, and temporal alignment for
    reproducibility and evaluation.
    """

    X = pd.DataFrame(index=df.index)

    # Time-based features (non-proprietary)
    if isinstance(df.index, pd.DatetimeIndex):
        X["hour_sin"] = np.sin(2 * np.pi * df.index.hour / 24)
        X["hour_cos"] = np.cos(2 * np.pi * df.index.hour / 24)
        X["dow_sin"] = np.sin(2 * np.pi * df.index.dayofweek / 7)
        X["dow_cos"] = np.cos(2 * np.pi * df.index.dayofweek / 7)

    # Placeholder feature set (fixed, deterministic)
    rng = np.random.default_rng(seed=42)
    N_FEATURES = 120

    for i in range(N_FEATURES):
        X[f"feature_{i:03d}"] = rng.standard_normal(len(X))

    X = X.dropna()

    print("Feature engineering complete:")
    print(f"  - Generated {len(X.columns)} features")
    print(f"  - Rows: {len(X)}")

    return X
