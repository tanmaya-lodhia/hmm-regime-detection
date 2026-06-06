"""
data.py — Download and prepare SPX return features for HMM training.
"""

import numpy as np
import pandas as pd
import yfinance as yf


def fetch_spx(start: str = "2000-01-01", end: str = "2024-12-31") -> pd.DataFrame:
    """Download SPX daily OHLCV and compute log returns."""
    raw = yf.download("^GSPC", start=start, end=end, progress=False, auto_adjust=True)
    # yfinance >=0.2 returns MultiIndex columns; flatten to single level
    raw.columns = [col[0] if isinstance(col, tuple) else col for col in raw.columns]
    raw.index = pd.to_datetime(raw.index).tz_localize(None)

    raw["log_return"] = np.log(raw["Close"] / raw["Close"].shift(1))
    raw = raw.dropna(subset=["log_return"])
    return raw


def build_features(df: pd.DataFrame, vol_window: int = 21) -> np.ndarray:
    """
    Build the observation matrix X that the HMM will be trained on.

    Features:
      - log_return      : daily log return (captures direction/magnitude)
      - rolling_vol     : 21-day rolling std of returns (captures volatility regime)

    Why two features?
    A single return series lets the HMM distinguish up vs. down days, but
    struggles to separate calm bull markets from volatile ones.  Adding
    realized vol gives the model a second axis so it can find a high-vol
    cluster that spans both bull and bear episodes (e.g. 2008 + 2020).
    """
    df = df.copy()
    df["rolling_vol"] = df["log_return"].rolling(vol_window).std()
    df = df.dropna(subset=["rolling_vol"])

    X = df[["log_return", "rolling_vol"]].values  # shape (T, 2)
    return X, df


if __name__ == "__main__":
    raw = fetch_spx()
    X, df = build_features(raw)
    print(f"Observations: {X.shape[0]} days | Features: {X.shape[1]}")
    print(df[["log_return", "rolling_vol"]].describe().round(5))
