import numpy as np
import pandas as pd
import yfinance as yf


def fetch_spx(start="2000-01-01", end="2024-12-31"):
    raw = yf.download("^GSPC", start=start, end=end, progress=False, auto_adjust=True)
    raw.columns = [col[0] if isinstance(col, tuple) else col for col in raw.columns]
    raw.index = pd.to_datetime(raw.index).tz_localize(None)
    raw["log_return"] = np.log(raw["Close"] / raw["Close"].shift(1))
    return raw.dropna(subset=["log_return"])


def build_features(df, vol_window=21):
    df = df.copy()
    df["rolling_vol"] = df["log_return"].rolling(vol_window).std()
    df = df.dropna(subset=["rolling_vol"])
    X = df[["log_return", "rolling_vol"]].values
    return X, df


if __name__ == "__main__":
    raw = fetch_spx()
    X, df = build_features(raw)
    print(f"{X.shape[0]} days, {X.shape[1]} features")
    print(df[["log_return", "rolling_vol"]].describe().round(5))
