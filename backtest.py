"""
backtest.py — Map HMM regimes to positions and evaluate strategy performance.
"""

import numpy as np
import pandas as pd


# Signal map: what position to hold in each regime
# +1 = fully long, 0 = flat (cash), -1 = short
SIGNAL_MAP = {
    "bull":     1.0,
    "high_vol": 0.0,   # sit out uncertain regimes
    "bear":    -1.0,
}


def attach_signals(df: pd.DataFrame, regime_labels: list[str]) -> pd.DataFrame:
    """
    Add regime labels, raw signals, and lagged (implementable) signals to df.

    Critical detail — the look-ahead bias problem:
    We predict today's regime using today's features (which include today's
    return!). If we trade on today's signal using today's close, we're using
    information we don't have until the day is over.

    Fix: shift signals by 1 day.  We observe today's regime at close,
    then trade at tomorrow's open (approximated as tomorrow's close here).
    This is still a simplification — a real system would use open prices —
    but it eliminates the worst form of look-ahead bias.
    """
    out = df.copy()
    out["regime"] = regime_labels
    out["signal_raw"] = out["regime"].map(SIGNAL_MAP)
    out["signal"] = out["signal_raw"].shift(1).fillna(0.0)  # lag by 1 day
    return out


def run_backtest(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute strategy and buy-and-hold daily returns, then cumulative performance.

    Strategy return = signal(t-1) * log_return(t)
    This is a daily rebalancing strategy: we hold a constant fractional
    position each day equal to the signal.

    Note on transaction costs:
    We don't model costs here. In practice, flipping from +1 to -1 costs
    ~2x the bid-ask spread plus market impact. The high_vol buffer (signal=0)
    naturally reduces turnover by acting as a waiting state between regimes.
    """
    out = df.copy()
    out["strat_return"] = out["signal"] * out["log_return"]
    out["bh_return"] = out["log_return"]  # buy-and-hold baseline

    out["strat_cum"] = out["strat_return"].cumsum().apply(np.exp)
    out["bh_cum"] = out["bh_return"].cumsum().apply(np.exp)
    return out


def performance_metrics(df: pd.DataFrame, trading_days: int = 252) -> pd.Series:
    """
    Standard quant performance metrics for strategy vs. buy-and-hold.

    Sharpe ratio: (mean daily return / std daily return) * sqrt(252)
    Max drawdown: largest peak-to-trough decline in cumulative wealth
    """
    results = {}

    for name, col in [("Strategy", "strat_return"), ("Buy & Hold", "bh_return")]:
        r = df[col].dropna()
        ann_return = r.mean() * trading_days
        ann_vol    = r.std() * np.sqrt(trading_days)
        sharpe     = ann_return / ann_vol if ann_vol > 0 else np.nan

        cum = np.exp(r.cumsum())
        rolling_max = cum.cummax()
        drawdown = (cum - rolling_max) / rolling_max
        max_dd = drawdown.min()

        results[name] = {
            "Ann. Return":  f"{ann_return*100:.2f}%",
            "Ann. Vol":     f"{ann_vol*100:.2f}%",
            "Sharpe":       f"{sharpe:.3f}",
            "Max Drawdown": f"{max_dd*100:.2f}%",
        }

    return pd.DataFrame(results)


if __name__ == "__main__":
    from data import fetch_spx, build_features
    from model import RegimeHMM

    raw = fetch_spx()
    X, df = build_features(raw)
    hmm = RegimeHMM().fit(X)
    labels = hmm.predict_named(X)

    df = attach_signals(df, labels)
    df = run_backtest(df)

    print("\n=== Performance Metrics ===")
    print(performance_metrics(df))

    print("\n=== Regime Distribution ===")
    print(df["regime"].value_counts())

    print("\n=== Signal Distribution ===")
    print(df["signal"].value_counts())
