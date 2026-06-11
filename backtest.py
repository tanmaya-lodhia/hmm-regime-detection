import numpy as np
import pandas as pd


SIGNAL_MAP = {
    "bull":     1.0,
    "high_vol": 0.0,
    "bear":    -1.0,
}


def attach_signals(df, regime_labels):
    out = df.copy()
    out["regime"] = regime_labels
    out["signal_raw"] = out["regime"].map(SIGNAL_MAP)
    # shift by 1 so we're trading on yesterday's regime call, not today's
    out["signal"] = out["signal_raw"].shift(1).fillna(0.0)
    return out


def run_backtest(df):
    out = df.copy()
    out["strat_return"] = out["signal"] * out["log_return"]
    out["bh_return"] = out["log_return"]
    out["strat_cum"] = out["strat_return"].cumsum().apply(np.exp)
    out["bh_cum"] = out["bh_return"].cumsum().apply(np.exp)
    return out


def performance_metrics(df, trading_days=252):
    results = {}
    for name, col in [("Strategy", "strat_return"), ("Buy & Hold", "bh_return")]:
        r = df[col].dropna()
        ann_return = r.mean() * trading_days
        ann_vol = r.std() * np.sqrt(trading_days)
        sharpe = ann_return / ann_vol if ann_vol > 0 else np.nan

        cum = np.exp(r.cumsum())
        max_dd = ((cum - cum.cummax()) / cum.cummax()).min()

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

    print(performance_metrics(df))
    print(df["regime"].value_counts())
