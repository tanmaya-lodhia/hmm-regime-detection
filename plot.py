import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd


REGIME_COLORS = {
    "bull":     "#2ecc71",
    "high_vol": "#f39c12",
    "bear":     "#e74c3c",
}


def plot_regimes_on_price(df, title="SPX Price with HMM Regimes"):
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(df.index, df["Close"], color="black", linewidth=0.8)

    regime_col = df["regime"].ffill()
    changes = regime_col != regime_col.shift(1)
    block_starts = df.index[changes].tolist()
    block_starts.append(df.index[-1])

    for i in range(len(block_starts) - 1):
        state = regime_col.loc[block_starts[i]]
        ax.axvspan(block_starts[i], block_starts[i + 1],
                   alpha=0.25, color=REGIME_COLORS[state], linewidth=0)

    patches = [mpatches.Patch(color=c, label=s, alpha=0.5) for s, c in REGIME_COLORS.items()]
    ax.legend(handles=patches, loc="upper left")
    ax.set_title(title)
    ax.set_ylabel("Price (USD)")
    ax.set_xlabel("Date")
    plt.tight_layout()
    return fig


def plot_cumulative_returns(df):
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(df.index, df["strat_cum"], label="HMM Strategy", color="#3498db", linewidth=1.2)
    ax.plot(df.index, df["bh_cum"], label="Buy & Hold", color="#95a5a6", linewidth=1.0, linestyle="--")
    ax.set_yscale("log")
    ax.set_title("Cumulative Returns — HMM Strategy vs. Buy & Hold (log scale)")
    ax.set_ylabel("Portfolio value (starting at 1)")
    ax.set_xlabel("Date")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def plot_drawdown(df):
    fig, ax = plt.subplots(figsize=(14, 4))
    for name, col, color in [("Strategy", "strat_cum", "#3498db"), ("Buy & Hold", "bh_cum", "#e74c3c")]:
        cum = df[col].dropna()
        dd = (cum - cum.cummax()) / cum.cummax() * 100
        ax.fill_between(cum.index, dd, 0, alpha=0.4, color=color, label=name)
    ax.set_title("Drawdown (%)")
    ax.set_ylabel("Drawdown (%)")
    ax.set_xlabel("Date")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def plot_state_scatter(X, labels):
    fig, ax = plt.subplots(figsize=(7, 5))
    for regime, color in REGIME_COLORS.items():
        mask = np.array(labels) == regime
        ax.scatter(X[mask, 0] * 100, X[mask, 1] * 100,
                   c=color, label=regime, alpha=0.3, s=5)
    ax.set_xlabel("Log Return (%)")
    ax.set_ylabel("21-day Rolling Vol (%)")
    ax.set_title("Feature Space — HMM State Clusters")
    ax.legend()
    plt.tight_layout()
    return fig


def plot_all(df, X, labels):
    figs = [
        ("regimes_on_price.png",   plot_regimes_on_price(df)),
        ("cumulative_returns.png", plot_cumulative_returns(df)),
        ("drawdown.png",           plot_drawdown(df)),
        ("state_scatter.png",      plot_state_scatter(X, labels)),
    ]
    for fname, fig in figs:
        fig.savefig(fname, dpi=150, bbox_inches="tight")
        print(f"Saved {fname}")
    plt.show()


if __name__ == "__main__":
    from data import fetch_spx, build_features
    from model import RegimeHMM
    from backtest import attach_signals, run_backtest

    raw = fetch_spx()
    X, df = build_features(raw)
    hmm = RegimeHMM().fit(X)
    labels = hmm.predict_named(X)
    df = attach_signals(df, labels)
    df = run_backtest(df)
    plot_all(df, X, labels)
