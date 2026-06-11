"""
HMM regime detection on SPX daily returns.

Usage:
    python main.py
    python main.py --start 2010-01-01 --end 2023-12-31
"""

import argparse
import matplotlib
matplotlib.use("Agg")

from data import fetch_spx, build_features
from model import RegimeHMM
from backtest import attach_signals, run_backtest, performance_metrics
from plot import plot_all


def main(start="2000-01-01", end="2024-12-31"):
    print(f"\nSPX HMM — {start} to {end}\n")

    raw = fetch_spx(start=start, end=end)
    X, df = build_features(raw)
    print(f"{len(df)} trading days loaded")

    hmm = RegimeHMM().fit(X)
    labels = hmm.predict_named(X)
    print(hmm.summary(X))
    print(hmm.transition_matrix())

    df = attach_signals(df, labels)
    df = run_backtest(df)
    print(performance_metrics(df))

    plot_all(df, X, labels)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2000-01-01")
    parser.add_argument("--end",   default="2024-12-31")
    args = parser.parse_args()
    main(start=args.start, end=args.end)
