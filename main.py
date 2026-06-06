"""
main.py — Run the full HMM regime detection pipeline end-to-end.

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


def main(start: str = "2000-01-01", end: str = "2024-12-31"):
    print(f"\n{'='*50}")
    print(f"  HMM Regime Detection — SPX {start} to {end}")
    print(f"{'='*50}\n")

    # 1. Data
    print("[1/4] Downloading SPX data...")
    raw = fetch_spx(start=start, end=end)
    X, df = build_features(raw)
    print(f"      {len(df)} trading days loaded.\n")

    # 2. Model
    print("[2/4] Fitting Gaussian HMM (3 states)...")
    hmm = RegimeHMM().fit(X)
    labels = hmm.predict_named(X)
    print("\n--- State Summary ---")
    print(hmm.summary(X))
    print("\n--- Transition Matrix ---")
    print(hmm.transition_matrix())

    # 3. Backtest
    print("\n[3/4] Running backtest...")
    df = attach_signals(df, labels)
    df = run_backtest(df)
    print("\n--- Performance Metrics ---")
    print(performance_metrics(df))

    # 4. Charts
    print("\n[4/4] Generating charts...")
    plot_all(df, X, labels)

    print("\nDone.  Open the PNG files to view results.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HMM Regime Detection on SPX")
    parser.add_argument("--start", default="2000-01-01")
    parser.add_argument("--end",   default="2024-12-31")
    args = parser.parse_args()
    main(start=args.start, end=args.end)
