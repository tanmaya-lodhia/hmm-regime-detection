# HMM Regime Detection + Mean Reversion Short Backtest

Two connected quantitative projects built in Python from scratch.

---

## Project 1 — HMM Regime Detection (`model.py`, `main.py`)

Fits a 3-state Gaussian Hidden Markov Model to S&P 500 daily returns to classify the market into three hidden regimes: **bull**, **bear**, and **high-volatility**.

**How it works:**
- Features: daily log return + 21-day realised volatility (captures both direction and risk level)
- Model: `GaussianHMM` with full covariance — each state gets its own covariance matrix so the model captures how return and vol co-move within a regime, not just independently
- Scaling: `StandardScaler` from scikit-learn so the EM algorithm isn't dominated by whichever feature has larger magnitude
- States are labelled post-hoc by mean return: highest → bull, lowest → bear, middle → high_vol
- Strategy: long in bull, flat in high-vol, short in bear — with a 1-day signal lag to avoid look-ahead bias

**Run it:**
```bash
python main.py
python main.py --start 2010-01-01 --end 2023-12-31
```

**Output charts:**
- `regimes_on_price.png` — SPX price shaded by regime
- `cumulative_returns.png` — HMM strategy vs buy-and-hold (log scale)
- `drawdown.png` — underwater chart
- `state_scatter.png` — feature space coloured by regime cluster

---

## Project 2 — Mean Reversion Short Backtest (`backtest-mk6.py`)

Systematic short strategy on Russell 2000 small caps. Targets stocks that spike 10%+ intraday with no fundamental catalyst behind the move.

**Logic:**
1. Screen for stocks up ≥10% open-to-close with $1+ price and $200K+ daily dollar volume
2. Check NewsAPI headlines for fundamental catalysts: earnings, FDA events, M&A, analyst actions, government contracts, management changes, capital raises — skip any name that triggers
3. Short the remaining "pure momentum, no-news" names at next morning's open
4. Cover on the first day the stock closes higher, or after 10 days maximum

**Stack:**
- `yfinance` + `curl_cffi` — price data with browser impersonation to handle rate limiting
- `NewsAPI` — headline fetching with automatic key rotation across multiple free-tier keys
- Custom regex classifier — 8 catalyst categories, 50+ patterns
- Backtesting engine built from scratch — no Backtrader/Zipline dependency

**Results (May–Jun 2026 | 30 trades):**
| Metric | All trades | Over $5 only |
|---|---|---|
| Win rate | 50.0% | 47.8% |
| Avg P&L / trade | −0.78% | −0.95% |
| Profit factor | 0.84 | 0.88 |
| Per-trade Sharpe | −0.065 | −0.081 |
| Max drawdown | −93.3% | −61.6% |

The strategy underperforms this window — the backtest period coincided with a strong market recovery where momentum names continued higher. The catalyst filter works in isolation; the broader regime context is the missing layer, which is what the HMM project addresses.

**Run it:**
```bash
pip install -r requirements.txt
cp .env.example .env   # add your NewsAPI keys
python backtest-mk6.py
```

---

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env
# Edit .env and add your NewsAPI key(s) from https://newsapi.org
```

### Requirements

```
yfinance
hmmlearn
scikit-learn
pandas
numpy
matplotlib
tqdm
requests
curl_cffi
python-dotenv
```

---

## Next step

Integrate the two projects: use the HMM regime label as a gate on the mean reversion short — only activate the strategy in bear or high-vol regimes, sit flat in bull. The hypothesis is that the edge is real but regime-conditional.
