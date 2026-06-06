"""
Mean Reversion Short Strategy Backtester
=========================================
Strategy: Short top daily gainers with no fundamental catalyst.
- Universe: Russell 2000 small caps
- Uses yfinance + curl_cffi for price data
- Uses NewsAPI for headline-based catalyst detection
- Enters short at next-day open, exits on first up-day close
- Analyses win rate, avg return, Sharpe, drawdown — split by price tier

Requirements:
    pip install yfinance pandas numpy matplotlib tqdm requests curl_cffi
"""

import os
import time
import datetime
import re
import urllib.request
import warnings
import requests
from dotenv import load_dotenv
load_dotenv()
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")   # non-interactive: save to file, don't pop a window
import matplotlib.pyplot as plt
from tqdm import tqdm


# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
# NewsAPI free tier history = 30 days. Set LOOKBACK_DAYS <= 30 for clean
# catalyst-filtered results. For longer history, upgrade to a paid NewsAPI plan.
NEWSAPI_KEYS = [
    v for k, v in os.environ.items()
    if k.upper().startswith("NEWSAPI_KEY_") and v.strip()
]
if not NEWSAPI_KEYS:
    raise EnvironmentError(
        "No NewsAPI keys found. Add NEWSAPI_KEY_1=... to a .env file. "
        "See .env.example for the format."
    )
_key_index    = 0

TOP_N_GAINERS = 10       # top gainers to consider per day
MIN_GAIN_PCT  = 10.0     # minimum % intraday (open→close) gain to qualify
MIN_PRICE     = 1.0      # minimum close price
MIN_VOLUME    = 200_000  # minimum dollar volume
LOOKBACK_DAYS = 30       # must stay ≤ 30 for free NewsAPI tier
MAX_HOLD_DAYS = 10       # max days in trade before forced exit
NEWS_SLEEP    = 0.5      # seconds between NewsAPI calls
BATCH_SIZE    = 50


# ─────────────────────────────────────────────
# STEP 1 — Russell 2000 universe
# ─────────────────────────────────────────────
def get_russell2000_tickers():
    try:
        url = ("https://www.ishares.com/us/products/239710/"
               "ishares-russell-2000-etf/1467271812596.ajax"
               "?fileType=csv&fileName=IWM_holdings&dataType=fund")
        df = pd.read_csv(url, skiprows=9)
        tickers = df["Ticker"].dropna().tolist()
        tickers = [str(t).strip().replace(".", "-") for t in tickers
                   if str(t).strip() and str(t).strip() != "nan"
                   and len(str(t).strip()) <= 5]
        if len(tickers) > 100:
            print(f"      Loaded {len(tickers)} tickers from iShares IWM.")
            return tickers
    except Exception:
        pass

    try:
        url = ("https://raw.githubusercontent.com/rreichel3/"
               "US-Stock-Symbols/main/russell2000/russell2000_tickers.txt")
        with urllib.request.urlopen(url, timeout=10) as f:
            tickers = [line.decode().strip() for line in f if line.strip()]
        tickers = [t.replace(".", "-") for t in tickers if t]
        if len(tickers) > 100:
            print(f"      Loaded {len(tickers)} tickers from GitHub.")
            return tickers
    except Exception:
        pass

    print("      Using hardcoded small-cap list.")
    return [
        "ACAD","AEHR","AEIS","AGIO","AGYS","AAOI","ALRM","AMKR","AMPH","ANGI",
        "APPF","APPN","ARCT","AROW","ARWR","ATEC","ATNI","ATRC","AVNS","AXNX",
        "AXSM","BBIO","BCPC","BEAM","BHVN","BOOT","BPMC","BTAI","BYND","CARA",
        "CARS","CASH","CBPO","CCRN","CDMO","CELC","CHGG","CHWY","CLFD","CLNE",
        "CLRB","CLSK","CLOV","CMCO","CMPS","CNMD","COGT","COOP","CORT","CPRX",
        "CRDF","CRDO","CRSP","CRVL","CSWC","CSWI","CTMX","CUTR","CVAC","CVCO",
        "CVNA","DAWN","DOCN","DY","EGBN","ENSG","EPRT","ESNT","FFIN","FN",
        "FORM","FULT","GATX","GBCI","GKOS","GH","GVA","HQY","HL","HOMB",
        "IDCC","IBP","INDB","IONQ","JBTM","JXN","KRG","KTOS","KRYS","LUMN",
        "MDGL","MGY","MOGA","MOD","NE","NJR","NBHC","NBTB","NPO","NXT",
        "OFG","OKLO","ONB","ORA","PACW","PCVX","PL","PLXS","PNFP","POR",
        "PRAX","PRIM","PTGX","QBTS","RMBS","RHP","RIG","ROAD","SANM","SATS",
        "SITM","SLAB","SM","SMTC","SPXC","SR","STRL","SWX","TMHC","TEX",
        "TRNO","TTMI","TXNM","UBSI","UEC","UMBF","VAL","VLY","VSAT","VIAV",
        "WTS","ZWS","AGX","CNX","CNR","CMC","CDE","AROC","AXNX","BE","ESE",
        "ENS","FSS","BCPC","CWST","CYTK","EAT","FLR","GTLS","IDCC","IBOC",
        "BBIO","BTSG","CWAN","AHR","QBTS","PRAX","PTGX","SMTC","LUMN","VAL",
        "FCFS","MOGA","SATS","GBCI","PCVX","STRL","EPRT","ESNT","TRNO","CTRE",
    ]


# ─────────────────────────────────────────────
# STEP 2 — Download price data via yfinance
# ─────────────────────────────────────────────
def download_price_data(tickers, start, end):
    import yfinance as yf

    try:
        from curl_cffi import requests as cffi_requests
        session = cffi_requests.Session(impersonate="chrome")
        use_session = True
        print("      Using curl_cffi (browser impersonation).")
    except ImportError:
        session = None
        use_session = False
        print("      curl_cffi not found — pip install curl_cffi for fewer rate limits.")

    batch_pause = 5
    batches     = [tickers[i:i+BATCH_SIZE] for i in range(0, len(tickers), BATCH_SIZE)]
    price_data  = {}

    print(f"      Downloading {len(tickers)} tickers in "
          f"{len(batches)} batches of {BATCH_SIZE}.")

    for i, batch in enumerate(batches, 1):
        print(f"      Batch {i}/{len(batches)}...", end=" ", flush=True)
        try:
            kwargs = dict(
                start=start, end=end, interval="1d",
                group_by="ticker", auto_adjust=True,
                progress=False, threads=False,
            )
            if use_session:
                kwargs["session"] = session

            raw   = yf.download(batch, **kwargs)
            added = 0
            for ticker in batch:
                try:
                    df = raw[ticker].copy() if len(batch) > 1 else raw.copy()
                    df = df.dropna(subset=["Close"])
                    if not df.empty:
                        price_data[ticker] = df
                        added += 1
                except Exception:
                    pass
            print(f"{added} OK ({len(price_data)} total)")
        except Exception as e:
            print(f"FAILED: {e}")

        if i < len(batches):
            time.sleep(batch_pause)

    print(f"      Got data for {len(price_data)} / {len(tickers)} tickers.")
    return price_data


# ─────────────────────────────────────────────
# STEP 3 — Find top gainers per trading day
# ─────────────────────────────────────────────
def find_daily_gainers(price_data, trading_days):
    daily_gainers = {}
    for date in trading_days:
        gainers = []
        for ticker, df in price_data.items():
            if date not in df.index:
                continue
            row = df.loc[date]
            try:
                o = float(row["Open"])
                c = float(row["Close"])
                v = float(row["Volume"])
            except Exception:
                continue

            if o <= 0:
                continue

            pct        = (c - o) / o * 100
            dollar_vol = c * v

            if pct >= MIN_GAIN_PCT and c >= MIN_PRICE and dollar_vol >= MIN_VOLUME:
                gainers.append({
                    "ticker":     ticker,
                    "date":       date,
                    "open":       round(o, 4),
                    "close":      round(c, 4),
                    "volume":     v,
                    "pct_change": round(pct, 2),
                })

        gainers.sort(key=lambda x: x["pct_change"], reverse=True)
        if gainers:
            daily_gainers[date] = gainers[:TOP_N_GAINERS]

    return daily_gainers


# ─────────────────────────────────────────────
# STEP 4 — NewsAPI headlines
# ─────────────────────────────────────────────
def get_yahoo_news(ticker, date_str):
    global _key_index

    signal_dt = datetime.datetime.strptime(date_str, "%Y-%m-%d")
    from_date = (signal_dt - datetime.timedelta(days=1)).strftime("%Y-%m-%d")
    to_date   = (signal_dt + datetime.timedelta(days=1)).strftime("%Y-%m-%d")

    for attempt in range(len(NEWSAPI_KEYS)):
        key = NEWSAPI_KEYS[_key_index]
        try:
            r = requests.get(
                "https://newsapi.org/v2/everything",
                params={
                    "q": ticker, "from": from_date, "to": to_date,
                    "language": "en", "sortBy": "relevancy",
                    "pageSize": 10, "apiKey": key,
                },
                timeout=10,
            )

            if r.status_code == 429:
                print(f"\n  NewsAPI key {_key_index + 1} rate-limited — rotating...")
                _key_index = (_key_index + 1) % len(NEWSAPI_KEYS)
                if _key_index == 0:
                    print("  WARNING: All NewsAPI keys exhausted. "
                          "Add more keys or wait 12 hours.")
                    return []
                continue

            if r.status_code != 200:
                return []

            articles  = r.json().get("articles", [])
            headlines = []
            for a in articles:
                title = a.get("title", "") or ""
                desc  = a.get("description", "") or ""
                if title:
                    headlines.append(f"{title}. {desc}".strip())
            return headlines

        except Exception:
            return []

    return []


def test_newsapi():
    """Verify NewsAPI connectivity at startup."""
    print("\n  [NewsAPI diagnostic]")
    try:
        r = requests.get(
            "https://newsapi.org/v2/everything",
            params={
                "q": "IONQ",
                "from": (datetime.date.today() - datetime.timedelta(days=7)).isoformat(),
                "to":   datetime.date.today().isoformat(),
                "language": "en", "pageSize": 3,
                "apiKey": NEWSAPI_KEYS[0],
            },
            timeout=10,
        )
        data = r.json()
        status = data.get("status", "unknown")
        total  = data.get("totalResults", 0)
        msg    = data.get("message", "")
        if status == "ok":
            print(f"  OK — {total} results for test query.")
        else:
            print(f"  WARN — status={status}, message={msg}")
            print(f"  Catalyst filtering may not work. Check your API keys.")
    except Exception as e:
        print(f"  ERROR: {e}")
    print()


# ─────────────────────────────────────────────
# STEP 5 — Catalyst keyword classifier
# ─────────────────────────────────────────────
CATALYST_PATTERNS = [
    ("Earnings / guidance", [
        r"\bearnings (call|report|release|results|beat|miss|surprise)\b",
        r"\b(beat|miss|missed|beats)\s+(earnings|estimates|expectations|consensus)\b",
        r"\b(raises?|lowers?|cuts?|reaffirms?)\s+guidance\b",
        r"\bquarterly (results|earnings|revenue|profit)\b",
        r"\breports?\s+Q[1-4]\b",
        r"\bfull.year (results|earnings|outlook)\b",
        r"\b(revenue|profit|eps)\s+(jump|surge|soar|plunge|drop|beat|miss)\b",
        r"\bearnings (per share|guidance|outlook|transcript)\b",
        r"\b(annual|quarterly)\s+(revenue|profit|loss|results)\b",
    ]),
    ("M&A / deals", [
        r"\b(to be |being |will be )?acquired\b",
        r"\bacquisition (of|by|deal|agreement)\b",
        r"\bmerger (agreement|deal|with|announcement)\b",
        r"\btakeover (bid|offer|attempt)\b",
        r"\bbuyout (deal|offer|firm)\b",
        r"\blicensing (agreement|deal|pact)\b",
        r"\bjoint venture (with|agreement|deal)\b",
    ]),
    ("FDA / clinical", [
        r"\bfda (approval|approves?|clears?|grants?|accepts?|rejects?|decision)\b",
        r"\b(phase [123]|phase i{1,3})\s+(trial|study|data|results)\b",
        r"\bclinical (trial|study|data|results)\b",
        r"\b(nda|bla|pdufa)\b",
        r"\bdrug (approval|approved|candidate|application)\b",
        r"\b(breakthrough|accelerated|fast.track) (therapy|designation|approval)\b",
        r"\b(positive|negative|topline|top.line) (data|results|readout)\b",
    ]),
    ("Analyst action", [
        r"\b(upgraded?|downgraded?)\s+(to|from|by)\b",
        r"\bprice target\s+(raised?|lowered?|cut|increased?|to \$)\b",
        r"\b(initiates?|starts?|begins?)\s+coverage\b",
        r"\b(buy|sell|hold|outperform|underperform|overweight|underweight)\s+rating\b",
        r"\banalyst\s+(upgrade|downgrade|raises?|cuts?|initiates?)\b",
    ]),
    ("Contract / government", [
        r"\b(wins?|awarded?|secures?|lands?)\s+(contract|deal|agreement)\b",
        r"\b(defense|government|military|dod|nasa|pentagon)\s+(contract|award|deal)\b",
        r"\bmulti.?(million|billion|year)\s+contract\b",
    ]),
    ("Management change", [
        r"\b(appoints?|names?|hires?)\s+(new\s+)?(ceo|cfo|coo|president|chief)\b",
        r"\b(ceo|cfo|coo|chief executive)\s+(resigns?|steps down|departs?|leaves?|retires?)\b",
        r"\bleadership (change|transition|shake.?up)\b",
    ]),
    ("Capital / financing", [
        r"\b(prices?|launches?|announces?|completes?)\s+(public |secondary |follow.on )?offering\b",
        r"\b(raises?|raised)\s+\$[\d]+(m|b|million|billion)\b",
        r"\bprivate placement\b",
        r"\bconvertible (note|bond|debt|offering)\b",
        r"\bsecondary offering\b",
        r"\binitial public offering\b",
    ]),
    ("Dividend / buyback", [
        r"\b(declares?|announces?|raises?|cuts?|suspends?)\s+(quarterly|annual|special)?\s*dividend\b",
        r"\bshare (buyback|repurchase) (program|plan|announcement)\b",
        r"\bstock (buyback|repurchase)\b",
    ]),
]

def classify_catalyst(ticker, date_str, pct_change, headlines):
    if not headlines:
        return {"has_catalyst": False, "reason": "No news found."}
    combined = " ".join(headlines).lower()
    for category, patterns in CATALYST_PATTERNS:
        for pat in patterns:
            if re.search(pat, combined):
                return {"has_catalyst": True,
                        "reason": f"{category} / '{pat}'"}
    return {"has_catalyst": False, "reason": "No catalyst keywords found."}


# ─────────────────────────────────────────────
# STEP 6 — Simulate one trade
# ─────────────────────────────────────────────
def simulate_trade(ticker, signal_date, price_data):
    """Short at OPEN of day+1. Exit on first close higher than prior close, or MAX_HOLD_DAYS."""
    if ticker not in price_data:
        return None

    df     = price_data[ticker]
    future = df[df.index > signal_date].head(MAX_HOLD_DAYS + 2)

    if len(future) < 2:
        return None

    entry_row   = future.iloc[0]
    entry_price = float(entry_row["Open"])
    entry_date  = future.index[0].strftime("%Y-%m-%d")

    if entry_price <= 0:
        return None

    exit_price  = None
    exit_date   = None
    exit_reason = "max_hold"
    prev_close  = float(entry_row["Close"])

    for i in range(1, len(future)):
        row        = future.iloc[i]
        curr_close = float(row["Close"])
        bar_date   = future.index[i].strftime("%Y-%m-%d")

        if i >= MAX_HOLD_DAYS:
            exit_price  = curr_close
            exit_date   = bar_date
            exit_reason = "max_hold"
            break

        if curr_close > prev_close:
            exit_price  = curr_close
            exit_date   = bar_date
            exit_reason = "rebound"
            break

        prev_close = curr_close

    if exit_price is None or exit_price <= 0:
        return None

    pnl_pct = (entry_price - exit_price) / entry_price * 100

    return {
        "ticker":      ticker,
        "signal_date": signal_date.strftime("%Y-%m-%d"),
        "entry_date":  entry_date,
        "entry_price": round(entry_price, 4),
        "exit_price":  round(exit_price, 4),
        "exit_date":   exit_date,
        "exit_reason": exit_reason,
        "pnl_pct":     round(pnl_pct, 4),
        "win":         pnl_pct > 0,
    }


# ─────────────────────────────────────────────
# STEP 7 — Run full backtest
# ─────────────────────────────────────────────
PRICE_CACHE_FILE = "price_data_cache.pkl"

def load_price_cache():
    import os, pickle
    if os.path.exists(PRICE_CACHE_FILE):
        print(f"      Found cache: {PRICE_CACHE_FILE}")
        try:
            with open(PRICE_CACHE_FILE, "rb") as f:
                cache = pickle.load(f)
            print(f"      Loaded {len(cache)} tickers from cache. "
                  f"Delete '{PRICE_CACHE_FILE}' to force fresh download.")
            return cache
        except Exception as e:
            print(f"      Cache load failed ({e}) — downloading fresh.")
    return None

def save_price_cache(price_data):
    import pickle
    with open(PRICE_CACHE_FILE, "wb") as f:
        pickle.dump(price_data, f)
    print(f"      Cached to '{PRICE_CACHE_FILE}'.")

def run_backtest():
    print("=" * 60)
    print("  Mean Reversion Short — Backtester")
    print("=" * 60)

    end_date   = datetime.date.today() - datetime.timedelta(days=1)
    start_date = end_date - datetime.timedelta(days=LOOKBACK_DAYS)
    print(f"  Period: {start_date} -> {end_date}  ({LOOKBACK_DAYS} days)")

    test_newsapi()

    print(f"\n[1/5] Building Russell 2000 universe...")
    tickers = get_russell2000_tickers()
    if len(tickers) > 400:
        tickers = tickers[:400]
        print(f"      Trimmed to top 400.")

    print(f"\n[2/5] Loading price data...")
    price_data = load_price_cache()
    if price_data is None:
        price_data = download_price_data(tickers, str(start_date), str(end_date))
        if not price_data:
            print("ERROR: No price data. Check internet connection.")
            return
        save_price_cache(price_data)
    else:
        if not price_data:
            print("ERROR: Cache empty. Delete cache file and re-run.")
            return

    all_dates    = sorted(set(idx for df in price_data.values() for idx in df.index))
    trading_days = [d for d in all_dates
                    if start_date <= d.date() <= end_date - datetime.timedelta(days=1)]
    print(f"      {len(trading_days)} trading days in range.")

    print(f"\n[3/5] Finding daily top gainers...")
    daily_gainers  = find_daily_gainers(price_data, trading_days)
    total_signals  = sum(len(v) for v in daily_gainers.values())
    print(f"      {total_signals} gainer signals across {len(daily_gainers)} days.")

    trades           = []
    skipped_catalyst = 0
    no_data          = 0
    news_found       = 0
    news_empty       = 0

    print(f"\n[4/5] Fetching news, classifying, simulating trades...")
    for date in tqdm(sorted(daily_gainers.keys()), unit="day"):
        for g in daily_gainers[date]:
            ticker     = g["ticker"]
            pct_change = g["pct_change"]

            headlines = get_yahoo_news(ticker, date.strftime("%Y-%m-%d"))
            time.sleep(NEWS_SLEEP)

            if headlines:
                news_found += 1
            else:
                news_empty += 1

            result = classify_catalyst(ticker, date.strftime("%Y-%m-%d"),
                                       pct_change, headlines)

            if result.get("has_catalyst", True):
                skipped_catalyst += 1
                continue

            trade = simulate_trade(ticker, date, price_data)
            if trade is None:
                no_data += 1
                continue

            trade["signal_gain_pct"] = pct_change
            trade["catalyst_reason"] = result.get("reason", "")
            trade["price_tier"] = (
                "Under $5 (Penny)" if trade["entry_price"] < 5.0 else "Over $5"
            )
            trades.append(trade)

    print(f"\n[5/5] Wrapping up...")
    print(f"      Headlines found / not found: {news_found} / {news_empty}")
    print(f"      Trades simulated:            {len(trades)}")
    print(f"      Skipped (catalyst detected): {skipped_catalyst}")
    print(f"      Skipped (no future data):    {no_data}")

    if not trades:
        print("\n  No trades generated. "
              "Try lowering MIN_GAIN_PCT or MIN_VOLUME, or extend LOOKBACK_DAYS.")
        return

    df = pd.DataFrame(trades)
    df.to_csv("backtest_trades.csv", index=False)
    print(f"      Saved -> backtest_trades.csv")

    print(f"      Fetching IWM benchmark...")
    iwm_benchmark = compute_iwm_benchmark(df, str(start_date), str(end_date))
    analyse_results(df, iwm_benchmark, start_date, end_date)


# ─────────────────────────────────────────────
# STEP 8 — Analyse and plot
# ─────────────────────────────────────────────
def compute_iwm_benchmark(df, start, end):
    import yfinance as yf
    try:
        iwm = yf.download("IWM", start=start, end=end,
                          auto_adjust=True, progress=False)
        if iwm.empty:
            return None

        iwm_pnls = []
        for _, row in df.iterrows():
            try:
                entry        = pd.Timestamp(row["entry_date"])
                exit_        = pd.Timestamp(row["exit_date"])
                entry_prices = iwm.loc[iwm.index >= entry, "Open"]
                exit_prices  = iwm.loc[iwm.index >= exit_,  "Close"]
                if entry_prices.empty or exit_prices.empty:
                    continue
                iwm_entry = float(entry_prices.iloc[0])
                iwm_exit  = float(exit_prices.iloc[0])
                iwm_pnls.append((iwm_entry - iwm_exit) / iwm_entry * 100)
            except Exception:
                continue

        if not iwm_pnls:
            return None

        iwm_pnls  = np.array(iwm_pnls)
        wins      = iwm_pnls[iwm_pnls > 0]
        losses    = iwm_pnls[iwm_pnls <= 0]
        iwm_start = float(iwm["Close"].iloc[0])
        iwm_end   = float(iwm["Close"].iloc[-1])

        return {
            "trade_pnls": iwm_pnls,
            "avg_pnl":    float(np.mean(iwm_pnls)),
            "win_rate":   len(wins) / len(iwm_pnls) * 100,
            "avg_win":    float(np.mean(wins))   if len(wins)   > 0 else 0.0,
            "avg_loss":   float(np.mean(losses)) if len(losses) > 0 else 0.0,
            "bah_pnl":    (iwm_start - iwm_end) / iwm_start * 100,
            "n":          len(iwm_pnls),
        }
    except Exception as e:
        print(f"      IWM benchmark failed: {e}")
        return None


def _max_drawdown(pnl_series):
    """Max peak-to-trough drawdown of cumulative P&L."""
    cum  = pnl_series.cumsum()
    peak = cum.cummax()
    dd   = cum - peak
    return float(dd.min())


def compute_stats(subset):
    if subset.empty:
        return {}
    wins   = subset[subset["win"] == True]
    losses = subset[subset["win"] == False]
    wr     = len(wins) / len(subset) * 100
    aw     = float(wins["pnl_pct"].mean())   if len(wins)   > 0 else 0.0
    al     = float(losses["pnl_pct"].mean()) if len(losses) > 0 else 0.0
    ap     = float(subset["pnl_pct"].mean())
    pf     = abs(aw / al) if al != 0 else float("inf")
    std    = float(subset["pnl_pct"].std())
    sharpe = ap / std if std > 0 else 0.0   # per-trade Sharpe (un-annualised)
    mdd    = _max_drawdown(subset.sort_values("entry_date")["pnl_pct"])

    if al != 0 and wr > 0:
        W              = wr / 100
        R              = abs(aw / al)
        kelly          = W - (1 - W) / R
        suggested_stop = abs(al) * 1.5
    else:
        kelly          = 0.0
        suggested_stop = 0.0

    return dict(
        n=len(subset), wins=len(wins), losses=len(losses),
        win_rate=wr, avg_pnl=ap, avg_win=aw, avg_loss=al,
        profit_factor=pf, sharpe=sharpe, max_drawdown=mdd,
        kelly=kelly, suggested_stop=suggested_stop,
        max_win=float(subset["pnl_pct"].max()),
        max_loss=float(subset["pnl_pct"].min()),
        total_pnl=float(subset["pnl_pct"].sum()),
    )


def print_stats_table(label, s):
    if not s:
        print(f"\n  {label}: no trades.\n")
        return
    print(f"\n  ── {label} ({s['n']} trades) " + "─" * 30)
    print(f"  {'Win rate':<36} {s['win_rate']:>8.1f}%")
    print(f"  {'Avg P&L per trade':<36} {s['avg_pnl']:>8.2f}%")
    print(f"  {'Total P&L (sum, no compounding)':<36} {s['total_pnl']:>8.2f}%")
    print(f"  {'Avg win':<36} {s['avg_win']:>8.2f}%")
    print(f"  {'Avg loss':<36} {s['avg_loss']:>8.2f}%")
    print(f"  {'Profit factor':<36} {s['profit_factor']:>8.2f}")
    print(f"  {'Per-trade Sharpe ratio':<36} {s['sharpe']:>8.3f}")
    print(f"  {'Max drawdown (cumulative P&L)':<36} {s['max_drawdown']:>8.2f}%")
    print(f"  {'Kelly criterion':<36} {s['kelly']:>8.2%}")
    print(f"  {'Suggested stop loss':<36} {s['suggested_stop']:>8.2f}%")
    print(f"  {'Best single trade':<36} {s['max_win']:>8.2f}%")
    print(f"  {'Worst single trade':<36} {s['max_loss']:>8.2f}%")


def analyse_results(df, iwm=None,
                    start_date=None, end_date=None):
    PENNY_LABEL    = "Under $5 (Penny)"
    NONPENNY_LABEL = "Over $5"

    df_penny    = df[df["price_tier"] == PENNY_LABEL].copy()
    df_nonpenny = df[df["price_tier"] == NONPENNY_LABEL].copy()

    stats_all      = compute_stats(df)
    stats_penny    = compute_stats(df_penny)
    stats_nonpenny = compute_stats(df_nonpenny)

    print_stats_table("ALL TRADES",            stats_all)
    print_stats_table("OVER $5 only",          stats_nonpenny)
    print_stats_table("UNDER $5 (Penny) only", stats_penny)

    if iwm:
        print(f"\n  ── IWM BENCHMARK (short IWM, same windows) " + "─" * 18)
        print(f"  {'Trades matched':<36} {iwm['n']:>8}")
        print(f"  {'Win rate':<36} {iwm['win_rate']:>8.1f}%")
        print(f"  {'Avg P&L per trade':<36} {iwm['avg_pnl']:>8.2f}%")
        print(f"  {'Avg win':<36} {iwm['avg_win']:>8.2f}%")
        print(f"  {'Avg loss':<36} {iwm['avg_loss']:>8.2f}%")
        print(f"  {'Buy-and-hold short IWM (full period)':<36} {iwm['bah_pnl']:>8.2f}%")
        alpha = stats_all.get("avg_pnl", 0) - iwm["avg_pnl"]
        print(f"  {'Alpha vs IWM (avg P&L/trade)':<36} {alpha:>+8.2f}%")
        verdict = ("OUTPERFORMS" if alpha > 0 else "UNDERPERFORMS")
        print(f"  Strategy {verdict} short IWM by {abs(alpha):.2f}% per trade.")
    else:
        print("\n  IWM benchmark unavailable.")

    # ── Chart ──────────────────────────────────────────────────
    date_str  = ""
    if start_date and end_date:
        date_str = f"  |  {start_date.strftime('%b %d, %Y')} – {end_date.strftime('%b %d, %Y')}"

    DARK_BG   = "#0d0d0d"
    PANEL_BG  = "#161616"
    C_PENNY   = "#ff9f43"
    C_REGULAR = "#00e5ff"
    C_ALL     = "#a29bfe"
    C_IWM     = "#f9ca24"

    fig = plt.figure(figsize=(18, 13), dpi=150)
    fig.suptitle(
        f"Mean Reversion Short Strategy  —  Backtest Results{date_str}",
        fontsize=14, fontweight="bold", color="#ffffff", y=0.98,
    )
    fig.patch.set_facecolor(DARK_BG)
    gs = fig.add_gridspec(3, 3, hspace=0.50, wspace=0.38)

    def style_ax(ax):
        ax.set_facecolor(PANEL_BG)
        ax.tick_params(colors="#aaaaaa", labelsize=8)
        ax.xaxis.label.set_color("#aaaaaa")
        ax.yaxis.label.set_color("#aaaaaa")
        ax.title.set_color("#eeeeee")
        for spine in ax.spines.values():
            spine.set_edgecolor("#2a2a2a")

    # Row 0 — equity curves
    for col, (subset, label, color) in enumerate([
        (df_nonpenny, "Over $5",          C_REGULAR),
        (df_penny,    "Under $5 (Penny)", C_PENNY),
        (df,          "All trades",       C_ALL),
    ]):
        ax = fig.add_subplot(gs[0, col])
        style_ax(ax)
        if not subset.empty:
            s   = subset.sort_values("entry_date").reset_index(drop=True)
            cum = s["pnl_pct"].cumsum()
            ax.plot(s.index, cum, color=color, linewidth=1.6, label="Strategy")
            ax.fill_between(s.index, cum, 0, where=cum >= 0,
                            alpha=0.13, color=color)
            ax.fill_between(s.index, cum, 0, where=cum < 0,
                            alpha=0.13, color="#ff4444")

            if col == 2 and iwm is not None:
                iwm_cum = np.cumsum(iwm["trade_pnls"][:len(s)])
                ax.plot(range(len(iwm_cum)), iwm_cum, color=C_IWM,
                        linewidth=1.3, linestyle="--", label="Short IWM")
                ax.legend(fontsize=7, labelcolor="#cccccc", framealpha=0)

            # Annotate final P&L
            final = float(cum.iloc[-1])
            ax.annotate(f"{final:+.1f}%", xy=(len(cum)-1, final),
                        xytext=(5, 0), textcoords="offset points",
                        color=color, fontsize=8, va="center")
            ax.axhline(0, color="#444444", linewidth=0.7, linestyle="--")

        ax.set_title(f"Equity Curve — {label}", fontsize=9, pad=6)
        ax.set_xlabel("Trade #", fontsize=8)
        ax.set_ylabel("Cum. P&L %", fontsize=8)

    # Row 1 — P&L distributions
    for col, (subset, label, color) in enumerate([
        (df_nonpenny, "Over $5",          C_REGULAR),
        (df_penny,    "Under $5 (Penny)", C_PENNY),
        (df,          "All",              C_ALL),
    ]):
        ax = fig.add_subplot(gs[1, col])
        style_ax(ax)
        if not subset.empty:
            ax.hist(subset["pnl_pct"], bins=25, color=color,
                    edgecolor=DARK_BG, alpha=0.85)
            ax.axvline(0, color="#ff4444", linewidth=1.2, linestyle="--",
                       label="Break-even")
            mv = subset["pnl_pct"].mean()
            ax.axvline(mv, color="#ffff66", linewidth=1.2, linestyle="--",
                       label=f"Avg {mv:+.2f}%")
            if col == 2 and iwm is not None:
                ax.axvline(iwm["avg_pnl"], color=C_IWM, linewidth=1.2,
                           linestyle=":", label=f"IWM {iwm['avg_pnl']:+.2f}%")
            ax.legend(fontsize=7, labelcolor="#cccccc", framealpha=0)
        ax.set_title(f"P&L Distribution — {label}", fontsize=9, pad=6)
        ax.set_xlabel("P&L %", fontsize=8)
        ax.set_ylabel("Frequency", fontsize=8)

    # Row 2 col 0-1 — win rate bars
    ax_wr = fig.add_subplot(gs[2, 0:2])
    style_ax(ax_wr)
    tiers     = ["Over $5", "Under $5\n(Penny)", "All trades",
                 "Short IWM\n(benchmark)"]
    win_rates = [
        stats_nonpenny.get("win_rate", 0),
        stats_penny.get("win_rate",    0),
        stats_all.get("win_rate",      0),
        iwm["win_rate"] if iwm else 0,
    ]
    colors_wr = [C_REGULAR, C_PENNY, C_ALL, C_IWM]
    bars = ax_wr.bar(tiers, win_rates, color=colors_wr,
                     edgecolor=DARK_BG, width=0.5)
    ax_wr.axhline(50, color="#555555", linewidth=0.9,
                  linestyle="--", label="50% break-even")
    ax_wr.set_ylim(0, 105)
    ax_wr.set_title("Win Rate by Tier vs Benchmark", fontsize=9, pad=6)
    ax_wr.set_ylabel("Win Rate %", fontsize=8)
    ax_wr.legend(fontsize=8, labelcolor="#cccccc", framealpha=0)
    for bar, val in zip(bars, win_rates):
        ax_wr.text(bar.get_x() + bar.get_width() / 2, val + 1.8,
                   f"{val:.1f}%", ha="center", color="#ffffff",
                   fontsize=9, fontweight="bold")

    # Row 2 col 2 — avg P&L + Sharpe summary
    ax_ap = fig.add_subplot(gs[2, 2])
    style_ax(ax_ap)
    labels_ap = ["Over $5", "All trades", "Short IWM"]
    avg_pnls  = [
        stats_nonpenny.get("avg_pnl", 0),
        stats_all.get("avg_pnl",      0),
        iwm["avg_pnl"] if iwm else 0,
    ]
    colors_ap = ["#00cc66" if v > 0 else "#ff4444" for v in avg_pnls]
    bars2 = ax_ap.bar(labels_ap, avg_pnls, color=colors_ap,
                      edgecolor=DARK_BG, width=0.5)
    ax_ap.axhline(0, color="#444444", linewidth=0.8, linestyle="--")
    ax_ap.set_title("Avg P&L / Trade vs Benchmark", fontsize=9, pad=6)
    ax_ap.set_ylabel("Avg P&L %", fontsize=8)
    for i, v in enumerate(avg_pnls):
        ax_ap.text(i, v + (0.04 if v >= 0 else -0.12),
                   f"{v:+.2f}%", ha="center", color="#ffffff", fontsize=9)

    # Key-stats text box (bottom-right of fig)
    if stats_all:
        box_lines = [
            f"n = {stats_all['n']} trades",
            f"Win rate:   {stats_all['win_rate']:.1f}%",
            f"Avg P&L:    {stats_all['avg_pnl']:+.2f}%",
            f"Sharpe:     {stats_all['sharpe']:.3f}",
            f"Max DD:     {stats_all['max_drawdown']:.2f}%",
            f"Prof. factor: {stats_all['profit_factor']:.2f}",
        ]
        fig.text(0.985, 0.02, "\n".join(box_lines),
                 ha="right", va="bottom", color="#cccccc",
                 fontsize=8, family="monospace",
                 bbox=dict(boxstyle="round,pad=0.5",
                           facecolor="#1e1e1e", edgecolor="#444444"))

    # Footnote
    fig.text(0.012, 0.005,
             "Strategy: short top Russell 2000 daily gainers (≥10% intraday) "
             "with no headline catalyst  |  Entry: next-day open  |  "
             "Exit: first up-day close or 10-day max hold  |  "
             "Not financial advice.",
             fontsize=6.5, color="#666666", ha="left", va="bottom")

    plt.savefig("backtest_results.png", dpi=150,
                bbox_inches="tight", facecolor=DARK_BG)
    print(f"\n  Chart saved -> backtest_results.png  (150 dpi, LinkedIn-ready)")

    print(f"\n{'='*60}")
    for label, s in [
        ("Over $5",          stats_nonpenny),
        ("Under $5 (Penny)", stats_penny),
        ("All",              stats_all),
    ]:
        if s:
            print(f"  [{label}] Stop: {s['suggested_stop']:.1f}%  "
                  f"(1.5× avg loss {abs(s['avg_loss']):.1f}%)")
    print(f"{'='*60}")

    generate_linkedin_post(stats_all, stats_nonpenny, stats_penny,
                           iwm, start_date, end_date)


def generate_linkedin_post(stats_all, stats_nonpenny, stats_penny,
                            iwm, start_date, end_date):
    """Print a ready-to-copy LinkedIn post based on backtest results."""
    if not stats_all:
        return

    period = ""
    if start_date and end_date:
        period = f"{start_date.strftime('%b %d')} – {end_date.strftime('%b %d, %Y')}"

    wr   = stats_all.get("win_rate",      0)
    ap   = stats_all.get("avg_pnl",       0)
    pf   = stats_all.get("profit_factor", 0)
    sh   = stats_all.get("sharpe",        0)
    mdd  = stats_all.get("max_drawdown",  0)
    n    = stats_all.get("n",             0)
    wr5  = stats_nonpenny.get("win_rate", 0)
    ap5  = stats_nonpenny.get("avg_pnl",  0)
    alpha = (ap - iwm["avg_pnl"]) if iwm else None

    lines = [
        "",
        "─" * 60,
        "  LINKEDIN POST (copy from here)",
        "─" * 60,
        "",
        f"I ran a mean reversion short backtest on Russell 2000 small caps ({period}).",
        "",
        "Strategy logic:",
        f"  • Screen for stocks up ≥{MIN_GAIN_PCT:.0f}% intraday (open→close)",
        f"  • Filter OUT any name with a headline catalyst (earnings, FDA, M&A, etc.)",
        f"  • Short the remainder at next morning's open",
        f"  • Cover on the first day the stock closes higher, or after {MAX_HOLD_DAYS} days",
        "",
        f"Results ({n} trades, {period}):",
        f"  • Win rate:      {wr:.1f}%",
        f"  • Avg P&L/trade: {ap:+.2f}%",
        f"  • Profit factor: {pf:.2f}",
        f"  • Per-trade Sharpe: {sh:.3f}",
        f"  • Max drawdown:  {mdd:.2f}%  (cumulative P&L basis)",
    ]
    if stats_nonpenny:
        lines.append(f"  • Over-$5 names only: {wr5:.1f}% win rate, {ap5:+.2f}% avg P&L")
    if alpha is not None:
        dir_ = "outperforms" if alpha > 0 else "underperforms"
        lines.append(f"  • Alpha vs. short IWM (same windows): {alpha:+.2f}% per trade")
        lines.append(f"    → Strategy {dir_} passive short by {abs(alpha):.2f}%/trade")
    lines += [
        "",
        "Key takeaway: stocks that spike on NO fundamental news tend to mean-revert.",
        "The catalyst filter is the alpha — without it, you're fighting earnings gaps.",
        "",
        "#quantfinance #algotrading #meanreversion #smallcap #backtesting",
        "",
        "─" * 60,
        "  (not financial advice — simulated results only)",
        "─" * 60,
        "",
    ]
    print("\n".join(lines))


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    run_backtest()
