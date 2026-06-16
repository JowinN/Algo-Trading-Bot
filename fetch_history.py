"""
Fetch 6 months of historical OHLCV data from Bybit and save to CSV.
Run this once before backtesting to avoid repeated API calls.

Usage:
    python fetch_history.py
    python fetch_history.py --days 90
    python fetch_history.py --symbols BTCUSDT ETHUSDT
"""

import os
import time
import argparse
from datetime import datetime, timedelta
import pandas as pd
import requests

from config import Config as c
from data import INTERVAL_MAP

BYBIT_BASE = "https://api.bybit.com/v5/market"
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "historical_data")


def fetch_symbol_data(symbol: str, interval: str = "15m", days: int = 180) -> pd.DataFrame:
    """
    Fetch historical candles from Bybit in batches.
    Bybit returns max 200 candles per request.
    Uses exponential backoff on rate limit errors.
    """
    bybit_interval = INTERVAL_MAP.get(interval, "15")
    all_candles = []
    current_end = datetime.now()
    target_start = datetime.now() - timedelta(days=days)

    # Calculate expected candle count
    interval_minutes = {"1m": 1, "3m": 3, "5m": 5, "15m": 15, "30m": 30,
                        "1h": 60, "4h": 240, "1d": 1440}
    mins = interval_minutes.get(interval, 15)
    expected_candles = (days * 24 * 60) // mins

    print(f"   Fetching ~{expected_candles} candles ({days} days of {interval})...")

    request_count = 0
    backoff = 0.25  # Start with 250ms between requests
    max_backoff = 30  # Max 30s backoff

    while True:
        try:
            url = f"{BYBIT_BASE}/kline"
            params = {
                "category": "linear",
                "symbol": symbol,
                "interval": bybit_interval,
                "limit": 200,
                "end": int(current_end.timestamp() * 1000)
            }
            resp = requests.get(url, params=params, timeout=15)
            resp.raise_for_status()

            raw = resp.json()
            if raw.get("retCode") != 0:
                ret_msg = raw.get("retMsg", "")
                if "Too many visits" in ret_msg or "Rate Limit" in ret_msg:
                    # Rate limited — exponential backoff
                    backoff = min(backoff * 2, max_backoff)
                    print(f"   ⚠ Rate limited — waiting {backoff:.0f}s...")
                    time.sleep(backoff)
                    continue
                else:
                    print(f"   Bybit error: {ret_msg}")
                    break

            candles = raw["result"]["list"]
            if not candles:
                break

            all_candles.extend(candles)
            request_count += 1
            
            # Reset backoff on success
            backoff = 0.25

            # Move end cursor to oldest candle timestamp
            oldest_ts = float(candles[-1][0])
            new_end = datetime.fromtimestamp(oldest_ts / 1000)

            # Stop if we're not making progress (hit Bybit's data limit)
            if new_end >= current_end:
                break
            current_end = new_end

            # Stop once we've gone back far enough
            if current_end <= target_start:
                break

            # Also stop if we have more than expected (safety)
            if len(all_candles) >= expected_candles * 1.05:
                break

            # Progress indicator every 10 requests
            if request_count % 10 == 0:
                pct = min(100, (len(all_candles) / expected_candles) * 100)
                print(f"   ... {len(all_candles)} candles fetched ({pct:.0f}%)")

            # Rate limiting: 5 requests/sec to stay well under Bybit limits
            time.sleep(0.2)

        except requests.exceptions.RequestException as e:
            backoff = min(backoff * 2, max_backoff)
            print(f"   Network error: {e} — retrying in {backoff:.0f}s...")
            time.sleep(backoff)
            continue
        except Exception as e:
            print(f"   Error: {e}")
            break

    if not all_candles:
        return pd.DataFrame()

    # Convert to DataFrame
    df = pd.DataFrame(all_candles, columns=[
        "timestamp", "open", "high", "low", "close", "volume", "turnover"
    ])
    df["timestamp"] = pd.to_datetime(df["timestamp"].astype(float), unit="ms")
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)

    # Sort chronologically and remove duplicates
    df = df.sort_values("timestamp").drop_duplicates(subset=["timestamp"]).reset_index(drop=True)

    # Trim to exact date range
    df = df[df["timestamp"] >= pd.Timestamp(target_start)].reset_index(drop=True)

    return df[["timestamp", "open", "high", "low", "close", "volume"]]


def save_to_csv(df: pd.DataFrame, symbol: str, interval: str):
    """Save dataframe to CSV in historical_data/ folder."""
    os.makedirs(DATA_DIR, exist_ok=True)
    filename = f"{symbol}_{interval}.csv"
    filepath = os.path.join(DATA_DIR, filename)
    df.to_csv(filepath, index=False)
    return filepath


def main():
    parser = argparse.ArgumentParser(description="Fetch historical data from Bybit")
    parser.add_argument("--days", type=int, default=180, help="Number of days to fetch (default: 180)")
    parser.add_argument("--symbols", nargs="+", default=None, help="Symbols to fetch (default: from config)")
    parser.add_argument("--interval", type=str, default="15m", help="Candle interval (default: 15m)")
    args = parser.parse_args()

    symbols = args.symbols or c.SYMBOLS
    days = args.days
    interval = args.interval

    print(f"\n{'='*60}")
    print(f"  FETCHING {days}-DAY HISTORICAL DATA FROM BYBIT")
    print(f"{'='*60}")
    print(f"  Symbols  : {', '.join(symbols)}")
    print(f"  Interval : {interval}")
    print(f"  Period   : {(datetime.now() - timedelta(days=days)).date()} → {datetime.now().date()}")
    print(f"  Output   : {DATA_DIR}/")
    print(f"{'='*60}\n")

    results = {}
    for symbol in symbols:
        print(f"📥 {symbol}:")
        df = fetch_symbol_data(symbol, interval=interval, days=days)

        if df.empty:
            print(f"   ❌ No data fetched\n")
            continue

        filepath = save_to_csv(df, symbol, interval)
        results[symbol] = len(df)
        print(f"   ✅ {len(df)} candles → {filepath}")
        print(f"   Date range: {df['timestamp'].min()} → {df['timestamp'].max()}\n")

    # Summary
    print(f"\n{'='*60}")
    print(f"  DOWNLOAD COMPLETE")
    print(f"{'='*60}")
    for sym, count in results.items():
        print(f"  {sym:12s} : {count:>6} candles")
    total = sum(results.values())
    print(f"  {'TOTAL':12s} : {total:>6} candles")
    print(f"\n  Files saved to: {DATA_DIR}/")
    print(f"  Run backtest with: python backtest.py")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
