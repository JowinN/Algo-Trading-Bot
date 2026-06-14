import requests
import pandas as pd

BYBIT_BASE = "https://api.bybit.com/v5/market"

# Bybit interval map
INTERVAL_MAP = {
    "1m": "1",   "3m": "3",   "5m": "5",
    "15m": "15", "30m": "30", "1h": "60",
    "4h": "240", "1d": "D"
}

def get_ohlcv(symbol: str, interval: str = "15m", limit: int = 200) -> pd.DataFrame:
    """Fetch OHLCV candles from Bybit (no API key needed)"""
    bybit_interval = INTERVAL_MAP.get(interval, "15")
    url    = f"{BYBIT_BASE}/kline"
    params = {
        "category": "linear",
        "symbol"  : symbol,
        "interval": bybit_interval,
        "limit"   : limit
    }
    resp = requests.get(url, params=params, timeout=10)
    resp.raise_for_status()

    raw = resp.json()
    if raw.get("retCode") != 0:
        raise ValueError(f"Bybit error: {raw.get('retMsg')}")

    # Bybit returns: [startTime, open, high, low, close, volume, turnover]
    candles = raw["result"]["list"]
    df = pd.DataFrame(candles, columns=[
        "timestamp", "open", "high", "low", "close", "volume", "turnover"
    ])
    df["timestamp"] = pd.to_datetime(df["timestamp"].astype(float), unit="ms")
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)

    df = df.sort_values("timestamp").set_index("timestamp")
    return df[["open", "high", "low", "close", "volume"]]

def get_current_price(symbol: str) -> float:
    """Get latest real-time price from Bybit"""
    url    = f"{BYBIT_BASE}/tickers"
    params = {"category": "linear", "symbol": symbol}
    resp   = requests.get(url, params=params, timeout=5)
    resp.raise_for_status()
    return float(resp.json()["result"]["list"][0]["lastPrice"])
