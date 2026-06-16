"""
INDICATORS v2 — Multi-Timeframe Crypto Trading System
=====================================================
Focused on: Volatility breakouts, trend pullbacks, and regime detection.
Designed for 15m primary + 1H/4H resampled higher timeframes.
"""

import pandas as pd
import numpy as np


# ══════════════════════════════════════════════════════════════════════
# CORE INDICATORS
# ══════════════════════════════════════════════════════════════════════

def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(period).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, adjust=False).mean()
    avg_loss = loss.ewm(com=period - 1, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period=14) -> pd.Series:
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)
    return tr.ewm(com=period - 1, adjust=False).mean()


def macd(series: pd.Series, fast=12, slow=26, signal=9):
    macd_line = ema(series, fast) - ema(series, slow)
    signal_line = ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def vwap(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, period: int = 20) -> pd.Series:
    """Rolling VWAP — volume weighted average price over period."""
    typical_price = (high + low + close) / 3.0
    tp_vol = typical_price * volume
    return tp_vol.rolling(period).sum() / volume.rolling(period).sum()


def bollinger_bands(series: pd.Series, period=20, std_dev=2):
    mid = series.rolling(period).mean()
    std = series.rolling(period).std()
    upper = mid + std_dev * std
    lower = mid - std_dev * std
    return upper, mid, lower


def keltner_channels(high: pd.Series, low: pd.Series, close: pd.Series,
                     ema_period=20, atr_period=14, atr_mult=1.5):
    """Keltner Channels — used with BB for squeeze detection."""
    mid = ema(close, ema_period)
    atr_val = atr(high, low, close, atr_period)
    upper = mid + atr_mult * atr_val
    lower = mid - atr_mult * atr_val
    return upper, mid, lower


def adx_system(high: pd.Series, low: pd.Series, close: pd.Series, period=14):
    """Full ADX system: ADX + DI+ + DI-"""
    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)

    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)

    atr_val = tr.ewm(com=period - 1, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(com=period - 1, adjust=False).mean() / atr_val)
    minus_di = 100 * (minus_dm.ewm(com=period - 1, adjust=False).mean() / atr_val)

    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    adx_val = dx.ewm(com=period - 1, adjust=False).mean()
    return adx_val, plus_di, minus_di


def squeeze_indicator(high: pd.Series, low: pd.Series, close: pd.Series,
                      bb_period=20, bb_std=2.0, kc_period=20, kc_mult=1.5):
    """
    TTM Squeeze: BB inside KC = squeeze (volatility compressed).
    When squeeze releases = explosive move incoming.
    
    Returns:
        squeeze_on: bool series (True = in squeeze)
        squeeze_momentum: momentum direction
    """
    bb_upper, bb_mid, bb_lower = bollinger_bands(close, bb_period, bb_std)
    kc_upper, kc_mid, kc_lower = keltner_channels(high, low, close, kc_period, kc_period, kc_mult)
    
    # Squeeze is ON when BB is inside KC
    squeeze_on = (bb_lower > kc_lower) & (bb_upper < kc_upper)
    
    # Momentum: close position relative to midline
    momentum = close - kc_mid
    
    return squeeze_on, momentum


def volume_profile(volume: pd.Series, period=20):
    """Relative volume: current vs average."""
    vol_ma = volume.rolling(period).mean()
    rel_volume = volume / vol_ma
    return rel_volume, vol_ma


def supertrend(high: pd.Series, low: pd.Series, close: pd.Series,
               period: int = 10, multiplier: float = 3.0):
    """
    Supertrend indicator — best trend-following filter.
    Returns:
        st_direction: 1 = bullish (price above), -1 = bearish (price below)
        st_value: the supertrend line value
    """
    atr_val = atr(high, low, close, period)
    hl2 = (high + low) / 2
    
    upper_band = hl2 + multiplier * atr_val
    lower_band = hl2 - multiplier * atr_val
    
    st_direction = pd.Series(index=close.index, dtype=float)
    st_value = pd.Series(index=close.index, dtype=float)
    final_upper = upper_band.copy()
    final_lower = lower_band.copy()
    
    for i in range(1, len(close)):
        # Final upper band: if current upper < prev final upper OR prev close > prev final upper
        if upper_band.iloc[i] < final_upper.iloc[i-1] or close.iloc[i-1] > final_upper.iloc[i-1]:
            final_upper.iloc[i] = upper_band.iloc[i]
        else:
            final_upper.iloc[i] = final_upper.iloc[i-1]
        
        # Final lower band: if current lower > prev final lower OR prev close < prev final lower
        if lower_band.iloc[i] > final_lower.iloc[i-1] or close.iloc[i-1] < final_lower.iloc[i-1]:
            final_lower.iloc[i] = lower_band.iloc[i]
        else:
            final_lower.iloc[i] = final_lower.iloc[i-1]
        
        # Direction
        if i == 1:
            st_direction.iloc[i] = 1 if close.iloc[i] > final_upper.iloc[i] else -1
        else:
            prev_st = st_value.iloc[i-1] if not pd.isna(st_value.iloc[i-1]) else final_upper.iloc[i-1]
            if prev_st == final_upper.iloc[i-1]:
                # Was bearish
                if close.iloc[i] > final_upper.iloc[i]:
                    st_direction.iloc[i] = 1  # Flip to bullish
                    st_value.iloc[i] = final_lower.iloc[i]
                else:
                    st_direction.iloc[i] = -1
                    st_value.iloc[i] = final_upper.iloc[i]
            else:
                # Was bullish
                if close.iloc[i] < final_lower.iloc[i]:
                    st_direction.iloc[i] = -1  # Flip to bearish
                    st_value.iloc[i] = final_upper.iloc[i]
                else:
                    st_direction.iloc[i] = 1
                    st_value.iloc[i] = final_lower.iloc[i]
    
    # Simpler approach: direction based on close vs bands
    # Recalculate cleanly
    direction = pd.Series(1, index=close.index, dtype=float)
    for i in range(1, len(close)):
        if direction.iloc[i-1] == 1:  # Was bullish
            if close.iloc[i] < final_lower.iloc[i]:
                direction.iloc[i] = -1
            else:
                direction.iloc[i] = 1
        else:  # Was bearish
            if close.iloc[i] > final_upper.iloc[i]:
                direction.iloc[i] = 1
            else:
                direction.iloc[i] = -1
    
    st_line = pd.Series(index=close.index, dtype=float)
    st_line[direction == 1] = final_lower[direction == 1]
    st_line[direction == -1] = final_upper[direction == -1]
    
    return direction, st_line


# ══════════════════════════════════════════════════════════════════════
# HIGHER TIMEFRAME RESAMPLING
# ══════════════════════════════════════════════════════════════════════

def resample_ohlcv(df: pd.DataFrame, timeframe: str = "1h") -> pd.DataFrame:
    """
    Resample 15m OHLCV data to higher timeframe.
    df must have DatetimeIndex or 'timestamp' column.
    """
    if "timestamp" in df.columns:
        temp = df.set_index("timestamp")
    else:
        temp = df.copy()
    
    resampled = temp.resample(timeframe).agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum"
    }).dropna()
    
    return resampled


# ══════════════════════════════════════════════════════════════════════
# COMPUTE ALL — Primary (15m)
# ══════════════════════════════════════════════════════════════════════

def compute_all(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all indicators for 15m data."""
    df = df.copy()
    
    # ── EMAs ──────────────────────────────────────────────────────
    df["ema9"] = ema(df["close"], 9)
    df["ema21"] = ema(df["close"], 21)
    df["ema50"] = ema(df["close"], 50)
    df["ema_fast"] = df["ema9"]
    df["ema_med"] = df["ema21"]
    df["ema_slow"] = df["ema50"]
    df["ema_trend"] = ema(df["close"], 200)
    
    # ── RSI ───────────────────────────────────────────────────────
    df["rsi"] = rsi(df["close"], 14)
    
    # ── MACD ──────────────────────────────────────────────────────
    df["macd"], df["macd_sig"], df["macd_hist"] = macd(df["close"])
    
    # ── ATR ───────────────────────────────────────────────────────
    df["atr"] = atr(df["high"], df["low"], df["close"], 14)
    
    # ── ADX System ────────────────────────────────────────────────
    df["adx"], df["di_plus"], df["di_minus"] = adx_system(
        df["high"], df["low"], df["close"], 14
    )
    
    # ── Bollinger Bands ───────────────────────────────────────────
    df["bb_upper"], df["bb_mid"], df["bb_lower"] = bollinger_bands(df["close"], 20, 2.0)
    bb_range = df["bb_upper"] - df["bb_lower"]
    df["bb_pct_b"] = ((df["close"] - df["bb_lower"]) / bb_range).where(bb_range > 0, 0.5)
    df["bb_width"] = bb_range / df["bb_mid"]
    
    # ── Squeeze Indicator (TTM Squeeze) ───────────────────────────
    df["squeeze_on"], df["squeeze_mom"] = squeeze_indicator(
        df["high"], df["low"], df["close"]
    )
    df["squeeze_fire"] = df["squeeze_on"].shift(1).fillna(False) & ~df["squeeze_on"]
    
    # ── Volume Analysis ───────────────────────────────────────────
    df["rel_volume"], df["vol_ma"] = volume_profile(df["volume"], 20)
    
    # ── Swing Points ──────────────────────────────────────────────
    df["swing_high"] = df["high"].rolling(10).max()
    df["swing_low"] = df["low"].rolling(10).min()
    
    # ── Supertrend (10, 3.0) ─ Primary trend filter ────────────
    df["st_direction"], df["st_value"] = supertrend(
        df["high"], df["low"], df["close"], period=10, multiplier=3.0
    )
    
    # ── Candle Analysis ───────────────────────────────────────────
    df["candle_body"] = abs(df["close"] - df["open"])
    df["candle_range"] = df["high"] - df["low"]
    df["body_pct"] = (df["candle_body"] / df["candle_range"]).where(df["candle_range"] > 0, 0)
    
    # ── Stochastic RSI ────────────────────────────────────────────
    rsi_vals = df["rsi"]
    rsi_min = rsi_vals.rolling(14).min()
    rsi_max = rsi_vals.rolling(14).max()
    rsi_range = rsi_max - rsi_min
    df["stoch_k"] = ((rsi_vals - rsi_min) / rsi_range).where(rsi_range > 0, 0.5) * 100
    df["stoch_k"] = df["stoch_k"].rolling(3).mean()
    df["stoch_d"] = df["stoch_k"].rolling(3).mean()
    
    # ── VWAP (20-period rolling) ────────────────────────────────
    df["vwap"] = vwap(df["high"], df["low"], df["close"], df["volume"], 20)

    # ── CMF ───────────────────────────────────────────────────────
    hl_range = df["high"] - df["low"]
    mfm = ((df["close"] - df["low"]) - (df["high"] - df["close"])) / hl_range.where(hl_range > 0, 1.0)
    mfv = mfm * df["volume"]
    df["cmf"] = mfv.rolling(20).sum() / df["volume"].rolling(20).sum()
    
    return df.dropna()


def compute_htf(df_15m: pd.DataFrame, timeframe: str = "1h") -> pd.DataFrame:
    """
    Compute higher timeframe indicators from 15m data.
    Supports: 1h, 4h, 1D
    Returns HTF dataframe with trend/momentum info.
    """
    htf = resample_ohlcv(df_15m, timeframe)
    
    # Need fewer candles for daily (60 days of daily data is a lot)
    min_candles = 30 if timeframe in ("1D", "D", "1d") else 60
    if len(htf) < min_candles:
        return pd.DataFrame()
    
    htf["ema20"] = ema(htf["close"], 20)
    htf["ema50"] = ema(htf["close"], 50)
    htf["rsi"] = rsi(htf["close"], 14)
    htf["atr"] = atr(htf["high"], htf["low"], htf["close"], 14)
    htf["adx"], htf["di_plus"], htf["di_minus"] = adx_system(
        htf["high"], htf["low"], htf["close"], 14
    )
    htf["macd"], htf["macd_sig"], htf["macd_hist"] = macd(htf["close"])
    
    # HTF Trend direction
    htf["trend_up"] = (htf["ema20"] > htf["ema50"]) & (htf["close"] > htf["ema20"])
    htf["trend_down"] = (htf["ema20"] < htf["ema50"]) & (htf["close"] < htf["ema20"])
    
    return htf.dropna()
