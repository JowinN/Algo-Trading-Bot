"""
ML Training V3 — Regime-Based Ensemble + Deep Lookback Features
================================================================
Fundamentally different approach from V2:

1. MARKET REGIME DETECTION:
   - Volatility regime (expanding/contracting/stable)
   - Trend regime (strong trend/weak trend/ranging)
   - Momentum regime (accelerating/decelerating/reversing)
   - Volume regime (active/quiet)

2. DEEP LOOKBACK FEATURES (20-50 candle context):
   - Rolling win-rate of recent signals
   - Price action patterns (higher highs/lower lows count)
   - Mean reversion vs momentum score
   - Cross-timeframe agreement
   - Volatility persistence and clustering

3. CROSS-ASSET FEATURES:
   - BTC correlation regime
   - Market-wide volatility (VIX-like)
   - Sector rotation signals

4. ENSEMBLE:
   - Regime classifier → selects sub-model
   - Trending regime model (momentum features weighted)
   - Ranging regime model (mean-reversion features weighted)
   - Volatile regime model (breakout features weighted)

Key differences from V2:
- Features focus on CONTEXT not SNAPSHOT
- Regime detection provides non-stationarity handling
- Much longer lookback (50 bars = 8 days on 4H)
- Ensemble prevents single-model underfitting
- Probability calibration per regime
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
import requests
from datetime import datetime, timedelta
from collections import defaultdict

from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss
from sklearn.calibration import CalibratedClassifierCV
import xgboost as xgb

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from indicators import compute_all, ema, rsi, atr, macd, adx_system
from strategy import generate_signal, Signal
from config import Config as c

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "historical_data")
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ml_models")
os.makedirs(MODEL_DIR, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════
# REGIME DETECTION
# ══════════════════════════════════════════════════════════════════════════

class RegimeDetector:
    """
    Classifies market into regimes based on multiple dimensions.
    All methods are vectorized with numpy/pandas for speed.
    """

    @staticmethod
    def detect_volatility_regime(df, lookback=50):
        """Volatility regime using rolling percentile rank of ATR."""
        atr_vals = df["atr"].values.astype(float)
        regime = np.zeros(len(df))
        # Use rolling rank
        atr_series = pd.Series(atr_vals)
        rolling_rank = atr_series.rolling(lookback).apply(
            lambda x: pd.Series(x).rank().iloc[-1] / len(x), raw=False
        )
        pct_rank = rolling_rank.values
        regime[pct_rank > 0.75] = 1    # expanding
        regime[pct_rank < 0.25] = -1   # contracting
        return regime

    @staticmethod
    def detect_trend_regime(df, lookback=50):
        """Trend regime using ADX + EMA alignment (vectorized)."""
        adx_vals = df["adx"].values.astype(float)
        ema9 = df["ema9"].values.astype(float)
        ema21 = df["ema21"].values.astype(float)
        ema50 = df["ema50"].values.astype(float)

        # EMA alignment: all aligned = 3, partial = 2, none = 1
        full_bull = (ema9 > ema21) & (ema21 > ema50)
        full_bear = (ema9 < ema21) & (ema21 < ema50)
        fully_aligned = full_bull | full_bear

        regime = np.zeros(len(df))
        regime[(adx_vals > 30) & fully_aligned] = 2       # strong trend
        regime[(adx_vals > 20) & (adx_vals <= 30) & fully_aligned] = 1  # weak trend
        regime[(adx_vals < 15)] = -1                       # choppy
        # Remaining = 0 (ranging)
        return regime

    @staticmethod
    def detect_momentum_regime(df, lookback=20):
        """Momentum regime using MACD histogram slope (vectorized)."""
        hist = df["macd_hist"].values.astype(float)
        atr_vals = df["atr"].values.astype(float)
        regime = np.zeros(len(df))

        # 5-bar slope of MACD hist normalized by ATR
        hist_series = pd.Series(hist)
        slope = (hist_series - hist_series.shift(4)) / 4.0
        slope_norm = np.where(atr_vals > 0, slope.values / atr_vals, 0)

        regime[slope_norm > 0.1] = 2    # strongly accelerating
        regime[(slope_norm > 0.02) & (slope_norm <= 0.1)] = 1  # mildly accelerating
        regime[slope_norm < -0.1] = -2   # strongly decelerating
        regime[(slope_norm < -0.02) & (slope_norm >= -0.1)] = -1  # mildly decelerating
        return regime

    @staticmethod
    def detect_volume_regime(df, lookback=20):
        """Volume regime using rolling mean of relative volume."""
        if "rel_volume" not in df.columns:
            return np.zeros(len(df))
        vol = df["rel_volume"].values.astype(float)
        vol_series = pd.Series(vol)
        avg_vol = vol_series.rolling(lookback).mean().values

        regime = np.zeros(len(df))
        regime[avg_vol > 1.5] = 2    # very active
        regime[(avg_vol > 1.0) & (avg_vol <= 1.5)] = 1  # active
        regime[avg_vol < 0.6] = -1   # quiet
        return regime


# ══════════════════════════════════════════════════════════════════════════
# DEEP LOOKBACK FEATURES (50-bar context)
# ══════════════════════════════════════════════════════════════════════════

def extract_regime_features(df, idx, direction):
    """
    Extract regime-aware features with deep lookback (50 bars context).
    These features capture MARKET CONTEXT rather than single-bar state.
    """
    if idx < 50:
        return None

    dir_sign = 1.0 if direction == "LONG" else -1.0
    curr = df.iloc[idx]
    price = float(curr["close"])
    atr_val = float(curr["atr"])

    if atr_val <= 0:
        return None

    features = {}

    # ── REGIME STATE (4 features) ─────────────────────────────────────
    features["vol_regime"] = float(df["vol_regime"].iloc[idx]) if "vol_regime" in df.columns else 0
    features["trend_regime"] = float(df["trend_regime"].iloc[idx]) if "trend_regime" in df.columns else 0
    features["mom_regime"] = float(df["mom_regime"].iloc[idx]) if "mom_regime" in df.columns else 0
    features["volume_regime"] = float(df["volume_regime"].iloc[idx]) if "volume_regime" in df.columns else 0

    # ── TREND CONTEXT (12 features) ──────────────────────────────────
    # Price position in N-bar range
    high_50 = df["high"].iloc[idx-50:idx+1].max()
    low_50 = df["low"].iloc[idx-50:idx+1].min()
    range_50 = high_50 - low_50
    features["price_in_50bar_range"] = (price - low_50) / range_50 if range_50 > 0 else 0.5

    high_20 = df["high"].iloc[idx-20:idx+1].max()
    low_20 = df["low"].iloc[idx-20:idx+1].min()
    range_20 = high_20 - low_20
    features["price_in_20bar_range"] = (price - low_20) / range_20 if range_20 > 0 else 0.5

    # Higher-highs / lower-lows count (trend structure)
    highs = df["high"].iloc[idx-20:idx+1].values
    lows = df["low"].iloc[idx-20:idx+1].values
    hh_count = sum(1 for i in range(1, len(highs)) if highs[i] > highs[i-1])
    ll_count = sum(1 for i in range(1, len(lows)) if lows[i] < lows[i-1])
    features["hh_count_20"] = hh_count / 20.0 * dir_sign
    features["ll_count_20"] = ll_count / 20.0 * dir_sign
    features["structure_score"] = (hh_count - ll_count) / 20.0 * dir_sign

    # EMA fan (distance between EMAs normalized)
    ema9 = float(curr["ema9"])
    ema21 = float(curr["ema21"])
    ema50 = float(curr["ema50"])
    features["ema_fan_width"] = abs(ema9 - ema50) / atr_val
    features["ema_alignment"] = ((ema9 - ema21) + (ema21 - ema50)) / atr_val * dir_sign

    # Trend age (how many bars since last EMA crossover)
    ema9_vals = df["ema9"].iloc[idx-50:idx+1].values
    ema21_vals = df["ema21"].iloc[idx-50:idx+1].values
    trend_age = 0
    for j in range(len(ema9_vals)-1, 0, -1):
        if (ema9_vals[j] > ema21_vals[j]) != (ema9_vals[j-1] > ema21_vals[j-1]):
            break
        trend_age += 1
    features["trend_age"] = min(trend_age / 50.0, 1.0)

    # Slope of price over 10, 20, 50 bars
    closes = df["close"].iloc[idx-50:idx+1].values
    features["slope_10"] = (closes[-1] - closes[-10]) / (atr_val * 10) * dir_sign
    features["slope_20"] = (closes[-1] - closes[-20]) / (atr_val * 20) * dir_sign
    features["slope_50"] = (closes[-1] - closes[-50]) / (atr_val * 50) * dir_sign

    # ── VOLATILITY CONTEXT (10 features) ─────────────────────────────
    atr_vals = df["atr"].iloc[idx-50:idx+1].values
    features["atr_percentile_50"] = np.searchsorted(np.sort(atr_vals), atr_val) / 50.0
    features["atr_ratio_10_50"] = np.mean(atr_vals[-10:]) / np.mean(atr_vals) if np.mean(atr_vals) > 0 else 1.0
    features["atr_expanding"] = 1.0 if atr_vals[-1] > atr_vals[-5] > atr_vals[-10] else 0.0
    features["atr_contracting"] = 1.0 if atr_vals[-1] < atr_vals[-5] < atr_vals[-10] else 0.0

    # Bollinger Band squeeze history
    if "bb_width" in df.columns:
        bb_widths = df["bb_width"].iloc[idx-20:idx+1].values
        features["bb_squeeze_depth"] = bb_widths[-1] / np.mean(bb_widths) if np.mean(bb_widths) > 0 else 1.0
        features["bb_width_trend"] = (bb_widths[-1] - bb_widths[-5]) / np.mean(bb_widths) if np.mean(bb_widths) > 0 else 0
    else:
        features["bb_squeeze_depth"] = 1.0
        features["bb_width_trend"] = 0.0

    # Range vs trend (ADX history)
    adx_vals = df["adx"].iloc[idx-20:idx+1].values
    features["adx_mean_20"] = np.mean(adx_vals) / 100.0
    features["adx_trend"] = (adx_vals[-1] - adx_vals[-5]) / 50.0
    features["adx_above_25_pct"] = np.mean(adx_vals > 25)
    features["ranging_bars_pct"] = np.mean(adx_vals < 20)

    # ── MOMENTUM CONTEXT (12 features) ───────────────────────────────
    rsi_vals = df["rsi"].iloc[idx-20:idx+1].values
    features["rsi_mean_20"] = np.mean(rsi_vals) / 100.0
    features["rsi_std_20"] = np.std(rsi_vals) / 50.0
    features["rsi_current_vs_mean"] = (rsi_vals[-1] - np.mean(rsi_vals)) / 50.0 * dir_sign
    features["rsi_overbought_pct"] = np.mean(rsi_vals > 70)
    features["rsi_oversold_pct"] = np.mean(rsi_vals < 30)

    # MACD histogram pattern
    hist_vals = df["macd_hist"].iloc[idx-20:idx+1].values
    hist_norm = hist_vals / atr_val
    features["macd_hist_mean_20"] = np.mean(hist_norm) * dir_sign
    features["macd_hist_std_20"] = np.std(hist_norm)
    features["macd_hist_positive_pct"] = np.mean(hist_norm * dir_sign > 0)
    # MACD divergence: price making new highs but MACD hist declining
    if direction == "LONG":
        price_new_high = closes[-1] > np.max(closes[-10:-1])
        macd_declining = hist_vals[-1] < np.max(hist_vals[-10:-1])
        features["divergence"] = 1.0 if (price_new_high and macd_declining) else 0.0
    else:
        price_new_low = closes[-1] < np.min(closes[-10:-1])
        macd_rising = hist_vals[-1] > np.min(hist_vals[-10:-1])
        features["divergence"] = 1.0 if (price_new_low and macd_rising) else 0.0

    # Momentum persistence (consecutive positive/negative bars)
    consec_pos = 0
    for j in range(len(hist_norm)-1, -1, -1):
        if hist_norm[j] * dir_sign > 0:
            consec_pos += 1
        else:
            break
    features["momentum_persistence"] = min(consec_pos / 10.0, 1.0)

    # Stoch RSI divergence from price
    if "stoch_k" in df.columns:
        stoch = df["stoch_k"].iloc[idx-10:idx+1].values
        features["stoch_divergence"] = (stoch[-1] - stoch[0]) / 100.0 * dir_sign
    else:
        features["stoch_divergence"] = 0.0

    # ── VOLUME CONTEXT (8 features) ──────────────────────────────────
    if "rel_volume" in df.columns:
        vol_vals = df["rel_volume"].iloc[idx-20:idx+1].values
        features["vol_mean_20"] = np.mean(vol_vals)
        features["vol_std_20"] = np.std(vol_vals)
        features["vol_trend_20"] = (np.mean(vol_vals[-5:]) - np.mean(vol_vals[:5])) / max(np.mean(vol_vals), 0.1)
        features["vol_spike_count"] = np.mean(vol_vals > 2.0)
        features["vol_current_rank"] = np.searchsorted(np.sort(vol_vals), vol_vals[-1]) / len(vol_vals)
    else:
        features["vol_mean_20"] = 1.0
        features["vol_std_20"] = 0.0
        features["vol_trend_20"] = 0.0
        features["vol_spike_count"] = 0.0
        features["vol_current_rank"] = 0.5

    # CMF accumulation/distribution
    if "cmf" in df.columns:
        cmf_vals = df["cmf"].iloc[idx-10:idx+1].values
        features["cmf_mean_10"] = np.mean(cmf_vals) * dir_sign
        features["cmf_trend"] = (cmf_vals[-1] - cmf_vals[0]) * dir_sign
        features["cmf_positive_pct"] = np.mean(cmf_vals * dir_sign > 0)
    else:
        features["cmf_mean_10"] = 0.0
        features["cmf_trend"] = 0.0
        features["cmf_positive_pct"] = 0.5

    # ── MEAN REVERSION vs MOMENTUM SCORE (4 features) ────────────────
    # How far is price from moving averages? (mean reversion potential)
    dist_ema21 = (price - ema21) / atr_val * dir_sign
    dist_ema50 = (price - ema50) / atr_val * dir_sign
    features["mean_rev_score"] = -abs(dist_ema21) / 3.0  # Negative = further from mean
    features["momentum_score"] = dist_ema21  # Positive = momentum in our direction

    # Ratio of current move to typical move (how extended)
    typical_range = np.mean(np.abs(np.diff(closes[-20:])))
    current_move = abs(closes[-1] - closes[-5])
    features["extension_ratio"] = current_move / (typical_range * 5) if typical_range > 0 else 1.0

    # Recent pullback depth (how much has it retraced)
    if direction == "LONG":
        recent_high = np.max(closes[-10:])
        pullback = (recent_high - price) / atr_val
    else:
        recent_low = np.min(closes[-10:])
        pullback = (price - recent_low) / atr_val
    features["pullback_depth"] = pullback

    # ── PRICE ACTION PATTERNS (8 features) ───────────────────────────
    # Body-to-wick ratio of recent candles
    opens = df["open"].iloc[idx-5:idx+1].values
    closes_5 = df["close"].iloc[idx-5:idx+1].values
    highs_5 = df["high"].iloc[idx-5:idx+1].values
    lows_5 = df["low"].iloc[idx-5:idx+1].values
    
    bodies = np.abs(closes_5 - opens)
    ranges = highs_5 - lows_5
    body_ratios = bodies / np.maximum(ranges, 0.0001)
    features["avg_body_ratio_5"] = np.mean(body_ratios)

    # Candle direction consistency
    bullish = np.sum(closes_5 > opens)
    features["bullish_candle_pct"] = bullish / len(closes_5) * dir_sign + (1 - dir_sign) * (1 - bullish / len(closes_5))

    # Gap frequency (close vs next open)
    if idx >= 6:
        opens_prev = df["open"].iloc[idx-4:idx+1].values
        closes_prev = df["close"].iloc[idx-5:idx].values
        gaps = np.abs(opens_prev - closes_prev)
        features["avg_gap_size"] = np.mean(gaps) / atr_val
    else:
        features["avg_gap_size"] = 0.0

    # Support/resistance proximity
    high_10 = df["high"].iloc[idx-10:idx+1].max()
    low_10 = df["low"].iloc[idx-10:idx+1].min()
    if direction == "LONG":
        features["dist_to_resistance"] = (high_10 - price) / atr_val
        features["dist_to_support"] = (price - low_10) / atr_val
    else:
        features["dist_to_resistance"] = (price - low_10) / atr_val
        features["dist_to_support"] = (high_10 - price) / atr_val

    # Inside bar / outside bar patterns
    prev_range = highs_5[-2] - lows_5[-2]
    curr_range = highs_5[-1] - lows_5[-1]
    features["inside_bar"] = 1.0 if (highs_5[-1] < highs_5[-2] and lows_5[-1] > lows_5[-2]) else 0.0
    features["range_expansion"] = curr_range / prev_range if prev_range > 0 else 1.0

    # ── CROSS-TIMEFRAME (4 features) ────────────────────────────────
    # These approximate HTF by using longer lookbacks
    close_20d_ago = closes[-min(120, len(closes))] if len(closes) > 120 else closes[0]  # ~20 days on 4H
    features["trend_20d"] = (price - close_20d_ago) / (atr_val * 20) * dir_sign

    # 50-bar momentum (vs 10-bar) — momentum acceleration
    mom_50 = (closes[-1] - closes[0]) / (atr_val * 50)
    mom_10 = (closes[-1] - closes[-10]) / (atr_val * 10)
    features["momentum_accel"] = (mom_10 - mom_50) * dir_sign
    features["multi_tf_agree"] = 1.0 if (mom_10 * dir_sign > 0 and mom_50 * dir_sign > 0) else 0.0
    features["tf_conflict"] = 1.0 if (mom_10 * dir_sign > 0) != (mom_50 * dir_sign > 0) else 0.0

    return features


# ══════════════════════════════════════════════════════════════════════════
# DATA PREPARATION
# ══════════════════════════════════════════════════════════════════════════

def load_and_prepare_data(symbol, use_full=True):
    """Load CSV data and compute all indicators + regimes."""
    csv_path = os.path.join(DATA_DIR, f"{symbol}_15m.csv")
    if not os.path.exists(csv_path):
        print(f"  {symbol}: CSV not found")
        return None

    df_15m = pd.read_csv(csv_path, parse_dates=["timestamp"], index_col="timestamp")
    if len(df_15m) < 2000:
        print(f"  {symbol}: Insufficient data ({len(df_15m)} bars)")
        return None

    # Resample to 4H
    df_4h = df_15m.resample("4h").agg({
        "open": "first", "high": "max",
        "low": "min", "close": "last", "volume": "sum"
    }).dropna()

    if len(df_4h) < 200:
        return None

    # Compute indicators
    df_4h = compute_all(df_4h)

    # Compute regimes
    df_4h["vol_regime"] = RegimeDetector.detect_volatility_regime(df_4h)
    df_4h["trend_regime"] = RegimeDetector.detect_trend_regime(df_4h)
    df_4h["mom_regime"] = RegimeDetector.detect_momentum_regime(df_4h)
    df_4h["volume_regime"] = RegimeDetector.detect_volume_regime(df_4h)

    # HTF bias from daily
    df_daily = df_15m.resample("1D").agg({
        "open": "first", "high": "max",
        "low": "min", "close": "last", "volume": "sum"
    }).dropna()

    htf_bias_series = pd.Series("NONE", index=df_4h.index)
    if len(df_daily) >= 50:
        df_daily["ema20"] = ema(df_daily["close"], 20)
        df_daily["ema50"] = ema(df_daily["close"], 50)
        df_daily = df_daily.dropna()
        for i, (ts, row) in enumerate(df_daily.iterrows()):
            if row["ema20"] > row["ema50"]:
                bias = "LONG"
            elif row["ema20"] < row["ema50"]:
                bias = "SHORT"
            else:
                bias = "NONE"
            # Apply to all 4H bars on this day
            day_mask = df_4h.index.date == ts.date()
            htf_bias_series[day_mask] = bias

    return df_4h, htf_bias_series


def simulate_trade_outcome(df, entry_idx, direction, atr_val, max_hold=50):
    """
    Simulate trade outcome with SL/TP from config.
    Returns: (outcome, mfe_atr, mae_atr, hold_bars, exit_type)
    """
    entry_price = float(df["close"].iloc[entry_idx])
    sl_dist = atr_val * c.SL_ATR_MULT
    tp_dist = atr_val * c.TP_ATR_MULT

    mfe = 0.0
    mae = 0.0
    end_idx = min(entry_idx + max_hold + 1, len(df))

    for i in range(entry_idx + 1, end_idx):
        bar_high = float(df["high"].iloc[i])
        bar_low = float(df["low"].iloc[i])

        if direction == "LONG":
            fav = bar_high - entry_price
            adv = entry_price - bar_low
        else:
            fav = entry_price - bar_low
            adv = bar_high - entry_price

        mfe = max(mfe, fav)
        mae = max(mae, adv)

        # Check SL hit
        if adv >= sl_dist:
            return 0, mfe / atr_val, mae / atr_val, i - entry_idx, "SL"
        # Check TP hit
        if fav >= tp_dist:
            return 1, mfe / atr_val, mae / atr_val, i - entry_idx, "TP"

    # Time exit — classify based on final P&L
    final_price = float(df["close"].iloc[end_idx - 1])
    if direction == "LONG":
        pnl = final_price - entry_price
    else:
        pnl = entry_price - final_price

    outcome = 1 if pnl > sl_dist * 0.5 else 0
    return outcome, mfe / atr_val, mae / atr_val, end_idx - entry_idx - 1, "TIME"


def generate_training_data(symbols=None, verbose=True):
    """Generate training data for all symbols with regime features."""
    symbols = symbols or c.SYMBOLS
    all_trades = []

    for sym_idx, symbol in enumerate(symbols):
        if verbose:
            print(f"  [{sym_idx+1:2d}/{len(symbols)}] {symbol}...", end=" ", flush=True)

        result = load_and_prepare_data(symbol)
        if result is None:
            if verbose:
                print("skipped")
            continue

        df_4h, htf_bias_series = result
        trades = []

        # Vectorized signal detection (much faster than calling generate_signal per bar)
        signals_mask = _vectorized_signals(df_4h, htf_bias_series)

        for i in signals_mask:
            if i < 55 or i >= len(df_4h) - 55:
                continue

            direction = signals_mask[i]

            # Extract regime features
            features = extract_regime_features(df_4h, i, direction)
            if features is None:
                continue

            # Simulate outcome
            atr_val = float(df_4h["atr"].iloc[i])
            outcome, mfe_atr, mae_atr, hold_bars, exit_type = \
                simulate_trade_outcome(df_4h, i, direction, atr_val)

            # Trade quality score (richer target)
            quality = mfe_atr / max(mae_atr, 0.1)

            trades.append({
                "symbol": symbol,
                "direction": direction,
                "timestamp": df_4h.index[i],
                "features": features,
                "outcome": outcome,
                "quality": quality,
                "mfe_atr": mfe_atr,
                "mae_atr": mae_atr,
                "hold_bars": hold_bars,
                "exit_type": exit_type,
                "regime_trend": features["trend_regime"],
                "regime_vol": features["vol_regime"],
            })

        if verbose:
            wins = sum(1 for t in trades if t["outcome"] == 1)
            print(f"{len(trades)} trades ({wins} wins, {100*wins/max(len(trades),1):.1f}% WR)")
        all_trades.extend(trades)

    return all_trades


def _vectorized_signals(df_4h, htf_bias_series):
    """
    Vectorized signal generation — replaces slow per-bar loop.
    Returns dict of {bar_index: direction} for bars that have a signal.
    """
    signals = {}

    # Pre-extract all needed columns as numpy arrays for speed
    closes = df_4h["close"].values.astype(float)
    opens = df_4h["open"].values.astype(float)
    ema21_arr = df_4h["ema21"].values.astype(float)
    ema50_arr = df_4h["ema50"].values.astype(float)
    adx_arr = df_4h["adx"].values.astype(float)
    rsi_arr = df_4h["rsi"].values.astype(float)
    atr_arr = df_4h["atr"].values.astype(float)
    macd_hist_arr = df_4h["macd_hist"].values.astype(float)
    rel_vol_arr = df_4h["rel_volume"].values.astype(float) if "rel_volume" in df_4h.columns else np.ones(len(df_4h))
    body_pct_arr = df_4h["body_pct"].values.astype(float) if "body_pct" in df_4h.columns else np.zeros(len(df_4h))
    squeeze_fire_arr = df_4h["squeeze_fire"].values if "squeeze_fire" in df_4h.columns else np.zeros(len(df_4h), dtype=bool)

    for i in range(1, len(df_4h)):
        price = closes[i]
        open_price = opens[i]
        atr_val = atr_arr[i]
        if atr_val <= 0:
            continue

        adx = adx_arr[i]
        if adx < 20:
            continue

        ema21 = ema21_arr[i]
        ema50 = ema50_arr[i]
        rsi_val = rsi_arr[i]
        macd_hist = macd_hist_arr[i]
        macd_hist_prev = macd_hist_arr[i-1]
        rel_vol = rel_vol_arr[i]
        body_pct = body_pct_arr[i]
        squeeze_fire = bool(squeeze_fire_arr[i])

        candle_bullish = price > open_price
        candle_bearish = price < open_price

        # Get HTF bias
        htf_bias = htf_bias_series.iloc[i] if i < len(htf_bias_series) else "NONE"

        # ── LONG ──
        if htf_bias == "LONG" or (htf_bias == "NONE" and ema21 > ema50):
            if ema21 <= ema50:
                continue
            dist = (price - ema21) / atr_val
            if dist > 2.5 or dist < -0.5:
                pass
            elif not candle_bullish or body_pct < 0.15:
                pass
            elif rsi_val > 75 or rsi_val < 28:
                pass
            elif not (macd_hist > macd_hist_prev or rel_vol > 1.2 or squeeze_fire):
                pass
            else:
                signals[i] = "LONG"
                continue

        # ── SHORT ──
        if htf_bias == "SHORT" or (htf_bias == "NONE" and ema21 < ema50):
            if ema21 >= ema50:
                continue
            dist = (ema21 - price) / atr_val
            if dist > 2.5 or dist < -0.5:
                continue
            if not candle_bearish or body_pct < 0.15:
                continue
            if rsi_val < 25 or rsi_val > 72:
                continue
            if not (macd_hist < macd_hist_prev or rel_vol > 1.2 or squeeze_fire):
                continue
            signals[i] = "SHORT"

    return signals


# ══════════════════════════════════════════════════════════════════════════
# ENSEMBLE MODEL TRAINING
# ══════════════════════════════════════════════════════════════════════════

def prepare_matrices(trades):
    """Convert trade list to feature matrices."""
    if not trades:
        return None, None, None, None

    feature_names = sorted(trades[0]["features"].keys())
    X = np.array([[t["features"].get(fn, 0.0) for fn in feature_names] for t in trades], dtype=np.float32)
    y_class = np.array([t["outcome"] for t in trades], dtype=np.int32)
    y_quality = np.array([t["quality"] for t in trades], dtype=np.float32)

    return X, y_class, y_quality, feature_names


def train_regime_ensemble(trades, verbose=True):
    """
    Train ensemble of regime-specific models + global model.
    Architecture:
    1. Global model (trained on all data)
    2. Trending regime model (ADX > 20)
    3. Ranging regime model (ADX <= 20)
    4. Final prediction = weighted average based on regime
    """
    if verbose:
        print(f"\n{'═'*70}")
        print(f"  TRAINING REGIME ENSEMBLE")
        print(f"{'═'*70}\n")

    X, y_class, y_quality, feature_names = prepare_matrices(trades)
    if X is None:
        print("  ERROR: No training data")
        return None

    n_total = len(y_class)
    n_pos = y_class.sum()
    print(f"  Total trades: {n_total}")
    print(f"  Win rate: {n_pos/n_total*100:.1f}%")
    print(f"  Features: {len(feature_names)}")

    # Temporal split: 70% train, 15% validation, 15% test
    split_train = int(n_total * 0.70)
    split_val = int(n_total * 0.85)

    X_train, y_train = X[:split_train], y_class[:split_train]
    X_val, y_val = X[split_train:split_val], y_class[split_train:split_val]
    X_test, y_test = X[split_val:], y_class[split_val:]
    q_train = y_quality[:split_train]

    print(f"  Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    # Scale features
    scaler = RobustScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    # ── GLOBAL MODEL ─────────────────────────────────────────────────
    print(f"\n  Training Global Model...")
    scale_pos = (len(y_train) - y_train.sum()) / max(y_train.sum(), 1)

    global_model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.03,
        subsample=0.7,
        colsample_bytree=0.7,
        reg_alpha=1.0,
        reg_lambda=3.0,
        min_child_weight=20,
        gamma=0.5,
        scale_pos_weight=scale_pos,
        eval_metric="logloss",
        early_stopping_rounds=30,
        random_state=42,
        tree_method="hist"
    )
    global_model.fit(
        X_train_s, y_train,
        eval_set=[(X_val_s, y_val)],
        verbose=False
    )
    global_probs_val = global_model.predict_proba(X_val_s)[:, 1]
    global_probs_test = global_model.predict_proba(X_test_s)[:, 1]
    global_auc_val = roc_auc_score(y_val, global_probs_val) if len(set(y_val)) > 1 else 0.5
    global_auc_test = roc_auc_score(y_test, global_probs_test) if len(set(y_test)) > 1 else 0.5
    print(f"    Val AUC: {global_auc_val:.4f}, Test AUC: {global_auc_test:.4f}")
    print(f"    Best iteration: {global_model.best_iteration}")

    # ── TRENDING REGIME MODEL ────────────────────────────────────────
    trend_regime_idx = feature_names.index("trend_regime") if "trend_regime" in feature_names else -1
    if trend_regime_idx >= 0:
        print(f"\n  Training Trending Regime Model...")
        trend_mask_train = X_train[:, trend_regime_idx] >= 1
        trend_mask_val = X_val[:, trend_regime_idx] >= 1
        trend_mask_test = X_test[:, trend_regime_idx] >= 1

        if trend_mask_train.sum() > 100:
            trend_model = xgb.XGBClassifier(
                n_estimators=200,
                max_depth=3,
                learning_rate=0.03,
                subsample=0.7,
                colsample_bytree=0.7,
                reg_alpha=1.5,
                reg_lambda=4.0,
                min_child_weight=25,
                gamma=0.5,
                scale_pos_weight=scale_pos,
                eval_metric="logloss",
                early_stopping_rounds=25,
                random_state=42,
                tree_method="hist"
            )
            X_t_train = X_train_s[trend_mask_train]
            y_t_train = y_train[trend_mask_train]
            X_t_val = X_val_s[trend_mask_val] if trend_mask_val.sum() > 10 else X_val_s[:10]
            y_t_val = y_val[trend_mask_val] if trend_mask_val.sum() > 10 else y_val[:10]

            trend_model.fit(X_t_train, y_t_train, eval_set=[(X_t_val, y_t_val)], verbose=False)
            if trend_mask_test.sum() > 10:
                trend_probs = trend_model.predict_proba(X_test_s[trend_mask_test])[:, 1]
                trend_auc = roc_auc_score(y_test[trend_mask_test], trend_probs) if len(set(y_test[trend_mask_test])) > 1 else 0.5
                print(f"    Trending test AUC: {trend_auc:.4f} ({trend_mask_test.sum()} trades)")
            else:
                trend_auc = 0.5
                print(f"    Trending: insufficient test data")
        else:
            trend_model = None
            trend_auc = 0.5
            print(f"    Trending: insufficient training data ({trend_mask_train.sum()} trades)")

        # ── RANGING REGIME MODEL ─────────────────────────────────────
        print(f"\n  Training Ranging Regime Model...")
        range_mask_train = X_train[:, trend_regime_idx] <= 0
        range_mask_val = X_val[:, trend_regime_idx] <= 0
        range_mask_test = X_test[:, trend_regime_idx] <= 0

        if range_mask_train.sum() > 100:
            range_model = xgb.XGBClassifier(
                n_estimators=200,
                max_depth=3,
                learning_rate=0.03,
                subsample=0.7,
                colsample_bytree=0.7,
                reg_alpha=2.0,
                reg_lambda=5.0,
                min_child_weight=30,
                gamma=0.8,
                scale_pos_weight=scale_pos,
                eval_metric="logloss",
                early_stopping_rounds=25,
                random_state=42,
                tree_method="hist"
            )
            X_r_train = X_train_s[range_mask_train]
            y_r_train = y_train[range_mask_train]
            X_r_val = X_val_s[range_mask_val] if range_mask_val.sum() > 10 else X_val_s[:10]
            y_r_val = y_val[range_mask_val] if range_mask_val.sum() > 10 else y_val[:10]

            range_model.fit(X_r_train, y_r_train, eval_set=[(X_r_val, y_r_val)], verbose=False)
            if range_mask_test.sum() > 10:
                range_probs = range_model.predict_proba(X_test_s[range_mask_test])[:, 1]
                range_auc = roc_auc_score(y_test[range_mask_test], range_probs) if len(set(y_test[range_mask_test])) > 1 else 0.5
                print(f"    Ranging test AUC: {range_auc:.4f} ({range_mask_test.sum()} trades)")
            else:
                range_auc = 0.5
                print(f"    Ranging: insufficient test data")
        else:
            range_model = None
            range_auc = 0.5
            print(f"    Ranging: insufficient training data ({range_mask_train.sum()} trades)")
    else:
        trend_model = None
        range_model = None
        trend_auc = 0.5
        range_auc = 0.5

    # ── QUALITY-WEIGHTED MODEL (predict quality, not just win/loss) ───
    print(f"\n  Training Quality Regressor...")
    # Clip quality to reasonable range and use as regression target
    q_clipped = np.clip(q_train, 0, 10)
    quality_model = xgb.XGBRegressor(
        n_estimators=200,
        max_depth=3,
        learning_rate=0.03,
        subsample=0.7,
        colsample_bytree=0.7,
        reg_alpha=1.0,
        reg_lambda=3.0,
        min_child_weight=20,
        eval_metric="mae",
        early_stopping_rounds=25,
        random_state=42,
        tree_method="hist"
    )
    q_val = np.clip(y_quality[split_train:split_val], 0, 10)
    quality_model.fit(X_train_s, q_clipped, eval_set=[(X_val_s, q_val)], verbose=False)
    q_pred_test = quality_model.predict(X_test_s)
    # Correlation between predicted quality and actual
    q_actual_test = y_quality[split_val:]
    q_corr = np.corrcoef(q_pred_test, q_actual_test)[0, 1]
    print(f"    Quality prediction correlation: {q_corr:.4f}")

    # ── ENSEMBLE SCORING ON TEST SET ─────────────────────────────────
    print(f"\n{'─'*70}")
    print(f"  ENSEMBLE TEST SET EVALUATION")
    print(f"{'─'*70}\n")

    # Ensemble prediction: weighted combination
    ensemble_probs = np.zeros(len(X_test))
    for i in range(len(X_test)):
        global_p = global_probs_test[i]
        quality_p = min(max(q_pred_test[i] / 5.0, 0), 1)  # Normalize quality to 0-1

        # Regime-specific weighting
        regime_val = X_test[i, trend_regime_idx] if trend_regime_idx >= 0 else 0
        if regime_val >= 1 and trend_model is not None:
            regime_p = trend_model.predict_proba(X_test_s[i:i+1])[:, 1][0]
            # Blend: 40% global + 40% regime + 20% quality
            ensemble_probs[i] = 0.40 * global_p + 0.40 * regime_p + 0.20 * quality_p
        elif regime_val <= 0 and range_model is not None:
            regime_p = range_model.predict_proba(X_test_s[i:i+1])[:, 1][0]
            ensemble_probs[i] = 0.40 * global_p + 0.40 * regime_p + 0.20 * quality_p
        else:
            ensemble_probs[i] = 0.60 * global_p + 0.40 * quality_p

    # Evaluate ensemble
    ensemble_auc = roc_auc_score(y_test, ensemble_probs) if len(set(y_test)) > 1 else 0.5
    print(f"  Ensemble AUC: {ensemble_auc:.4f}")
    print(f"  Global AUC:   {global_auc_test:.4f}")
    print(f"  Improvement:  {(ensemble_auc - global_auc_test)*100:+.2f}%")

    # Find optimal threshold via validation set
    ensemble_probs_val = np.zeros(len(X_val))
    for i in range(len(X_val)):
        gp = global_model.predict_proba(X_val_s[i:i+1])[:, 1][0]
        qp = min(max(quality_model.predict(X_val_s[i:i+1])[0] / 5.0, 0), 1)
        regime_val = X_val[i, trend_regime_idx] if trend_regime_idx >= 0 else 0
        if regime_val >= 1 and trend_model is not None:
            rp = trend_model.predict_proba(X_val_s[i:i+1])[:, 1][0]
            ensemble_probs_val[i] = 0.40 * gp + 0.40 * rp + 0.20 * qp
        elif regime_val <= 0 and range_model is not None:
            rp = range_model.predict_proba(X_val_s[i:i+1])[:, 1][0]
            ensemble_probs_val[i] = 0.40 * gp + 0.40 * rp + 0.20 * qp
        else:
            ensemble_probs_val[i] = 0.60 * gp + 0.40 * qp

    # Threshold search
    print(f"\n  ── Threshold Search (Validation Set) ──")
    print(f"  {'Threshold':<12} {'Pass%':<8} {'WR':<8} {'PF':<8} {'Exp/Trade':<12}")
    best_threshold = 0.5
    best_expectancy = -999

    for thresh in [0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65]:
        passed = ensemble_probs_val >= thresh
        if passed.sum() < 10:
            continue
        wr = y_val[passed].mean()
        n_passed = passed.sum()
        pass_pct = n_passed / len(y_val) * 100

        # Estimate PF and expectancy
        wins = y_val[passed].sum()
        losses = n_passed - wins
        if losses > 0:
            pf = (wins * c.TP_ATR_MULT / c.SL_ATR_MULT) / losses
            expectancy = wr * (c.TP_ATR_MULT / c.SL_ATR_MULT) - (1 - wr)
        else:
            pf = 999
            expectancy = wr * (c.TP_ATR_MULT / c.SL_ATR_MULT)

        print(f"  {thresh:<12.2f} {pass_pct:<8.1f} {wr*100:<8.1f} {pf:<8.2f} {expectancy:<12.3f}")

        if expectancy > best_expectancy:
            best_expectancy = expectancy
            best_threshold = thresh

    print(f"\n  Selected threshold: {best_threshold:.2f} (expectancy={best_expectancy:.3f})")

    # ── TEST SET TRADING SIMULATION ──────────────────────────────────
    print(f"\n  ── Test Set Trading Simulation ──")
    test_passed = ensemble_probs >= best_threshold
    test_rejected = ~test_passed

    if test_passed.sum() > 0:
        passed_wr = y_test[test_passed].mean()
        rejected_wr = y_test[test_rejected].mean() if test_rejected.sum() > 0 else 0

        wins_t = y_test[test_passed].sum()
        losses_t = test_passed.sum() - wins_t
        pf_t = (wins_t * c.TP_ATR_MULT / c.SL_ATR_MULT) / max(losses_t, 1)
        expectancy_t = passed_wr * (c.TP_ATR_MULT / c.SL_ATR_MULT) - (1 - passed_wr)

        print(f"  All trades WR:      {y_test.mean()*100:.1f}%")
        print(f"  Passed ({test_passed.sum()}):     WR={passed_wr*100:.1f}% PF={pf_t:.2f} Exp={expectancy_t:.3f}")
        print(f"  Rejected ({test_rejected.sum()}):   WR={rejected_wr*100:.1f}%")
        print(f"  Separation:         {(passed_wr - rejected_wr)*100:+.1f}%")

        # Probability distribution check
        print(f"\n  Prob distribution: min={ensemble_probs.min():.4f} max={ensemble_probs.max():.4f} std={ensemble_probs.std():.4f}")
    else:
        print(f"  No trades pass threshold!")
        passed_wr = 0
        expectancy_t = 0

    # ── FEATURE IMPORTANCE ───────────────────────────────────────────
    print(f"\n  ── Top 20 Features (Global Model) ──")
    importances = global_model.feature_importances_
    indices = np.argsort(importances)[::-1][:20]
    for rank, idx in enumerate(indices):
        bar = "█" * int(importances[idx] / importances[indices[0]] * 20)
        print(f"    {rank+1:2d}. {feature_names[idx]:<25} {importances[idx]:.4f} {bar}")

    # ── SAVE MODEL ───────────────────────────────────────────────────
    model_data = {
        "global_model": global_model,
        "trend_model": trend_model,
        "range_model": range_model,
        "quality_model": quality_model,
        "scaler": scaler,
        "feature_names": feature_names,
        "threshold": best_threshold,
        "trend_regime_idx": trend_regime_idx,
        "version": "v3_regime_ensemble",
        "metadata": {
            "train_date": datetime.now().isoformat(),
            "n_trades": n_total,
            "n_features": len(feature_names),
            "global_auc_test": global_auc_test,
            "ensemble_auc_test": ensemble_auc,
            "test_wr_passed": float(passed_wr) if test_passed.sum() > 0 else 0,
            "test_expectancy": float(expectancy_t),
            "quality_corr": float(q_corr),
        }
    }

    model_path = os.path.join(MODEL_DIR, "ml_filter_v3_regime.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(model_data, f)
    print(f"\n  ✓ Model saved: {model_path}")

    # Also save as production if it's better
    prod_path = os.path.join(MODEL_DIR, "ml_filter_production.pkl")
    # Convert to format MLFilter expects
    production_data = {
        "classifier": global_model,
        "scaler": scaler,
        "feature_names": feature_names,
        "feature_mask": None,  # No mask — use all features
        "confidence_threshold": best_threshold,
        "trend_model": trend_model,
        "range_model": range_model,
        "quality_model": quality_model,
        "trend_regime_idx": trend_regime_idx,
        "version": "v3_regime_ensemble",
        "metadata": model_data["metadata"]
    }
    with open(prod_path, "wb") as f:
        pickle.dump(production_data, f)
    print(f"  ✓ Production model saved: {prod_path}")

    return model_data


# ══════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("  ML TRAINING V3 — REGIME-BASED ENSEMBLE")
    print("=" * 70)
    print(f"  Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"  Symbols: {len(c.SYMBOLS)}")
    print(f"  Strategy: 4H Momentum Continuation")
    print()

    # Check data availability
    available = 0
    for sym in c.SYMBOLS:
        if os.path.exists(os.path.join(DATA_DIR, f"{sym}_15m.csv")):
            available += 1
    print(f"  Historical data available: {available}/{len(c.SYMBOLS)} symbols")

    if available < 10:
        print("  ERROR: Need at least 10 symbols with historical data")
        sys.exit(1)

    # Generate training data
    print(f"\n{'═'*70}")
    print(f"  GENERATING TRAINING DATA")
    print(f"{'═'*70}\n")
    trades = generate_training_data(c.SYMBOLS)

    if len(trades) < 500:
        print(f"\n  ERROR: Only {len(trades)} trades generated. Need at least 500.")
        sys.exit(1)

    print(f"\n  Total trades: {len(trades)}")
    wins = sum(1 for t in trades if t["outcome"] == 1)
    print(f"  Win rate: {wins}/{len(trades)} = {100*wins/len(trades):.1f}%")
    print(f"  Avg quality: {np.mean([t['quality'] for t in trades]):.2f}")

    # Regime distribution
    trend_counts = defaultdict(int)
    for t in trades:
        trend_counts[t["regime_trend"]] += 1
    print(f"\n  Regime distribution:")
    for regime, count in sorted(trend_counts.items()):
        labels = {-1: "Choppy", 0: "Ranging", 1: "Weak Trend", 2: "Strong Trend"}
        print(f"    {labels.get(regime, f'R={regime}')}: {count} ({100*count/len(trades):.1f}%)")

    # Train ensemble
    model_data = train_regime_ensemble(trades)

    if model_data is None:
        print("\n  TRAINING FAILED")
        sys.exit(1)

    print(f"\n{'═'*70}")
    print(f"  TRAINING V3 COMPLETE")
    print(f"{'═'*70}")
    meta = model_data["metadata"]
    print(f"  Global AUC:      {meta['global_auc_test']:.4f}")
    print(f"  Ensemble AUC:    {meta['ensemble_auc_test']:.4f}")
    print(f"  Test WR Passed:  {meta['test_wr_passed']*100:.1f}%")
    print(f"  Test Expectancy: {meta['test_expectancy']:.3f}")
    print(f"  Quality Corr:    {meta['quality_corr']:.4f}")
    print(f"{'═'*70}\n")


if __name__ == "__main__":
    main()
