"""
ML Signal Filter + Dynamic TP/SL Optimizer
===========================================
Two-model approach:
1. CLASSIFIER: Should we take this trade? (win probability)
2. REGRESSOR: What's the optimal SL (ATR mult) and TP (R-multiple)?

Uses MFE/MAE (Maximum Favorable/Adverse Excursion) analysis
to learn what TP/SL values maximize expectancy for each setup.

Features: 50+ market-state indicators for robust generalization.
"""

import numpy as np
import pickle
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
    HAS_SKLEARN_GB = True
except ImportError:
    HAS_SKLEARN_GB = False


# ══════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════════════════

MIN_TRAINING_SAMPLES = 50
RETRAIN_INTERVAL = 15
CONFIDENCE_THRESHOLD = 0.15

# SL/TP bounds (in ATR multiples for SL, R-multiples for TP)
MIN_SL_ATR = 1.0
MAX_SL_ATR = 4.0
MIN_TP_R = 1.5
MAX_TP_R = 6.0


# ══════════════════════════════════════════════════════════════════════════
# EXPANDED FEATURE EXTRACTION (50+ features)
# ══════════════════════════════════════════════════════════════════════════

def extract_features(curr, prev, price, atr_val, direction):
    """
    Extract comprehensive feature vector from current market state.
    50+ features covering: trend, momentum, volatility, volume, structure.
    All features are normalized/scale-independent.
    """
    if atr_val <= 0:
        atr_val = 0.0001  # Safety

    dir_sign = 1.0 if direction == "LONG" else -1.0

    # ── Core Price Data ────────────────────────────────────────────────
    ema9 = float(curr["ema9"])
    ema21 = float(curr["ema21"])
    ema50 = float(curr["ema50"])
    prev_ema21 = float(prev["ema21"])
    prev_ema50 = float(prev["ema50"])

    # ── Trend Indicators ───────────────────────────────────────────────
    adx = float(curr["adx"])
    di_plus = float(curr["di_plus"])
    di_minus = float(curr["di_minus"])
    prev_adx = float(prev["adx"]) if "adx" in prev.index else adx
    prev_di_plus = float(prev["di_plus"]) if "di_plus" in prev.index else di_plus
    prev_di_minus = float(prev["di_minus"]) if "di_minus" in prev.index else di_minus

    # ── Momentum ──────────────────────────────────────────────────────
    rsi = float(curr["rsi"])
    prev_rsi = float(prev["rsi"]) if "rsi" in prev.index else rsi
    macd_val = float(curr["macd"])
    macd_hist = float(curr["macd_hist"])
    prev_macd_hist = float(prev["macd_hist"])
    macd_sig = float(curr["macd_sig"]) if "macd_sig" in curr.index else 0

    # ── Volatility ────────────────────────────────────────────────────
    bb_pct_b = float(curr["bb_pct_b"]) if "bb_pct_b" in curr.index else 0.5
    bb_width = float(curr["bb_width"]) if "bb_width" in curr.index else 0
    prev_bb_width = float(prev["bb_width"]) if "bb_width" in prev.index else bb_width

    # ── Volume ────────────────────────────────────────────────────────
    rel_vol = float(curr["rel_volume"]) if "rel_volume" in curr.index else 1.0
    cmf = float(curr["cmf"]) if "cmf" in curr.index else 0

    # ── VWAP ──────────────────────────────────────────────────────────
    vwap = float(curr["vwap"]) if "vwap" in curr.index else price

    # ── Structure ─────────────────────────────────────────────────────
    swing_high = float(curr["swing_high"]) if "swing_high" in curr.index else price
    swing_low = float(curr["swing_low"]) if "swing_low" in curr.index else price
    body_pct = float(curr["body_pct"]) if curr.get("body_pct", float('nan')) == curr.get("body_pct", float('nan')) else 0

    # ── Stochastic RSI ────────────────────────────────────────────────
    stoch_k = float(curr["stoch_k"]) if "stoch_k" in curr.index else 50
    stoch_d = float(curr["stoch_d"]) if "stoch_d" in curr.index else 50

    # ── Squeeze ──────────────────────────────────────────────────────
    squeeze_on = 1.0 if curr.get("squeeze_on", False) else 0.0
    squeeze_mom = float(curr["squeeze_mom"]) if "squeeze_mom" in curr.index else 0

    # ── Supertrend ───────────────────────────────────────────────────
    st_direction = float(curr["st_direction"]) if "st_direction" in curr.index else 0

    # ══════════════════════════════════════════════════════════════════
    # BUILD FEATURE VECTOR (50+ features)
    # ══════════════════════════════════════════════════════════════════

    features = {}

    # ── TREND (12 features) ─────────────────────────────────────────
    features["adx"] = adx
    features["adx_change"] = adx - prev_adx
    features["di_spread"] = (di_plus - di_minus) * dir_sign
    features["di_spread_change"] = ((di_plus - di_minus) - (prev_di_plus - prev_di_minus)) * dir_sign
    features["ema9_21_dist"] = (ema9 - ema21) / atr_val * dir_sign
    features["ema21_50_dist"] = (ema21 - ema50) / atr_val * dir_sign
    features["price_ema9_dist"] = (price - ema9) / atr_val * dir_sign
    features["price_ema21_dist"] = (price - ema21) / atr_val * dir_sign
    features["price_ema50_dist"] = (price - ema50) / atr_val * dir_sign
    features["ema21_slope"] = (ema21 - prev_ema21) / atr_val * dir_sign
    features["ema50_slope"] = (ema50 - prev_ema50) / atr_val * dir_sign
    features["supertrend_align"] = st_direction * dir_sign

    # ── MOMENTUM (12 features) ──────────────────────────────────────
    features["rsi"] = rsi
    features["rsi_change"] = rsi - prev_rsi
    features["rsi_dist_50"] = (rsi - 50) * dir_sign  # Distance from neutral
    features["macd_norm"] = macd_val / atr_val * dir_sign
    features["macd_hist_norm"] = macd_hist / atr_val * dir_sign
    features["macd_accel"] = (macd_hist - prev_macd_hist) / atr_val * dir_sign
    features["macd_hist_sign"] = 1.0 if (macd_hist * dir_sign > 0) else -1.0
    features["stoch_k"] = stoch_k
    features["stoch_d"] = stoch_d
    features["stoch_cross"] = (stoch_k - stoch_d) * dir_sign
    features["squeeze_on"] = squeeze_on
    features["squeeze_mom_dir"] = squeeze_mom / atr_val * dir_sign if atr_val > 0 else 0

    # ── VOLATILITY (8 features) ─────────────────────────────────────
    features["bb_pct_b"] = bb_pct_b
    features["bb_width"] = bb_width
    features["bb_width_change"] = bb_width - prev_bb_width
    features["bb_position"] = (bb_pct_b - 0.5) * 2 * dir_sign  # -1 to 1, positive = favorable
    features["atr_normalized"] = atr_val / price * 1000  # ATR as % of price (scaled)
    features["price_range_ratio"] = (float(curr["high"]) - float(curr["low"])) / atr_val
    features["body_pct"] = body_pct
    features["candle_direction"] = 1.0 if float(curr["close"]) > float(curr["open"]) else -1.0

    # ── VOLUME (6 features) ──────────────────────────────────────────
    features["rel_volume"] = rel_vol
    features["cmf"] = cmf * dir_sign
    features["vol_above_avg"] = 1.0 if rel_vol > 1.0 else 0.0
    features["vol_spike"] = 1.0 if rel_vol > 2.0 else 0.0
    features["price_vwap_dist"] = (price - vwap) / atr_val * dir_sign
    features["cmf_strong"] = 1.0 if (cmf * dir_sign > 0.1) else 0.0

    # ── STRUCTURE (8 features) ───────────────────────────────────────
    features["dist_swing_high"] = (swing_high - price) / atr_val
    features["dist_swing_low"] = (price - swing_low) / atr_val
    features["swing_range"] = (swing_high - swing_low) / atr_val
    features["price_in_range"] = (price - swing_low) / (swing_high - swing_low) if (swing_high - swing_low) > 0 else 0.5
    # Room to target (how much space in favorable direction)
    if direction == "LONG":
        features["room_to_target"] = (swing_high - price) / atr_val
        features["room_to_stop"] = (price - swing_low) / atr_val
    else:
        features["room_to_target"] = (price - swing_low) / atr_val
        features["room_to_stop"] = (swing_high - price) / atr_val
    features["structure_rr"] = features["room_to_target"] / max(features["room_to_stop"], 0.1)
    features["pullback_depth"] = features["price_ema21_dist"]  # How far from EMA21

    # ── REGIME (4 features) ──────────────────────────────────────────
    features["trend_strength"] = adx * abs(di_plus - di_minus) / 100  # Combined
    features["momentum_regime"] = 1.0 if (macd_hist * dir_sign > 0 and rsi * dir_sign > 50 * dir_sign) else 0.0
    features["vol_regime"] = min(rel_vol, 3.0) / 3.0  # Capped normalized
    features["squeeze_fire"] = 1.0 if (curr.get("squeeze_fire", False)) else 0.0
    features["direction_long"] = 1.0 if direction == "LONG" else 0.0

    return features


def extract_features_extended(curr, prev, price, atr_val, direction, df_slice=None):
    """
    Extended features that include lookback context.
    Uses recent history for pattern detection.
    """
    features = extract_features(curr, prev, price, atr_val, direction)

    if df_slice is not None and len(df_slice) >= 10:
        dir_sign = 1.0 if direction == "LONG" else -1.0

        # Last 5 candles RSI trend
        rsi_5 = df_slice["rsi"].iloc[-5:].values
        features["rsi_slope_5"] = (rsi_5[-1] - rsi_5[0]) / 5.0

        # ATR expansion/contraction (current vs 10-bar avg)
        atr_10 = df_slice["atr"].iloc[-10:].mean()
        features["atr_expansion"] = float(curr["atr"]) / atr_10 if atr_10 > 0 else 1.0

        # Consecutive candles in direction
        closes = df_slice["close"].iloc[-5:].values
        if direction == "LONG":
            consec = sum(1 for i in range(1, len(closes)) if closes[i] > closes[i-1])
        else:
            consec = sum(1 for i in range(1, len(closes)) if closes[i] < closes[i-1])
        features["consecutive_direction"] = consec / 4.0

        # Volatility percentile (where is current ATR vs last 50)
        if len(df_slice) >= 50:
            atr_50 = df_slice["atr"].iloc[-50:].values
            features["atr_percentile"] = np.searchsorted(np.sort(atr_50), float(curr["atr"])) / 50.0
        else:
            features["atr_percentile"] = 0.5

        # Volume trend (5-bar)
        vol_5 = df_slice["rel_volume"].iloc[-5:].values
        features["vol_trend"] = (vol_5[-1] - vol_5[0]) / max(vol_5.mean(), 0.1)

        # MACD histogram trend (accelerating or decelerating)
        hist_5 = df_slice["macd_hist"].iloc[-5:].values
        features["macd_hist_trend"] = (hist_5[-1] - hist_5[0]) / atr_val * dir_sign

    return features


# ======================================================================
# REGIME DETECTION (V3)
# ======================================================================

class RegimeDetector:
    """Classifies market into regimes. All methods are vectorized."""

    @staticmethod
    def detect_volatility_regime(df, lookback=50):
        atr_vals = df["atr"].values.astype(float)
        regime = np.zeros(len(df))
        atr_series = pd.Series(atr_vals)
        rolling_rank = atr_series.rolling(lookback).apply(
            lambda x: pd.Series(x).rank().iloc[-1] / len(x), raw=False
        )
        pct_rank = rolling_rank.values
        regime[pct_rank > 0.75] = 1
        regime[pct_rank < 0.25] = -1
        return regime

    @staticmethod
    def detect_trend_regime(df, lookback=50):
        adx_vals = df["adx"].values.astype(float)
        ema9 = df["ema9"].values.astype(float)
        ema21 = df["ema21"].values.astype(float)
        ema50 = df["ema50"].values.astype(float)
        full_bull = (ema9 > ema21) & (ema21 > ema50)
        full_bear = (ema9 < ema21) & (ema21 < ema50)
        fully_aligned = full_bull | full_bear
        regime = np.zeros(len(df))
        regime[(adx_vals > 30) & fully_aligned] = 2
        regime[(adx_vals > 20) & (adx_vals <= 30) & fully_aligned] = 1
        regime[(adx_vals < 15)] = -1
        return regime

    @staticmethod
    def detect_momentum_regime(df, lookback=20):
        hist = df["macd_hist"].values.astype(float)
        atr_vals = df["atr"].values.astype(float)
        regime = np.zeros(len(df))
        hist_series = pd.Series(hist)
        slope = (hist_series - hist_series.shift(4)) / 4.0
        slope_norm = np.where(atr_vals > 0, slope.values / atr_vals, 0)
        regime[slope_norm > 0.1] = 2
        regime[(slope_norm > 0.02) & (slope_norm <= 0.1)] = 1
        regime[slope_norm < -0.1] = -2
        regime[(slope_norm < -0.02) & (slope_norm >= -0.1)] = -1
        return regime

    @staticmethod
    def detect_volume_regime(df, lookback=20):
        if "rel_volume" not in df.columns:
            return np.zeros(len(df))
        vol = df["rel_volume"].values.astype(float)
        vol_series = pd.Series(vol)
        avg_vol = vol_series.rolling(lookback).mean().values
        regime = np.zeros(len(df))
        regime[avg_vol > 1.5] = 2
        regime[(avg_vol > 1.0) & (avg_vol <= 1.5)] = 1
        regime[avg_vol < 0.6] = -1
        return regime


def compute_regimes(df):
    """Add regime columns to a DataFrame. Returns modified df."""
    df["vol_regime"] = RegimeDetector.detect_volatility_regime(df)
    df["trend_regime"] = RegimeDetector.detect_trend_regime(df)
    df["mom_regime"] = RegimeDetector.detect_momentum_regime(df)
    df["volume_regime"] = RegimeDetector.detect_volume_regime(df)
    return df



# ══════════════════════════════════════════════════════════════════════════
# REGIME-AWARE FEATURE EXTRACTION (V3 — 60+ contextual features)
# ══════════════════════════════════════════════════════════════════════════

def extract_regime_features(df, idx, direction):
    """
    Extract regime-aware features with deep lookback (50 bars context).
    These features capture MARKET CONTEXT rather than single-bar state.
    Used by V3 regime ensemble model.
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
    high_50 = df["high"].iloc[idx-50:idx+1].max()
    low_50 = df["low"].iloc[idx-50:idx+1].min()
    range_50 = high_50 - low_50
    features["price_in_50bar_range"] = (price - low_50) / range_50 if range_50 > 0 else 0.5

    high_20 = df["high"].iloc[idx-20:idx+1].max()
    low_20 = df["low"].iloc[idx-20:idx+1].min()
    range_20 = high_20 - low_20
    features["price_in_20bar_range"] = (price - low_20) / range_20 if range_20 > 0 else 0.5

    highs = df["high"].iloc[idx-20:idx+1].values
    lows = df["low"].iloc[idx-20:idx+1].values
    hh_count = sum(1 for i in range(1, len(highs)) if highs[i] > highs[i-1])
    ll_count = sum(1 for i in range(1, len(lows)) if lows[i] < lows[i-1])
    features["hh_count_20"] = hh_count / 20.0 * dir_sign
    features["ll_count_20"] = ll_count / 20.0 * dir_sign
    features["structure_score"] = (hh_count - ll_count) / 20.0 * dir_sign

    ema9 = float(curr["ema9"])
    ema21 = float(curr["ema21"])
    ema50 = float(curr["ema50"])
    features["ema_fan_width"] = abs(ema9 - ema50) / atr_val
    features["ema_alignment"] = ((ema9 - ema21) + (ema21 - ema50)) / atr_val * dir_sign

    ema9_vals = df["ema9"].iloc[idx-50:idx+1].values
    ema21_vals = df["ema21"].iloc[idx-50:idx+1].values
    trend_age = 0
    for j in range(len(ema9_vals)-1, 0, -1):
        if (ema9_vals[j] > ema21_vals[j]) != (ema9_vals[j-1] > ema21_vals[j-1]):
            break
        trend_age += 1
    features["trend_age"] = min(trend_age / 50.0, 1.0)

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

    if "bb_width" in df.columns:
        bb_widths = df["bb_width"].iloc[idx-20:idx+1].values
        features["bb_squeeze_depth"] = bb_widths[-1] / np.mean(bb_widths) if np.mean(bb_widths) > 0 else 1.0
        features["bb_width_trend"] = (bb_widths[-1] - bb_widths[-5]) / np.mean(bb_widths) if np.mean(bb_widths) > 0 else 0
    else:
        features["bb_squeeze_depth"] = 1.0
        features["bb_width_trend"] = 0.0

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

    hist_vals = df["macd_hist"].iloc[idx-20:idx+1].values
    hist_norm = hist_vals / atr_val
    features["macd_hist_mean_20"] = np.mean(hist_norm) * dir_sign
    features["macd_hist_std_20"] = np.std(hist_norm)
    features["macd_hist_positive_pct"] = np.mean(hist_norm * dir_sign > 0)

    if direction == "LONG":
        price_new_high = closes[-1] > np.max(closes[-10:-1])
        macd_declining = hist_vals[-1] < np.max(hist_vals[-10:-1])
        features["divergence"] = 1.0 if (price_new_high and macd_declining) else 0.0
    else:
        price_new_low = closes[-1] < np.min(closes[-10:-1])
        macd_rising = hist_vals[-1] > np.min(hist_vals[-10:-1])
        features["divergence"] = 1.0 if (price_new_low and macd_rising) else 0.0

    consec_pos = 0
    for j in range(len(hist_norm)-1, -1, -1):
        if hist_norm[j] * dir_sign > 0:
            consec_pos += 1
        else:
            break
    features["momentum_persistence"] = min(consec_pos / 10.0, 1.0)

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

    if "cmf" in df.columns:
        cmf_vals = df["cmf"].iloc[idx-10:idx+1].values
        features["cmf_mean_10"] = np.mean(cmf_vals) * dir_sign
        features["cmf_trend"] = (cmf_vals[-1] - cmf_vals[0]) * dir_sign
        features["cmf_positive_pct"] = np.mean(cmf_vals * dir_sign > 0)
    else:
        features["cmf_mean_10"] = 0.0
        features["cmf_trend"] = 0.0
        features["cmf_positive_pct"] = 0.5

    # ── MEAN REVERSION vs MOMENTUM (4 features) ─────────────────────
    dist_ema21 = (price - ema21) / atr_val * dir_sign
    dist_ema50 = (price - ema50) / atr_val * dir_sign
    features["mean_rev_score"] = -abs(dist_ema21) / 3.0
    features["momentum_score"] = dist_ema21

    typical_range = np.mean(np.abs(np.diff(closes[-20:])))
    current_move = abs(closes[-1] - closes[-5])
    features["extension_ratio"] = current_move / (typical_range * 5) if typical_range > 0 else 1.0

    if direction == "LONG":
        recent_high = np.max(closes[-10:])
        pullback = (recent_high - price) / atr_val
    else:
        recent_low = np.min(closes[-10:])
        pullback = (price - recent_low) / atr_val
    features["pullback_depth"] = pullback

    # ── PRICE ACTION PATTERNS (8 features) ───────────────────────────
    opens = df["open"].iloc[idx-5:idx+1].values
    closes_5 = df["close"].iloc[idx-5:idx+1].values
    highs_5 = df["high"].iloc[idx-5:idx+1].values
    lows_5 = df["low"].iloc[idx-5:idx+1].values

    bodies = np.abs(closes_5 - opens)
    ranges = highs_5 - lows_5
    body_ratios = bodies / np.maximum(ranges, 0.0001)
    features["avg_body_ratio_5"] = np.mean(body_ratios)

    bullish = np.sum(closes_5 > opens)
    features["bullish_candle_pct"] = bullish / len(closes_5) * dir_sign + (1 - dir_sign) * (1 - bullish / len(closes_5))

    if idx >= 6:
        opens_prev = df["open"].iloc[idx-4:idx+1].values
        closes_prev = df["close"].iloc[idx-5:idx].values
        gaps = np.abs(opens_prev - closes_prev)
        features["avg_gap_size"] = np.mean(gaps) / atr_val
    else:
        features["avg_gap_size"] = 0.0

    high_10 = df["high"].iloc[idx-10:idx+1].max()
    low_10 = df["low"].iloc[idx-10:idx+1].min()
    if direction == "LONG":
        features["dist_to_resistance"] = (high_10 - price) / atr_val
        features["dist_to_support"] = (price - low_10) / atr_val
    else:
        features["dist_to_resistance"] = (price - low_10) / atr_val
        features["dist_to_support"] = (high_10 - price) / atr_val

    features["inside_bar"] = 1.0 if (highs_5[-1] < highs_5[-2] and lows_5[-1] > lows_5[-2]) else 0.0
    prev_range = highs_5[-2] - lows_5[-2]
    curr_range = highs_5[-1] - lows_5[-1]
    features["range_expansion"] = curr_range / prev_range if prev_range > 0 else 1.0

    # ── CROSS-TIMEFRAME (4 features) ────────────────────────────────
    close_20d_ago = closes[-min(120, len(closes))] if len(closes) > 120 else closes[0]
    features["trend_20d"] = (price - close_20d_ago) / (atr_val * 20) * dir_sign

    mom_50 = (closes[-1] - closes[0]) / (atr_val * 50)
    mom_10 = (closes[-1] - closes[-10]) / (atr_val * 10)
    features["momentum_accel"] = (mom_10 - mom_50) * dir_sign
    features["multi_tf_agree"] = 1.0 if (mom_10 * dir_sign > 0 and mom_50 * dir_sign > 0) else 0.0
    features["tf_conflict"] = 1.0 if (mom_10 * dir_sign > 0) != (mom_50 * dir_sign > 0) else 0.0
    features["direction_long"] = 1.0 if direction == "LONG" else 0.0

    return features


# ══════════════════════════════════════════════════════════════════════════
# MFE/MAE CALCULATION (for TP/SL target generation)
# ══════════════════════════════════════════════════════════════════════════

def calculate_mfe_mae(df, entry_idx, direction, entry_price, atr_val, max_bars=30):
    """
    Calculate Maximum Favorable Excursion (MFE) and Maximum Adverse Excursion (MAE)
    in ATR units for a given entry.

    MFE = max profit reached before exit (in ATR)
    MAE = max drawdown before recovery or exit (in ATR)

    Returns: (mfe_atr, mae_atr) or None if insufficient data
    """
    if entry_idx + 2 >= len(df):
        return None

    end_idx = min(entry_idx + max_bars + 1, len(df))
    mfe = 0.0
    mae = 0.0

    for j in range(entry_idx + 1, end_idx):
        row = df.iloc[j]
        high = float(row["high"])
        low = float(row["low"])

        if direction == "LONG":
            favorable = (high - entry_price) / atr_val
            adverse = (entry_price - low) / atr_val
        else:
            favorable = (entry_price - low) / atr_val
            adverse = (high - entry_price) / atr_val

        mfe = max(mfe, favorable)
        mae = max(mae, adverse)

    return mfe, mae


def calculate_optimal_exit(df, entry_idx, direction, entry_price, atr_val, max_bars=30):
    """
    Fast optimal SL/TP using MFE/MAE heuristic.
    Instead of brute-force grid, uses the actual price excursions to derive optimal levels.
    Returns: (optimal_sl_atr, optimal_tp_r, actual_pnl_r)
    """
    if entry_idx + 2 >= len(df):
        return None

    end_idx = min(entry_idx + max_bars + 1, len(df))

    # Get highs and lows as numpy arrays for speed
    future_slice = df.iloc[entry_idx + 1:end_idx]
    if future_slice.empty:
        return None

    highs = future_slice["high"].values.astype(float)
    lows = future_slice["low"].values.astype(float)

    # Calculate running MFE and MAE bar by bar
    if direction == "LONG":
        favorable = (highs - entry_price) / atr_val  # How far price went up
        adverse = (entry_price - lows) / atr_val     # How far price went down
    else:
        favorable = (entry_price - lows) / atr_val
        adverse = (highs - entry_price) / atr_val

    max_favorable = np.maximum.accumulate(favorable)  # Running MFE
    max_adverse = np.maximum.accumulate(adverse)      # Running MAE

    total_mfe = max_favorable[-1] if len(max_favorable) > 0 else 0
    total_mae = max_adverse[-1] if len(max_adverse) > 0 else 0

    # Derive optimal SL: slightly beyond the MAE that was recovered from
    # Find the MAE at the point of max profit
    mfe_idx = np.argmax(max_favorable)
    mae_at_mfe = max_adverse[mfe_idx] if mfe_idx > 0 else max_adverse[0]

    # Optimal SL = MAE needed to survive + buffer
    optimal_sl = np.clip(mae_at_mfe + 0.3, MIN_SL_ATR, MAX_SL_ATR)

    # Optimal TP: based on achievable MFE in R-multiples of the SL
    if optimal_sl > 0:
        optimal_tp_r = np.clip(total_mfe / optimal_sl, MIN_TP_R, MAX_TP_R)
    else:
        optimal_tp_r = 3.0

    # Calculate actual PnL with these levels
    sl_dist = atr_val * optimal_sl
    tp_dist = sl_dist * optimal_tp_r

    if direction == "LONG":
        sl_price = entry_price - sl_dist
        tp_price = entry_price + tp_dist
    else:
        sl_price = entry_price + sl_dist
        tp_price = entry_price - tp_dist

    actual_pnl = -1.0  # Default: SL hit
    for j in range(len(highs)):
        if direction == "LONG":
            if lows[j] <= sl_price:
                actual_pnl = -1.0
                break
            if highs[j] >= tp_price:
                actual_pnl = optimal_tp_r
                break
        else:
            if highs[j] >= sl_price:
                actual_pnl = -1.0
                break
            if lows[j] <= tp_price:
                actual_pnl = optimal_tp_r
                break

    return optimal_sl, optimal_tp_r, actual_pnl


# ══════════════════════════════════════════════════════════════════════════
# ML FILTER CLASS (production use)
# ══════════════════════════════════════════════════════════════════════════

class MLFilter:
    """
    Production ML filter supporting:
    - V3 Regime Ensemble (from train_ml_v3.py)
    - XGBoost models (from train_ml_v2.py with feature_mask)
    - sklearn GradientBoosting models (legacy online learning)
    """

    def __init__(self, confidence_threshold=CONFIDENCE_THRESHOLD):
        self.classifier = None
        self.sl_regressor = None
        self.tp_regressor = None
        self.scaler = StandardScaler()
        self.confidence_threshold = confidence_threshold
        self.feature_mask = None  # Boolean mask for feature selection
        self.training_features = []
        self.training_labels = []
        self.training_sl = []
        self.training_tp = []
        self.trades_since_retrain = 0
        self.is_trained = False
        self.feature_names = None
        # V3 regime ensemble fields
        self.trend_model = None
        self.range_model = None
        self.quality_model = None
        self.trend_regime_idx = -1
        # V4 direction ensemble fields
        self.classifier_long = None
        self.classifier_short = None
        self.sl_regressor_long = None
        self.sl_regressor_short = None
        self.tp_regressor_long = None
        self.tp_regressor_short = None
        self.scaler_long = StandardScaler()
        self.scaler_short = StandardScaler()
        self.confidence_threshold_long = confidence_threshold
        self.confidence_threshold_short = confidence_threshold
        self.version = "v2"

    def add_completed_trade(self, features: dict, won: bool, optimal_sl=None, optimal_tp=None):
        """Record a completed trade for future training."""
        self.training_features.append(list(features.values()))
        self.training_labels.append(1 if won else 0)
        if optimal_sl is not None:
            self.training_sl.append(optimal_sl)
        if optimal_tp is not None:
            self.training_tp.append(optimal_tp)
        self.trades_since_retrain += 1

        if self.feature_names is None:
            self.feature_names = list(features.keys())

        if (len(self.training_labels) >= MIN_TRAINING_SAMPLES and
                self.trades_since_retrain >= RETRAIN_INTERVAL):
            self._train()

    def _apply_feature_mask(self, feature_vals):
        """Apply feature mask to select relevant features from full vector."""
        if self.feature_mask is not None:
            # Ensure we have enough features for the mask
            if len(feature_vals) < len(self.feature_mask):
                feature_vals = feature_vals + [0.0] * (len(self.feature_mask) - len(feature_vals))
            return [v for v, m in zip(feature_vals, self.feature_mask) if m]
        return feature_vals

    def should_take_trade(self, features: dict) -> tuple:
        """
        Returns (should_take, confidence, suggested_sl_atr, suggested_tp_r).
        If not trained, returns defaults.
        Supports V4 direction ensemble, V3 regime ensemble and V2 single-model inference.
        """
        if not self.is_trained:
            return True, 0.5, 2.0, 3.0

        # V4 direction ensemble logic
        if getattr(self, "version", "v2") == "v4_direction_ensemble" and self.classifier_long is not None and self.classifier_short is not None:
            direction_long_val = features.get("direction_long", 1.0)
            is_long = direction_long_val == 1.0
            
            feature_vals = list(features.values())
            feature_vals = self._apply_feature_mask(feature_vals)
            
            if is_long:
                scaler_to_use = self.scaler_long
                clf_to_use = self.classifier_long
                sl_to_use = self.sl_regressor_long
                tp_to_use = self.tp_regressor_long
                threshold = getattr(self, "confidence_threshold_long", self.confidence_threshold)
            else:
                scaler_to_use = self.scaler_short
                clf_to_use = self.classifier_short
                sl_to_use = self.sl_regressor_short
                tp_to_use = self.tp_regressor_short
                threshold = getattr(self, "confidence_threshold_short", self.confidence_threshold)
                
            expected_n = scaler_to_use.n_features_in_
            if len(feature_vals) > expected_n:
                feature_vals = feature_vals[:expected_n]
            elif len(feature_vals) < expected_n:
                feature_vals.extend([0.0] * (expected_n - len(feature_vals)))
                
            X = np.array([feature_vals]).reshape(1, -1)
            X_scaled = scaler_to_use.transform(X)
            
            prob = clf_to_use.predict_proba(X_scaled)[0][1]
            should_take = prob >= threshold
            
            suggested_sl = 2.0
            suggested_tp = 3.0
            if sl_to_use is not None:
                suggested_sl = float(sl_to_use.predict(X_scaled)[0])
                suggested_sl = np.clip(suggested_sl, MIN_SL_ATR, MAX_SL_ATR)
            if tp_to_use is not None:
                suggested_tp = float(tp_to_use.predict(X_scaled)[0])
                suggested_tp = np.clip(suggested_tp, MIN_TP_R, MAX_TP_R)
                
            return should_take, prob, suggested_sl, suggested_tp

        # Legacy/V2/V3 single model logic
        feature_vals = list(features.values())

        # Apply feature mask if present (V2 XGBoost production model)
        feature_vals = self._apply_feature_mask(feature_vals)

        # Handle remaining size mismatch
        expected_n = self.scaler.n_features_in_
        if len(feature_vals) > expected_n:
            feature_vals = feature_vals[:expected_n]
        elif len(feature_vals) < expected_n:
            feature_vals.extend([0.0] * (expected_n - len(feature_vals)))

        X = np.array([feature_vals]).reshape(1, -1)
        X_scaled = self.scaler.transform(X)

        # V3 Regime Ensemble: weighted prediction from multiple models
        if self.version == "v3_regime_ensemble" and self.trend_model is not None:
            global_prob = self.classifier.predict_proba(X_scaled)[0][1]

            # Quality score as probability
            quality_p = 0.5
            if self.quality_model is not None:
                q_pred = self.quality_model.predict(X_scaled)[0]
                quality_p = min(max(q_pred / 5.0, 0), 1)

            # Determine regime and blend
            if self.trend_regime_idx >= 0 and self.trend_regime_idx < len(feature_vals):
                regime_val = feature_vals[self.trend_regime_idx]
            else:
                regime_val = 0

            if regime_val >= 1 and self.trend_model is not None:
                regime_prob = self.trend_model.predict_proba(X_scaled)[0][1]
                prob = 0.40 * global_prob + 0.40 * regime_prob + 0.20 * quality_p
            elif regime_val <= 0 and self.range_model is not None:
                regime_prob = self.range_model.predict_proba(X_scaled)[0][1]
                prob = 0.40 * global_prob + 0.40 * regime_prob + 0.20 * quality_p
            else:
                prob = 0.60 * global_prob + 0.40 * quality_p
        else:
            # V2 single model
            prob = self.classifier.predict_proba(X_scaled)[0][1]

        should_take = prob >= self.confidence_threshold

        # SL/TP regression
        suggested_sl = 2.0
        suggested_tp = 3.0

        if self.sl_regressor is not None:
            suggested_sl = float(self.sl_regressor.predict(X_scaled)[0])
            suggested_sl = np.clip(suggested_sl, MIN_SL_ATR, MAX_SL_ATR)

        if self.tp_regressor is not None:
            suggested_tp = float(self.tp_regressor.predict(X_scaled)[0])
            suggested_tp = np.clip(suggested_tp, MIN_TP_R, MAX_TP_R)

        return should_take, prob, suggested_sl, suggested_tp

    def _train(self):
        """Train/retrain all models on accumulated online data.
        V4: supports separate LONG/SHORT models based on direction_long feature.
        V3: retrains global classifier with XGBoost.
        V2: uses sklearn GradientBoosting.
        """
        X = np.array(self.training_features)
        y = np.array(self.training_labels)
        y_sl = np.array(self.training_sl) if self.training_sl else None
        y_tp = np.array(self.training_tp) if self.training_tp else None

        if len(set(y)) < 2:
            return

        # V4 direction ensemble online retraining
        if getattr(self, "version", "v2") == "v4_direction_ensemble" and HAS_XGB:
            dir_idx = -1
            if self.feature_names:
                try:
                    dir_idx = self.feature_names.index("direction_long")
                except ValueError:
                    pass
            
            if dir_idx != -1:
                long_mask = X[:, dir_idx] == 1.0
                short_mask = X[:, dir_idx] == 0.0

                # Train LONG models
                if long_mask.sum() >= MIN_TRAINING_SAMPLES and len(set(y[long_mask])) >= 2:
                    X_long = X[long_mask]
                    y_long = y[long_mask]
                    self.scaler_long.fit(X_long)
                    X_long_s = self.scaler_long.transform(X_long)
                    
                    self.classifier_long = xgb.XGBClassifier(
                        n_estimators=100, max_depth=3, learning_rate=0.05, verbosity=0
                    )
                    self.classifier_long.fit(X_long_s, y_long)

                    if y_sl is not None and len(y_sl) == len(X):
                        y_sl_long = y_sl[long_mask]
                        self.sl_regressor_long = xgb.XGBRegressor(
                            n_estimators=50, max_depth=3, learning_rate=0.05, verbosity=0
                        )
                        self.sl_regressor_long.fit(X_long_s, y_sl_long)

                    if y_tp is not None and len(y_tp) == len(X):
                        y_tp_long = y_tp[long_mask]
                        self.tp_regressor_long = xgb.XGBRegressor(
                            n_estimators=50, max_depth=3, learning_rate=0.05, verbosity=0
                        )
                        self.tp_regressor_long.fit(X_long_s, y_tp_long)

                # Train SHORT models
                if short_mask.sum() >= MIN_TRAINING_SAMPLES and len(set(y[short_mask])) >= 2:
                    X_short = X[short_mask]
                    y_short = y[short_mask]
                    self.scaler_short.fit(X_short)
                    X_short_s = self.scaler_short.transform(X_short)
                    
                    self.classifier_short = xgb.XGBClassifier(
                        n_estimators=100, max_depth=3, learning_rate=0.05, verbosity=0
                    )
                    self.classifier_short.fit(X_short_s, y_short)

                    if y_sl is not None and len(y_sl) == len(X):
                        y_sl_short = y_sl[short_mask]
                        self.sl_regressor_short = xgb.XGBRegressor(
                            n_estimators=50, max_depth=3, learning_rate=0.05, verbosity=0
                        )
                        self.sl_regressor_short.fit(X_short_s, y_sl_short)

                    if y_tp is not None and len(y_tp) == len(X):
                        y_tp_short = y_tp[short_mask]
                        self.tp_regressor_short = xgb.XGBRegressor(
                            n_estimators=50, max_depth=3, learning_rate=0.05, verbosity=0
                        )
                        self.tp_regressor_short.fit(X_short_s, y_tp_short)

                self.feature_mask = None
                self.is_trained = self.classifier_long is not None or self.classifier_short is not None
                self.trades_since_retrain = 0
                return

        # Fallback to single scaler training
        self.scaler.fit(X)
        X_scaled = self.scaler.transform(X)

        if self.version == "v3_regime_ensemble" and HAS_XGB:
            self.classifier = xgb.XGBClassifier(
                n_estimators=200,
                max_depth=4,
                learning_rate=0.05,
                min_child_weight=10,
                subsample=0.8,
                colsample_bytree=0.7,
                reg_alpha=0.5,
                reg_lambda=2.0,
                eval_metric="logloss",
                random_state=42,
            )
            self.classifier.fit(X_scaled, y)

            if self.training_sl and len(self.training_sl) >= MIN_TRAINING_SAMPLES:
                from sklearn.ensemble import GradientBoostingRegressor as GBR
                y_quality = np.array(self.training_sl[:len(X)])
                self.quality_model = GBR(
                    n_estimators=50, max_depth=3, learning_rate=0.05,
                    min_samples_leaf=10, random_state=42,
                )
                self.quality_model.fit(X_scaled[:len(y_quality)], y_quality)

            print(f"   ML V3 retrained global classifier on {len(y)} trades | "
                  f"Acc: {self.classifier.score(X_scaled, y):.1%}")

        elif HAS_SKLEARN_GB:
            self.classifier = GradientBoostingClassifier(
                n_estimators=50,
                max_depth=2,
                learning_rate=0.1,
                min_samples_leaf=8,
                subsample=0.8,
                max_features=0.6,
                random_state=42,
            )
            self.classifier.fit(X_scaled, y)
            print(f"   ML retrained on {len(y)} trades | Acc: {self.classifier.score(X_scaled, y):.1%}")
        else:
            return

        if len(self.training_sl) >= MIN_TRAINING_SAMPLES and HAS_SKLEARN_GB:
            y_sl = np.array(self.training_sl[:len(X)])
            self.sl_regressor = GradientBoostingRegressor(
                n_estimators=30,
                max_depth=2,
                learning_rate=0.1,
                min_samples_leaf=8,
                subsample=0.8,
                random_state=42,
            )
            self.sl_regressor.fit(X_scaled[:len(y_sl)], y_sl)

        if len(self.training_tp) >= MIN_TRAINING_SAMPLES and HAS_SKLEARN_GB:
            y_tp = np.array(self.training_tp[:len(X)])
            self.tp_regressor = GradientBoostingRegressor(
                n_estimators=30,
                max_depth=2,
                learning_rate=0.1,
                min_samples_leaf=8,
                subsample=0.8,
                random_state=42,
            )
            self.tp_regressor.fit(X_scaled[:len(y_tp)], y_tp)

        self.feature_mask = None
        self.is_trained = True
        self.trades_since_retrain = 0

    def get_feature_importance(self):
        """Get feature importance from classifier."""
        if not self.is_trained or self.feature_names is None:
            return {}
        if getattr(self, "version", "v2") == "v4_direction_ensemble":
            imp_long = None
            if self.classifier_long is not None:
                if hasattr(self.classifier_long, "estimator"):
                    imp_long = self.classifier_long.estimator.feature_importances_
                elif hasattr(self.classifier_long, "feature_importances_"):
                    imp_long = self.classifier_long.feature_importances_

            imp_short = None
            if self.classifier_short is not None:
                if hasattr(self.classifier_short, "estimator"):
                    imp_short = self.classifier_short.estimator.feature_importances_
                elif hasattr(self.classifier_short, "feature_importances_"):
                    imp_short = self.classifier_short.feature_importances_
            
            if imp_long is not None and imp_short is not None:
                importances = (imp_long + imp_short) / 2.0
            elif imp_long is not None:
                importances = imp_long
            elif imp_short is not None:
                importances = imp_short
            else:
                return {}
        else:
            clf = self.classifier
            if clf is None:
                return {}
            if hasattr(clf, "estimator"):
                importances = clf.estimator.feature_importances_
            elif hasattr(clf, "feature_importances_"):
                importances = clf.feature_importances_
            else:
                return {}
            
        return dict(sorted(
            zip(self.feature_names, importances),
            key=lambda x: x[1], reverse=True
        ))

    def save(self, filepath="ml_filter.pkl"):
        """Save all models to disk."""
        if not self.is_trained:
            return
        data = {
            "classifier": self.classifier,
            "sl_regressor": self.sl_regressor,
            "tp_regressor": self.tp_regressor,
            "scaler": self.scaler,
            "feature_names": self.feature_names,
            "feature_mask": self.feature_mask,
            "confidence_threshold": self.confidence_threshold,
            "training_features": self.training_features,
            "training_labels": self.training_labels,
            "training_sl": self.training_sl,
            "training_tp": self.training_tp,
            # V3 ensemble fields
            "trend_model": getattr(self, "trend_model", None),
            "range_model": getattr(self, "range_model", None),
            "quality_model": getattr(self, "quality_model", None),
            "trend_regime_idx": getattr(self, "trend_regime_idx", -1),
            "version": getattr(self, "version", "v2"),
            # V4 direction ensemble fields
            "classifier_long": getattr(self, "classifier_long", None),
            "classifier_short": getattr(self, "classifier_short", None),
            "sl_regressor_long": getattr(self, "sl_regressor_long", None),
            "sl_regressor_short": getattr(self, "sl_regressor_short", None),
            "tp_regressor_long": getattr(self, "tp_regressor_long", None),
            "tp_regressor_short": getattr(self, "tp_regressor_short", None),
            "scaler_long": getattr(self, "scaler_long", None),
            "scaler_short": getattr(self, "scaler_short", None),
            "confidence_threshold_long": getattr(self, "confidence_threshold_long", self.confidence_threshold),
            "confidence_threshold_short": getattr(self, "confidence_threshold_short", self.confidence_threshold),
        }
        with open(filepath, "wb") as f:
            pickle.dump(data, f)

    def load(self, filepath="ml_filter.pkl"):
        """Load trained models from disk. Supports V4, V3 regime ensemble, V2 XGBoost, and sklearn formats."""
        script_dir = os.path.dirname(os.path.abspath(__file__))
        full_path = os.path.join(script_dir, filepath) if not os.path.isabs(filepath) else filepath
        if not os.path.exists(full_path):
            # Try ml_models/ directory
            alt_path = os.path.join(script_dir, "ml_models", "ml_filter_production.pkl")
            if os.path.exists(alt_path):
                full_path = alt_path
            else:
                return False
        with open(full_path, "rb") as f:
            data = pickle.load(f)
        self.classifier = data.get("classifier")
        self.sl_regressor = data.get("sl_regressor")
        self.tp_regressor = data.get("tp_regressor")
        self.scaler = data.get("scaler", StandardScaler())
        self.feature_names = data.get("feature_names")
        self.feature_mask = data.get("feature_mask")
        self.confidence_threshold = data.get("confidence_threshold", CONFIDENCE_THRESHOLD)
        self.training_features = data.get("training_features", [])
        self.training_labels = data.get("training_labels", [])
        self.training_sl = data.get("training_sl", [])
        self.training_tp = data.get("training_tp", [])
        # V3 regime ensemble fields
        self.trend_model = data.get("trend_model")
        self.range_model = data.get("range_model")
        self.quality_model = data.get("quality_model")
        self.trend_regime_idx = data.get("trend_regime_idx", -1)
        self.version = data.get("version", "v2")
        # V4 direction ensemble fields
        self.classifier_long = data.get("classifier_long")
        self.classifier_short = data.get("classifier_short")
        self.sl_regressor_long = data.get("sl_regressor_long")
        self.sl_regressor_short = data.get("sl_regressor_short")
        self.tp_regressor_long = data.get("tp_regressor_long")
        self.tp_regressor_short = data.get("tp_regressor_short")
        self.scaler_long = data.get("scaler_long", StandardScaler())
        self.scaler_short = data.get("scaler_short", StandardScaler())
        self.confidence_threshold_long = data.get("confidence_threshold_long", self.confidence_threshold)
        self.confidence_threshold_short = data.get("confidence_threshold_short", self.confidence_threshold)
        self.is_trained = (self.classifier is not None) or (self.classifier_long is not None)
        return True
