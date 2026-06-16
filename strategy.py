"""
STRATEGY V10 - 4H Momentum Continuation
======================================
Catch multi-day trends. Hold 1-3 days. Fewer filters, more trades, bigger wins.

Architecture:
- PRIMARY: 4H candles
- HTF: Daily trend bias (EMA21 vs EMA50)
- ENTRY: Price near EMA21 in established trend + momentum resumption
- SL: 2.0 ATR on 4H (wide enough for intraday noise)
- TP: 3.5R (multi-day target)
- HOLD: 1-3 days typical

Key changes from V9:
- Removed VWAP filter (meaningless on 4H rolling)
- Removed DI+/DI- filter (redundant with ADX + HTF)
- Removed EMA21 slope requirement (kills pullbacks)
- Removed strict MACD > 0 (kills pullback entries)
- Relaxed price vs EMA21 to allow actual pullbacks
- Lowered ADX threshold to 20
- Simplified to 5 core conditions: Trend + Pullback Zone + Momentum Resumption + Body + RSI

Math: 35% WR at 3.5R = PF 1.88
      30% WR at 3.5R = PF 1.50
      40% WR at 3.5R = PF 2.33
"""

import numpy as np


class Signal:
    LONG = "LONG"
    SHORT = "SHORT"
    NONE = "NONE"


def generate_signal(df, htf_bias=None):
    """
    4H Momentum Continuation:
    - Established trend (EMA21 > EMA50 + ADX >= 20)
    - Price in pullback zone (within 2.5 ATR of EMA21)
    - Momentum resuming (MACD hist improving OR squeeze fire OR volume surge)
    - Confirmation candle (some body in direction)
    - RSI not extreme
    """
    if len(df) < 50:
        return Signal.NONE, 0, 0

    curr = df.iloc[-1]
    prev = df.iloc[-2]

    price = float(curr["close"])
    open_price = float(curr["open"])
    atr_val = float(curr["atr"])

    if atr_val <= 0:
        return Signal.NONE, 0, 0

    # Core indicators
    ema21 = float(curr["ema21"])
    ema50 = float(curr["ema50"])
    adx = float(curr["adx"])
    rsi_val = float(curr["rsi"])
    macd_hist = float(curr["macd_hist"])
    macd_hist_prev = float(prev["macd_hist"])
    rel_vol = float(curr["rel_volume"])

    # Candle analysis
    body_pct = float(curr["body_pct"]) if curr["body_pct"] == curr["body_pct"] else 0
    candle_bullish = price > open_price
    candle_bearish = price < open_price

    # Squeeze fire (volatility expansion after compression)
    squeeze_fire = bool(curr.get("squeeze_fire", False))

    # ═══════════════════════════════════════════════════════════════
    # FILTER 1: ADX — trend must have some strength
    # ═══════════════════════════════════════════════════════════════
    if adx < 20:
        return Signal.NONE, 0, 0

    # ═══════════════════════════════════════════════════════════════
    # LONG SETUP
    # ═══════════════════════════════════════════════════════════════
    if htf_bias == "LONG" or (htf_bias is None and ema21 > ema50):

        # ── TREND: EMA21 above EMA50 ──────────────────────────────
        if ema21 <= ema50:
            return Signal.NONE, 0, 0

        # ── PULLBACK ZONE: price within -0.5 to +2.5 ATR of EMA21 ─
        dist_from_ema21 = (price - ema21) / atr_val
        if dist_from_ema21 > 2.5 or dist_from_ema21 < -0.5:
            return Signal.NONE, 0, 0

        # ── CONFIRMATION CANDLE: bullish with some body ───────────
        if not candle_bullish:
            return Signal.NONE, 0, 0
        if body_pct < 0.15:
            return Signal.NONE, 0, 0

        # ── RSI: not overbought, not deeply oversold ──────────────
        if rsi_val > 75 or rsi_val < 28:
            return Signal.NONE, 0, 0

        # ── MOMENTUM RESUMPTION (any one of these) ────────────────
        macd_improving = macd_hist > macd_hist_prev
        vol_surge = rel_vol > 1.2
        squeeze_released = squeeze_fire
        if not (macd_improving or vol_surge or squeeze_released):
            return Signal.NONE, 0, 0

        sl, tp = _calculate_levels(Signal.LONG, price, atr_val, df)
        return Signal.LONG, sl, tp

    # ═══════════════════════════════════════════════════════════════
    # SHORT SETUP
    # ═══════════════════════════════════════════════════════════════
    elif htf_bias == "SHORT" or (htf_bias is None and ema21 < ema50):

        # ── TREND: EMA21 below EMA50 ─────────────────────────────
        if ema21 >= ema50:
            return Signal.NONE, 0, 0

        # ── PULLBACK ZONE: price within -0.5 to +2.5 ATR of EMA21 ─
        dist_from_ema21 = (ema21 - price) / atr_val
        if dist_from_ema21 > 2.5 or dist_from_ema21 < -0.5:
            return Signal.NONE, 0, 0

        # ── CONFIRMATION CANDLE: bearish with some body ───────────
        if not candle_bearish:
            return Signal.NONE, 0, 0
        if body_pct < 0.15:
            return Signal.NONE, 0, 0

        # ── RSI: not oversold, not deeply overbought ──────────────
        if rsi_val < 25 or rsi_val > 72:
            return Signal.NONE, 0, 0

        # ── MOMENTUM RESUMPTION (any one of these) ────────────────
        macd_improving = macd_hist < macd_hist_prev
        vol_surge = rel_vol > 1.2
        squeeze_released = squeeze_fire
        if not (macd_improving or vol_surge or squeeze_released):
            return Signal.NONE, 0, 0

        sl, tp = _calculate_levels(Signal.SHORT, price, atr_val, df)
        return Signal.SHORT, sl, tp

    return Signal.NONE, 0, 0


def _calculate_levels(signal, price, atr_val, df):
    """
    4H SL/TP levels:
    SL: 2.0 ATR (wide — beyond intraday noise on 4H)
    TP: 3.5R (7.0 ATR — multi-day target)

    Structure-adjusted: Uses swing points when available.
    """
    sl_dist = atr_val * 2.0

    if signal == Signal.LONG:
        swing_low = float(df.iloc[-1]["swing_low"])
        struct_dist = price - swing_low
        # Use structure if between 1.2 and 2.5 ATR
        if 1.2 * atr_val <= struct_dist <= 2.5 * atr_val:
            sl_dist = struct_dist + atr_val * 0.1  # Small buffer
        sl = price - sl_dist
        tp = price + sl_dist * 3.5  # 3.5:1 R:R
    else:
        swing_high = float(df.iloc[-1]["swing_high"])
        struct_dist = swing_high - price
        if 1.2 * atr_val <= struct_dist <= 2.5 * atr_val:
            sl_dist = struct_dist + atr_val * 0.1
        sl = price + sl_dist
        tp = price - sl_dist * 3.5  # 3.5:1 R:R

    return sl, tp