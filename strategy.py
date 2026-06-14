import pandas as pd
from indicators import compute_all, detect_candle_patterns
from config import Config as c

class Signal:
    LONG  = "LONG"
    SHORT = "SHORT"
    NONE  = "NONE"

def market_regime(df: pd.DataFrame) -> str:
    last   = df.iloc[-1]
    spread = abs(last["ema_fast"] - last["ema_slow"]) / last["ema_slow"] * 100
    return "TRENDING" if spread > 0.1 else "RANGING"

def _check_early_fire(conds, df, direction):
    """
    Returns True if 8/10 or 9/10 conditions pass
    AND a matching candle pattern confirms the direction.
    """
    passed = sum(bool(v) for v in conds)
    total  = len(conds)

    if passed < total - 2:
        return False, None

    patterns = detect_candle_patterns(df)
    for pname, ptype in patterns.items():
        if direction == Signal.LONG  and ptype == "bullish":
            return True, pname
        if direction == Signal.SHORT and ptype == "bearish":
            return True, pname
        # Doji only confirms at 9/10
        if ptype == "neutral" and passed == total - 1:
            return True, f"{pname} (Doji)"

    return False, None

def generate_signal(df: pd.DataFrame) -> tuple:
    df = compute_all(df)
    if len(df) < 3:
        return Signal.NONE, 0, 0

    curr  = df.iloc[-1]
    prev  = df.iloc[-2]
    prev2 = df.iloc[-3]
    price = curr["close"]
    atr_v = curr["atr"]

    if market_regime(df) == "RANGING":
        return Signal.NONE, 0, 0

    sl_dist = atr_v * c.SL_ATR_MULT
    tp_dist = atr_v * c.TP_ATR_MULT

    # ── Candle helpers ─────────────────────────────────────────────
    bullish_candle  = curr["close"] > curr["open"]
    bearish_candle  = curr["close"] < curr["open"]
    pullback_long   = prev["close"] < prev["open"] and bullish_candle
    pullback_short  = prev["close"] > prev["open"] and bearish_candle
    macd_growing_up = curr["macd_hist"] > prev["macd_hist"] > prev2["macd_hist"]
    macd_growing_dn = curr["macd_hist"] < prev["macd_hist"] < prev2["macd_hist"]
    rsi_rising      = curr["rsi"] > prev["rsi"]
    rsi_falling     = curr["rsi"] < prev["rsi"]
    recent_high     = df["high"].iloc[-6:-1].max()
    recent_low      = df["low"].iloc[-6:-1].min()
    breakout_up     = curr["close"] > recent_high
    breakout_dn     = curr["close"] < recent_low

    # ── SIGNAL A: STANDARD LONG ────────────────────────────────────
    long_conds = [
        curr["ema_fast"] > curr["ema_med"],
        curr["ema_med"]  > curr["ema_slow"],
        c.RSI_OVERSOLD < curr["rsi"] < c.RSI_OVERBOUGHT,
        curr["macd_hist"] > 0,
        curr["close"] > curr["bb_mid"],
        curr["volume"] > curr["vol_ma"],
        bullish_candle,
        rsi_rising,
        macd_growing_up,
        pullback_long or breakout_up,
    ]

    # ── SIGNAL B: STANDARD SHORT ───────────────────────────────────
    short_conds = [
        curr["ema_fast"] < curr["ema_med"],
        curr["ema_med"]  < curr["ema_slow"],
        c.RSI_OVERSOLD < curr["rsi"] < c.RSI_OVERBOUGHT,
        curr["macd_hist"] < 0,
        curr["close"] < curr["bb_mid"],
        curr["volume"] > curr["vol_ma"],
        bearish_candle,
        rsi_falling,
        macd_growing_dn,
        pullback_short or breakout_dn,
    ]

    # ── SIGNAL C: DOWNTREND SHORT ──────────────────────────────────
    downtrend_short_conds = [
        curr["ema_fast"] < curr["ema_med"],
        curr["ema_med"]  < curr["ema_slow"],
        c.RSI_EXTREME_LOW < curr["rsi"] < c.RSI_OVERSOLD,
        curr["macd_hist"] < 0,
        curr["close"] < curr["bb_mid"],
        macd_growing_dn,
        bearish_candle,
    ]

    # ── SIGNAL D: UPTREND LONG ─────────────────────────────────────
    uptrend_long_conds = [
        curr["ema_fast"] > curr["ema_med"],
        curr["ema_med"]  > curr["ema_slow"],
        c.RSI_OVERBOUGHT < curr["rsi"] < c.RSI_EXTREME_HIGH,
        curr["macd_hist"] > 0,
        curr["close"] > curr["bb_mid"],
        macd_growing_up,
        bullish_candle,
    ]

    # ── FULL FIRE (all conditions met) ─────────────────────────────
    if all(long_conds):
        return Signal.LONG,  round(price - sl_dist, 5), round(price + tp_dist, 5)
    if all(short_conds):
        return Signal.SHORT, round(price + sl_dist, 5), round(price - tp_dist, 5)
    if all(downtrend_short_conds):
        return Signal.SHORT, round(price + sl_dist, 5), round(price - tp_dist, 5)
    if all(uptrend_long_conds):
        return Signal.LONG,  round(price - sl_dist, 5), round(price + tp_dist, 5)

    # ── EARLY FIRE (8/10 or 9/10 + candle pattern) ─────────────────
    fired, pattern = _check_early_fire(long_conds,           df, Signal.LONG)
    if fired:
        print(f"[EARLY FIRE] STD LONG  via pattern: {pattern}")
        return Signal.LONG,  round(price - sl_dist, 5), round(price + tp_dist, 5)

    fired, pattern = _check_early_fire(short_conds,          df, Signal.SHORT)
    if fired:
        print(f"[EARLY FIRE] STD SHORT via pattern: {pattern}")
        return Signal.SHORT, round(price + sl_dist, 5), round(price - tp_dist, 5)

    fired, pattern = _check_early_fire(downtrend_short_conds, df, Signal.SHORT)
    if fired:
        print(f"[EARLY FIRE] DT SHORT  via pattern: {pattern}")
        return Signal.SHORT, round(price + sl_dist, 5), round(price - tp_dist, 5)

    fired, pattern = _check_early_fire(uptrend_long_conds,   df, Signal.LONG)
    if fired:
        print(f"[EARLY FIRE] UT LONG   via pattern: {pattern}")
        return Signal.LONG,  round(price - sl_dist, 5), round(price + tp_dist, 5)

    return Signal.NONE, 0, 0
