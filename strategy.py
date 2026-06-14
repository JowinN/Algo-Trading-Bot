import pandas as pd
from indicators import compute_all
from config import Config as c

class Signal:
    LONG  = "LONG"
    SHORT = "SHORT"
    NONE  = "NONE"

def market_regime(df: pd.DataFrame) -> str:
    """
    Determines if market is TRENDING or RANGING.
    More robust: uses multiple timeframes and EMA separation.
    """
    if len(df) < 50:
        return "RANGING"
    
    last = df.iloc[-1]
    
    # EMA separation (tight trend = strong trend)
    ema_fast = last["ema_fast"]
    ema_med = last["ema_med"]
    ema_slow = last["ema_slow"]
    ema_trend = last["ema_trend"]
    
    fast_med_pct = abs(ema_fast - ema_med) / ema_med * 100 if ema_med != 0 else 0
    med_slow_pct = abs(ema_med - ema_slow) / ema_slow * 100 if ema_slow != 0 else 0
    slow_trend_pct = abs(ema_slow - ema_trend) / ema_trend * 100 if ema_trend != 0 else 0
    
    # TRENDING: all EMAs aligned, good separation
    if (fast_med_pct > c.MIN_EMA_SPREAD and 
        med_slow_pct > c.MIN_EMA_SPREAD and
        slow_trend_pct > 0.05):
        return "TRENDING"
    
    return "RANGING"

def is_strong_uptrend(df: pd.DataFrame) -> bool:
    """
    Confirms strong uptrend: fast > med > slow > 200ema, with good separation.
    """
    if len(df) < 50:
        return False
    
    last = df.iloc[-1]
    ema_fast = last["ema_fast"]
    ema_med = last["ema_med"]
    ema_slow = last["ema_slow"]
    ema_trend = last["ema_trend"]
    
    return (
        ema_fast > ema_med > ema_slow > ema_trend and
        (ema_fast - ema_med) / ema_med * 100 > c.MIN_EMA_SPREAD and
        (ema_med - ema_slow) / ema_slow * 100 > c.MIN_EMA_SPREAD
    )

def is_strong_downtrend(df: pd.DataFrame) -> bool:
    """
    Confirms strong downtrend: fast < med < slow < 200ema, with good separation.
    """
    if len(df) < 50:
        return False
    
    last = df.iloc[-1]
    ema_fast = last["ema_fast"]
    ema_med = last["ema_med"]
    ema_slow = last["ema_slow"]
    ema_trend = last["ema_trend"]
    
    return (
        ema_fast < ema_med < ema_slow < ema_trend and
        (ema_med - ema_fast) / ema_fast * 100 > c.MIN_EMA_SPREAD and
        (ema_slow - ema_med) / ema_med * 100 > c.MIN_EMA_SPREAD
    )

def is_price_near_ema_support(close: float, ema_fast: float, ema_med: float) -> bool:
    """
    Check if price is near EMAs (pullback) - good entry for LONG.
    """
    support_range = (ema_med - ema_fast) * 0.5
    return close > ema_fast - support_range and close < ema_med + support_range

def is_price_near_ema_resistance(close: float, ema_fast: float, ema_med: float) -> bool:
    """
    Check if price is near EMAs (pullback) - good entry for SHORT.
    """
    resistance_range = (ema_med - ema_fast) * 0.5
    return close < ema_fast + resistance_range and close > ema_med - resistance_range

def generate_signal(df: pd.DataFrame) -> tuple:
    """
    Advanced signal generation with:
    - Trend confirmation via multiple EMAs
    - Volume confirmation
    - MACD acceleration
    - RSI momentum
    - Support/resistance levels
    """
    df = compute_all(df)
    if len(df) < 50:
        return Signal.NONE, 0, 0

    curr = df.iloc[-1]
    prev = df.iloc[-2]
    prev2 = df.iloc[-3]
    price = curr["close"]
    atr_v = curr["atr"]

    # ── MARKET REGIME CHECK ────────────────────────────────────────
    regime = market_regime(df)
    if regime == "RANGING":
        return Signal.NONE, 0, 0

    sl_dist = atr_v * c.SL_ATR_MULT
    tp_dist = atr_v * c.TP_ATR_MULT

    # ── CANDLE HELPERS ────────────────────────────────────────────
    bullish_candle = curr["close"] > curr["open"]
    bearish_candle = curr["close"] < curr["open"]
    
    # Volume confirmation
    volume_strong = curr["volume"] > curr["vol_ma"] * c.VOLUME_MIN
    
    # MACD momentum (stronger threshold)
    macd_hist = curr["macd_hist"]
    macd_hist_prev = prev["macd_hist"]
    macd_hist_prev2 = prev2["macd_hist"]
    macd_up_momentum = macd_hist > macd_hist_prev > macd_hist_prev2
    macd_dn_momentum = macd_hist < macd_hist_prev < macd_hist_prev2
    
    # RSI momentum
    rsi_curr = curr["rsi"]
    rsi_prev = prev["rsi"]
    rsi_moving_up = rsi_curr > rsi_prev
    rsi_moving_dn = rsi_curr < rsi_prev

    # Recent highs/lows
    recent_high = df["high"].iloc[-c.PULLBACK_LOOKBACK:-1].max()
    recent_low = df["low"].iloc[-c.PULLBACK_LOOKBACK:-1].min()
    recent_range = recent_high - recent_low
    
    # Breakout: price broke above/below recent range
    breakout_strength = (price - recent_low) / recent_range if recent_range > 0 else 0
    strong_breakout_up = breakout_strength > c.BREAKOUT_STRENGTH
    strong_breakout_dn = breakout_strength < (1 - c.BREAKOUT_STRENGTH)

    # ── SIGNAL A: STRONG UPTREND LONG ─────────────────────────────
    uptrend_long = (
        is_strong_uptrend(df) and
        curr["close"] > curr["ema_fast"] and
        rsi_curr > 40 and rsi_curr < 70 and  # RSI sweet spot
        macd_up_momentum and
        volume_strong and
        bullish_candle and
        (strong_breakout_up or is_price_near_ema_support(price, curr["ema_fast"], curr["ema_med"]))
    )

    if uptrend_long:
        return Signal.LONG, round(price - sl_dist, 5), round(price + tp_dist, 5)

    # ── SIGNAL B: STRONG DOWNTREND SHORT ───────────────────────────
    downtrend_short = (
        is_strong_downtrend(df) and
        curr["close"] < curr["ema_fast"] and
        rsi_curr < 60 and rsi_curr > 30 and  # RSI sweet spot
        macd_dn_momentum and
        volume_strong and
        bearish_candle and
        (strong_breakout_dn or is_price_near_ema_resistance(price, curr["ema_fast"], curr["ema_med"]))
    )

    if downtrend_short:
        return Signal.SHORT, round(price + sl_dist, 5), round(price - tp_dist, 5)

    return Signal.NONE, 0, 0
