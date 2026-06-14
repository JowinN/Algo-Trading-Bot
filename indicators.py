import pandas as pd

def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta    = series.diff()
    gain     = delta.clip(lower=0)
    loss     = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, adjust=False).mean()
    avg_loss = loss.ewm(com=period - 1, adjust=False).mean()
    rs       = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def macd(series: pd.Series, fast=12, slow=26, signal=9):
    macd_line   = ema(series, fast) - ema(series, slow)
    signal_line = ema(macd_line, signal)
    histogram   = macd_line - signal_line
    return macd_line, signal_line, histogram

def bollinger_bands(series: pd.Series, period=20, std_dev=2):
    mid = series.rolling(period).mean()
    std = series.rolling(period).std()
    return mid + std_dev * std, mid, mid - std_dev * std

def atr(high: pd.Series, low: pd.Series, close: pd.Series, period=14) -> pd.Series:
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low  - close.shift()).abs()
    ], axis=1).max(axis=1)
    return tr.ewm(com=period - 1, adjust=False).mean()

def compute_all(df: pd.DataFrame) -> pd.DataFrame:
    """Attach all indicators to the OHLCV dataframe"""
    from config import Config as c
    df = df.copy()
    df["ema_fast"]                              = ema(df["close"], c.EMA_FAST)
    df["ema_med"]                               = ema(df["close"], c.EMA_MED)
    df["ema_slow"]                              = ema(df["close"], c.EMA_SLOW)
    df["rsi"]                                   = rsi(df["close"], c.RSI_PERIOD)
    df["macd"], df["macd_sig"], df["macd_hist"] = macd(df["close"])
    df["bb_upper"], df["bb_mid"], df["bb_lower"] = bollinger_bands(df["close"])
    df["atr"]                                   = atr(df["high"], df["low"], df["close"], c.ATR_PERIOD)
    df["vol_ma"]                                = df["volume"].rolling(c.VOLUME_MA).mean()
    return df.dropna()

def detect_candle_patterns(df):
    """
    Detects candle patterns on the last 3 candles.
    Returns a dict: { pattern_name: "bullish"/"bearish"/"neutral" }
    Only returns patterns that are detected (non-empty).
    """
    c0 = df.iloc[-1]  # current
    c1 = df.iloc[-2]  # previous
    c2 = df.iloc[-3]  # two back

    o0, h0, c0p, l0 = float(c0["open"]), float(c0["high"]), float(c0["close"]), float(c0["low"])
    o1, h1, c1p, l1 = float(c1["open"]), float(c1["high"]), float(c1["close"]), float(c1["low"])
    o2, h2, c2p, l2 = float(c2["open"]), float(c2["high"]), float(c2["close"]), float(c2["low"])

    body0  = abs(c0p - o0)
    body1  = abs(c1p - o1)
    body2  = abs(c2p - o2)
    range0 = h0 - l0 if h0 != l0 else 0.0001
    range1 = h1 - l1 if h1 != l1 else 0.0001
    range2 = h2 - l2 if h2 != l2 else 0.0001

    upper_wick0 = h0 - max(o0, c0p)
    lower_wick0 = min(o0, c0p) - l0
    upper_wick1 = h1 - max(o1, c1p)
    lower_wick1 = min(o1, c1p) - l1

    bull0 = c0p > o0
    bear0 = c0p < o0
    bull1 = c1p > o1
    bear1 = c1p < o1

    patterns = {}

    # ── SINGLE CANDLE ─────────────────────────────────────────────

    # Doji — body very small relative to range
    if body0 / range0 < 0.1:
        patterns["Doji"] = "neutral"

    # Hammer — small body at top, long lower wick, in downtrend
    if (lower_wick0 >= 2 * body0 and
        upper_wick0 <= 0.3 * body0 and
        body0 / range0 < 0.4):
        patterns["Hammer"] = "bullish"

    # Inverted Hammer — small body at bottom, long upper wick
    if (upper_wick0 >= 2 * body0 and
        lower_wick0 <= 0.3 * body0 and
        body0 / range0 < 0.4):
        patterns["Inverted Hammer"] = "bullish"

    # Shooting Star — small body at bottom, long upper wick, bearish context
    if (upper_wick0 >= 2 * body0 and
        lower_wick0 <= 0.3 * body0 and
        bear0 and
        body0 / range0 < 0.4):
        patterns["Shooting Star"] = "bearish"

    # Bullish Pinbar — long lower wick, close near high
    if (lower_wick0 >= 2.5 * body0 and
        upper_wick0 <= 0.5 * body0 and
        (c0p - l0) / range0 > 0.6):
        patterns["Pinbar Bullish"] = "bullish"

    # Bearish Pinbar — long upper wick, close near low
    if (upper_wick0 >= 2.5 * body0 and
        lower_wick0 <= 0.5 * body0 and
        (h0 - c0p) / range0 > 0.6):
        patterns["Pinbar Bearish"] = "bearish"

    # ── TWO CANDLE ────────────────────────────────────────────────

    # Bullish Engulfing — prev bearish, curr bullish and engulfs prev body
    if (bear1 and bull0 and
        o0 <= c1p and c0p >= o1 and
        body0 > body1):
        patterns["Bullish Engulfing"] = "bullish"

    # Bearish Engulfing — prev bullish, curr bearish and engulfs prev body
    if (bull1 and bear0 and
        o0 >= c1p and c0p <= o1 and
        body0 > body1):
        patterns["Bearish Engulfing"] = "bearish"

    # ── THREE CANDLE ──────────────────────────────────────────────

    # Morning Star — bearish, small doji/body, bullish
    if (bear1 and                          # c1 bearish
        body1 / range1 < 0.35 and          # c1 small body (star)
        bull0 and                          # c0 bullish
        c0p > (o2 + c2p) / 2):            # c0 closes above midpoint of c2
        patterns["Morning Star"] = "bullish"

    # Evening Star — bullish, small doji/body, bearish
    if (bull1 and
        body1 / range1 < 0.35 and
        bear0 and
        c0p < (o2 + c2p) / 2):
        patterns["Evening Star"] = "bearish"

    # Three White Soldiers — three consecutive bullish candles, each closing higher
    if (bull0 and bull1 and (c2p > o2) and
        c0p > c1p > c2p and
        o0 > o1 and o1 > o2 and
        body0 / range0 > 0.5 and
        body1 / range1 > 0.5):
        patterns["Three White Soldiers"] = "bullish"

    # Three Black Crows — three consecutive bearish candles, each closing lower
    if (bear0 and bear1 and (c2p < o2) and
        c0p < c1p < c2p and
        o0 < o1 and o1 < o2 and
        body0 / range0 > 0.5 and
        body1 / range1 > 0.5):
        patterns["Three Black Crows"] = "bearish"

    return patterns
