class Config:
    # ── TRADING PAIRS ──────────────────────────────────────────────
    # Focus on high-liquidity pairs only
    SYMBOLS = [
        # Ultra-high volume majors (most predictable)
        "BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT",
        "SOLUSDT", "AVAXUSDT",
    ]

    # ── PER-SYMBOL LEVERAGE ────────────────────────────────────────
    # Conservative leverage - reduced from 20x to 5x-10x
    # Lower leverage = fewer stop-outs = better risk management
    SYMBOL_LEVERAGE = {
        "BTCUSDT"  : 5,    # Was 20x → 5x (less risky)
        "ETHUSDT"  : 5,    # Was 20x → 5x
        "BNBUSDT"  : 5,    # Was 20x → 5x
        "XRPUSDT"  : 10,   # Was 20x → 10x (slightly higher)
        "SOLUSDT"  : 10,   # Was 20x → 10x
        "AVAXUSDT" : 5,    # Was 10x → 5x
    }
    LEVERAGE = 3   # fallback default (was 2)

    # ── PER-SYMBOL QTY STEPS ───────────────────────────────────────
    QTY_STEPS = {
        "BTCUSDT"  : 0.001,
        "ETHUSDT"  : 0.01,
        "BNBUSDT"  : 0.01,
        "SOLUSDT"  : 0.1,
        "XRPUSDT"  : 1,
        "AVAXUSDT" : 0.1,
    }
    QTY_STEP = 1   # fallback default

    # ── TIMEFRAMES ─────────────────────────────────────────────────
    PRIMARY_TF = "15m"      # Entry signals
    PATTERN_TF = "1h"       # Pattern confirmation (added)
    TREND_TF   = "4h"       # Macro trend (changed from 1h)

    # ── INDICATORS ──────────────────────────────────────────────────
    EMA_FAST    = 9         # Short-term momentum
    EMA_MED     = 21        # Medium-term direction
    EMA_SLOW    = 50        # Long-term trend (was 50, unchanged)
    EMA_TREND   = 200       # ADDED: Ultra-long term trend filter
    
    RSI_PERIOD       = 14
    RSI_OVERSOLD     = 30   # Changed from 35 (more aggressive buy signal)
    RSI_OVERBOUGHT   = 70   # Changed from 65 (more aggressive sell signal)
    RSI_EXTREME_LOW  = 20   # For extreme conditions
    RSI_EXTREME_HIGH = 80   # For extreme conditions
    
    ATR_PERIOD  = 14        # Volatility measure
    BB_PERIOD   = 20        # Bollinger Bands
    BB_STD      = 2         # Standard deviations
    
    MACD_FAST   = 12
    MACD_SLOW   = 26
    MACD_SIGNAL = 9
    
    VOLUME_MA   = 20        # Volume moving average
    VOLUME_MIN  = 1.0       # ADDED: Minimum volume ratio (1.0x MA)

    # ── RISK MANAGEMENT ─────────────────────────────────────────────
    RISK_PER_TRADE   = 0.02       # Reduced from 0.05 → 0.02 (2% per trade)
    SL_ATR_MULT      = 2.5        # INCREASED from 1.75 → 2.5 (give room to breathe)
    TP_ATR_MULT      = 3.5        # INCREASED from 0.90 → 3.5 (let winners run)
    # Risk/Reward ratio: 1:1.4 (was 1:0.51) = MUCH better
    
    DAILY_LOSS_LIMIT = 0.03       # Reduced from 0.05 → 0.03 (3% max daily loss)
    MAX_POSITIONS    = 1          # One position at a time
    MIN_NOTIONAL     = 10.0        # Increased from $5 → $10 (better fills)

    # ── TREND QUALITY FILTERS (NEW) ─────────────────────────────────
    MIN_EMA_SPREAD   = 0.15        # ADDED: EMA separation (tighter trends only)
    MIN_RSI_RANGE    = 10          # ADDED: RSI movement threshold
    MIN_MACD_HISTO   = 0.001       # ADDED: Minimum MACD histogram strength

    # ── ENTRY FILTERS (NEW) ─────────────────────────────────────────
    REQUIRE_HILO_CONFIRMATION = True  # ADDED: Close must be near H/L
    PULLBACK_LOOKBACK = 5            # ADDED: Pullback from recent 5 candles
    BREAKOUT_STRENGTH = 0.5          # ADDED: % of range for breakout (50%)

    # ── BOT LOOP ────────────────────────────────────────────────────
    LOOP_INTERVAL    = 60            # Check every 60 seconds (unchanged)
    MIN_CANDLES_SINCE_TRADE = 5      # ADDED: Wait 75 min (5 × 15m) between trades
