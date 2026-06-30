class Config:
    # ── TRADING PAIRS ──────────────────────────────────────────────
    SYMBOLS = [
        "BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT",
        "SOLUSDT", "AVAXUSDT", "DOGEUSDT", "ADAUSDT",
        "DOTUSDT", "LINKUSDT", "LTCUSDT",
        "UNIUSDT", "ATOMUSDT", "APTUSDT", "ARBUSDT",
        "OPUSDT", "NEARUSDT", "FILUSDT",
        "AAVEUSDT", "INJUSDT", "SUIUSDT", "SEIUSDT",
        "TIAUSDT", "RUNEUSDT", "LDOUSDT",
        "IMXUSDT", "GRTUSDT", "RENDERUSDT",
        "WLDUSDT", "ONDOUSDT", "JUPUSDT",
    ]

    # ── PER-SYMBOL LEVERAGE ────────────────────────────────────────
    SYMBOL_LEVERAGE = {s: 20 for s in SYMBOLS}
    LEVERAGE = 20

    # ── PER-SYMBOL QTY STEPS ───────────────────────────────────────
    QTY_STEPS = {
        "BTCUSDT"   : 0.001,
        "ETHUSDT"   : 0.01,
        "BNBUSDT"   : 0.01,
        "SOLUSDT"   : 0.1,
        "XRPUSDT"   : 1,
        "AVAXUSDT"  : 0.1,
        "DOGEUSDT"  : 1,
        "ADAUSDT"   : 1,
        "DOTUSDT"   : 0.1,
        "LINKUSDT"  : 0.1,
        "LTCUSDT"   : 0.01,
        "UNIUSDT"   : 0.1,
        "ATOMUSDT"  : 0.1,
        "APTUSDT"   : 0.1,
        "ARBUSDT"   : 1,
        "OPUSDT"    : 0.1,
        "NEARUSDT"  : 0.1,
        "FILUSDT"   : 0.1,
        "AAVEUSDT"  : 0.01,
        "INJUSDT"   : 0.1,
        "SUIUSDT"   : 0.1,
        "SEIUSDT"   : 1,
        "TIAUSDT"   : 0.1,
        "RUNEUSDT"  : 0.1,
        "LDOUSDT"   : 0.1,
        "IMXUSDT"   : 1,
        "GRTUSDT"   : 1,
        "RENDERUSDT": 0.1,
        "WLDUSDT"   : 0.1,
        "ONDOUSDT"  : 0.1,
        "JUPUSDT"   : 1,
    }
    QTY_STEP = 1

    # ── TIMEFRAMES ─────────────────────────────────────────────────
    PRIMARY_TF = "4h"       # Entry signals (4H candles)
    HTF_TF     = "1d"       # Higher timeframe trend bias (Daily)
    DATA_TF    = "15m"      # Raw data resolution (resampled to 4H)

    # ── INDICATORS ──────────────────────────────────────────────────
    EMA_FAST    = 9
    EMA_MED     = 21
    EMA_SLOW    = 50
    EMA_TREND   = 200
    RSI_PERIOD  = 14
    ATR_PERIOD  = 14
    BB_PERIOD   = 20
    BB_STD      = 2
    MACD_FAST   = 12
    MACD_SLOW   = 26
    MACD_SIGNAL = 9
    VOLUME_MA   = 20

    # ── RISK MANAGEMENT ───────────────────────────────────────────
    RISK_PER_TRADE   = 0.03      # 1.5% risk per trade (aggressive but controlled)
    SL_ATR_MULT      = 1.0        # 1.0 ATR SL
    TP_ATR_MULT      = 1.0        # 1.0 ATR TP (1:1 R:R)
    USE_ML_DYNAMIC_SL_TP = False  # Use ML-suggested dynamic SL/TP levels if True

    DAILY_LOSS_LIMIT = 0.08       # 8% max daily loss
    MAX_POSITIONS    = 3          # 4 concurrent (more opportunities)
    MAX_TRADES_PER_DAY = 10        # 6 daily cap
    MIN_NOTIONAL     = 10.0

    # ── ENTRY FILTERS ─────────────────────────────────────
    ADX_MIN          = 20
    VOLUME_MIN       = 0.8
    REQUIRE_HTF_BIAS = False      # Trade with HTF when available, but allow EMA-based entries
    MIN_CANDLES_SINCE_TRADE = 1   # 4 hour cooldown (1 x 4H candle)

    # ── TRAILING STOP ────────────────────────────────────────────────
    TRAILING_ACTIVATE_R = 1.5     # Activate trailing after 1.5R profit
    TRAILING_DISTANCE_R = 0.75    # Trail 0.75R behind (locks in ~0.75R minimum)
    TRAILING_DISTANCE_ATR = 1.5   # Trail 1.5 ATR behind current price

    # ── CONSECUTIVE LOSS CIRCUIT BREAKER ────────────────────────────
    MAX_CONSECUTIVE_LOSSES = 3
    LOSS_COOLDOWN_CANDLES = 3     # 12 hours cooldown (3 x 4H)

    # ── BOT LOOP ────────────────────────────────────────────────────
    LOOP_INTERVAL    = 360        # Check every 2min
