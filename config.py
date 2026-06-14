class Config:
    # ── TRADING PAIRS ──────────────────────────────────────────────
    SYMBOLS = [
        # High volume majors
        "BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT",
        "SOLUSDT", "DOGEUSDT", "ADAUSDT", "AVAXUSDT",
        # Mid volume
        "LINKUSDT", "DOTUSDT", "LTCUSDT", "NEARUSDT",
        "ATOMUSDT", "UNIUSDT", "ARBUSDT", "OPUSDT",
        "INJUSDT", "SUIUSDT",
        # Lower volume
        "BTWUSDT", "CTRUSDT",
    ]

    # ── PER-SYMBOL LEVERAGE ────────────────────────────────────────
    # 10x → battle-tested high liquidity majors
    # 5x  → solid mid caps with decent volume
    # 2x  → low cap / volatile / thin liquidity
    SYMBOL_LEVERAGE = {
        "BTCUSDT"  : 20,
        "ETHUSDT"  : 20,
        "BNBUSDT"  : 20,
        "XRPUSDT"  : 20,
        "SOLUSDT"  : 20,
        "DOGEUSDT" : 10,
        "ADAUSDT"  : 10,
        "AVAXUSDT" : 10,
        "LINKUSDT" : 10,
        "DOTUSDT"  : 10,
        "LTCUSDT"  : 10,
        "NEARUSDT" : 10,
        "ATOMUSDT" : 10,
        "UNIUSDT"  : 10,
        "ARBUSDT"  : 5,
        "OPUSDT"   : 5,
        "INJUSDT"  : 5,
        "SUIUSDT"  : 5,
        "BTWUSDT"  : 5,
        "CTRUSDT"  : 5,
    }
    LEVERAGE = 2   # fallback default

    # ── PER-SYMBOL QTY STEPS ───────────────────────────────────────
    QTY_STEPS = {
        "BTCUSDT"  : 0.001,
        "ETHUSDT"  : 0.01,
        "BNBUSDT"  : 0.01,
        "SOLUSDT"  : 0.1,
        "XRPUSDT"  : 1,
        "ADAUSDT"  : 1,
        "DOGEUSDT" : 1,
        "AVAXUSDT" : 0.1,
        "DOTUSDT"  : 0.1,
        "LINKUSDT" : 0.1,
        "LTCUSDT"  : 0.01,
        "UNIUSDT"  : 0.1,
        "ATOMUSDT" : 0.1,
        "NEARUSDT" : 0.1,
        "OPUSDT"   : 1,
        "INJUSDT"  : 0.1,
        "SUIUSDT"  : 1,
        "ARBUSDT"  : 1,
        "BTWUSDT"  : 10,
        "CTRUSDT"  : 10,
    }
    QTY_STEP = 10   # fallback default

    PRIMARY_TF       = "15m"
    PATTERN_TF  = "1h"
    TREND_TF         = "1h"

    # ── INDICATORS ─────────────────────────────────────────────────
    EMA_FAST         = 9
    EMA_MED          = 21
    EMA_SLOW         = 50
    RSI_PERIOD       = 14
    RSI_OVERSOLD     = 35
    RSI_OVERBOUGHT   = 65
    RSI_EXTREME_LOW  = 20
    RSI_EXTREME_HIGH = 80
    ATR_PERIOD       = 14
    BB_PERIOD        = 20
    BB_STD           = 2
    MACD_FAST        = 12
    MACD_SLOW        = 26
    MACD_SIGNAL      = 9
    VOLUME_MA        = 20

    # ── RISK MANAGEMENT ────────────────────────────────────────────
    RISK_PER_TRADE   = 0.05
    SL_ATR_MULT      = 1.75
    TP_ATR_MULT      = 0.90
    DAILY_LOSS_LIMIT = 0.05
    MAX_POSITIONS    = 1
    MIN_NOTIONAL     = 5.0

    # ── BOT LOOP ───────────────────────────────────────────────────
    LOOP_INTERVAL    = 60
