from config import Config as c

def position_size(balance: float, price: float, sl_price: float, symbol: str = None) -> float:
    """
    Fixed-fractional sizing with minimum notional enforcement.
    Returns quantity rounded to nearest QTY_STEP multiple.
    """
    risk_usdt = balance * c.RISK_PER_TRADE
    sl_pct    = abs(price - sl_price) / price

    if sl_pct == 0:
        return 0.0

    position_usdt = risk_usdt / sl_pct

    # Enforce Mudrex minimum notional ($5 USDT)
    position_usdt = max(position_usdt, c.MIN_NOTIONAL)

    # Can't spend more than we have (with leverage)
    max_usdt      = balance * c.SYMBOL_LEVERAGE.get(symbol, c.LEVERAGE)
    position_usdt = min(position_usdt, max_usdt)

    raw_qty = position_usdt / price

    # ← Use per-symbol step if available, else fallback
    step = c.QTY_STEPS.get(symbol, c.QTY_STEP) if symbol else c.QTY_STEP

    # Round DOWN to nearest step
    qty = (raw_qty // step) * step

    # Must be at least one step
    qty = max(qty, step)

    # If rounding down put us below min notional, round UP one step
    if qty * price < c.MIN_NOTIONAL:
        qty += step

    return float(qty)

def daily_limit_ok(daily_pnl: float, balance: float) -> bool:
    """Returns False when daily loss exceeds the configured limit"""
    return (daily_pnl / balance) > -c.DAILY_LOSS_LIMIT
