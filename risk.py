from config import Config as c

def position_size(balance: float, price: float, sl_price: float, symbol: str = None) -> float:
    """
    Conservative position sizing with improved risk management.
    - Uses fixed-fractional method
    - Enforces minimum notional
    - Respects leverage limits
    """
    if price == 0 or sl_price == 0:
        return 0.0
    
    # Risk amount based on balance
    risk_usdt = balance * c.RISK_PER_TRADE
    
    # Distance to stop loss in price
    sl_pct = abs(price - sl_price) / price
    
    if sl_pct == 0:
        return 0.0

    # Position size based on risk
    position_usdt = risk_usdt / sl_pct

    # Apply leverage limit
    max_usdt = balance * c.SYMBOL_LEVERAGE.get(symbol, c.LEVERAGE)
    position_usdt = min(position_usdt, max_usdt)

    # Enforce minimum notional ($10 USDT minimum)
    position_usdt = max(position_usdt, c.MIN_NOTIONAL)

    # Calculate raw quantity
    raw_qty = position_usdt / price

    # Get step size for symbol
    step = c.QTY_STEPS.get(symbol, c.QTY_STEP) if symbol else c.QTY_STEP

    # Round DOWN to nearest step
    qty = (raw_qty // step) * step

    # Ensure at least one step
    qty = max(qty, step)

    # If rounding down violated min notional, round up
    if qty * price < c.MIN_NOTIONAL:
        qty += step

    return float(qty)

def daily_limit_ok(daily_pnl: float, balance: float) -> bool:
    """
    Check if daily loss hasn't exceeded limit.
    Returns False when daily loss >= configured limit.
    """
    if balance == 0:
        return True
    return (daily_pnl / balance) > -c.DAILY_LOSS_LIMIT
