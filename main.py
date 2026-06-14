import time
import os
import logging
from datetime import datetime, date
from dotenv import load_dotenv
from mudrex import TradeClient

from config   import Config as c
from data     import get_ohlcv, get_current_price
from strategy import generate_signal, Signal
from risk     import position_size, daily_limit_ok

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    handlers=[logging.FileHandler("bot.log"), logging.StreamHandler()]
)
log = logging.getLogger(__name__)

client = TradeClient(api_secret=os.getenv("MUDREX_API_SECRET"))

daily_pnl  = 0.0
daily_date = date.today()

# ─────────────────────────────────────────────────────────────────────

def get_balance() -> float:
    resp = client.get_available_funds()
    return float(resp.balance)

def get_open_position(symbol: str = None):
    try:
        resp      = client.get_positions()
        # ← get_positions() returns a plain list, not resp.result
        positions = resp if isinstance(resp, list) else resp.result
        if not positions:
            return None
        for pos in positions:
            sym = pos.get("symbol") if isinstance(pos, dict) else getattr(pos, "symbol", None)
            qty = pos.get("quantity", 0) if isinstance(pos, dict) else getattr(pos, "quantity", 0)
            if float(qty) != 0:
                if symbol is None or sym == symbol:
                    return pos
    except Exception as e:
        log.warning(f"  Could not fetch positions: {e}")
    return None

def set_leverage():
    for symbol in c.SYMBOLS:
        try:
            client.set_leverage(
                symbol=symbol,
                margin_type="ISOLATED",
                leverage=str(c.SYMBOL_LEVERAGE.get(symbol, c.LEVERAGE))
            )
            log.info(f"  Leverage set → {c.SYMBOL_LEVERAGE.get(symbol, c.LEVERAGE)}x ISOLATED on {symbol}")
        except Exception as e:
            log.warning(f"  Could not set leverage on {symbol}: {e}")

def open_trade(symbol: str, signal: str, price: float, sl: float, tp: float, qty: float, lev: int) -> bool:
    try:
        resp = client.place_order(
            symbol=symbol,
            leverage=str(lev),   # ← per-symbol leverage
            quantity=str(int(qty)),
            order_type=signal,
            trigger_type="MARKET",
            is_stoploss=True,
            stoploss_price=str(sl),
            is_takeprofit=True,
            takeprofit_price=str(tp),
            reduce_only=False
        )
        icon = "🟢" if signal == "LONG" else "🔴"
        log.info(f"{icon} [{signal}] {symbol} ORDER PLACED @ ${price:.5f}")
        log.info(f"   Qty: {int(qty)}  |  SL: ${sl:.5f}  |  TP: ${tp:.5f}  |  Lev: {lev}x")
        log.info(f"   Order ID: {resp.order_id}")
        return True
    except Exception as e:
        log.error(f"⚠  place_order failed ({symbol}): {e}")
        return False

def close_trade(symbol: str):
    try:
        resp = client.close_position(symbol=symbol)
        log.info(f"🔄  Position closed → {symbol} | {resp}")
    except Exception as e:
        log.error(f"⚠  close_position failed ({symbol}): {e}")

# ─────────────────────────────────────────────────────────────────────

def run():
    global daily_pnl, daily_date

    balance = get_balance()
    set_leverage()

    print("\n" + "═"*52)
    print("  🤖  MUDREX ALGO BOT  —  LIVE TRADING MODE")
    print("═"*52)
    print(f"  Symbols   : {', '.join(c.SYMBOLS)}")
    print(f"  Timeframe : {c.PRIMARY_TF}")
    print(f"  Balance   : ${balance:.4f} USDT")
    print(f"  Leverage  : per-symbol (2x/5x/10x)")
    print(f"  Risk/Trade: {c.RISK_PER_TRADE*100:.0f}% of balance")
    print(f"  Strategy  : First signal wins")
    print("═"*52 + "\n")

    loop_count = 0

    while True:
        try:
            # ── Midnight reset ──────────────────────────────────────
            if date.today() != daily_date:
                daily_pnl  = 0.0
                daily_date = date.today()

            now     = datetime.now().strftime("%H:%M:%S")
            balance = get_balance()

            # ── Daily loss guard ────────────────────────────────────
            if not daily_limit_ok(daily_pnl, balance):
                log.warning("⛔  Daily loss limit hit — pausing 1 hour.")
                time.sleep(3600)
                daily_pnl = 0.0
                continue

            # ── Check open positions vs MAX_POSITIONS ────────────────
            all_positions = []
            try:
                resp = client.get_positions()
                all_positions = resp if isinstance(resp, list) else resp.result
                all_positions = [p for p in all_positions if (p.get("quantity", 0) if isinstance(p, dict) else getattr(p, "quantity", 0)) != 0]
            except Exception as e:
                log.warning(f"  Could not fetch positions: {e}")

            open_count = len(all_positions)

            if open_count > 0:
                for op in all_positions:
                    sym = op.get("symbol") if isinstance(op, dict) else getattr(op, "symbol", "?")
                    qty = op.get("quantity", 0) if isinstance(op, dict) else getattr(op, "quantity", 0)
                    log.info(f"[{now}]  📊 Holding {sym}  qty={qty}  |  Bal: ${balance:.4f}")

            if open_count >= c.MAX_POSITIONS:
                time.sleep(c.LOOP_INTERVAL)
                continue

            slots_free = c.MAX_POSITIONS - open_count
            open_symbols = [(p.get("symbol") if isinstance(p, dict) else getattr(p, "symbol", "?")) for p in all_positions]

            # ── Scan all symbols — fill free slots ───────────────────
            log.info(f"[{now}]  🔍 Scanning {', '.join(c.SYMBOLS)}  |  Slots free: {slots_free}  |  Bal: ${balance:.4f}")

            trades_opened = 0
            for symbol in c.SYMBOLS:
                if trades_opened >= slots_free:
                    break
                if symbol in open_symbols:
                    continue  # already holding this symbol
                try:
                    df = get_ohlcv(symbol, c.PRIMARY_TF, limit=100)
                    if df.empty or len(df) < 3:
                        log.info(f"         {symbol}: not enough candles")
                        continue

                    signal, sl, tp = generate_signal(df)
                    price          = get_current_price(symbol)

                    log.info(f"         {symbol}: ${price:.5f}  Signal: {signal}")

                    if signal != Signal.NONE:
                        qty = position_size(balance, price, sl, symbol)
                        if qty > 0:
                            lev    = c.SYMBOL_LEVERAGE.get(symbol, c.LEVERAGE)
                            opened = open_trade(symbol, signal, price, sl, tp, qty, lev)
                            if opened:
                                trades_opened += 1
                        else:
                            log.warning(f"         {symbol}: qty=0, balance too low?")

                except Exception as e:
                    log.error(f"         {symbol}: error → {e}")
                    continue

            loop_count += 1
            if loop_count % 30 == 0:
                log.info(f"\n  💰  Balance: ${balance:.5f} USDT\n")

            time.sleep(c.LOOP_INTERVAL)

        except KeyboardInterrupt:
            print("\n\n🛑  Bot stopped.")
            print(f"  Final balance: ${get_balance():.5f} USDT")
            break

        except Exception as e:
            log.error(f"⚠  Error: {e}")
            time.sleep(30)

if __name__ == "__main__":
    run()
