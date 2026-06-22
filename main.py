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
from ml_model import MLFilter, extract_features_extended, extract_regime_features, compute_regimes, calculate_mfe_mae
import json
from collections import deque
import requests
from indicators import compute_all, compute_htf, compute_htf

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    handlers=[logging.FileHandler("bot.log")],
    force=True
)
log = logging.getLogger(__name__)

client = TradeClient(api_secret=os.getenv("MUDREX_API_SECRET"))

# ── Instrument Info Cache ─────────────────────────────────────────────
INSTRUMENT_INFO = {}  # {symbol: {"tickSize": float, "qtyStep": float}}

def load_instrument_info():
    """Fetch tick sizes and qty steps from Bybit for all configured symbols."""
    global INSTRUMENT_INFO
    try:
        resp = requests.get(
            "https://api.bybit.com/v5/market/instruments-info",
            params={"category": "linear"},
            timeout=10
        )
        data = resp.json()
        if data["retCode"] == 0:
            for item in data["result"]["list"]:
                sym = item["symbol"]
                INSTRUMENT_INFO[sym] = {
                    "tickSize": float(item["priceFilter"]["tickSize"]),
                    "qtyStep": float(item["lotSizeFilter"]["qtyStep"]),
                }
            log.info(f"  📏  Loaded instrument info for {len(INSTRUMENT_INFO)} symbols")
        else:
            log.warning(f"  Failed to load instrument info: {data['retMsg']}")
    except Exception as e:
        log.warning(f"  Failed to load instrument info: {e}")

load_instrument_info()


# ── ML Filter ────────────────────────────────────────────────────────
ml_filter = MLFilter()
ml_filter.load()
if ml_filter.is_trained:
    log.info(f"✅  ML model loaded (threshold={ml_filter.confidence_threshold:.0%})")

# ── Online Learning State ─────────────────────────────────────────────
# Store pending samples: {symbol: [{features, direction, entry_price, atr, timestamp, bar_idx}]}
ONLINE_SAMPLES_FILE = os.path.join(os.path.dirname(__file__), "ml_online_samples.json")
ONLINE_LOOKBACK_BARS = 20  # Check outcome after 20 bars (4H = ~3.3 days)
ONLINE_RETRAIN_EVERY = 50  # Retrain after 50 new labeled samples
online_pending = []  # Samples waiting for outcome evaluation
online_new_labels = 0  # Counter for new labels since last retrain

def load_online_samples():
    global online_pending
    if os.path.exists(ONLINE_SAMPLES_FILE):
        try:
            with open(ONLINE_SAMPLES_FILE, "r") as f:
                online_pending = json.load(f)
            # Discard samples with wrong feature count for current model
            if ml_filter.is_trained and ml_filter.feature_names and online_pending:
                expected = len(ml_filter.feature_names)
                valid = [s for s in online_pending if len(s.get("features", {})) == expected]
                if len(valid) < len(online_pending):
                    log.info(f"  🗑️  Discarded {len(online_pending) - len(valid)} stale samples (wrong feature count)")
                    online_pending = valid
            log.info(f"  📚  Loaded {len(online_pending)} pending online samples")
        except:
            online_pending = []

def save_online_samples():
    try:
        with open(ONLINE_SAMPLES_FILE, "w") as f:
            json.dump(online_pending, f)
    except Exception as e:
        log.warning(f"  Failed to save online samples: {e}")

def process_online_learning():
    """Check pending samples that now have enough future bars to label."""
    global online_pending, online_new_labels

    if not online_pending:
        return

    remaining = []
    labeled = 0

    for sample in online_pending:
        symbol = sample["symbol"]
        direction = sample["direction"]
        entry_price = sample["entry_price"]
        atr_val = sample["atr"]
        bars_elapsed = sample.get("bars_elapsed", 0) + 1
        sample["bars_elapsed"] = bars_elapsed

        if bars_elapsed < ONLINE_LOOKBACK_BARS:
            remaining.append(sample)
            continue

        # Enough bars passed — fetch current data and evaluate outcome
        try:
            df = get_ohlcv(symbol, c.PRIMARY_TF, limit=ONLINE_LOOKBACK_BARS + 10)
            if df is None or len(df) < ONLINE_LOOKBACK_BARS:
                remaining.append(sample)
                continue
            df = compute_all(df)

            # Use the entry bar (approximately ONLINE_LOOKBACK_BARS ago)
            entry_idx = max(0, len(df) - ONLINE_LOOKBACK_BARS - 1)
            result = calculate_mfe_mae(df, entry_idx, direction, entry_price, atr_val, max_bars=ONLINE_LOOKBACK_BARS)

            if result is None:
                remaining.append(sample)
                continue

            mfe, mae = result
            # Label: WIN if MFE >= 2*MAE and MFE >= 1.5 ATR (decent R:R achieved)
            won = (mfe >= 1.5 and mfe >= 2.0 * mae)

            # Optimal SL/TP from observed excursions
            optimal_sl = min(max(mae + 0.3, 1.0), 4.0)
            optimal_tp_r = min(max(mfe / optimal_sl if optimal_sl > 0 else 2.0, 1.5), 6.0)

            # Add to model training data
            features_dict = sample["features"]
            ml_filter.training_features.append(list(features_dict.values()))
            ml_filter.training_labels.append(1 if won else 0)
            ml_filter.training_sl.append(optimal_sl)
            ml_filter.training_tp.append(optimal_tp_r)
            if ml_filter.feature_names is None:
                ml_filter.feature_names = list(features_dict.keys())

            online_new_labels += 1
            labeled += 1

        except Exception:
            remaining.append(sample)
            continue

    online_pending = remaining

    if labeled > 0:
        log.info(f"  🧠  Online learning: labeled {labeled} samples (total training: {len(ml_filter.training_labels)}, new since retrain: {online_new_labels})")

    # Retrain if enough new samples
    if online_new_labels >= ONLINE_RETRAIN_EVERY and len(ml_filter.training_labels) >= 80:
        log.info(f"  🔄  Retraining ML model with {len(ml_filter.training_labels)} total samples...")
        ml_filter._pretrained = False  # Allow retraining
        ml_filter.trades_since_retrain = ONLINE_RETRAIN_EVERY  # Force retrain
        ml_filter._train()
        ml_filter._pretrained = True  # Re-lock
        ml_filter.save()
        online_new_labels = 0
        log.info(f"  ✅  ML model retrained and saved!")

    save_online_samples()

load_online_samples()

if not ml_filter.is_trained:
    log.warning("⚠  ML model not trained — will accept all signals")

daily_pnl  = 0.0
daily_date = date.today()
consecutive_losses = 0  # Track consecutive losses for circuit breaker
loss_cooldown_until = None  # Timestamp when cooldown expires

# ─────────────────────────────────────────────────────────────────────

def get_balance() -> float:
    try:
        resp = client.get_available_funds()
        return float(resp.balance)
    except Exception:
        return 0.0

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

def round_price(price: float, symbol: str = None) -> float:
    """Round price to valid tick size using Bybit instrument info."""
    tick = 0.0001  # fallback
    if symbol and symbol in INSTRUMENT_INFO:
        tick = INSTRUMENT_INFO[symbol]["tickSize"]
    else:
        # Fallback heuristic if instrument info missing
        if price >= 1000:
            tick = 0.10
        elif price >= 100:
            tick = 0.01
        elif price >= 1:
            tick = 0.001
        elif price >= 0.1:
            tick = 0.0001
        else:
            tick = 0.00001
    return round(round(price / tick) * tick, 8)

def round_qty(qty: float, symbol: str = None) -> float:
    """Round quantity to valid step size using Bybit instrument info."""
    step = 0.1  # fallback
    if symbol and symbol in INSTRUMENT_INFO:
        step = INSTRUMENT_INFO[symbol]["qtyStep"]
    import math
    return round(math.floor(qty / step) * step, 8)

def open_trade(symbol: str, signal: str, price: float, sl: float, tp: float, qty: float, lev: int) -> bool:
    try:
        sl = round_price(sl, symbol)
        tp = round_price(tp, symbol)
        qty = round_qty(qty, symbol)

        # Cap SL so it doesn't exceed liquidation distance (80% of max margin)
        max_sl_pct = (1.0 / lev) * 0.75  # 75% of margin to stay above liq price
        if signal == "LONG":
            min_sl = price * (1 - max_sl_pct)
            if sl < min_sl:
                log.info(f"   📐 SL capped: {sl:.6f} → {min_sl:.6f} (lev={lev}x limit)")
                sl = round_price(min_sl, symbol)
        elif signal == "SHORT":
            max_sl = price * (1 + max_sl_pct)
            if sl > max_sl:
                log.info(f"   📐 SL capped: {sl:.6f} → {max_sl:.6f} (lev={lev}x limit)")
                sl = round_price(max_sl, symbol)
        log.info(f"   📐 ORDER DEBUG: {symbol} {signal} price={price} sl={sl} tp={tp} qty={qty} lev={lev}")
        # Validate SL direction
        if signal == "LONG" and sl >= price:
            log.error(f"   ⚠ INVALID: LONG SL ({sl}) >= price ({price})")
            return False
        if signal == "SHORT" and sl <= price:
            log.error(f"   ⚠ INVALID: SHORT SL ({sl}) <= price ({price})")
            return False
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

# ── Trailing Stop State ──────────────────────────────────────────────
# Tracks entry prices and ATR for trailing stop calculation
trailing_state = {}  # { symbol: { entry_price, atr, direction, current_sl } }

def manage_trailing_stop(symbol: str, position: dict, current_price: float):
    """
    Moves SL to breakeven after 1.5R profit, then trails at 1.5 ATR.
    Updates the exchange SL if the new SL is better than the current one.
    """
    global trailing_state
    
    if symbol not in trailing_state:
        return  # No trailing info yet
    
    state = trailing_state[symbol]
    entry = state["entry_price"]
    atr_v = state["atr"]
    direction = state["direction"]
    current_sl = state["current_sl"]
    
    sl_dist = atr_v * c.SL_ATR_MULT
    activation_dist = sl_dist * c.TRAILING_ACTIVATE_R  # 1.5R
    trail_dist = atr_v * c.TRAILING_DISTANCE_ATR
    
    new_sl = None
    
    if direction == "LONG":
        profit_dist = current_price - entry
        if profit_dist >= activation_dist:
            # Trail SL at 1.5 ATR below current price
            candidate_sl = current_price - trail_dist
            if candidate_sl > current_sl:
                new_sl = candidate_sl
    elif direction == "SHORT":
        profit_dist = entry - current_price
        if profit_dist >= activation_dist:
            # Trail SL at 1.5 ATR above current price
            candidate_sl = current_price + trail_dist
            if candidate_sl < current_sl:
                new_sl = candidate_sl
    
    if new_sl:
        try:
            pos_id = position.get("id") if isinstance(position, dict) else getattr(position, "id", None)
            if not pos_id:
                log.warning(f"  No position ID for {symbol}, cannot trail SL")
                return
            new_sl = round_price(new_sl, symbol)
            client.amend_risk_order(
                position_id=pos_id,
                is_stoploss=True,
                stoploss_price=str(new_sl)
            )
            trailing_state[symbol]["current_sl"] = new_sl
            log.info(f"📈  Trailing SL moved → {symbol} new SL: ${new_sl:.5f} (pos={pos_id[:8]})")
        except Exception as e:
            log.warning(f"  Could not update trailing SL for {symbol}: {e}")

# ─────────────────────────────────────────────────────────────────────

def run():
    global daily_pnl, daily_date, consecutive_losses, loss_cooldown_until, online_pending

    balance = get_balance()
    set_leverage()

    print("\n" + "═"*52)
    print("  🤖  MUDREX ALGO BOT  —  LIVE TRADING MODE")
    print("═"*52)
    print(f"  Symbols   : {', '.join(c.SYMBOLS)}")
    print(f"  Timeframe : {c.PRIMARY_TF} (HTF: {c.HTF_TF})")
    print(f"  Balance   : ${balance:.4f} USDT")
    print(f"  Leverage  : per-symbol (5x/10x)")
    print(f"  Risk/Trade: {c.RISK_PER_TRADE*100:.1f}% of balance")
    print(f"  R:R Ratio : 1:{c.TP_ATR_MULT/c.SL_ATR_MULT:.1f}")
    print(f"  Strategy  : ADX+EMA+MACD trend following")
    print("═"*52 + "\n")

    loop_count = 0
    prev_balance = balance  # Track balance changes to detect wins/losses
    had_position = False    # Track if we had an open position last loop

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
                if resp is not None:
                    raw_positions = resp if isinstance(resp, list) else getattr(resp, "result", None) or []
                    if raw_positions:
                        all_positions = [p for p in raw_positions if float(p.get("quantity", 0) if isinstance(p, dict) else getattr(p, "quantity", 0)) != 0]
            except Exception as e:
                log.warning(f"  Could not fetch positions: {e}")
                all_positions = []

            open_count = len(all_positions)

            # ── Detect position close & track wins/losses ────────────
            if had_position and open_count == 0:
                # A position just closed — check if it was a win or loss
                pnl_change = balance - prev_balance
                if pnl_change > 0:
                    consecutive_losses = 0  # Reset on win
                    log.info(f"[{now}]  ✅ Trade closed PROFIT +${pnl_change:.2f} | Streak reset")
                elif pnl_change < 0:
                    consecutive_losses += 1
                    daily_pnl += pnl_change
                    log.info(f"[{now}]  ❌ Trade closed LOSS ${pnl_change:.2f} | Consecutive: {consecutive_losses}")
                prev_balance = balance
                # Clean up trailing state
                trailing_state.clear()
            
            had_position = open_count > 0

            if open_count > 0:
                for op in all_positions:
                    sym = op.get("symbol") if isinstance(op, dict) else getattr(op, "symbol", "?")
                    qty = op.get("quantity", 0) if isinstance(op, dict) else getattr(op, "quantity", 0)
                    log.info(f"[{now}]  📊 Holding {sym}  qty={qty}  |  Bal: ${balance:.4f}")
                    
                    # ── Trailing stop management ──────────────────────
                    try:
                        current_price = get_current_price(sym)
                        manage_trailing_stop(sym, op, current_price)
                    except Exception as e:
                        log.warning(f"  Trailing stop check failed for {sym}: {e}")

            if open_count >= c.MAX_POSITIONS:
                time.sleep(c.LOOP_INTERVAL)
                continue

            # ── Consecutive loss circuit breaker ─────────────────────
            if consecutive_losses >= c.MAX_CONSECUTIVE_LOSSES:
                if loss_cooldown_until is None:
                    loss_cooldown_until = datetime.now()
                    log.warning(f"⛔  {consecutive_losses} consecutive losses — cooling down for {c.LOSS_COOLDOWN_CANDLES} candles")
                
                # Wait LOSS_COOLDOWN_CANDLES × 15min = 4 hours
                cooldown_seconds = c.LOSS_COOLDOWN_CANDLES * 60 * 15  # 15m candles
                elapsed = (datetime.now() - loss_cooldown_until).total_seconds()
                if elapsed < cooldown_seconds:
                    time.sleep(c.LOOP_INTERVAL)
                    continue
                else:
                    log.info("✅  Cooldown expired — resuming trading")
                    consecutive_losses = 0
                    loss_cooldown_until = None

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
                    if df is None or df.empty or len(df) < 3:
                        log.info(f"         {symbol}: not enough candles")
                        continue
                    df = compute_all(df)

                    # Fetch higher-timeframe data for trend confirmation
                    htf_bias = None
                    try:
                        htf_df = get_ohlcv(symbol, c.HTF_TF, limit=100)
                        if htf_df is not None and not htf_df.empty:
                            htf_df = compute_all(htf_df)
                            if len(htf_df) >= 20:
                                last_htf = htf_df.iloc[-1]
                                # Trend: EMA20 > EMA50 and close > EMA20
                                ema20_htf = float(last_htf.get("ema_med", last_htf.get("ema21", 0)))
                                ema50_htf = float(last_htf.get("ema_slow", last_htf.get("ema50", 0)))
                                close_htf = float(last_htf["close"])
                                # LONG: price above both EMAs OR classic trend (EMA20>EMA50 + price>EMA20)
                                if (ema20_htf > ema50_htf and close_htf > ema20_htf) or                                    (close_htf > ema20_htf and close_htf > ema50_htf):
                                    htf_bias = "LONG"
                                # SHORT: price below both EMAs OR classic trend
                                elif (ema20_htf < ema50_htf and close_htf < ema20_htf) or                                      (close_htf < ema20_htf and close_htf < ema50_htf):
                                    htf_bias = "SHORT"
                    except Exception:
                        pass  # HTF is optional — proceed without it

                    signal, sl, tp = generate_signal(df, htf_bias)
                    price          = get_current_price(symbol)

                    log.info(f"         {symbol}: ${price:.5f}  Signal: {signal}")

                    # ── Online Learning: store sample for ALL signals ────
                    try:
                        atr_live = float(df.iloc[-1]["atr"])
                        for direction_sample in ["LONG", "SHORT"]:
                            if ml_filter.version in ["v3_regime_ensemble", "v4_direction_ensemble"]:
                                df_regime = compute_regimes(df.copy())
                                feats = extract_regime_features(df_regime, len(df_regime) - 1, direction_sample)
                                if feats is None:
                                    continue
                            else:
                                curr_row = df.iloc[-1]
                                prev_row = df.iloc[-2]
                                df_slice = df.iloc[-15:] if len(df) >= 15 else df
                                feats = extract_features_extended(
                                    curr_row, prev_row, price, atr_live,
                                    direction_sample, df_slice=df_slice
                                )
                            online_pending.append({
                                "symbol": symbol,
                                "direction": direction_sample,
                                "entry_price": float(price),
                                "atr": float(atr_live),
                                "features": feats,
                                "bars_elapsed": 0,
                                "timestamp": datetime.now().isoformat(),
                            })
                    except Exception:
                        pass

                    if signal != Signal.NONE:
                        # ── ML Filter Gate ──────────────────────────────────
                        atr_v = float(df.iloc[-1]["atr"])  # Use actual ATR from indicators
                        ml_conf = 0.5
                        ml_passed = True

                        if ml_filter.is_trained:
                            try:
                                if ml_filter.version in ["v3_regime_ensemble", "v4_direction_ensemble"]:
                                    # V3/V4: compute regimes and use deep regime features
                                    df_regime = compute_regimes(df.copy())
                                    direction_str = "LONG" if signal == Signal.LONG else "SHORT"
                                    ml_features = extract_regime_features(df_regime, len(df_regime) - 1, direction_str)
                                    if ml_features is None:
                                        ml_features = {}  # fallback - will be padded with zeros
                                else:
                                    # V2: use extended features
                                    curr_row = df.iloc[-1]
                                    prev_row = df.iloc[-2]
                                    df_slice = df.iloc[-15:] if len(df) >= 15 else df
                                    ml_features = extract_features_extended(
                                        curr_row, prev_row, price, atr_v, signal, df_slice=df_slice
                                    )
                                should_take, ml_conf, ml_sl_mult, ml_tp_r = ml_filter.should_take_trade(ml_features)

                                if not should_take:
                                    log.info(f"         {symbol}: ❌ ML BLOCKED (conf={ml_conf:.0%} < threshold)")
                                    ml_passed = False
                                else:
                                    # Use ML-suggested SL/TP
                                    sl_dist = atr_v * ml_sl_mult
                                    if signal == Signal.LONG:
                                        sl = price - sl_dist
                                        tp = price + sl_dist * ml_tp_r
                                    else:
                                        sl = price + sl_dist
                                        tp = price - sl_dist * ml_tp_r
                                    log.info(f"         {symbol}: ✅ ML PASS (conf={ml_conf:.0%}) SL={ml_sl_mult:.1f}ATR TP={ml_tp_r:.1f}R")
                            except Exception as e:
                                log.warning(f"         {symbol}: ML error ({e}) — using strategy SL/TP")

                        if not ml_passed:
                            continue

                        qty = position_size(balance, price, sl, symbol)
                        if qty > 0:
                            lev    = c.SYMBOL_LEVERAGE.get(symbol, c.LEVERAGE)
                            opened = open_trade(symbol, signal, price, sl, tp, qty, lev)
                            if opened:
                                trades_opened += 1
                                prev_balance = balance
                                # Store trailing stop state
                                trailing_state[symbol] = {
                                    "entry_price": price,
                                    "atr": atr_v,
                                    "direction": signal,
                                    "current_sl": sl
                                }
                        else:
                            log.warning(f"         {symbol}: qty=0, balance too low?")

                except Exception as e:
                    log.error(f"         {symbol}: error → {e}")
                    continue

            # ── Online Learning: process pending samples ────────
            if loop_count % 5 == 0:  # Every 5 loops (~10 min)
                try:
                    process_online_learning()
                except Exception as e:
                    log.warning(f"  Online learning error: {e}")

            # Cap pending samples to avoid memory bloat (keep last 2000)
            if len(online_pending) > 2000:
                online_pending = online_pending[-2000:]
                save_online_samples()

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
