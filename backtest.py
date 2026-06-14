import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
from data import INTERVAL_MAP
from indicators import compute_all
from strategy import generate_signal, Signal
from risk import position_size
from config import Config as c
import json

BYBIT_BASE = "https://api.bybit.com/v5/market"

class BacktestEngine:
    """
    Backtests the algo trading bot on 1 month of historical data.
    """
    
    def __init__(self, symbols=None, start_date=None, balance=1000, days=30):
        self.symbols = symbols or c.SYMBOLS
        self.balance = balance
        self.days = days
        self.start_date = start_date or (datetime.now() - timedelta(days=days))
        self.trades = []
        self.equity_curve = []
        
    def fetch_historical_data(self, symbol, interval="15m", days=30):
        """
        Fetch 1 month of historical 15m candles from Bybit.
        Returns dataframe sorted by timestamp.
        """
        bybit_interval = INTERVAL_MAP.get(interval, "15")
        all_candles = []
        current_date = datetime.now()
        
        # Bybit API limits to ~200 candles per request
        # We need to fetch multiple batches
        while len(all_candles) < (days * 24 * 60 // 15):  # 15m candles in 1 month
            try:
                url = f"{BYBIT_BASE}/kline"
                params = {
                    "category": "linear",
                    "symbol": symbol,
                    "interval": bybit_interval,
                    "limit": 200,
                    "end": int(current_date.timestamp() * 1000)
                }
                resp = requests.get(url, params=params, timeout=10)
                resp.raise_for_status()
                
                raw = resp.json()
                if raw.get("retCode") != 0:
                    print(f"Bybit error: {raw.get('retMsg')}")
                    break
                
                candles = raw["result"]["list"]
                if not candles:
                    break
                
                all_candles.extend(candles)
                
                # Move back to the timestamp of the oldest candle
                oldest_timestamp = float(candles[-1][0])
                current_date = datetime.fromtimestamp(oldest_timestamp / 1000)
                
                # Stop if we've gone back days+ days
                if (datetime.now() - current_date).days >= days:
                    break
                    
            except Exception as e:
                print(f"Error fetching {symbol}: {e}")
                break
        
        # Convert to DataFrame
        df = pd.DataFrame(all_candles, columns=[
            "timestamp", "open", "high", "low", "close", "volume", "turnover"
        ])
        df["timestamp"] = pd.to_datetime(df["timestamp"].astype(float), unit="ms")
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = df[col].astype(float)
        
        df = df.sort_values("timestamp").reset_index(drop=True)
        return df[["timestamp", "open", "high", "low", "close", "volume"]]
    
    def run_backtest(self):
        """
        Simulate the bot trading across 1 month of data.
        """
        print(f"\n{'='*70}")
        print(f"  BACKTESTING ALGO BOT — {self.days} DAYS")
        print(f"  Start Balance: ${self.balance:.2f}")
        print(f"  Symbols: {len(self.symbols)}")
        print(f"{'='*70}\n")
        
        # Fetch data for all symbols
        print("📥 Fetching historical data from Bybit...")
        all_data = {}
        for symbol in self.symbols:
            print(f"   {symbol}...", end=" ", flush=True)
            try:
                df = self.fetch_historical_data(symbol, days=self.days)
                all_data[symbol] = df
                print(f"✓ ({len(df)} candles)")
            except Exception as e:
                print(f"✗ {e}")
        
        if not all_data:
            print("❌ No data fetched. Aborting.")
            return
        
        # Find the date range (intersection of all symbols)
        min_date = max([df["timestamp"].min() for df in all_data.values()])
        max_date = min([df["timestamp"].max() for df in all_data.values()])
        
        print(f"\n📊 Backtest period: {min_date.date()} to {max_date.date()}")
        print(f"   Duration: {(max_date - min_date).days} days\n")
        
        # Align all dataframes to same date range
        for symbol in all_data:
            df = all_data[symbol]
            all_data[symbol] = df[(df["timestamp"] >= min_date) & (df["timestamp"] <= max_date)].reset_index(drop=True)
        
        # Simulate the bot logic
        current_balance = self.balance
        open_position = None  # { symbol, entry_price, qty, sl, tp, entry_time }
        daily_pnl = 0.0
        last_daily_reset = min_date.date()
        last_trade_time = None
        
        # Get all unique timestamps across all symbols
        all_timestamps = sorted(set([ts for df in all_data.values() for ts in df["timestamp"]]))
        
        for ts in all_timestamps[100:]:  # Skip first 100 candles for indicator warmup
            current_date = ts.date()
            
            # ── Daily reset ────────────────────────────────────────────
            if current_date != last_daily_reset:
                daily_pnl = 0.0
                last_daily_reset = current_date
            
            # ── Daily loss guard ───────────────────────────────────────
            if not self._daily_limit_ok(daily_pnl, current_balance):
                continue
            
            # ── Close position if SL/TP hit ───────────────────────────
            if open_position:
                sym = open_position["symbol"]
                df = all_data[sym]
                candle = df[df["timestamp"] == ts]
                
                if candle.empty:
                    continue
                
                high = float(candle["high"].iloc[0])
                low = float(candle["low"].iloc[0])
                close = float(candle["close"].iloc[0])
                
                exit_price = None
                exit_reason = None
                
                # Check stoploss
                if open_position["direction"] == Signal.LONG:
                    if low <= open_position["sl"]:
                        exit_price = open_position["sl"]
                        exit_reason = "STOPLOSS"
                    elif high >= open_position["tp"]:
                        exit_price = open_position["tp"]
                        exit_reason = "TAKEPROFIT"
                elif open_position["direction"] == Signal.SHORT:
                    if high >= open_position["sl"]:
                        exit_price = open_position["sl"]
                        exit_reason = "STOPLOSS"
                    elif low <= open_position["tp"]:
                        exit_price = open_position["tp"]
                        exit_reason = "TAKEPROFIT"
                
                if exit_price:
                    # Calculate P&L
                    if open_position["direction"] == Signal.LONG:
                        pnl = (exit_price - open_position["entry_price"]) * open_position["qty"]
                    else:
                        pnl = (open_position["entry_price"] - exit_price) * open_position["qty"]
                    
                    current_balance += pnl
                    daily_pnl += pnl
                    
                    self.trades.append({
                        "symbol": sym,
                        "direction": open_position["direction"],
                        "entry_time": open_position["entry_time"],
                        "entry_price": open_position["entry_price"],
                        "exit_time": ts,
                        "exit_price": exit_price,
                        "qty": open_position["qty"],
                        "pnl": pnl,
                        "pnl_pct": (pnl / (open_position["entry_price"] * open_position["qty"])) * 100,
                        "reason": exit_reason
                    })
                    
                    print(f"[{ts}] 🔄 CLOSED {sym} {open_position['direction']} | "
                          f"Entry: ${open_position['entry_price']:.5f} → Exit: ${exit_price:.5f} | "
                          f"P&L: ${pnl:+.2f} ({exit_reason})")
                    
                    open_position = None
                    last_trade_time = ts
            
            # ── Check for new entries ──────────────────────────────────
            if open_position is None:
                # Check minimum candles between trades (cooldown)
                if last_trade_time is not None:
                    candles_since = len([t for df in all_data.values() if (t > last_trade_time) and (t <= ts) for t in df["timestamp"]])
                    if candles_since < c.MIN_CANDLES_SINCE_TRADE:
                        continue
                
                for symbol in self.symbols:
                    if symbol not in all_data:
                        continue
                    
                    df = all_data[symbol]
                    candle_idx = df[df["timestamp"] == ts].index
                    
                    if candle_idx.empty:
                        continue
                    
                    idx = candle_idx[0]
                    if idx < 50:  # Need at least 50 candles for indicators
                        continue
                    
                    # Get last 100 candles up to this point (better for 200 EMA)
                    hist_df = df.iloc[max(0, idx-100):idx+1].copy()
                    
                    try:
                        # Compute indicators and generate signal
                        signal_df = compute_all(hist_df)
                        if signal_df.empty:
                            continue
                        
                        signal, sl, tp = generate_signal(signal_df)
                        
                        if signal != Signal.NONE:
                            entry_price = float(signal_df.iloc[-1]["close"])
                            qty = position_size(current_balance, entry_price, sl, symbol)
                            
                            if qty > 0:
                                lev = c.SYMBOL_LEVERAGE.get(symbol, c.LEVERAGE)
                                
                                open_position = {
                                    "symbol": symbol,
                                    "direction": signal,
                                    "entry_price": entry_price,
                                    "qty": qty,
                                    "sl": sl,
                                    "tp": tp,
                                    "entry_time": ts,
                                    "leverage": lev
                                }
                                
                                print(f"[{ts}] 🟢 OPENED {symbol} {signal} | "
                                      f"Price: ${entry_price:.5f} | Qty: {qty:.2f} | "
                                      f"SL: ${sl:.5f} | TP: ${tp:.5f} | Lev: {lev}x")
                                break  # Only 1 position at a time
                    
                    except Exception as e:
                        pass  # Skip symbols with errors
            
            # ── Record equity ──────────────────────────────────────────
            self.equity_curve.append({
                "timestamp": ts,
                "balance": current_balance,
                "open_position": open_position["symbol"] if open_position else None
            })
        
        # ── Close any remaining open position ──────────────────────────
        if open_position:
            last_price = float(all_data[open_position["symbol"]].iloc[-1]["close"])
            pnl = (last_price - open_position["entry_price"]) * open_position["qty"]
            current_balance += pnl
            self.trades.append({
                "symbol": open_position["symbol"],
                "direction": open_position["direction"],
                "entry_time": open_position["entry_time"],
                "entry_price": open_position["entry_price"],
                "exit_time": max_date,
                "exit_price": last_price,
                "qty": open_position["qty"],
                "pnl": pnl,
                "pnl_pct": (pnl / (open_position["entry_price"] * open_position["qty"])) * 100,
                "reason": "BACKTEST_END"
            })
        
        self.print_results(current_balance, self.balance)
    
    def _daily_limit_ok(self, daily_pnl: float, balance: float) -> bool:
        """Check daily loss limit"""
        if balance == 0:
            return True
        return (daily_pnl / balance) > -c.DAILY_LOSS_LIMIT
    
    def print_results(self, final_balance, initial_balance):
        """
        Print backtest summary statistics.
        """
        if not self.trades:
            print("\n❌ No trades executed during backtest.")
            return
        
        df_trades = pd.DataFrame(self.trades)
        
        total_pnl = final_balance - initial_balance
        total_return_pct = (total_pnl / initial_balance) * 100
        
        winning_trades = df_trades[df_trades["pnl"] > 0]
        losing_trades = df_trades[df_trades["pnl"] < 0]
        
        win_rate = (len(winning_trades) / len(df_trades)) * 100 if len(df_trades) > 0 else 0
        avg_win = winning_trades["pnl"].mean() if len(winning_trades) > 0 else 0
        avg_loss = abs(losing_trades["pnl"].mean()) if len(losing_trades) > 0 else 0
        profit_factor = (winning_trades["pnl"].sum() / abs(losing_trades["pnl"].sum())) if len(losing_trades) > 0 and losing_trades["pnl"].sum() != 0 else 0
        
        equity_df = pd.DataFrame(self.equity_curve)
        max_balance = equity_df["balance"].max()
        max_drawdown = ((max_balance - equity_df["balance"].min()) / max_balance) * 100 if max_balance > 0 else 0
        
        print(f"\n{'='*70}")
        print(f"  BACKTEST RESULTS — {self.days} DAYS")
        print(f"{'='*70}")
        print(f"  Initial Balance:     ${initial_balance:.2f}")
        print(f"  Final Balance:       ${final_balance:.2f}")
        print(f"  Total P&L:           ${total_pnl:+.2f}")
        print(f"  Total Return:        {total_return_pct:+.2f}%")
        print(f"\n  Total Trades:        {len(df_trades)}")
        print(f"  Winning Trades:      {len(winning_trades)} ({win_rate:.1f}%)")
        print(f"  Losing Trades:       {len(losing_trades)} ({100-win_rate:.1f}%)")
        print(f"\n  Avg Win:             ${avg_win:+.2f}")
        print(f"  Avg Loss:            ${avg_loss:+.2f}")
        print(f"  Profit Factor:       {profit_factor:.2f} (>1.5 is excellent)")
        print(f"  Max Drawdown:        {max_drawdown:.2f}%")
        print(f"{'='*70}\n")
        
        # Show trades table
        print("📋 All Trades:")
        print(df_trades[["symbol", "direction", "entry_price", "exit_price", "pnl", "pnl_pct", "reason"]].to_string())
        
        # Save results to CSV
        df_trades.to_csv("backtest_trades.csv", index=False)
        equity_df.to_csv("backtest_equity_curve.csv", index=False)
        print("\n✅ Results saved to backtest_trades.csv and backtest_equity_curve.csv")

if __name__ == "__main__":
    # Run 30-day backtest
    backtester = BacktestEngine(balance=1000, days=30)
    backtester.run_backtest()
