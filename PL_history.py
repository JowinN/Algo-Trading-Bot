import sys, os
sys.path.insert(0, "/home/jowin/mudrex-bot")
os.chdir("/home/jowin/mudrex-bot")
from dotenv import load_dotenv
load_dotenv("/home/jowin/mudrex-bot/.env")
from mudrex import TradeClient
client = TradeClient(api_secret=os.getenv("MUDREX_API_SECRET"))

resp   = client.get_order_history()
trades = resp if isinstance(resp, list) else getattr(resp, "result", None) or getattr(resp, "orders", None) or []

from collections import defaultdict
positions = defaultdict(list)
for t in trades:
    def g(k): return t.get(k,"") if isinstance(t, dict) else getattr(t, k, "")
    positions[g("future_position_uuid")].append({
        "symbol"      : g("symbol"),
        "order_type"  : g("order_type"),
        "qty"         : float(g("filled_quantity") or 0),
        "filled_price": float(g("filled_price") or 0),
        "leverage"    : float(g("leverage") or 1),
        "created_at"  : g("created_at"),
    })

EXIT_TYPES = {"TAKEPROFIT", "STOPLOSS", "SHORT"}
results = []
total_pnl = 0.0

for pid, orders in positions.items():
    entry = next((o for o in orders if o["order_type"] == "LONG"), None)
    exit_ = next((o for o in orders if o["order_type"] in EXIT_TYPES and o["order_type"] != "LONG"), None)

    if not entry:
        # SHORT entry
        entry = next((o for o in orders if o["order_type"] == "SHORT"), None)
        exit_ = next((o for o in orders if o["order_type"] not in ("SHORT","LONG")), None)
        if entry and exit_:
            pnl = (entry["filled_price"] - exit_["filled_price"]) * entry["qty"]
        elif entry:
            pnl = None
        else:
            continue
    else:
        if exit_:
            pnl = (exit_["filled_price"] - entry["filled_price"]) * entry["qty"]
        else:
            pnl = None

    lev    = entry["leverage"]
    margin = (entry["filled_price"] * entry["qty"] / lev) if lev > 0 else 1
    pnl_pct = (pnl / margin * 100) if pnl is not None and margin else None

    results.append({
        "symbol"     : entry["symbol"],
        "side"       : entry["order_type"],
        "qty"        : entry["qty"],
        "entry_price": entry["filled_price"],
        "exit_price" : exit_["filled_price"] if exit_ else None,
        "exit_type"  : exit_["order_type"]   if exit_ else "OPEN",
        "lev"        : lev,
        "pnl"        : pnl,
        "pnl_pct"    : pnl_pct,
        "time"       : entry["created_at"],
    })
    if pnl: total_pnl += pnl

results.sort(key=lambda x: x["time"], reverse=True)

print(f"{'Symbol':<12} {'Side':<6} {'Qty':<6} {'Entry':<10} {'Exit':<10} {'Exit Type':<12} {'Lev':<4} {'P&L':>10} {'P&L%':>8}  Time")
print("─"*100)
for r in results:
    pnl_str     = (f"+${r['pnl']:.4f}" if r['pnl'] >= 0 else f"-${abs(r['pnl']):.4f}") if r['pnl'] is not None else "OPEN"
    pnl_pct_str = (f"+{r['pnl_pct']:.2f}%" if r['pnl_pct'] >= 0 else f"{r['pnl_pct']:.2f}%") if r['pnl_pct'] is not None else ""
    print(f"  {r['symbol']:<12} {r['side']:<6} {r['qty']:<6} {r['entry_price']:<10} {str(r['exit_price'] or ''):<10} {r['exit_type']:<12} {int(r['lev']):<4} {pnl_str:>10} {pnl_pct_str:>8}  {r['time'][:19]}")

print("─"*100)
sign = "+" if total_pnl >= 0 else ""
print(f"  {'TOTAL P&L':>50}  {sign}${total_pnl:.4f}")
