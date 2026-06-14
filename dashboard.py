import os
import time
import threading
from datetime import datetime
from flask import Flask, jsonify, render_template_string
from dotenv import load_dotenv
from mudrex import TradeClient

from config     import Config as c
from data       import get_ohlcv, get_current_price
from indicators import compute_all, detect_candle_patterns

load_dotenv()

app    = Flask(__name__)
client = TradeClient(api_secret=os.getenv("MUDREX_API_SECRET"))

state = {
    "balance"      : 0.0,
    "open_positions": [],
    "indicators"   : [],
    "last_updated" : "-",
}

def fetch_balance():
    try:
        resp = client.get_available_funds()
        return float(resp.balance)
    except:
        return 0.0

def fetch_position():
    try:
        resp      = client.get_positions()
        positions = resp if isinstance(resp, list) else getattr(resp, "result", None)
        if not positions:
            return []
        open_pos = []
        for pos in positions:
            status = pos.get("status") if isinstance(pos, dict) else getattr(pos, "status", "")
            if str(status).upper() == "OPEN":
                open_pos.append(pos)
        return open_pos
    except:
        return []

PATTERN_PRIORITY = [
    "Three White Soldiers", "Three Black Crows",
    "Morning Star",         "Evening Star",
    "Bullish Engulfing",    "Bearish Engulfing",
    "Pinbar Bullish",       "Pinbar Bearish",
    "Hammer",               "Shooting Star",
    "Inverted Hammer",      "Doji",
]

def fetch_indicators():
    results = []
    for symbol in c.SYMBOLS:
        try:
            df    = get_ohlcv(symbol, c.PRIMARY_TF, limit=100)
            df    = compute_all(df)
            curr  = df.iloc[-1]
            prev  = df.iloc[-2]
            prev2 = df.iloc[-3]
            price = get_current_price(symbol)

            spread = abs(curr["ema_fast"] - curr["ema_slow"]) / curr["ema_slow"] * 100
            regime = "TREND" if spread > 0.1 else "RANGE"

            bullish_candle  = bool(curr["close"] > curr["open"])
            bearish_candle  = bool(curr["close"] < curr["open"])
            pullback_long   = bool(prev["close"] < prev["open"] and bullish_candle)
            pullback_short  = bool(prev["close"] > prev["open"] and bearish_candle)
            macd_growing_up = bool(curr["macd_hist"] > prev["macd_hist"] > prev2["macd_hist"])
            macd_growing_dn = bool(curr["macd_hist"] < prev["macd_hist"] < prev2["macd_hist"])
            rsi_rising      = bool(curr["rsi"] > prev["rsi"])
            rsi_falling     = bool(curr["rsi"] < prev["rsi"])
            recent_high     = float(df["high"].iloc[-6:-1].max())
            recent_low      = float(df["low"].iloc[-6:-1].min())
            breakout_up     = bool(curr["close"] > recent_high)
            breakout_dn     = bool(curr["close"] < recent_low)

            signals = {
                "A: STD LONG": [
                    curr["ema_fast"] > curr["ema_med"],
                    curr["ema_med"]  > curr["ema_slow"],
                    c.RSI_OVERSOLD < curr["rsi"] < c.RSI_OVERBOUGHT,
                    curr["macd_hist"] > 0,
                    curr["close"] > curr["bb_mid"],
                    curr["volume"] > curr["vol_ma"],
                    bullish_candle,
                    rsi_rising,
                    macd_growing_up,
                    pullback_long or breakout_up,
                ],
                "B: STD SHORT": [
                    curr["ema_fast"] < curr["ema_med"],
                    curr["ema_med"]  < curr["ema_slow"],
                    c.RSI_OVERSOLD < curr["rsi"] < c.RSI_OVERBOUGHT,
                    curr["macd_hist"] < 0,
                    curr["close"] < curr["bb_mid"],
                    curr["volume"] > curr["vol_ma"],
                    bearish_candle,
                    rsi_falling,
                    macd_growing_dn,
                    pullback_short or breakout_dn,
                ],
                "C: DT SHORT": [
                    curr["ema_fast"] < curr["ema_med"],
                    curr["ema_med"]  < curr["ema_slow"],
                    c.RSI_EXTREME_LOW < curr["rsi"] < c.RSI_OVERSOLD,
                    curr["macd_hist"] < 0,
                    curr["close"] < curr["bb_mid"],
                    macd_growing_dn,
                    bearish_candle,
                ],
                "D: UT LONG": [
                    curr["ema_fast"] > curr["ema_med"],
                    curr["ema_med"]  > curr["ema_slow"],
                    c.RSI_OVERBOUGHT < curr["rsi"] < c.RSI_EXTREME_HIGH,
                    curr["macd_hist"] > 0,
                    curr["close"] > curr["bb_mid"],
                    macd_growing_up,
                    bullish_candle,
                ],
            }

            sig_scores = {}
            for sname, conds in signals.items():
                passed = sum(bool(v) for v in conds)
                total  = len(conds)
                sig_scores[sname] = {"passed": passed, "total": total, "fire": passed == total}

            # ── Candle pattern detection ───────────────────────────
            df_htf   = get_ohlcv(symbol, c.PATTERN_TF, limit=50)
            df_htf   = compute_all(df_htf)
            patterns = detect_candle_patterns(df_htf)
            top_pattern  = next((p for p in PATTERN_PRIORITY if p in patterns), None)
            pattern_type = patterns.get(top_pattern, None) if top_pattern else None

            results.append({
                "symbol"      : symbol,
                "price"       : round(price, 5),
                "regime"      : regime,
                "spread"      : round(spread, 3),
                "rsi"         : round(float(curr["rsi"]), 1),
                "macd"        : round(float(curr["macd_hist"]), 6),
                "atr"         : round(float(curr["atr"]), 6),
                "vol"         : int(curr["volume"]),
                "vol_ma"      : int(curr["vol_ma"]),
                "bb_mid"      : round(float(curr["bb_mid"]), 5),
                "ema_fast"    : round(float(curr["ema_fast"]), 5),
                "ema_slow"    : round(float(curr["ema_slow"]), 5),
                "signals"     : sig_scores,
                "lev"         : c.SYMBOL_LEVERAGE.get(symbol, c.LEVERAGE),
                "pattern"     : top_pattern,
                "pattern_type": pattern_type,
            })
            time.sleep(0.2)
        except Exception as e:
            results.append({"symbol": symbol, "error": str(e)})
    return results

def background_refresh():
    while True:
        try:
            state["balance"]       = fetch_balance()
            state["open_positions"] = fetch_position()
            state["indicators"]    = fetch_indicators()
            state["last_updated"]  = datetime.now().strftime("%H:%M:%S")
        except Exception as e:
            print(f"Dashboard refresh error: {e}")
        time.sleep(60)

@app.route("/")
def index():
    return render_template_string(HTML)

@app.route("/api/state")
def api_state():
    positions = state.get("open_positions", [])
    pos_data  = []
    for pos in positions:
        def g(d, k, fallback):
            return d.get(k, fallback) if isinstance(d, dict) else getattr(d, k, fallback)
        def gf(d, k):
            v = g(d, k, 0)
            try: return float(v)
            except: return 0.0
        def gs(d, k):
            raw = g(d, k, {})
            return raw if isinstance(raw, dict) else vars(raw) if hasattr(raw, "__dict__") else {}

        sym   = g(pos, "symbol", "?")
        qty   = gf(pos, "quantity")
        side  = g(pos, "order_type", "?")
        entry = gf(pos, "entry_price")
        lev   = int(gf(pos, "leverage") or 1)
        sl    = float(gs(pos, "stoploss") .get("price", 0) or 0)
        tp    = float(gs(pos, "takeprofit").get("price", 0) or 0)
        try:
            curr_price = get_current_price(sym)
        except:
            curr_price = entry
        if str(side).upper() == "LONG":
            pnl = (curr_price - entry) * qty
        else:
            pnl = (entry - curr_price) * qty
        margin  = (entry * qty / lev) if lev > 0 and entry > 0 else 1
        pnl_pct = (pnl / margin) * 100 if margin else 0
        pos_data.append({
            "symbol" : sym,
            "qty"    : qty,
            "side"   : side,
            "entry"  : round(entry, 6),
            "price"  : round(curr_price, 6),
            "pnl"    : round(pnl, 4),
            "pnl_pct": round(pnl_pct, 2),
            "lev"    : lev,
            "sl"     : round(sl, 6),
            "tp"     : round(tp, 6),
        })
    return jsonify({
        "balance"       : round(state["balance"], 4),
        "open_positions": pos_data,
        "indicators"    : state["indicators"],
        "last_updated"  : state["last_updated"],
    })

@app.route("/api/logs")
def api_logs():
    log_path = os.path.join(os.path.dirname(__file__), "bot.log")
    try:
        with open(log_path, "r") as f:
            lines = f.readlines()
        lines = lines[-200:][::-1]
        return jsonify({"lines": [l.rstrip() for l in lines]})
    except FileNotFoundError:
        return jsonify({"lines": ["⚠ bot.log not found"]})
    except Exception as e:
        return jsonify({"lines": [f"⚠ Error reading log: {e}"]})

HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>Mudrex Algo Dashboard</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
  :root {
    --bg:     #080b0f;
    --card:   #0f1318;
    --border: #1e2530;
    --green:  #00d49a;
    --red:    #ff5050;
    --yellow: #ffb84d;
    --blue:   #4db8ff;
    --purple: #a78bfa;
    --text:   #dde3ed;
    --muted:  #6b7785;
    --fire:   #00d49a12;
    --hdr:    48px;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    background: var(--bg); color: var(--text);
    font-family: 'Inter', sans-serif; font-size: 13px;
    min-height: 100vh;
  }

  /* ── HEADER ── */
  header {
    height: var(--hdr);
    display: flex; align-items: center; justify-content: space-between;
    padding: 0 20px; border-bottom: 1px solid var(--border);
    background: var(--card); position: sticky; top: 0; z-index: 100;
  }
  .hdr-left { display: flex; align-items: center; gap: 8px; }
  .hdr-left h1 { font-size: 14px; font-weight: 700; color: var(--green); }
  .dot {
    width: 7px; height: 7px; border-radius: 50%; background: var(--green);
    animation: blink 1.5s infinite;
  }
  #last-updated { color: var(--muted); font-size: 11px; }

  /* ── STATS ── */
  .stats-row {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
    gap: 12px; padding: 16px 20px 0;
  }
  .stat-card {
    background: var(--card); border: 1px solid var(--border);
    border-radius: 10px; padding: 14px 16px;
  }
  .stat-label {
    font-size: 10px; font-weight: 600; color: var(--muted);
    text-transform: uppercase; letter-spacing: 0.7px; margin-bottom: 7px;
  }
  .stat-value { font-size: 18px; font-weight: 700; }
  .g { color: var(--green); }
  .r { color: var(--red);   }
  .b { color: var(--blue);  }
  .y { color: var(--yellow);}
  .p { color: var(--purple);}

  /* ── POSITION ── */
  .section { margin: 16px 20px 0; }
  .section-label {
    font-size: 10px; font-weight: 600; color: var(--muted);
    text-transform: uppercase; letter-spacing: 0.7px; margin-bottom: 8px;
  }
  .pos-card {
    background: var(--card); border: 1px solid var(--border);
    border-radius: 10px; padding: 14px 18px;
  }
  .pos-empty { color: var(--muted); font-size: 12px; }
  .pos-grid  { display: flex; gap: 24px; flex-wrap: wrap; align-items: center; }
  .pos-item  { display: flex; flex-direction: column; gap: 3px; }
  .plabel    { font-size: 10px; color: var(--muted); text-transform: uppercase; }
  .pvalue    { font-size: 14px; font-weight: 600; }
  .badge {
    display: inline-block; padding: 2px 10px; border-radius: 5px;
    font-size: 11px; font-weight: 700; text-transform: uppercase;
  }
  .badge.long  { background:#00d49a18; color:var(--green);  border:1px solid #00d49a44; }
  .badge.short { background:#ff505018; color:var(--red);    border:1px solid #ff505044; }
  .badge.none  { background:#6b778518; color:var(--muted);  border:1px solid var(--border); }

  /* ── TAB BAR ── */
  .table-section { margin: 16px 20px 28px; }
  .tab-bar { display: flex; gap: 8px; margin-bottom: 12px; }
  .tab-btn {
    background: var(--card); border: 1px solid var(--border);
    color: var(--muted); font-size: 12px; font-weight: 600;
    padding: 7px 18px; border-radius: 7px; cursor: pointer;
    transition: all 0.2s; font-family: inherit;
  }
  .tab-btn:hover  { border-color: #2e3a4a; color: var(--text); }
  .tab-btn.active { background: #00d49a18; border-color: #00d49a55; color: var(--green); }

  /* ── SLIDER WRAPPER ── */
  .slider-outer { overflow: hidden; }
  .slider-inner {
    display: flex;
    transition: transform 0.35s cubic-bezier(0.4, 0, 0.2, 1);
    will-change: transform;
    align-items: flex-start;
  }
  .slide { min-width: 100%; width: 100%; }

  /* ── DESKTOP TABLE ��─ */
  .table-wrap {
    overflow-x: auto; border-radius: 10px;
    border: 1px solid var(--border);
  }
  table { width: 100%; border-collapse: collapse; }
  thead { background: #0c1015; }
  th {
    text-align: left; padding: 10px 14px;
    color: var(--muted); font-size: 10px; font-weight: 600;
    text-transform: uppercase; letter-spacing: 0.6px;
    border-bottom: 1px solid var(--border); white-space: nowrap;
  }
  td {
    padding: 9px 14px; border-bottom: 1px solid #111820;
    vertical-align: middle; white-space: nowrap;
  }
  tr:last-child td     { border-bottom: none; }
  tr:hover td          { background: #0f141a; }
  tr.fire-row td       { background: var(--fire); }
  tr.fire-row:hover td { background: #00d49a1e; }
  tr.active-row td        { background: #ff505010; }
  tr.active-row:hover td  { background: #ff50501e; }

  .sym     { font-weight: 700; font-size: 13px; }
  .sym-sub { font-size: 10px; color: var(--muted); margin-top: 1px; }
  .regime-trend { color: var(--blue);  font-size: 11px; font-weight: 600; }
  .regime-range { color: var(--muted); font-size: 11px; }
  .rsi-wrap     { display: flex; align-items: center; gap: 6px; }
  .rsi-bar-bg   { width: 46px; height: 4px; background: #1e2530; border-radius: 2px; overflow: hidden; }
  .rsi-bar-fill { height: 100%; border-radius: 2px; }
  .sig-row { display: flex; gap: 5px; flex-wrap: wrap; }
  .sig-pip {
    font-size: 10px; font-weight: 600; padding: 2px 7px;
    border-radius: 4px; border: 1px solid transparent; white-space: nowrap;
  }
  .sig-pip.fire  { background:#00d49a18; color:var(--green);  border-color:#00d49a55; }
  .sig-pip.close { background:#ffb84d18; color:var(--yellow); border-color:#ffb84d55; }
  .sig-pip.mid   { background:#4db8ff10; color:#4db8ff77;     border-color:#4db8ff22; }
  .sig-pip.low   { background:transparent; color:#2e3a4a;     border-color:#1e2530; }
  .lev { font-size: 10px; font-weight: 700; padding: 2px 8px; border-radius: 4px; }
  .lev-10 { background:#00d49a18; color:var(--green);  border:1px solid #00d49a44; }
  .lev-5  { background:#ffb84d18; color:var(--yellow); border:1px solid #ffb84d44; }
  .lev-2  { background:#4db8ff10; color:var(--blue);   border:1px solid #4db8ff33; }
  .error-cell { color: var(--red); font-size: 11px; }

  /* ── LOG PANEL ── */
  .log-toolbar {
    display: flex; align-items: center; gap: 14px;
    margin-bottom: 10px; flex-wrap: wrap;
  }
  .log-refresh-btn {
    background: var(--card); border: 1px solid var(--border);
    color: var(--text); font-size: 11px; padding: 5px 14px;
    border-radius: 6px; cursor: pointer; font-family: inherit;
    transition: border-color 0.15s;
  }
  .log-refresh-btn:hover { border-color: #2e3a4a; }
  .log-box {
    background: #050709; border: 1px solid var(--border);
    border-radius: 10px; padding: 14px 16px;
    height: 480px; overflow-y: auto;
    font-family: 'JetBrains Mono', 'Fira Code', 'Courier New', monospace;
    font-size: 11.5px; line-height: 1.75;
  }
  .log-line       { display: block; white-space: pre-wrap; word-break: break-all; }
  .log-line.trade { color: var(--purple); font-weight: 600; }
  .log-line.fire  { color: var(--green);  font-weight: 600; }
  .log-line.error { color: var(--red); }
  .log-line.warn  { color: var(--yellow); }
  .log-line.info  { color: #7a9ab0; }
  .log-line.dim   { color: #2a3340; }

  /* ── MOBILE CARDS ── */
  .mobile-cards { display: none; }
  .m-card {
    background: var(--card); border: 1px solid var(--border);
    border-radius: 10px; padding: 14px 16px; margin-bottom: 10px;
  }
  .m-card.fire-row   { border-color: #00d49a44; background: var(--fire); }
  .m-card.active-row { border-color: #ff505044; background: #ff505010; }
  .m-card-top {
    display: flex; justify-content: space-between;
    align-items: flex-start; margin-bottom: 10px;
  }
  .m-sym   { font-weight: 700; font-size: 15px; }
  .m-price { font-size: 13px; font-weight: 600; }
  .m-grid  { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 8px; margin-bottom: 10px; }
  .m-item  { display: flex; flex-direction: column; gap: 2px; }
  .m-lbl   { font-size: 9px; color: var(--muted); text-transform: uppercase; letter-spacing: 0.5px; }
  .m-val   { font-size: 12px; font-weight: 600; }
  .m-sigs  { display: flex; gap: 5px; flex-wrap: wrap; }

  /* ── FOOTER ── */
  footer {
    text-align: center; padding: 14px; color: #2a3340;
    font-size: 11px; border-top: 1px solid var(--border);
  }

  /* ── RESPONSIVE ── */
  @media (max-width: 768px) {
    header { padding: 0 14px; }
    .hdr-left h1 { font-size: 12px; }
    .stats-row { padding: 12px 14px 0; gap: 8px; }
    .stat-card { padding: 12px 14px; }
    .stat-value { font-size: 16px; }
    .section { margin: 12px 14px 0; }
    .table-section { margin: 12px 14px 20px; }
    .table-wrap { display: none; }
    .mobile-cards { display: block; }
    .log-box { height: 360px; font-size: 10.5px; }
  }

  @keyframes blink { 0%,100%{opacity:1} 50%{opacity:0.2} }
</style>
</head>
<body>

<header>
  <div class="hdr-left">
    <div class="dot"></div>
    <h1>MUDREX ALGO DASHBOARD</h1>
  </div>
  <span id="last-updated">Loading...</span>
</header>

<div class="stats-row">
  <div class="stat-card">
    <div class="stat-label">💰 Balance</div>
    <div class="stat-value g" id="balance">—</div>
  </div>
  <div class="stat-card">
    <div class="stat-label">📊 Position</div>
    <div class="stat-value b" id="pos-symbol">—</div>
  </div>
  <div class="stat-card">
    <div class="stat-label">💼 Order Value</div>
    <div class="stat-value y" id="pos-entry">—</div>
  </div>
  <div class="stat-card">
    <div class="stat-label">📈 Total P&L</div>
    <div class="stat-value" id="pos-qty">—</div>
  </div>
  <div class="stat-card">
    <div class="stat-label">🔍 Scanning</div>
    <div class="stat-value b" id="pair-count">—</div>
  </div>
</div>

<div class="section">
  <div class="section-label">Active Position</div>
  <div class="pos-card" id="pos-card">
    <span class="pos-empty">No open position</span>
  </div>
</div>

<div class="table-section">

  <div class="tab-bar">
    <button class="tab-btn active" id="btn-scanner" onclick="switchTab('scanner')">📊 Indicator Scanner</button>
    <button class="tab-btn"        id="btn-logs"    onclick="switchTab('logs')">📋 Bot Logs</button>
  </div>

  <div class="slider-outer">
    <div class="slider-inner" id="slider-inner">

      <!-- Slide 1: Scanner -->
      <div class="slide">
        <div class="table-wrap">
          <table>
            <thead>
              <tr>
                <th>#</th><th>Symbol</th><th>Price</th><th>Regime</th>
                <th>RSI</th><th>MACD Hist</th><th>Vol Ratio</th>
                <th>EMA Spread</th><th>Lev</th><th>Signal Scores</th>
                <th>Candle Pattern</th><th>Status</th>
              </tr>
            </thead>
            <tbody id="indicator-table">
              <tr><td colspan="12" style="color:var(--muted);text-align:center;padding:40px;font-size:12px">
                ⏳ Fetching data...
              </td></tr>
            </tbody>
          </table>
        </div>
        <div class="mobile-cards" id="mobile-cards">
          <div style="color:var(--muted);text-align:center;padding:30px;font-size:12px">⏳ Fetching data...</div>
        </div>
      </div>

      <!-- Slide 2: Logs -->
      <div class="slide">
        <div class="log-toolbar">
          <span id="log-count" style="color:var(--muted);font-size:11px"></span>
          <button class="log-refresh-btn" onclick="fetchLogs()">↻ Refresh</button>
          <label style="color:var(--muted);font-size:11px;display:flex;align-items:center;gap:5px;cursor:pointer">
            <input type="checkbox" id="auto-scroll" checked> Auto-scroll
          </label>
        </div>
        <div class="log-box" id="log-box">
          <span class="log-line dim">Switch to this tab to load logs...</span>
        </div>
      </div>

    </div>
  </div>

</div>

<footer>Auto-refreshes every 60s &nbsp;·&nbsp; Mudrex Algo Bot &nbsp;·&nbsp; 15m timeframe</footer>

<script>
// ── STATE ────────────────────────────────────────────────────────
let currentPos    = null;
let allPositions  = [];

// ── HELPERS ──────────────────────────────────────────────────────
function rsiColor(v) {
  if (v >= 70) return "var(--red)";
  if (v >= 65) return "var(--yellow)";
  if (v <= 30) return "var(--blue)";
  if (v <= 35) return "#4db8ffaa";
  return "var(--text)";
}
function rsiBarColor(v) {
  if (v >= 65) return "var(--yellow)";
  if (v <= 35) return "var(--blue)";
  return "var(--green)";
}
function sigClass(passed, total, fire) {
  if (fire)              return "fire";
  if (passed >= total-1) return "close";
  if (passed >= total-3) return "mid";
  return "low";
}
function levClass(lev) {
  if (lev >= 10) return "lev-10";
  if (lev >= 5)  return "lev-5";
  return "lev-2";
}
function spreadColor(s) {
  if (s > 0.5) return "var(--green)";
  if (s > 0.1) return "var(--yellow)";
  return "var(--muted)";
}
function patternHTML(name, type) {
  if (!name) return '<span style="color:#2e3a4a;font-size:10px">—</span>';
  const color = type === "bullish" ? "var(--green)"
              : type === "bearish" ? "var(--red)"
              : "var(--yellow)";
  const icon  = type === "bullish" ? "▲" : type === "bearish" ? "▼" : "◆";
  return `<span style="color:${color};font-size:10px;font-weight:600;
    background:${color}18;border:1px solid ${color}44;
    padding:2px 7px;border-radius:4px;white-space:nowrap">
    ${icon} ${name}</span>`;
}
function statusHTML(symbol, anyFire) {
  const isActive = allPositions.some(p => p.symbol === symbol);
  if (isActive) {
    return `<span style="display:inline-flex;align-items:center;gap:5px;
      font-size:10px;font-weight:700;color:var(--red);
      background:#ff505018;border:1px solid #ff505044;
      padding:3px 10px;border-radius:20px;white-space:nowrap">
      <span style="width:7px;height:7px;border-radius:50%;background:var(--red);
      box-shadow:0 0 6px var(--red);flex-shrink:0;
      animation:blink 1s infinite;display:inline-block"></span>
      ACTIVE</span>`;
  }
  if (anyFire) {
    return `<span style="display:inline-flex;align-items:center;gap:5px;
      font-size:10px;font-weight:700;color:var(--yellow);
      background:#ffb84d18;border:1px solid #ffb84d44;
      padding:3px 10px;border-radius:20px;white-space:nowrap">
      <span style="width:7px;height:7px;border-radius:50%;background:var(--yellow);
      box-shadow:0 0 6px var(--yellow);flex-shrink:0;
      animation:blink 1.5s infinite;display:inline-block"></span>
      FIRE</span>`;
  }
  return `<span style="display:inline-flex;align-items:center;gap:5px;
    font-size:10px;font-weight:600;color:#3a4a5a;
    background:#12181f;border:1px solid #1e2530;
    padding:3px 10px;border-radius:20px;white-space:nowrap">
    <span style="width:7px;height:7px;border-radius:50%;background:#2e3a4a;
    flex-shrink:0;display:inline-block"></span>
    IDLE</span>`;
}

// ── TAB SLIDE ────────────────────────────────────────────────────
let currentTab = "scanner";
function switchTab(tab) {
  currentTab = tab;
  document.getElementById("slider-inner").style.transform =
    tab === "scanner" ? "translateX(0%)" : "translateX(-100%)";
  document.getElementById("btn-scanner").classList.toggle("active", tab === "scanner");
  document.getElementById("btn-logs")   .classList.toggle("active", tab === "logs");
  if (tab === "logs") fetchLogs();
}

// ── LOGS ─────────────────────────────────────────────────────────
function logClass(line) {
  if (/LONG|SHORT|ORDER PLACED|🟢|🔴/.test(line)) return "trade";
  if (/FIRE|signal fired/i.test(line))             return "fire";
  if (/ERROR|error|⚠|failed/i.test(line))          return "error";
  if (/WARNING|warning/i.test(line))               return "warn";
  return "info";
}
async function fetchLogs() {
  try {
    const r    = await fetch("/api/logs");
    const data = await r.json();
    const box  = document.getElementById("log-box");
    document.getElementById("log-count").textContent =
      data.lines.length + " lines (newest first)";
    box.innerHTML = data.lines.map(line => {
      const safe = line.replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;");
      return `<span class="log-line ${logClass(line)}">${safe}</span>`;
    }).join("<br>");
    if (document.getElementById("auto-scroll").checked) box.scrollTop = 0;
  } catch(e) {
    document.getElementById("log-box").innerHTML =
      '<span class="log-line error">⚠ Failed to fetch logs</span>';
  }
}
setInterval(() => { if (currentTab === "logs") fetchLogs(); }, 15000);

// ── SCANNER REFRESH ──────────────────────────────────────────────
async function refresh() {
  try {
    const r    = await fetch("/api/state");
    const data = await r.json();

    document.getElementById("last-updated").textContent = "Updated: " + data.last_updated;
    document.getElementById("balance").textContent = "$" + Number(data.balance).toFixed(4) + " USDT";
    document.getElementById("pair-count").textContent = (data.indicators || []).length + " pairs";

    // Store position globally so statusHTML can access it
    currentPos = (data.open_positions && data.open_positions.length > 0) ? data.open_positions[0] : null;
    allPositions = data.open_positions || [];
    const pos  = currentPos;

    if (allPositions.length > 0) {
      const first  = allPositions[0];
      const side0  = (first.side || "").toUpperCase();
      document.getElementById("pos-symbol").textContent = allPositions.length > 1 ? allPositions.length + " positions" : first.symbol || "—";
      const totalValue = allPositions.reduce((s, p) => s + (parseFloat(p.entry||0) * parseFloat(p.qty||0)), 0);
      const totalPnl   = allPositions.reduce((s, p) => s + (parseFloat(p.pnl||0)), 0);
      const pnlSign    = totalPnl >= 0 ? "+" : "";
      const pnlCol     = totalPnl >= 0 ? "var(--green)" : "var(--red)";
      document.getElementById("pos-entry").textContent  = "$" + totalValue.toFixed(4);
      document.getElementById("pos-qty").textContent    = pnlSign + "$" + totalPnl.toFixed(4);
      document.getElementById("pos-qty").style.color    = pnlCol;
      document.getElementById("pos-card").innerHTML = allPositions.map(pos => {
        const side     = (pos.side || "").toUpperCase();
        const badgeC   = side === "LONG" ? "long" : side === "SHORT" ? "short" : "none";
        const pnlColor = pos.pnl >= 0 ? "var(--green)" : "var(--red)";
        const pnlSign  = pos.pnl >= 0 ? "+" : "";
        const priceCol = pos.price > pos.entry ? "var(--green)" : pos.price < pos.entry ? "var(--red)" : "var(--text)";
        return `<div class="pos-grid" style="margin-bottom:${allPositions.length>1?'14px':'0'};gap:20px;flex-wrap:wrap">
          <div class="pos-item"><span class="plabel">Symbol</span><span class="pvalue">${pos.symbol}</span></div>
          <div class="pos-item"><span class="plabel">Direction</span><span class="badge ${badgeC}">${side} ${pos.lev}x</span></div>
          <div class="pos-item"><span class="plabel">Quantity</span><span class="pvalue">${pos.qty}</span></div>
          <div class="pos-item"><span class="plabel">Entry</span><span class="pvalue">$${parseFloat(pos.entry||0).toFixed(5)}</span></div>
          <div class="pos-item"><span class="plabel">Current Price</span><span class="pvalue" style="color:${priceCol}">$${parseFloat(pos.price||0).toFixed(5)}</span></div>
          <div class="pos-item"><span class="plabel">Unrealized P&L</span><span class="pvalue" style="color:${pnlColor}">${pnlSign}$${pos.pnl} <span style="font-size:11px;opacity:0.8">(${pnlSign}${pos.pnl_pct}%)</span></span></div>
          <div class="pos-item"><span class="plabel">Stop Loss</span><span class="pvalue" style="color:var(--red)">$${parseFloat(pos.sl||0).toFixed(5)}</span></div>
          <div class="pos-item"><span class="plabel">Take Profit</span><span class="pvalue" style="color:var(--green)">$${parseFloat(pos.tp||0).toFixed(5)}</span></div>
        </div>`;
      }).join('<hr style="border-color:var(--border);margin:0 0 14px">');
    } else {
      document.getElementById("pos-symbol").textContent = "None";
      document.getElementById("pos-entry").textContent  = "—";
      document.getElementById("pos-qty").textContent    = "—";
      document.getElementById("pos-card").innerHTML = '<span class="pos-empty">No open position</span>';
    }

    const rows   = data.indicators || [];
    const tbody  = document.getElementById("indicator-table");
    const mCards = document.getElementById("mobile-cards");
    tbody.innerHTML  = "";
    mCards.innerHTML = "";

    rows.forEach((row, idx) => {
      if (row.error) {
        const tr = document.createElement("tr");
        tr.innerHTML = `<td style="color:var(--muted)">${idx+1}</td>
          <td><span class="sym">${row.symbol}</span></td>
          <td colspan="10" class="error-cell">⚠ ${row.error}</td>`;
        tbody.appendChild(tr);
        const mc = document.createElement("div");
        mc.className = "m-card";
        mc.innerHTML = `<div class="m-card-top"><span class="m-sym">${row.symbol}</span></div>
          <span style="color:var(--red);font-size:11px">⚠ ${row.error}</span>`;
        mCards.appendChild(mc);
        return;
      }

      const isActive  = allPositions.some(p => p.symbol === row.symbol);
      const anyFire   = Object.values(row.signals).some(s => s.fire);
      const volRatio  = row.vol_ma > 0 ? (row.vol / row.vol_ma).toFixed(2) : "—";
      const volColor  = row.vol > row.vol_ma ? "var(--green)" : "var(--muted)";
      const macdColor = row.macd > 0 ? "var(--green)" : "var(--red)";
      const rsiPct    = Math.min(100, Math.max(0, row.rsi));

      const rsiHTML = `
        <div class="rsi-wrap">
          <span style="color:${rsiColor(row.rsi)};font-weight:600;min-width:30px">${row.rsi}</span>
          <div class="rsi-bar-bg">
            <div class="rsi-bar-fill" style="width:${rsiPct}%;background:${rsiBarColor(row.rsi)}"></div>
          </div>
        </div>`;

      const sigHTML = Object.entries(row.signals).map(([name, s]) => {
        const cls   = sigClass(s.passed, s.total, s.fire);
        const label = s.fire ? "🟢 " + name : `${name} ${s.passed}/${s.total}`;
        return `<span class="sig-pip ${cls}">${label}</span>`;
      }).join("");

      const pHTML = patternHTML(row.pattern, row.pattern_type);
      const sHTML = statusHTML(row.symbol, anyFire);

      // Desktop row — active-row takes priority over fire-row
      const tr = document.createElement("tr");
      if      (isActive) tr.classList.add("active-row");
      else if (anyFire)  tr.classList.add("fire-row");
      tr.innerHTML = `
        <td style="color:var(--muted);font-size:11px">${idx+1}</td>
        <td>
          <div class="sym">${row.symbol.replace("USDT","")}<span style="color:var(--muted);font-weight:400">/USDT</span></div>
          <div class="sym-sub">ATR: ${row.atr}</div>
        </td>
        <td style="font-weight:600">$${row.price}</td>
        <td>
          <span class="regime-${row.regime.toLowerCase()}">${row.regime}</span>
          <div style="font-size:10px;color:var(--muted);margin-top:2px">${row.spread}%</div>
        </td>
        <td>${rsiHTML}</td>
        <td style="color:${macdColor};font-size:11px">${row.macd > 0 ? "+" : ""}${row.macd}</td>
        <td style="color:${volColor}">${volRatio}x</td>
        <td style="color:${spreadColor(row.spread)};font-size:11px">${row.spread}%</td>
        <td><span class="lev ${levClass(row.lev)}">${row.lev}x</span></td>
        <td><div class="sig-row">${sigHTML}</div></td>
        <td>${pHTML}</td>
        <td>${sHTML}</td>`;
      tbody.appendChild(tr);

      // Mobile card
      const mc = document.createElement("div");
      mc.className = "m-card" + (isActive ? " active-row" : anyFire ? " fire-row" : "");
      mc.innerHTML = `
        <div class="m-card-top">
          <div>
            <div class="m-sym">${row.symbol.replace("USDT","")}<span style="color:var(--muted);font-size:12px;font-weight:400">/USDT</span></div>
            <div style="font-size:10px;color:var(--muted);margin-top:2px">ATR: ${row.atr}</div>
          </div>
          <div style="text-align:right">
            <div class="m-price">$${row.price}</div>
            <div style="margin-top:4px"><span class="lev ${levClass(row.lev)}">${row.lev}x</span></div>
          </div>
        </div>
        <div class="m-grid">
          <div class="m-item"><span class="m-lbl">Regime</span><span class="m-val regime-${row.regime.toLowerCase()}">${row.regime}</span></div>
          <div class="m-item"><span class="m-lbl">RSI</span><span class="m-val" style="color:${rsiColor(row.rsi)}">${row.rsi}</span></div>
          <div class="m-item"><span class="m-lbl">MACD</span><span class="m-val" style="color:${macdColor}">${row.macd > 0 ? "+" : ""}${row.macd}</span></div>
          <div class="m-item"><span class="m-lbl">Vol Ratio</span><span class="m-val" style="color:${volColor}">${volRatio}x</span></div>
          <div class="m-item"><span class="m-lbl">EMA Spread</span><span class="m-val" style="color:${spreadColor(row.spread)}">${row.spread}%</span></div>
        </div>
        <div class="m-sigs">${sigHTML}</div>
        <div style="margin-top:8px;display:flex;gap:8px;align-items:center;flex-wrap:wrap">
          ${pHTML}
          ${sHTML}
        </div>`;
      mCards.appendChild(mc);
    });

  } catch(e) {
    console.error("Refresh error:", e);
    document.getElementById("last-updated").textContent = "⚠ Refresh failed";
  }
}

refresh();
setInterval(refresh, 60000);
</script>
</body>
</html>
"""

if __name__ == "__main__":
    t = threading.Thread(target=background_refresh, daemon=True)
    t.start()
    app.run(host="0.0.0.0", port=5000, debug=False)
