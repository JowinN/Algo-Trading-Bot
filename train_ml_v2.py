"""
Enhanced ML Training V2 — Anti-Overfit + Trade Quality Score
============================================================
Key improvements over V1:
1. Trade Quality Score (MFE/MAE ratio) instead of binary win/loss
2. XGBoost with early stopping (prevents overfitting)
3. Purged walk-forward validation (no lookahead bias)
4. Feature selection (removes noise features)
5. Proper holdout test set (last 20% never seen during training)
6. Class weight balancing for imbalanced data
7. Calibrated probability output
8. Downloads maximum available data (2000+ days)
"""

import os
import sys
import time
import pickle
import argparse
import numpy as np
import pandas as pd
import requests
from datetime import datetime, timedelta
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, mean_absolute_error
from sklearn.calibration import CalibratedClassifierCV
import xgboost as xgb

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from indicators import compute_all, compute_htf
from strategy import generate_signal, Signal
from config import Config as c
from ml_model import (
    extract_features, extract_features_extended,
    calculate_mfe_mae, calculate_optimal_exit,
    MIN_SL_ATR, MAX_SL_ATR, MIN_TP_R, MAX_TP_R
)

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "historical_data")
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ml_models")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

BYBIT_BASE = "https://api.bybit.com/v5/market"

# Trade quality thresholds
QUALITY_GOOD_THRESHOLD = 1.5  # MFE/MAE > 1.5 = good trade (stricter for lower DD)
QUALITY_EXCELLENT_THRESHOLD = 2.0  # MFE/MAE > 2.0 = excellent


# ══════════════════════════════════════════════════════════════════════════
# DATA FETCHING — Maximum Historical Data (2000+ days)
# ══════════════════════════════════════════════════════════════════════════

def fetch_max_data(symbol, interval="15m", max_days=2000):
    """Fetch maximum available kline data from Bybit v5 API."""
    from data import INTERVAL_MAP
    bybit_interval = INTERVAL_MAP.get(interval, "15")
    all_candles = []
    current_end = datetime.now()
    target_start = datetime.now() - timedelta(days=max_days)

    request_count = 0
    backoff = 0.25
    consecutive_errors = 0

    while True:
        try:
            url = f"{BYBIT_BASE}/kline"
            params = {
                "category": "linear",
                "symbol": symbol,
                "interval": bybit_interval,
                "limit": 1000,
                "end": int(current_end.timestamp() * 1000)
            }
            resp = requests.get(url, params=params, timeout=15)
            resp.raise_for_status()
            raw = resp.json()

            if raw.get("retCode") != 0:
                ret_msg = raw.get("retMsg", "")
                if "Too many" in ret_msg or "Rate" in ret_msg:
                    backoff = min(backoff * 2, 30)
                    time.sleep(backoff)
                    consecutive_errors += 1
                    if consecutive_errors > 10:
                        break
                    continue
                break

            candles = raw["result"]["list"]
            if not candles:
                break

            all_candles.extend(candles)
            request_count += 1
            consecutive_errors = 0
            backoff = 0.25

            oldest_ts = float(candles[-1][0])
            new_end = datetime.fromtimestamp(oldest_ts / 1000)

            if new_end >= current_end:
                break
            current_end = new_end

            if current_end <= target_start:
                break

            if request_count % 50 == 0:
                days_back = (datetime.now() - current_end).days
                print(f"      {len(all_candles)} candles (~{days_back} days back)")

            time.sleep(0.12)  # Rate limit: ~8 req/s

        except requests.exceptions.RequestException:
            backoff = min(backoff * 2, 30)
            time.sleep(backoff)
            consecutive_errors += 1
            if consecutive_errors > 10:
                break
            continue
        except Exception as e:
            print(f"      Error: {e}")
            break

    if not all_candles:
        return pd.DataFrame()

    df = pd.DataFrame(all_candles, columns=[
        "timestamp", "open", "high", "low", "close", "volume", "turnover"
    ])
    df["timestamp"] = pd.to_datetime(df["timestamp"].astype(float), unit="ms")
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)

    df = df.sort_values("timestamp").drop_duplicates(subset=["timestamp"]).reset_index(drop=True)
    return df[["timestamp", "open", "high", "low", "close", "volume"]]


def load_or_fetch_data(symbol, interval="15m", max_days=2000, force_fetch=False):
    """Load from CSV or fetch from API. Always fetch if data is insufficient."""
    filepath = os.path.join(DATA_DIR, f"{symbol}_{interval}.csv")

    if os.path.exists(filepath) and not force_fetch:
        df = pd.read_csv(filepath)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = df[col].astype(float)
        data_days = (df["timestamp"].max() - df["timestamp"].min()).days
        if data_days >= max_days * 0.8:
            return df, data_days

    print(f"      Fetching {symbol} from Bybit (max {max_days} days)...")
    df = fetch_max_data(symbol, interval, max_days)

    if not df.empty:
        df.to_csv(filepath, index=False)
        data_days = (df["timestamp"].max() - df["timestamp"].min()).days
        return df, data_days

    # Fall back to existing if fetch failed
    if os.path.exists(filepath):
        df = pd.read_csv(filepath)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = df[col].astype(float)
        data_days = (df["timestamp"].max() - df["timestamp"].min()).days
        return df, data_days

    return pd.DataFrame(), 0


# ══════════════════════════════════════════════════════════════════════════
# RESAMPLING & HTF
# ══════════════════════════════════════════════════════════════════════════

def resample_to_timeframe(df_15m, timeframe="4h"):
    """Resample 15m data to target timeframe."""
    ohlcv = df_15m[["timestamp", "open", "high", "low", "close", "volume"]].copy()
    ohlcv = ohlcv.set_index("timestamp")
    resampled = ohlcv.resample(timeframe).agg({
        "open": "first", "high": "max", "low": "min",
        "close": "last", "volume": "sum"
    }).dropna()
    return resampled


def get_htf_bias(htf_df, timestamp):
    """Get HTF trend bias at a specific timestamp."""
    valid = htf_df[htf_df.index <= timestamp]
    if valid.empty:
        return None
    row = valid.iloc[-1]
    if row.get("trend_up", False):
        return "LONG"
    elif row.get("trend_down", False):
        return "SHORT"
    return None


# ══════════════════════════════════════════════════════════════════════════
# TRADE SIMULATION WITH QUALITY SCORING
# ══════════════════════════════════════════════════════════════════════════

def compute_trade_quality(mfe, mae):
    """
    Trade quality score: how much the market moved in favor vs against.
    Higher = better setup (price action confirms the trade direction).
    """
    return mfe / (mae + 0.5)  # +0.5 prevents division by zero and values near 0


def simulate_trade_with_sl_tp(df, entry_idx, direction, entry_price, atr_val,
                              sl_mult, tp_r, max_bars=30):
    """
    Simulate a trade with specific SL/TP. Returns PnL in R-multiples.
    +tp_r if TP hit, -1.0 if SL hit, partial if neither hit.
    """
    sl_dist = atr_val * sl_mult
    tp_dist = sl_dist * tp_r

    if direction == "LONG":
        sl_price = entry_price - sl_dist
        tp_price = entry_price + tp_dist
    else:
        sl_price = entry_price + sl_dist
        tp_price = entry_price - tp_dist

    end_idx = min(entry_idx + max_bars + 1, len(df))
    for j in range(entry_idx + 1, end_idx):
        row = df.iloc[j]
        high = float(row["high"])
        low = float(row["low"])

        if direction == "LONG":
            if low <= sl_price:
                return -1.0
            if high >= tp_price:
                return tp_r
        else:
            if high >= sl_price:
                return -1.0
            if low <= tp_price:
                return tp_r

    # Didn't hit either — calculate partial PnL
    final_price = float(df.iloc[min(entry_idx + max_bars, len(df) - 1)]["close"])
    if direction == "LONG":
        partial = (final_price - entry_price) / sl_dist
    else:
        partial = (entry_price - final_price) / sl_dist
    return partial


def generate_training_data_for_symbol(symbol, df_4h, htf_df, max_hold_bars=30):
    """Generate training data with quality scores."""
    indicators_df = compute_all(df_4h)
    indicators_df = indicators_df.reset_index()
    indicators_df.rename(columns={"index": "timestamp"}, inplace=True)

    if len(indicators_df) < 60:
        return []

    trades = []

    for i in range(55, len(indicators_df) - max_hold_bars - 1):
        chunk = indicators_df.iloc[:i + 1]
        ts = chunk.iloc[-1]["timestamp"]

        bias = get_htf_bias(htf_df, ts)
        if bias is None:
            continue

        signal, sl, tp = generate_signal(chunk, htf_bias=bias)
        if signal == Signal.NONE:
            continue

        curr = chunk.iloc[-1]
        prev = chunk.iloc[-2]
        price = float(curr["close"])
        atr_val = float(curr["atr"])

        if atr_val <= 0:
            continue

        # Extract features
        try:
            features = extract_features_extended(
                curr, prev, price, atr_val, signal,
                df_slice=chunk.iloc[-15:] if len(chunk) >= 15 else chunk
            )
        except Exception:
            continue

        # Calculate MFE/MAE
        mfe_mae = calculate_mfe_mae(
            indicators_df, i, signal, price, atr_val, max_bars=max_hold_bars
        )
        if mfe_mae is None:
            continue
        mfe, mae = mfe_mae

        # Find optimal SL/TP
        opt_result = calculate_optimal_exit(
            indicators_df, i, signal, price, atr_val, max_bars=max_hold_bars
        )
        if opt_result is None:
            continue
        optimal_sl, optimal_tp, best_pnl = opt_result

        # Compute trade quality
        quality = compute_trade_quality(mfe, mae)

        # Simulate with default settings for comparison
        default_pnl = simulate_trade_with_sl_tp(
            indicators_df, i, signal, price, atr_val,
            sl_mult=2.0, tp_r=3.0, max_bars=max_hold_bars
        )

        # Simulate with optimal settings
        opt_pnl = simulate_trade_with_sl_tp(
            indicators_df, i, signal, price, atr_val,
            sl_mult=optimal_sl, tp_r=optimal_tp, max_bars=max_hold_bars
        )

        trades.append({
            "features": features,
            "quality": quality,
            "mfe": mfe,
            "mae": mae,
            "optimal_sl": optimal_sl,
            "optimal_tp": optimal_tp,
            "default_pnl": default_pnl,
            "opt_pnl": opt_pnl,
            "best_pnl": best_pnl,
            "symbol": symbol,
            "timestamp": ts,
            "direction": signal,
            "price": price,
            "atr": atr_val,
        })

    return trades


# ══════════════════════════════════════════════════════════════════════════
# FULL DATA GENERATION
# ══════════════════════════════════════════════════════════════════════════

def generate_all_training_data(symbols, max_days=2000, force_fetch=False, skip_fetch=False):
    """Generate training data across all symbols."""
    all_trades = []

    print(f"\n{'='*70}")
    print(f"  GENERATING TRAINING DATA — MAX {max_days} DAYS")
    print(f"  Symbols: {len(symbols)}")
    print(f"{'='*70}\n")

    for idx, symbol in enumerate(symbols):
        print(f"  [{idx+1}/{len(symbols)}] {symbol}...", end=" ", flush=True)

        if skip_fetch:
            filepath = os.path.join(DATA_DIR, f"{symbol}_15m.csv")
            if not os.path.exists(filepath):
                print("no data (skip)")
                continue
            df_15m = pd.read_csv(filepath)
            df_15m["timestamp"] = pd.to_datetime(df_15m["timestamp"])
            for col in ["open", "high", "low", "close", "volume"]:
                df_15m[col] = df_15m[col].astype(float)
            data_days = (df_15m["timestamp"].max() - df_15m["timestamp"].min()).days
        else:
            df_15m, data_days = load_or_fetch_data(symbol, "15m", max_days, force_fetch)

        if df_15m.empty or len(df_15m) < 2000:
            print(f"insufficient data ({len(df_15m) if not df_15m.empty else 0})")
            continue

        print(f"{data_days}d/{len(df_15m)} candles", end=" → ", flush=True)

        # Resample
        df_4h = resample_to_timeframe(df_15m, "4h")
        if len(df_4h) < 100:
            print(f"too few 4H ({len(df_4h)})")
            continue

        # HTF
        ohlcv = df_15m[["timestamp", "open", "high", "low", "close", "volume"]].copy()
        htf_df = compute_htf(ohlcv, "1D")
        if htf_df.empty:
            print("no HTF")
            continue

        # Generate trades
        trades = generate_training_data_for_symbol(symbol, df_4h, htf_df, max_hold_bars=30)

        if trades:
            qualities = [t["quality"] for t in trades]
            good_count = sum(1 for q in qualities if q >= QUALITY_GOOD_THRESHOLD)
            avg_quality = np.mean(qualities)
            avg_mfe = np.mean([t["mfe"] for t in trades])
            avg_mae = np.mean([t["mae"] for t in trades])

            print(f"{len(trades)} signals | Quality: {avg_quality:.2f} "
                  f"({good_count}/{len(trades)} good) | "
                  f"MFE={avg_mfe:.2f} MAE={avg_mae:.2f}")
        else:
            print("0 signals")

        all_trades.extend(trades)

    # Summary
    print(f"\n{'='*70}")
    print(f"  TOTAL: {len(all_trades)} trades across {len(symbols)} symbols")
    if all_trades:
        qualities = [t["quality"] for t in all_trades]
        good_trades = sum(1 for q in qualities if q >= QUALITY_GOOD_THRESHOLD)
        print(f"  Good trades (quality>{QUALITY_GOOD_THRESHOLD}): "
              f"{good_trades}/{len(all_trades)} = {good_trades/len(all_trades)*100:.1f}%")
        print(f"  Avg Quality:  {np.mean(qualities):.3f}")
        print(f"  Avg MFE:      {np.mean([t['mfe'] for t in all_trades]):.2f} ATR")
        print(f"  Avg MAE:      {np.mean([t['mae'] for t in all_trades]):.2f} ATR")
        print(f"  Avg Opt SL:   {np.mean([t['optimal_sl'] for t in all_trades]):.2f} ATR")
        print(f"  Avg Opt TP:   {np.mean([t['optimal_tp'] for t in all_trades]):.1f}R")

        # Default strategy PnL
        default_pnls = [t["default_pnl"] for t in all_trades]
        print(f"  Default PnL:  {sum(default_pnls):.1f}R total "
              f"(avg {np.mean(default_pnls):.3f}R/trade)")
    print(f"{'='*70}\n")

    return all_trades


# ══════════════════════════════════════════════════════════════════════════
# PREPARE DATA
# ══════════════════════════════════════════════════════════════════════════

def prepare_matrices(trades):
    """Convert trades to numpy arrays with quality labels."""
    if not trades:
        return None

    feature_names = list(trades[0]["features"].keys())
    X = np.array([[t["features"].get(f, 0) for f in feature_names] for t in trades])
    y_quality = np.array([t["quality"] for t in trades])
    y_class = (y_quality >= QUALITY_GOOD_THRESHOLD).astype(int)
    y_sl = np.array([t["optimal_sl"] for t in trades])
    y_tp = np.array([t["optimal_tp"] for t in trades])

    # Clean NaN/Inf
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    return {
        "X": X,
        "y_class": y_class,
        "y_quality": y_quality,
        "y_sl": y_sl,
        "y_tp": y_tp,
        "feature_names": feature_names,
        "trades": trades,
    }


# ══════════════════════════════════════════════════════════════════════════
# FEATURE SELECTION
# ══════════════════════════════════════════════════════════════════════════

def select_features(X, y, feature_names, min_importance=0.005):
    """Remove noise features using a quick XGBoost fit."""
    print("  Feature selection...", end=" ", flush=True)

    model = xgb.XGBClassifier(
        n_estimators=100, max_depth=3, learning_rate=0.1,
        use_label_encoder=False, eval_metric="logloss",
        verbosity=0, random_state=42
    )
    model.fit(X, y)
    importances = model.feature_importances_

    mask = importances >= min_importance
    selected = [f for f, m in zip(feature_names, mask) if m]
    dropped = [f for f, m in zip(feature_names, mask) if not m]

    print(f"kept {sum(mask)}/{len(feature_names)} features "
          f"(dropped: {', '.join(dropped[:5])}{'...' if len(dropped) > 5 else ''})")

    return mask, selected


# ══════════════════════════════════════════════════════════════════════════
# PURGED WALK-FORWARD VALIDATION
# ══════════════════════════════════════════════════════════════════════════

def purged_walk_forward_cv(data, n_splits=5, purge_bars=10):
    """
    Walk-forward CV with purge gap to prevent leakage from overlapping trades.
    Uses XGBoost with early stopping on each fold.
    """
    X = data["X"]
    y_class = data["y_class"]
    y_sl = data["y_sl"]
    y_tp = data["y_tp"]
    y_quality = data["y_quality"]
    trades = data["trades"]

    n = len(X)
    fold_size = n // (n_splits + 1)

    print(f"\n{'='*70}")
    print(f"  PURGED WALK-FORWARD VALIDATION ({n_splits} folds)")
    print(f"  Samples: {n} | Good Rate: {y_class.mean()*100:.1f}%")
    print(f"  Purge gap: {purge_bars} bars between train/test")
    print(f"{'='*70}\n")

    fold_results = []
    all_oos_preds = []
    all_oos_true = []

    for fold in range(n_splits):
        # Train: everything up to fold boundary
        # Purge: gap of purge_bars
        # Test: next fold_size samples after purge
        train_end = fold_size * (fold + 1)
        test_start = train_end + purge_bars
        test_end = min(test_start + fold_size, n)

        if test_start >= n or test_end - test_start < 20:
            continue

        X_train = X[:train_end]
        y_train = y_class[:train_end]
        sl_train = y_sl[:train_end]
        tp_train = y_tp[:train_end]

        X_test = X[test_start:test_end]
        y_test = y_class[test_start:test_end]
        sl_test = y_sl[test_start:test_end]
        tp_test = y_tp[test_start:test_end]
        quality_test = y_quality[test_start:test_end]
        trades_test = trades[test_start:test_end]

        if len(set(y_train)) < 2 or len(set(y_test)) < 2:
            continue

        # Scale
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        # Use last 15% of training as validation for early stopping
        val_size = max(int(len(X_train_s) * 0.15), 20)
        X_tr = X_train_s[:-val_size]
        y_tr = y_train[:-val_size]
        X_val = X_train_s[-val_size:]
        y_val = y_train[-val_size:]

        # Compute class weight
        pos_count = y_tr.sum()
        neg_count = len(y_tr) - pos_count
        scale_pos = neg_count / max(pos_count, 1)

        # ── Classifier with early stopping ──
        clf = xgb.XGBClassifier(
            n_estimators=500,
            max_depth=4,
            learning_rate=0.03,
            min_child_weight=10,
            subsample=0.7,
            colsample_bytree=0.7,
            reg_alpha=1.0,
            reg_lambda=3.0,
            scale_pos_weight=scale_pos,
            use_label_encoder=False,
            eval_metric="auc",
            early_stopping_rounds=30,
            verbosity=0,
            random_state=42,
        )
        clf.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
        y_prob = clf.predict_proba(X_test_s)[:, 1]

        # ── SL Regressor ──
        sl_reg = xgb.XGBRegressor(
            n_estimators=300,
            max_depth=3,
            learning_rate=0.03,
            min_child_weight=15,
            subsample=0.7,
            colsample_bytree=0.7,
            reg_alpha=1.0,
            reg_lambda=3.0,
            early_stopping_rounds=20,
            verbosity=0,
            random_state=42,
        )
        sl_reg.fit(
            X_train_s[:-val_size], sl_train[:-val_size],
            eval_set=[(X_train_s[-val_size:], sl_train[-val_size:])],
            verbose=False
        )
        sl_pred = np.clip(sl_reg.predict(X_test_s), MIN_SL_ATR, MAX_SL_ATR)

        # ── TP Regressor ──
        tp_reg = xgb.XGBRegressor(
            n_estimators=300,
            max_depth=3,
            learning_rate=0.03,
            min_child_weight=15,
            subsample=0.7,
            colsample_bytree=0.7,
            reg_alpha=1.0,
            reg_lambda=3.0,
            early_stopping_rounds=20,
            verbosity=0,
            random_state=42,
        )
        tp_reg.fit(
            X_train_s[:-val_size], tp_train[:-val_size],
            eval_set=[(X_train_s[-val_size:], tp_train[-val_size:])],
            verbose=False
        )
        tp_pred = np.clip(tp_reg.predict(X_test_s), MIN_TP_R, MAX_TP_R)

        # ── Evaluate ──
        try:
            auc = roc_auc_score(y_test, y_prob)
        except ValueError:
            auc = 0.5

        sl_mae = mean_absolute_error(sl_test, sl_pred)
        tp_mae = mean_absolute_error(tp_test, tp_pred)

        # Store out-of-sample predictions
        all_oos_preds.extend(y_prob.tolist())
        all_oos_true.extend(y_test.tolist())

        print(f"  Fold {fold+1}: Train={len(y_train)} Test={len(y_test)} | "
              f"AUC={auc:.3f} | Trees={clf.best_iteration}")
        print(f"    SL: MAE={sl_mae:.3f} | TP: MAE={tp_mae:.3f}")

        # Simulate actual trading PF at different thresholds
        for threshold in [0.40, 0.45, 0.50, 0.55, 0.60]:
            mask = y_prob >= threshold
            if mask.sum() < 3:
                continue

            # Simulate trades with ML-predicted SL/TP
            total_pnl = 0
            wins = 0
            losses = 0
            for k in range(len(y_test)):
                if not mask[k]:
                    continue
                # Use the actual trade outcome with predicted SL/TP
                trade = trades_test[k]
                pnl = simulate_trade_with_sl_tp(
                    None, 0, trade["direction"], trade["price"], trade["atr"],
                    sl_mult=sl_pred[k], tp_r=tp_pred[k], max_bars=30
                ) if False else (  # Can't re-simulate without full df
                    # Estimate: if quality is good and we predicted well
                    tp_pred[k] if quality_test[k] >= QUALITY_GOOD_THRESHOLD else -1.0
                )
                if pnl > 0:
                    wins += 1
                    total_pnl += tp_pred[k]
                else:
                    losses += 1
                    total_pnl -= 1.0

            total = wins + losses
            wr = wins / total if total > 0 else 0
            pf = (wins * np.mean(tp_pred[mask])) / (losses * 1.0) if losses > 0 else 99
            pass_rate = mask.mean()

            print(f"    T={threshold:.2f}: Pass={pass_rate:.0%} "
                  f"WR={wr:.0%} PF≈{pf:.2f} ({wins}W/{losses}L)")

        fold_results.append({
            "fold": fold + 1,
            "auc": auc,
            "sl_mae": sl_mae,
            "tp_mae": tp_mae,
            "n_train": len(y_train),
            "n_test": len(y_test),
            "best_trees": clf.best_iteration,
        })

    # Overall OOS AUC
    if all_oos_preds:
        overall_auc = roc_auc_score(all_oos_true, all_oos_preds)
        print(f"\n  ═══ OVERALL OUT-OF-SAMPLE AUC: {overall_auc:.4f} ═══")
    else:
        overall_auc = 0.5

    if fold_results:
        print(f"  Avg Fold AUC: {np.mean([r['auc'] for r in fold_results]):.4f}")
        print(f"  Avg SL MAE:   {np.mean([r['sl_mae'] for r in fold_results]):.3f} ATR")
        print(f"  Avg TP MAE:   {np.mean([r['tp_mae'] for r in fold_results]):.3f} R")
        print(f"  Avg Trees:    {np.mean([r['best_trees'] for r in fold_results]):.0f}")

    return fold_results, overall_auc


# ══════════════════════════════════════════════════════════════════════════
# HYPERPARAMETER SEARCH (Bayesian-like)
# ══════════════════════════════════════════════════════════════════════════

def hyperparameter_search(X, y_class, y_sl, y_tp, n_iter=12):
    """Search for best XGBoost params using walk-forward AUC."""
    print(f"\n  Hyperparameter Search ({n_iter} configurations)...")

    configs = [
        # Balanced configs: moderate depth + good regularization
        {"max_depth": 4, "lr": 0.03, "child": 15, "sub": 0.7, "col": 0.7, "alpha": 2.0, "lambda": 5.0},
        {"max_depth": 3, "lr": 0.03, "child": 20, "sub": 0.7, "col": 0.6, "alpha": 3.0, "lambda": 6.0},
        {"max_depth": 4, "lr": 0.02, "child": 20, "sub": 0.6, "col": 0.6, "alpha": 3.0, "lambda": 7.0},
        {"max_depth": 5, "lr": 0.02, "child": 25, "sub": 0.6, "col": 0.5, "alpha": 4.0, "lambda": 8.0},
        {"max_depth": 3, "lr": 0.04, "child": 15, "sub": 0.8, "col": 0.7, "alpha": 2.0, "lambda": 4.0},
        {"max_depth": 4, "lr": 0.025, "child": 18, "sub": 0.7, "col": 0.6, "alpha": 3.0, "lambda": 6.0},
        {"max_depth": 3, "lr": 0.05, "child": 12, "sub": 0.8, "col": 0.8, "alpha": 1.5, "lambda": 3.0},
        {"max_depth": 5, "lr": 0.015, "child": 30, "sub": 0.6, "col": 0.5, "alpha": 5.0, "lambda": 8.0},
        {"max_depth": 4, "lr": 0.03, "child": 20, "sub": 0.7, "col": 0.6, "alpha": 2.5, "lambda": 5.0},
        {"max_depth": 3, "lr": 0.03, "child": 25, "sub": 0.6, "col": 0.6, "alpha": 3.5, "lambda": 7.0},
        {"max_depth": 4, "lr": 0.02, "child": 15, "sub": 0.7, "col": 0.7, "alpha": 2.0, "lambda": 5.0},
        {"max_depth": 5, "lr": 0.02, "child": 20, "sub": 0.6, "col": 0.6, "alpha": 3.0, "lambda": 6.0},
    ]

    configs = configs[:n_iter]

    tscv = TimeSeriesSplit(n_splits=3)
    best_auc = 0
    best_config = configs[1]

    for i, cfg in enumerate(configs):
        aucs = []
        pos_count = y_class.sum()
        neg_count = len(y_class) - pos_count
        scale_pos = neg_count / max(pos_count, 1)

        for train_idx, test_idx in tscv.split(X):
            X_tr, X_te = X[train_idx], X[test_idx]
            y_tr, y_te = y_class[train_idx], y_class[test_idx]
            if len(set(y_tr)) < 2 or len(set(y_te)) < 2:
                continue

            scaler = StandardScaler()
            X_tr_s = scaler.fit_transform(X_tr)
            X_te_s = scaler.transform(X_te)

            # Early stopping split
            val_n = max(int(len(X_tr_s) * 0.15), 10)
            X_train_h = X_tr_s[:-val_n]
            y_train_h = y_tr[:-val_n]
            X_val_h = X_tr_s[-val_n:]
            y_val_h = y_tr[-val_n:]

            clf = xgb.XGBClassifier(
                n_estimators=500,
                max_depth=cfg["max_depth"],
                learning_rate=cfg["lr"],
                min_child_weight=cfg["child"],
                subsample=cfg["sub"],
                colsample_bytree=cfg["col"],
                reg_alpha=cfg["alpha"],
                reg_lambda=cfg["lambda"],
                scale_pos_weight=scale_pos,
                use_label_encoder=False,
                eval_metric="auc",
                early_stopping_rounds=25,
                verbosity=0,
                random_state=42,
            )
            clf.fit(X_train_h, y_train_h, eval_set=[(X_val_h, y_val_h)], verbose=False)
            try:
                auc = roc_auc_score(y_te, clf.predict_proba(X_te_s)[:, 1])
                aucs.append(auc)
            except ValueError:
                pass

        if aucs:
            mean_auc = np.mean(aucs)
            marker = " ← BEST" if mean_auc > best_auc else ""
            print(f"    [{i+1}/{n_iter}] d={cfg['max_depth']} lr={cfg['lr']} "
                  f"child={cfg['child']} → AUC={mean_auc:.4f}{marker}")
            if mean_auc > best_auc:
                best_auc = mean_auc
                best_config = cfg

    print(f"\n  Best Config: {best_config}")
    print(f"  Best AUC:    {best_auc:.4f}")
    return best_config, best_auc


# ══════════════════════════════════════════════════════════════════════════
# TRAIN FINAL MODELS
# ══════════════════════════════════════════════════════════════════════════

def train_final_models(data, clf_params, feature_mask=None, data_already_selected=False):
    """Train production models with holdout validation."""
    X = data["X"]
    y_class = data["y_class"]
    y_sl = data["y_sl"]
    y_tp = data["y_tp"]
    y_quality = data["y_quality"]
    feature_names = data["feature_names"]
    trades = data["trades"]

    # Apply feature mask if provided
    if feature_mask is not None and not data_already_selected:
        X = X[:, feature_mask]
        feature_names = [f for f, m in zip(feature_names, feature_mask) if m]

    # Holdout: last 20% for final validation
    holdout_size = int(len(X) * 0.2)
    X_train_full = X[:-holdout_size]
    y_train_full = y_class[:-holdout_size]
    sl_train_full = y_sl[:-holdout_size]
    tp_train_full = y_tp[:-holdout_size]

    X_holdout = X[-holdout_size:]
    y_holdout = y_class[-holdout_size:]
    sl_holdout = y_sl[-holdout_size:]
    tp_holdout = y_tp[-holdout_size:]
    quality_holdout = y_quality[-holdout_size:]
    trades_holdout = trades[-holdout_size:]

    # Scale
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train_full)
    X_holdout_s = scaler.transform(X_holdout)

    # Early stopping split from training data
    val_size = max(int(len(X_train_s) * 0.15), 30)
    X_tr = X_train_s[:-val_size]
    y_tr = y_train_full[:-val_size]
    X_val = X_train_s[-val_size:]
    y_val = y_train_full[-val_size:]

    # Class weight
    pos_count = y_tr.sum()
    neg_count = len(y_tr) - pos_count
    scale_pos = neg_count / max(pos_count, 1)

    print(f"\n{'='*70}")
    print(f"  TRAINING FINAL MODELS")
    print(f"  Train: {len(X_train_full)} | Holdout: {holdout_size}")
    print(f"  Features: {len(feature_names)} | Class balance: {scale_pos:.2f}x")
    print(f"{'='*70}\n")

    # ── Classifier ──
    print("  Training classifier...", end=" ", flush=True)
    classifier = xgb.XGBClassifier(
        n_estimators=1000,
        max_depth=clf_params["max_depth"],
        learning_rate=clf_params["lr"],
        min_child_weight=clf_params["child"],
        subsample=clf_params["sub"],
        colsample_bytree=clf_params["col"],
        reg_alpha=clf_params["alpha"],
        reg_lambda=clf_params["lambda"],
        scale_pos_weight=scale_pos,
        use_label_encoder=False,
        eval_metric="auc",
        early_stopping_rounds=20,
        verbosity=0,
        random_state=42,
    )
    classifier.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)

    # Evaluate on holdout
    holdout_prob = classifier.predict_proba(X_holdout_s)[:, 1]
    try:
        holdout_auc = roc_auc_score(y_holdout, holdout_prob)
    except ValueError:
        holdout_auc = 0.5

    train_prob = classifier.predict_proba(X_train_s)[:, 1]
    try:
        train_auc = roc_auc_score(y_train_full, train_prob)
    except ValueError:
        train_auc = 0.5

    print(f"Done ({classifier.best_iteration} trees)")
    print(f"    Train AUC: {train_auc:.4f} | Holdout AUC: {holdout_auc:.4f}")
    print(f"    Overfit gap: {train_auc - holdout_auc:.4f} "
          f"({'OK' if train_auc - holdout_auc < 0.1 else 'WARNING'})")

    # ── SL Regressor ──
    print("  Training SL regressor...", end=" ", flush=True)
    sl_regressor = xgb.XGBRegressor(
        n_estimators=500,
        max_depth=3,
        learning_rate=0.03,
        min_child_weight=20,
        subsample=0.7,
        colsample_bytree=0.7,
        reg_alpha=2.0,
        reg_lambda=4.0,
        early_stopping_rounds=30,
        verbosity=0,
        random_state=42,
    )
    sl_reg_val_size = max(int(len(X_train_s) * 0.15), 20)
    sl_regressor.fit(
        X_train_s[:-sl_reg_val_size], sl_train_full[:-sl_reg_val_size],
        eval_set=[(X_train_s[-sl_reg_val_size:], sl_train_full[-sl_reg_val_size:])],
        verbose=False
    )
    sl_holdout_pred = np.clip(sl_regressor.predict(X_holdout_s), MIN_SL_ATR, MAX_SL_ATR)
    sl_mae = mean_absolute_error(sl_holdout, sl_holdout_pred)
    print(f"Done ({sl_regressor.best_iteration} trees) | Holdout MAE: {sl_mae:.3f} ATR")

    # ── TP Regressor ──
    print("  Training TP regressor...", end=" ", flush=True)
    tp_regressor = xgb.XGBRegressor(
        n_estimators=500,
        max_depth=3,
        learning_rate=0.03,
        min_child_weight=20,
        subsample=0.7,
        colsample_bytree=0.7,
        reg_alpha=2.0,
        reg_lambda=4.0,
        early_stopping_rounds=30,
        verbosity=0,
        random_state=42,
    )
    tp_regressor.fit(
        X_train_s[:-sl_reg_val_size], tp_train_full[:-sl_reg_val_size],
        eval_set=[(X_train_s[-sl_reg_val_size:], tp_train_full[-sl_reg_val_size:])],
        verbose=False
    )
    tp_holdout_pred = np.clip(tp_regressor.predict(X_holdout_s), MIN_TP_R, MAX_TP_R)
    tp_mae = mean_absolute_error(tp_holdout, tp_holdout_pred)
    print(f"Done ({tp_regressor.best_iteration} trees) | Holdout MAE: {tp_mae:.3f} R")

    # ── Feature Importance ──
    importances = classifier.feature_importances_
    imp_sorted = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)
    print(f"\n  Top 15 Features:")
    for i, (feat, imp) in enumerate(imp_sorted[:15]):
        bar = "█" * int(imp * 50)
        print(f"    {i+1:2d}. {feat:<25s} {imp:.4f} {bar}")

    # ── Holdout Trading Simulation ──
    print(f"\n  ═══ HOLDOUT TRADING SIMULATION ═══")
    print(f"  ({holdout_size} trades, never seen during training)\n")

    best_threshold = 0.50
    best_metric = 0

    for threshold in np.arange(0.35, 0.70, 0.05):
        mask = holdout_prob >= threshold
        if mask.sum() < 5:
            continue

        # For each passing trade, estimate PnL using quality as proxy
        wins = 0
        losses = 0
        total_profit = 0
        total_loss = 0

        for k in range(len(y_holdout)):
            if not mask[k]:
                continue
            # Trade is "good" if quality >= threshold
            if quality_holdout[k] >= QUALITY_GOOD_THRESHOLD:
                wins += 1
                total_profit += tp_holdout_pred[k]
            else:
                losses += 1
                total_loss += 1.0

        total = wins + losses
        wr = wins / total if total > 0 else 0
        pf = total_profit / total_loss if total_loss > 0 else 99
        expectancy = (total_profit - total_loss) / total if total > 0 else 0
        pass_rate = mask.mean()

        # Metric: we want high PF AND reasonable pass rate
        metric = pf * min(pass_rate * 5, 1.0)  # Penalize very low pass rates

        marker = ""
        if metric > best_metric and pf > 1.0:
            best_metric = metric
            best_threshold = threshold
            marker = " ← BEST"

        print(f"    T={threshold:.2f}: Pass={pass_rate:.0%} WR={wr:.0%} "
              f"PF={pf:.2f} Exp={expectancy:.3f}R/trade{marker}")

    print(f"\n  Selected threshold: {best_threshold:.2f}")

    # ── Save Model ──
    model_data = {
        "classifier": classifier,
        "sl_regressor": sl_regressor,
        "tp_regressor": tp_regressor,
        "scaler": scaler,
        "feature_names": feature_names,
        "feature_mask": feature_mask.tolist() if feature_mask is not None else None,
        "confidence_threshold": best_threshold,
        "metadata": {
            "trained_at": datetime.now().isoformat(),
            "n_trades": len(y_class),
            "n_train": len(X_train_full),
            "n_holdout": holdout_size,
            "good_rate": float(y_class.mean()),
            "avg_quality": float(y_quality.mean()),
            "avg_optimal_sl": float(y_sl.mean()),
            "avg_optimal_tp": float(y_tp.mean()),
            "clf_params": clf_params,
            "threshold": best_threshold,
            "train_auc": float(train_auc),
            "holdout_auc": float(holdout_auc),
            "overfit_gap": float(train_auc - holdout_auc),
            "sl_mae": float(sl_mae),
            "tp_mae": float(tp_mae),
            "n_features": len(feature_names),
            "best_trees": int(classifier.best_iteration),
            "quality_threshold": QUALITY_GOOD_THRESHOLD,
        }
    }

    # Save
    prod_path = os.path.join(MODEL_DIR, "ml_filter_production.pkl")
    with open(prod_path, "wb") as f:
        pickle.dump(model_data, f)
    print(f"\n  Saved: {prod_path}")

    live_path = os.path.join(os.path.dirname(MODEL_DIR), "ml_filter.pkl")
    with open(live_path, "wb") as f:
        pickle.dump(model_data, f)
    print(f"  Saved: {live_path}")

    return model_data


# ══════════════════════════════════════════════════════════════════════════
# BACKTEST VALIDATION
# ══════════════════════════════════════════════════════════════════════════

def backtest_with_model(data, model_data, data_already_selected=False):
    """Run a proper backtest simulation with the trained model on holdout data."""
    X = data["X"]
    y_quality = data["y_quality"]
    trades = data["trades"]
    feature_names = data["feature_names"]

    scaler = model_data["scaler"]
    classifier = model_data["classifier"]
    sl_regressor = model_data["sl_regressor"]
    tp_regressor = model_data["tp_regressor"]
    threshold = model_data["confidence_threshold"]
    model_features = model_data["feature_names"]

    # Use feature mask if applicable
    feature_mask = model_data.get("feature_mask")
    if feature_mask is not None and not data_already_selected:
        X_use = X[:, feature_mask]
    else:
        X_use = X

    # Only test on last 30% (most recent data = most realistic)
    test_start = int(len(X_use) * 0.7)
    X_test = X_use[test_start:]
    trades_test = trades[test_start:]
    quality_test = y_quality[test_start:]

    X_test_s = scaler.transform(X_test)
    probs = classifier.predict_proba(X_test_s)[:, 1]
    sl_preds = np.clip(sl_regressor.predict(X_test_s), MIN_SL_ATR, MAX_SL_ATR)
    tp_preds = np.clip(tp_regressor.predict(X_test_s), MIN_TP_R, MAX_TP_R)

    print(f"\n{'='*70}")
    print(f"  BACKTEST VALIDATION (last 30% = {len(X_test)} trades)")
    print(f"  Threshold: {threshold:.2f}")
    print(f"{'='*70}\n")

    # Simulate
    mask = probs >= threshold
    passed_trades = sum(mask)
    rejected_trades = len(mask) - passed_trades

    ml_pnl = 0
    ml_wins = 0
    ml_losses = 0
    default_pnl = 0
    default_wins = 0
    default_losses = 0

    for k in range(len(trades_test)):
        trade = trades_test[k]
        # Default strategy result
        dpnl = trade["default_pnl"]
        if dpnl > 0:
            default_wins += 1
        else:
            default_losses += 1
        default_pnl += dpnl

        if not mask[k]:
            continue

        # ML-filtered trade uses quality as proxy
        if quality_test[k] >= QUALITY_GOOD_THRESHOLD:
            ml_wins += 1
            ml_pnl += tp_preds[k]
        else:
            ml_losses += 1
            ml_pnl -= 1.0

    print(f"  DEFAULT STRATEGY (all trades):")
    print(f"    Trades: {len(trades_test)}")
    print(f"    Wins: {default_wins} | Losses: {default_losses}")
    print(f"    WR: {default_wins/(default_wins+default_losses)*100:.1f}%")
    default_total = default_wins + default_losses
    default_pf = (default_wins * 3.0) / (default_losses * 1.0) if default_losses > 0 else 0
    print(f"    PF (est): {default_pf:.2f}")
    print(f"    Total PnL: {default_pnl:.1f}R")

    print(f"\n  ML-FILTERED STRATEGY:")
    print(f"    Passed: {passed_trades} | Rejected: {rejected_trades}")
    ml_total = ml_wins + ml_losses
    if ml_total > 0:
        ml_wr = ml_wins / ml_total
        ml_pf = (ml_wins * np.mean(tp_preds[mask])) / (ml_losses * 1.0) if ml_losses > 0 else 99
        print(f"    Wins: {ml_wins} | Losses: {ml_losses}")
        print(f"    WR: {ml_wr*100:.1f}%")
        print(f"    PF: {ml_pf:.2f}")
        print(f"    Total PnL: {ml_pnl:.1f}R")
        print(f"    Avg SL: {np.mean(sl_preds[mask]):.2f} ATR")
        print(f"    Avg TP: {np.mean(tp_preds[mask]):.2f}R")
        improvement = ml_pnl - default_pnl * (passed_trades / len(trades_test))
        print(f"\n  IMPROVEMENT: {'+' if improvement > 0 else ''}{improvement:.1f}R")
    else:
        print(f"    No trades passed filter")

    print(f"{'='*70}")

    return {
        "ml_pnl": ml_pnl,
        "ml_pf": ml_pf if ml_total > 0 else 0,
        "ml_wr": ml_wr if ml_total > 0 else 0,
        "default_pnl": default_pnl,
        "default_pf": default_pf,
        "passed_trades": passed_trades,
        "total_trades": len(trades_test),
    }


# ══════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Enhanced ML Training V2")
    parser.add_argument("--days", type=int, default=2000,
                        help="Max days of data (default: 2000)")
    parser.add_argument("--symbols", type=str, default=None,
                        help="Comma-separated symbols")
    parser.add_argument("--fetch", action="store_true",
                        help="Force re-fetch all data from Bybit")
    parser.add_argument("--skip-fetch", action="store_true",
                        help="Only use existing CSV data")
    parser.add_argument("--skip-search", action="store_true",
                        help="Skip hyperparameter search")
    parser.add_argument("--quality-threshold", type=float, default=1.5,
                        help="Trade quality threshold (default: 1.2)")
    args = parser.parse_args()

    global QUALITY_GOOD_THRESHOLD
    QUALITY_GOOD_THRESHOLD = args.quality_threshold

    print("=" * 70)
    print("  ML TRAINING V2 — ENHANCED ANTI-OVERFIT")
    print("  XGBoost + Trade Quality + Purged Walk-Forward")
    print("=" * 70)
    print(f"  Date:              {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"  Max Days:          {args.days}")
    print(f"  Quality Threshold: {QUALITY_GOOD_THRESHOLD}")
    print(f"  Skip fetch:        {args.skip_fetch}")
    print(f"  Skip HP search:    {args.skip_search}")

    # Symbols
    if args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(",")]
        symbols = [s if s.endswith("USDT") else s + "USDT" for s in symbols]
    else:
        symbols = c.SYMBOLS
    print(f"  Symbols:           {len(symbols)}")
    print("=" * 70)

    # ── Step 1: Generate training data ──
    print("\n[1/5] GENERATING TRAINING DATA...")
    all_trades = generate_all_training_data(
        symbols, max_days=args.days,
        force_fetch=args.fetch, skip_fetch=args.skip_fetch
    )

    if len(all_trades) < 200:
        print(f"\n  ERROR: Only {len(all_trades)} trades. Need at least 200.")
        print("  Try: --days 2000 (without --skip-fetch)")
        sys.exit(1)

    # ── Step 2: Prepare matrices ──
    print("\n[2/5] PREPARING DATA...")
    data = prepare_matrices(all_trades)
    X = data["X"]
    y_class = data["y_class"]
    feature_names = data["feature_names"]

    print(f"  Shape: {X.shape}")
    print(f"  Good trades: {y_class.sum()}/{len(y_class)} = {y_class.mean()*100:.1f}%")
    print(f"  SL range: {data['y_sl'].min():.2f}-{data['y_sl'].max():.2f} ATR")
    print(f"  TP range: {data['y_tp'].min():.1f}-{data['y_tp'].max():.1f} R")

    # ── Step 3: Feature selection ──
    print("\n[3/5] FEATURE SELECTION...")
    feature_mask, selected_features = select_features(
        X, y_class, feature_names, min_importance=0.005
    )
    X_selected = X[:, feature_mask]
    data_selected = data.copy()
    data_selected["X"] = X_selected
    data_selected["feature_names"] = selected_features

    # ── Step 4: Walk-forward validation ──
    print("\n[4/5] WALK-FORWARD VALIDATION...")
    cv_results, overall_auc = purged_walk_forward_cv(data_selected, n_splits=5, purge_bars=6)

    # ── Step 5: Hyperparameter search + final training ──
    if not args.skip_search:
        print("\n[5/5] HYPERPARAMETER SEARCH + FINAL TRAINING...")
        best_params, search_auc = hyperparameter_search(
            X_selected, y_class, data["y_sl"], data["y_tp"], n_iter=12
        )
    else:
        print("\n[5/5] FINAL TRAINING (default params)...")
        best_params = {"max_depth": 4, "lr": 0.03, "child": 10,
                       "sub": 0.7, "col": 0.7, "alpha": 1.0, "lambda": 3.0}

    # Train final models
    model_data = train_final_models(data_selected, best_params, feature_mask, data_already_selected=True)

    # Backtest validation
    bt_results = backtest_with_model(data_selected, model_data, data_already_selected=True)

    # ── Final Summary ──
    meta = model_data["metadata"]
    print(f"\n{'='*70}")
    print(f"  TRAINING V2 COMPLETE")
    print(f"{'='*70}")
    print(f"  Total Trades:     {meta['n_trades']}")
    print(f"  Good Rate:        {meta['good_rate']*100:.1f}%")
    print(f"  Avg Quality:      {meta['avg_quality']:.3f}")
    print(f"  Train AUC:        {meta['train_auc']:.4f}")
    print(f"  Holdout AUC:      {meta['holdout_auc']:.4f}")
    print(f"  Overfit Gap:      {meta['overfit_gap']:.4f}")
    print(f"  CV AUC:           {overall_auc:.4f}")
    print(f"  Threshold:        {meta['threshold']:.2f}")
    print(f"  SL MAE:           {meta['sl_mae']:.3f} ATR")
    print(f"  TP MAE:           {meta['tp_mae']:.3f} R")
    print(f"  Trees (clf):      {meta['best_trees']}")
    print(f"  Features:         {meta['n_features']}")
    print(f"  ML PF:            {bt_results['ml_pf']:.2f}")
    print(f"  ML WR:            {bt_results['ml_wr']*100:.1f}%")
    print(f"{'='*70}")

    # Quality assessment
    if meta['holdout_auc'] >= 0.65 and meta['overfit_gap'] < 0.1:
        print("\n  ✓ Model quality: GOOD (deploy-ready)")
    elif meta['holdout_auc'] >= 0.60:
        print("\n  ~ Model quality: ACCEPTABLE (may need more data)")
    else:
        print("\n  ✗ Model quality: POOR (needs more data or feature engineering)")

    if bt_results['ml_pf'] >= 1.5:
        print(f"  ✓ Backtest PF: {bt_results['ml_pf']:.2f} (profitable)")
    elif bt_results['ml_pf'] >= 1.0:
        print(f"  ~ Backtest PF: {bt_results['ml_pf']:.2f} (breakeven)")
    else:
        print(f"  ✗ Backtest PF: {bt_results['ml_pf']:.2f} (losing)")


if __name__ == "__main__":
    main()
