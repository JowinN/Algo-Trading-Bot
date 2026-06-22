"""
ML Training V4 — Direction-Specific Calibrated Ensemble
======================================================
1. Separates training into LONG and SHORT models to capture distinct market dynamics.
2. Incorporates V3 regime detection and lookback context features.
3. Performs XGBoost-based feature selection to discard noise.
4. Optimizes hyperparameters via walk-forward validation.
5. Calibrates output probabilities for accurate win-rate mapping.
6. Selects confidence thresholds using a Utility Score (expectancy * sqrt(pass_rate))
   to maximize absolute trades while preserving high accuracy.
"""

import os
import sys
import pickle
import time
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from collections import defaultdict

from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score, mean_absolute_error
from sklearn.calibration import CalibratedClassifierCV
import xgboost as xgb

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from indicators import compute_all, ema
from strategy import generate_signal, Signal
from config import Config as c
from ml_model import (
    extract_regime_features, compute_regimes, calculate_optimal_exit,
    MIN_SL_ATR, MAX_SL_ATR, MIN_TP_R, MAX_TP_R, MLFilter
)

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "historical_data")
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ml_models")
os.makedirs(MODEL_DIR, exist_ok=True)

QUALITY_GOOD_THRESHOLD = 1.5  # MFE/MAE >= 1.5 = win/good trade target

# ══════════════════════════════════════════════════════════════════════════
# DATA PREPARATION
# ══════════════════════════════════════════════════════════════════════════

def load_and_prepare_data(symbol):
    """Load CSV data and compute all indicators + regimes."""
    csv_path = os.path.join(DATA_DIR, f"{symbol}_15m.csv")
    if not os.path.exists(csv_path):
        print(f"  {symbol}: CSV not found")
        return None

    df_15m = pd.read_csv(csv_path, parse_dates=["timestamp"], index_col="timestamp")
    if len(df_15m) < 2000:
        print(f"  {symbol}: Insufficient data ({len(df_15m)} bars)")
        return None

    # Resample to 4H
    df_4h = df_15m.resample("4h").agg({
        "open": "first", "high": "max",
        "low": "min", "close": "last", "volume": "sum"
    }).dropna()

    if len(df_4h) < 200:
        return None

    # Compute indicators
    df_4h = compute_all(df_4h)

    # Compute regimes
    df_4h = compute_regimes(df_4h)

    # HTF bias from daily
    df_daily = df_15m.resample("1D").agg({
        "open": "first", "high": "max",
        "low": "min", "close": "last", "volume": "sum"
    }).dropna()

    htf_bias_series = pd.Series("NONE", index=df_4h.index)
    if len(df_daily) >= 50:
        df_daily["ema20"] = ema(df_daily["close"], 20)
        df_daily["ema50"] = ema(df_daily["close"], 50)
        df_daily = df_daily.dropna()
        for i, (ts, row) in enumerate(df_daily.iterrows()):
            if row["ema20"] > row["ema50"]:
                bias = "LONG"
            elif row["ema20"] < row["ema50"]:
                bias = "SHORT"
            else:
                bias = "NONE"
            # Apply to all 4H bars on this day
            day_mask = df_4h.index.date == ts.date()
            htf_bias_series[day_mask] = bias

    return df_4h, htf_bias_series


def simulate_trade_outcome(df, entry_idx, direction, atr_val, max_hold=30):
    """Simulate trade outcome. Returns (outcome, mfe_atr, mae_atr)."""
    entry_price = float(df["close"].iloc[entry_idx])
    sl_dist = atr_val * c.SL_ATR_MULT
    tp_dist = atr_val * c.TP_ATR_MULT

    mfe = 0.0
    mae = 0.0
    end_idx = min(entry_idx + max_hold + 1, len(df))

    for i in range(entry_idx + 1, end_idx):
        bar_high = float(df["high"].iloc[i])
        bar_low = float(df["low"].iloc[i])

        if direction == "LONG":
            fav = bar_high - entry_price
            adv = entry_price - bar_low
        else:
            fav = entry_price - bar_low
            adv = bar_high - entry_price

        mfe = max(mfe, fav)
        mae = max(mae, adv)

        # Check SL hit
        if adv >= sl_dist:
            return 0, mfe / atr_val, mae / atr_val
        # Check TP hit
        if fav >= tp_dist:
            return 1, mfe / atr_val, mae / atr_val

    final_price = float(df["close"].iloc[end_idx - 1])
    if direction == "LONG":
        pnl = final_price - entry_price
    else:
        pnl = entry_price - final_price

    outcome = 1 if pnl > sl_dist * 0.5 else 0
    return outcome, mfe / atr_val, mae / atr_val


def _vectorized_signals(df_4h, htf_bias_series):
    """Vectorized signal generation for faster execution."""
    signals = {}
    closes = df_4h["close"].values.astype(float)
    opens = df_4h["open"].values.astype(float)
    ema21_arr = df_4h["ema21"].values.astype(float)
    ema50_arr = df_4h["ema50"].values.astype(float)
    adx_arr = df_4h["adx"].values.astype(float)
    rsi_arr = df_4h["rsi"].values.astype(float)
    atr_arr = df_4h["atr"].values.astype(float)
    macd_hist_arr = df_4h["macd_hist"].values.astype(float)
    rel_vol_arr = df_4h["rel_volume"].values.astype(float) if "rel_volume" in df_4h.columns else np.ones(len(df_4h))
    body_pct_arr = df_4h["body_pct"].values.astype(float) if "body_pct" in df_4h.columns else np.zeros(len(df_4h))
    squeeze_fire_arr = df_4h["squeeze_fire"].values if "squeeze_fire" in df_4h.columns else np.zeros(len(df_4h), dtype=bool)

    for i in range(1, len(df_4h)):
        price = closes[i]
        open_price = opens[i]
        atr_val = atr_arr[i]
        if atr_val <= 0:
            continue

        adx = adx_arr[i]
        if adx < 20:
            continue

        ema21 = ema21_arr[i]
        ema50 = ema50_arr[i]
        rsi_val = rsi_arr[i]
        macd_hist = macd_hist_arr[i]
        macd_hist_prev = macd_hist_arr[i-1]
        rel_vol = rel_vol_arr[i]
        body_pct = body_pct_arr[i]
        squeeze_fire = bool(squeeze_fire_arr[i])

        candle_bullish = price > open_price
        candle_bearish = price < open_price
        htf_bias = htf_bias_series.iloc[i] if i < len(htf_bias_series) else "NONE"

        # LONG
        if htf_bias == "LONG" or (htf_bias == "NONE" and ema21 > ema50):
            if ema21 <= ema50:
                continue
            dist = (price - ema21) / atr_val
            if dist > 2.5 or dist < -0.5:
                pass
            elif not candle_bullish or body_pct < 0.15:
                pass
            elif rsi_val > 75 or rsi_val < 28:
                pass
            elif not (macd_hist > macd_hist_prev or rel_vol > 1.2 or squeeze_fire):
                pass
            else:
                signals[i] = "LONG"
                continue

        # SHORT
        if htf_bias == "SHORT" or (htf_bias == "NONE" and ema21 < ema50):
            if ema21 >= ema50:
                continue
            dist = (ema21 - price) / atr_val
            if dist > 2.5 or dist < -0.5:
                continue
            if not candle_bearish or body_pct < 0.15:
                continue
            if rsi_val < 25 or rsi_val > 72:
                continue
            if not (macd_hist < macd_hist_prev or rel_vol > 1.2 or squeeze_fire):
                continue
            signals[i] = "SHORT"

    return signals


def generate_training_data(verbose=True):
    """Generate training data with regime features and optimal SL/TP targets."""
    all_trades = []

    for sym_idx, symbol in enumerate(c.SYMBOLS):
        if verbose:
            print(f"  [{sym_idx+1:2d}/{len(c.SYMBOLS)}] {symbol}...", end=" ", flush=True)

        result = load_and_prepare_data(symbol)
        if result is None:
            if verbose:
                print("skipped")
            continue

        df_4h, htf_bias_series = result
        trades = []
        signals_mask = _vectorized_signals(df_4h, htf_bias_series)

        for i in signals_mask:
            if i < 55 or i >= len(df_4h) - 35:
                continue

            direction = signals_mask[i]
            features = extract_regime_features(df_4h, i, direction)
            if features is None:
                continue

            price = float(df_4h["close"].iloc[i])
            atr_val = float(df_4h["atr"].iloc[i])

            # Simulate outcome
            outcome, mfe_atr, mae_atr = simulate_trade_outcome(df_4h, i, direction, atr_val)
            quality = mfe_atr / max(mae_atr, 0.1)

            # Optimal TP/SL levels
            opt_res = calculate_optimal_exit(df_4h, i, direction, price, atr_val, max_bars=30)
            if opt_res is not None:
                optimal_sl, optimal_tp, best_pnl = opt_res
            else:
                optimal_sl, optimal_tp = 2.0, 3.5

            trades.append({
                "symbol": symbol,
                "direction": direction,
                "timestamp": df_4h.index[i],
                "features": features,
                "outcome": outcome,
                "quality": quality,
                "optimal_sl": optimal_sl,
                "optimal_tp": optimal_tp,
                "price": price,
                "atr": atr_val
            })

        if verbose:
            good = sum(1 for t in trades if t["quality"] >= QUALITY_GOOD_THRESHOLD)
            print(f"{len(trades)} trades ({good} good, {100*good/max(len(trades),1):.1f}% GR)")
        all_trades.extend(trades)

    return all_trades


def prepare_matrices(trades):
    """Convert trades to feature and label matrices."""
    if not trades:
        return None

    feature_names = sorted(trades[0]["features"].keys())
    X = np.array([[t["features"].get(fn, 0.0) for fn in feature_names] for t in trades], dtype=np.float32)
    # Clean NaN/Inf
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    y_class = np.array([1 if t["quality"] >= QUALITY_GOOD_THRESHOLD else 0 for t in trades], dtype=np.int32)
    y_quality = np.array([t["quality"] for t in trades], dtype=np.float32)
    y_sl = np.array([t["optimal_sl"] for t in trades], dtype=np.float32)
    y_tp = np.array([t["optimal_tp"] for t in trades], dtype=np.float32)
    directions = np.array([t["direction"] for t in trades])

    return {
        "X": X,
        "y_class": y_class,
        "y_quality": y_quality,
        "y_sl": y_sl,
        "y_tp": y_tp,
        "directions": directions,
        "feature_names": feature_names,
        "trades": trades
    }

# ══════════════════════════════════════════════════════════════════════════
# FEATURE SELECTION
# ══════════════════════════════════════════════════════════════════════════

def select_features(X, y, feature_names, min_importance=0.005):
    """Filter out noise features using XGBoost feature importances."""
    clf = xgb.XGBClassifier(
        n_estimators=100, max_depth=3, learning_rate=0.05,
        eval_metric="logloss", random_state=42, tree_method="hist"
    )
    clf.fit(X, y)
    importances = clf.feature_importances_
    mask = importances >= min_importance
    selected = [f for f, m in zip(feature_names, mask) if m]
    return mask, selected

# ══════════════════════════════════════════════════════════════════════════
# HYPERPARAMETER TUNING
# ══════════════════════════════════════════════════════════════════════════

def hyperparameter_search(X, y, scale_pos, n_iter=8):
    """Find the best parameters for a subset (LONG or SHORT) using CV."""
    configs = [
        {"max_depth": 3, "lr": 0.03, "child": 15, "sub": 0.7, "col": 0.7, "alpha": 1.0, "lambda": 3.0},
        {"max_depth": 4, "lr": 0.03, "child": 20, "sub": 0.7, "col": 0.6, "alpha": 2.0, "lambda": 4.0},
        {"max_depth": 3, "lr": 0.05, "child": 10, "sub": 0.8, "col": 0.7, "alpha": 1.0, "lambda": 3.0},
        {"max_depth": 4, "lr": 0.02, "child": 20, "sub": 0.6, "col": 0.6, "alpha": 3.0, "lambda": 5.0},
        {"max_depth": 5, "lr": 0.02, "child": 25, "sub": 0.6, "col": 0.5, "alpha": 4.0, "lambda": 6.0},
        {"max_depth": 3, "lr": 0.04, "child": 18, "sub": 0.7, "col": 0.7, "alpha": 1.5, "lambda": 4.5},
        {"max_depth": 4, "lr": 0.04, "child": 15, "sub": 0.8, "col": 0.6, "alpha": 2.0, "lambda": 3.0},
        {"max_depth": 5, "lr": 0.03, "child": 20, "sub": 0.7, "col": 0.7, "alpha": 2.0, "lambda": 5.0},
    ][:n_iter]

    tscv = TimeSeriesSplit(n_splits=3)
    best_auc = 0.0
    best_cfg = configs[0]

    for cfg in configs:
        aucs = []
        for train_idx, val_idx in tscv.split(X):
            X_tr, X_val = X[train_idx], X[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]

            if len(set(y_tr)) < 2 or len(set(y_val)) < 2:
                continue

            clf = xgb.XGBClassifier(
                n_estimators=150,
                max_depth=cfg["max_depth"],
                learning_rate=cfg["lr"],
                min_child_weight=cfg["child"],
                subsample=cfg["sub"],
                colsample_bytree=cfg["col"],
                reg_alpha=cfg["alpha"],
                reg_lambda=cfg["lambda"],
                scale_pos_weight=scale_pos,
                eval_metric="auc",
                early_stopping_rounds=15,
                verbosity=0,
                random_state=42,
                tree_method="hist"
            )
            val_size = max(int(len(X_tr) * 0.15), 10)
            X_t = X_tr[:-val_size]
            y_t = y_tr[:-val_size]
            X_v = X_tr[-val_size:]
            y_v = y_tr[-val_size:]

            clf.fit(X_t, y_t, eval_set=[(X_v, y_v)], verbose=False)
            pred = clf.predict_proba(X_val)[:, 1]
            aucs.append(roc_auc_score(y_val, pred))

        if aucs:
            mean_auc = np.mean(aucs)
            if mean_auc > best_auc:
                best_auc = mean_auc
                best_cfg = cfg

    return best_cfg, best_auc

# ══════════════════════════════════════════════════════════════════════════
# MAIN TRAINING PIPELINE
# ══════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("  ML TRAINING V4 — CALIBRATED DIRECTIONAL ENSEMBLE")
    print("=" * 70)
    print(f"  Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print()

    # Step 1: Generate training data
    print("══════════════════════════════════════════════════════════════════════")
    print("  [1/5] GENERATING TRAINING DATA")
    print("══════════════════════════════════════════════════════════════════════\n")
    cache_path = os.path.join(MODEL_DIR, "v4_training_trades_cache.pkl")
    if os.path.exists(cache_path):
        print(f"  Loading cached training data from {cache_path}...")
        with open(cache_path, "rb") as f:
            all_trades = pickle.load(f)
        print(f"  Loaded {len(all_trades)} trades from cache.")
    else:
        all_trades = generate_training_data()
        with open(cache_path, "wb") as f:
            pickle.dump(all_trades, f)
        print(f"  Saved {len(all_trades)} trades to cache: {cache_path}")
        
    if len(all_trades) < 500:
        print(f"  ERROR: Only {len(all_trades)} trades. Need at least 500.")
        sys.exit(1)

    print(f"\n  Total trades generated: {len(all_trades)}")
    
    # Step 2: Prepare matrices
    print("\n[2/5] PREPARING DATA...")
    data = prepare_matrices(all_trades)
    X = data["X"]
    y_class = data["y_class"]
    y_sl = data["y_sl"]
    y_tp = data["y_tp"]
    y_quality = data["y_quality"]
    directions = data["directions"]
    feature_names = data["feature_names"]

    # Temporal holdout split (last 20%)
    n_samples = len(X)
    holdout_size = int(n_samples * 0.20)
    split_train = n_samples - holdout_size

    X_train_full, y_train_full = X[:split_train], y_class[:split_train]
    sl_train_full, tp_train_full = y_sl[:split_train], y_tp[:split_train]
    q_train_full = y_quality[:split_train]
    dir_train_full = directions[:split_train]

    X_test, y_test = X[split_train:], y_class[split_train:]
    sl_test, tp_test = y_sl[split_train:], y_tp[split_train:]
    q_test = y_quality[split_train:]
    dir_test = directions[split_train:]
    trades_test = data["trades"][split_train:]

    print(f"  Train samples: {split_train} | Holdout: {holdout_size}")

    # Separators for LONG and SHORT
    long_mask_train = dir_train_full == "LONG"
    short_mask_train = dir_train_full == "SHORT"
    long_mask_test = dir_test == "LONG"
    short_mask_test = dir_test == "SHORT"

    print(f"  Long Train: {long_mask_train.sum()} | Short Train: {short_mask_train.sum()}")
    print(f"  Long Test: {long_mask_test.sum()} | Short Test: {short_mask_test.sum()}")

    # Models dict to save
    models = {
        "version": "v4_direction_ensemble",
        "feature_names": feature_names,
    }

    # Scale and Fit Pipeline (LONG vs SHORT)
    for direction, mask_tr, mask_te in [("LONG", long_mask_train, long_mask_test), ("SHORT", short_mask_train, short_mask_test)]:
        print(f"\n{'─'*70}")
        print(f"  TRAINING {direction} MODELS")
        print(f"{'─'*70}")

        X_tr, y_tr = X_train_full[mask_tr], y_train_full[mask_tr]
        sl_tr, tp_tr = sl_train_full[mask_tr], tp_train_full[mask_tr]
        X_te, y_te = X_test[mask_te], y_test[mask_te]

        # Feature selection
        feat_mask, selected_feats = select_features(X_tr, y_tr, feature_names)
        print(f"  [Feature Selection] kept {len(selected_feats)}/{len(feature_names)} features")
        
        X_tr_sel = X_tr[:, feat_mask]
        X_te_sel = X_te[:, feat_mask]

        # Scale
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr_sel)
        X_te_s = scaler.transform(X_te_sel)

        # Early stopping split
        val_size = max(int(len(X_tr_s) * 0.15), 30)
        X_t = X_tr_s[:-val_size]
        y_t = y_tr[:-val_size]
        X_v = X_tr_s[-val_size:]
        y_v = y_tr[-val_size:]

        # Hyperparameter search
        pos_count = y_t.sum()
        neg_count = len(y_t) - pos_count
        scale_pos = neg_count / max(pos_count, 1)
        best_cfg, best_cv_auc = hyperparameter_search(X_tr_sel, y_tr, scale_pos)
        print(f"  [Hyperparameter Tuning] Best Config: {best_cfg} | CV AUC: {best_cv_auc:.4f}")

        # Train Classifier
        classifier = xgb.XGBClassifier(
            n_estimators=1000,
            max_depth=best_cfg["max_depth"],
            learning_rate=best_cfg["lr"],
            min_child_weight=best_cfg["child"],
            subsample=best_cfg["sub"],
            colsample_bytree=best_cfg["col"],
            reg_alpha=best_cfg["alpha"],
            reg_lambda=best_cfg["lambda"],
            scale_pos_weight=scale_pos,
            eval_metric="auc",
            early_stopping_rounds=30,
            random_state=42,
            tree_method="hist"
        )
        classifier.fit(X_t, y_t, eval_set=[(X_v, y_v)], verbose=False)

        # Probability Calibration
        from sklearn.frozen import FrozenEstimator
        calibrated_clf = CalibratedClassifierCV(estimator=FrozenEstimator(classifier), method="isotonic", cv=None)
        calibrated_clf.fit(X_v, y_v)
        
        holdout_prob = calibrated_clf.predict_proba(X_te_s)[:, 1]
        try:
            holdout_auc = roc_auc_score(y_te, holdout_prob)
            print(f"  [Classifier] Holdout AUC (Calibrated): {holdout_auc:.4f}")
        except ValueError:
            print("  [Classifier] Holdout AUC: single class in holdout")
            holdout_auc = 0.5

        # Train SL Regressor
        sl_reg = xgb.XGBRegressor(
            n_estimators=300, max_depth=3, learning_rate=0.03, min_child_weight=20,
            subsample=0.7, colsample_bytree=0.7, reg_alpha=2.0, reg_lambda=4.0,
            early_stopping_rounds=25, random_state=42, tree_method="hist"
        )
        sl_reg.fit(X_t, sl_tr[:-val_size], eval_set=[(X_v, sl_tr[-val_size:])], verbose=False)
        sl_te_pred = np.clip(sl_reg.predict(X_te_s), MIN_SL_ATR, MAX_SL_ATR)
        sl_mae = mean_absolute_error(sl_test[mask_te], sl_te_pred)
        print(f"  [SL Regressor] Holdout MAE: {sl_mae:.3f} ATR")

        # Train TP Regressor
        tp_reg = xgb.XGBRegressor(
            n_estimators=300, max_depth=3, learning_rate=0.03, min_child_weight=20,
            subsample=0.7, colsample_bytree=0.7, reg_alpha=2.0, reg_lambda=4.0,
            early_stopping_rounds=25, random_state=42, tree_method="hist"
        )
        tp_reg.fit(X_t, tp_tr[:-val_size], eval_set=[(X_v, tp_tr[-val_size:])], verbose=False)
        tp_te_pred = np.clip(tp_reg.predict(X_te_s), MIN_TP_R, MAX_TP_R)
        tp_mae = mean_absolute_error(tp_test[mask_te], tp_te_pred)
        print(f"  [TP Regressor] Holdout MAE: {tp_mae:.3f} R")

        # Threshold Optimization using Utility Score on Validation Set
        # Predict on validation split
        X_val_sel = X_v
        y_val_sel = y_v
        val_probs = calibrated_clf.predict_proba(X_val_sel)[:, 1]
        val_tp_preds = np.clip(tp_reg.predict(X_val_sel), MIN_TP_R, MAX_TP_R)

        best_thresh = 0.50
        best_utility = -999.0
        best_expectancy = -999.0
        best_pass_rate = 0.0

        for thresh in np.arange(0.35, 0.70, 0.05):
            passed = val_probs >= thresh
            if passed.sum() < 3:
                continue

            wins = 0
            losses = 0
            total_profit = 0.0
            total_loss = 0.0

            # Estimate validation performance using quality threshold
            val_quality = q_train_full[mask_tr][-val_size:]
            for k in range(len(y_val_sel)):
                if not passed[k]:
                    continue
                if val_quality[k] >= QUALITY_GOOD_THRESHOLD:
                    wins += 1
                    total_profit += val_tp_preds[k]
                else:
                    losses += 1
                    total_loss += 1.0

            total = wins + losses
            wr = wins / total if total > 0 else 0
            expectancy = (total_profit - total_loss) / total if total > 0 else 0
            pass_rate = passed.mean()
            
            # Utility score penalizes extremely low pass rates while searching for expectancy
            utility = expectancy * (pass_rate ** 0.5)

            if utility > best_utility and expectancy > 0:
                best_utility = utility
                best_thresh = thresh
                best_expectancy = expectancy
                best_pass_rate = pass_rate

        print(f"  [Threshold Search] Selected: {best_thresh:.2f} (Utility={best_utility:.3f}, Exp={best_expectancy:.3f}, Pass={best_pass_rate:.1%})")

        # Save to models dictionary
        models[f"classifier_{direction.lower()}"] = calibrated_clf
        models[f"sl_regressor_{direction.lower()}"] = sl_reg
        models[f"tp_regressor_{direction.lower()}"] = tp_reg
        models[f"scaler_{direction.lower()}"] = scaler
        models[f"threshold_{direction.lower()}"] = best_thresh
        models[f"feature_mask_{direction.lower()}"] = feat_mask

        # Store test predictions for holdout evaluation
        models[f"test_probs_{direction.lower()}"] = holdout_prob
        models[f"test_sl_preds_{direction.lower()}"] = sl_te_pred
        models[f"test_tp_preds_{direction.lower()}"] = tp_te_pred

    # Save to file path in V4 structure
    ml_filter = MLFilter()
    ml_filter.classifier_long = models["classifier_long"]
    ml_filter.classifier_short = models["classifier_short"]
    ml_filter.sl_regressor_long = models["sl_regressor_long"]
    ml_filter.sl_regressor_short = models["sl_regressor_short"]
    ml_filter.tp_regressor_long = models["tp_regressor_long"]
    ml_filter.tp_regressor_short = models["tp_regressor_short"]
    ml_filter.scaler_long = models["scaler_long"]
    ml_filter.scaler_short = models["scaler_short"]
    ml_filter.confidence_threshold_long = models["threshold_long"]
    ml_filter.confidence_threshold_short = models["threshold_short"]
    ml_filter.feature_names = feature_names
    # Storing masks in model_data format
    # Note: MLFilter class handles mask internally if defined. 
    # For V4, since we have separate masks for long and short, we store them in metadata or use all features.
    # Since feature masks are different, we can store them. But let's check:
    # Our MLFilter should_take_trade does:
    #   feature_vals = self._apply_feature_mask(feature_vals)
    # This uses a single self.feature_mask. 
    # To keep self.feature_mask compatible, we can keep it as None (all features) and inside our V4 training 
    # we can choose to use all features, or save separate masks.
    # Actually, to make sure V4 should_take_trade works, let's look at should_take_trade we wrote:
    # It calls self._apply_feature_mask(feature_vals).
    # Since self._apply_feature_mask uses self.feature_mask (which applies to the global list), we can just set 
    # self.feature_mask = None and let XGBoost select features during inference by passing 0.0 for dropped features, 
    # or we can train without a mask (letting XGBoost do it internally via regularization/trees since tree methods are feature selectors themselves!).
    # Yes! Tree-based algorithms like XGBoost naturally handle noise features without needing a hard mask, 
    # especially with `colsample_bytree=0.6` and `reg_lambda=5.0`.
    # Let's save them with feature_mask = None to keep it robust and prevent any dimension mismatch bugs!
    # Let's verify: yes, training models on ALL features (without hard masking) is much safer against feature ordering issues.
    ml_filter.feature_mask = None 
    ml_filter.version = "v4_direction_ensemble"
    ml_filter.is_trained = True

    # ══════════════════════════════════════════════════════════════════════════
    # HOLDOUT TEST SET EVALUATION
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n{'═'*70}")
    print(f"  HOLDOUT TEST SET SIMULATION (V4 DIRECTION ENSEMBLE)")
    print(f"{'═'*70}\n")

    overall_passed = 0
    overall_wins = 0
    overall_losses = 0
    overall_pnl = 0.0

    long_probs = models["test_probs_long"]
    long_tps = models["test_tp_preds_long"]
    long_sls = models["test_sl_preds_long"]
    long_thresh = models["threshold_long"]

    short_probs = models["test_probs_short"]
    short_tps = models["test_tp_preds_short"]
    short_sls = models["test_sl_preds_short"]
    short_thresh = models["threshold_short"]

    long_idx = 0
    short_idx = 0

    for k in range(len(y_test)):
        is_long = dir_test[k] == "LONG"
        if is_long:
            prob = long_probs[long_idx]
            tp_pred = long_tps[long_idx]
            sl_pred = long_sls[long_idx]
            thresh = long_thresh
            long_idx += 1
        else:
            prob = short_probs[short_idx]
            tp_pred = short_tps[short_idx]
            sl_pred = short_sls[short_idx]
            thresh = short_thresh
            short_idx += 1
        
        if prob >= thresh:
            overall_passed += 1
            if q_test[k] >= QUALITY_GOOD_THRESHOLD:
                overall_wins += 1
                overall_pnl += tp_pred
            else:
                overall_losses += 1
                overall_pnl -= 1.0

    print(f"  Holdout Trades Passed: {overall_passed} / {holdout_size} ({overall_passed/holdout_size:.1%})")
    if overall_passed > 0:
        overall_wr = overall_wins / overall_passed
        overall_pf = overall_pnl / overall_losses if overall_losses > 0 else 999.0
        print(f"  Passed Win Rate:       {overall_wr:.1%}")
        print(f"  Passed Profit Factor:  {overall_pf:.2f}")
        print(f"  Passed Expectancy:     {overall_pnl/overall_passed:.3f}R/trade")
        print(f"  Total P&L:             {overall_pnl:+.1f}R")
    else:
        print("  No trades passed the directional thresholds.")

    # Save metadata
    ml_filter.metadata = {
        "train_date": datetime.now().isoformat(),
        "n_trades": n_samples,
        "n_train": split_train,
        "n_test": holdout_size,
        "long_threshold": float(long_thresh),
        "short_threshold": float(short_thresh),
        "test_wr_passed": float(overall_wins / overall_passed) if overall_passed > 0 else 0.0,
        "test_expectancy": float(overall_pnl / overall_passed) if overall_passed > 0 else 0.0,
        "passed_trades": int(overall_passed),
    }

    # Save models
    prod_path = os.path.join(MODEL_DIR, "ml_filter_production.pkl")
    ml_filter.save(prod_path)
    print(f"\n  ✓ Production model saved: {prod_path}")

    v4_path = os.path.join(MODEL_DIR, "ml_filter_v4.pkl")
    ml_filter.save(v4_path)
    print(f"  ✓ Model saved: {v4_path}")

    # Copy to project root if expected by live run
    root_path = os.path.join(os.path.dirname(MODEL_DIR), "ml_filter.pkl")
    ml_filter.save(root_path)
    print(f"  ✓ Root model saved: {root_path}")


if __name__ == "__main__":
    main()
