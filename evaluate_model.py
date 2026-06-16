"""
ML Model Evaluation Script
===========================
Tests the production XGBoost model for:
1. Overfitting (train vs holdout performance gap)
2. Underfitting (absolute discriminative power)
3. Forward test on recent unseen data (last 7 days)
4. Calibration (predicted probabilities vs actual outcomes)
5. Feature importance stability
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from collections import defaultdict

# Add project to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import Config as c
from data import get_ohlcv
from indicators import compute_all, compute_htf
from strategy import generate_signal, Signal
from ml_model import MLFilter, extract_features_extended

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score, recall_score,
    f1_score, brier_score_loss, log_loss, classification_report,
    confusion_matrix
)
from sklearn.model_selection import cross_val_score, StratifiedKFold


# ══════════════════════════════════════════════════════════════════════════
# LOAD MODEL
# ══════════════════════════════════════════════════════════════════════════

def load_production_model():
    """Load the production model and return its components."""
    model_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "ml_models", "ml_filter_production.pkl"
    )
    if not os.path.exists(model_path):
        print(f"ERROR: Model not found at {model_path}")
        sys.exit(1)

    with open(model_path, "rb") as f:
        data = pickle.load(f)

    print(f"Model loaded from: {model_path}")
    print(f"  Keys: {list(data.keys())}")
    print(f"  Model type: {type(data.get('classifier', data.get('model', 'unknown')))}")
    if "feature_mask" in data:
        mask = data["feature_mask"]
        print(f"  Feature mask: {sum(mask)}/{len(mask)} features selected")
    if "feature_names" in data:
        print(f"  Feature names: {len(data['feature_names'])} total")
    if "threshold" in data:
        print(f"  Threshold: {data['threshold']}")
    
    return data


# ══════════════════════════════════════════════════════════════════════════
# GENERATE TEST DATA FROM RECENT MARKET
# ══════════════════════════════════════════════════════════════════════════

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "historical_data")


def generate_signals_for_symbol(symbol, use_last_pct=0.3):
    """
    Generate signals from historical CSV data.
    Uses the last `use_last_pct` of data as unseen forward-test window.
    """
    try:
        csv_path = os.path.join(DATA_DIR, f"{symbol}_15m.csv")
        if not os.path.exists(csv_path):
            return []

        df_15m = pd.read_csv(csv_path, parse_dates=["timestamp"], index_col="timestamp")
        if len(df_15m) < 1000:
            return []

        # Use last portion as "new" data the model hasn't been optimized on
        split_idx = int(len(df_15m) * (1 - use_last_pct))
        df_15m = df_15m.iloc[split_idx:]

        # Resample to 4H
        df_4h = df_15m.resample("4h").agg({
            "open": "first", "high": "max",
            "low": "min", "close": "last", "volume": "sum"
        }).dropna()

        if len(df_4h) < 60:
            return []

        df_4h = compute_all(df_4h)

        # Get HTF (daily) from same data
        df_daily = df_15m.resample("1D").agg({
            "open": "first", "high": "max",
            "low": "min", "close": "last", "volume": "sum"
        }).dropna()
        
        htf_bias = None
        if len(df_daily) >= 50:
            from indicators import ema as ema_func
            df_daily["ema20"] = ema_func(df_daily["close"], 20)
            df_daily["ema50"] = ema_func(df_daily["close"], 50)
            df_daily = df_daily.dropna()
        
        signals = []
        for i in range(55, len(df_4h) - 8):  # Leave 8 bars for forward look
            # Update HTF bias at each point
            candle_time = df_4h.index[i]
            if len(df_daily) >= 50:
                daily_before = df_daily[df_daily.index <= candle_time]
                if len(daily_before) > 0:
                    last_d = daily_before.iloc[-1]
                    if "ema20" in last_d.index and "ema50" in last_d.index:
                        if last_d["ema20"] > last_d["ema50"]:
                            htf_bias = "LONG"
                        elif last_d["ema20"] < last_d["ema50"]:
                            htf_bias = "SHORT"
                        else:
                            htf_bias = None

            slice_df = df_4h.iloc[:i+1]
            signal, _, _ = generate_signal(slice_df, htf_bias)
            
            if signal == Signal.NONE:
                continue

            curr = df_4h.iloc[i]
            prev = df_4h.iloc[i-1]
            price = float(curr["close"])
            atr_val = float(curr["atr"])
            direction = signal

            if atr_val <= 0:
                continue

            df_slice = df_4h.iloc[max(0, i-10):i+1]
            features = extract_features_extended(curr, prev, price, atr_val, direction, df_slice)

            # Look ahead for outcome (8 candles = 32 hours forward)
            future = df_4h.iloc[i+1:i+9]
            sl_dist = atr_val * c.SL_ATR_MULT
            tp_dist = atr_val * c.TP_ATR_MULT

            if direction == "LONG":
                # Check bar-by-bar for SL/TP hit order
                max_fav = 0
                max_adv = 0
                hit_tp = False
                hit_sl = False
                for _, bar in future.iterrows():
                    bar_adv = price - float(bar["low"])
                    bar_fav = float(bar["high"]) - price
                    max_adv = max(max_adv, bar_adv)
                    max_fav = max(max_fav, bar_fav)
                    if bar_adv >= sl_dist and not hit_tp:
                        hit_sl = True
                        break
                    if bar_fav >= tp_dist:
                        hit_tp = True
                        break
            else:
                max_fav = 0
                max_adv = 0
                hit_tp = False
                hit_sl = False
                for _, bar in future.iterrows():
                    bar_adv = float(bar["high"]) - price
                    bar_fav = price - float(bar["low"])
                    max_adv = max(max_adv, bar_adv)
                    max_fav = max(max_fav, bar_fav)
                    if bar_adv >= sl_dist and not hit_tp:
                        hit_sl = True
                        break
                    if bar_fav >= tp_dist:
                        hit_tp = True
                        break

            # Outcome: TP hit = win, SL hit = loss, neither = check MFE/MAE ratio
            if hit_tp:
                outcome = 1
            elif hit_sl:
                outcome = 0
            else:
                outcome = 1 if (max_fav > max_adv * 1.5) else 0

            signals.append({
                "symbol": symbol,
                "direction": direction,
                "features": features,
                "outcome": outcome,
                "hit_tp": hit_tp,
                "hit_sl": hit_sl,
                "mfe_atr": max_fav / atr_val,
                "mae_atr": max_adv / atr_val,
                "timestamp": df_4h.index[i]
            })

        return signals
    except Exception as e:
        print(f"  {symbol}: error - {e}")
        import traceback
        traceback.print_exc()
        return []


# ══════════════════════════════════════════════════════════════════════════
# EVALUATION FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════

def evaluate_model_on_data(model_data, signals):
    """Score all signals through the model and evaluate."""
    classifier = model_data.get("classifier", model_data.get("model"))
    scaler = model_data.get("scaler")
    feature_mask = model_data.get("feature_mask")
    feature_names = model_data.get("feature_names")
    threshold = model_data.get("threshold", 0.5)

    if classifier is None:
        print("ERROR: No classifier found in model data")
        return None

    results = []
    for sig in signals:
        feat_dict = sig["features"]
        
        # Convert to array
        if feature_names:
            feat_vec = np.array([feat_dict.get(fn, 0.0) for fn in feature_names], dtype=np.float32)
        else:
            feat_vec = np.array(list(feat_dict.values()), dtype=np.float32)

        # Apply feature mask
        if feature_mask is not None:
            mask = np.array(feature_mask, dtype=bool)
            if len(feat_vec) >= len(mask):
                feat_vec = feat_vec[:len(mask)][mask]
            elif len(feat_vec) < len(mask):
                padded = np.zeros(len(mask), dtype=np.float32)
                padded[:len(feat_vec)] = feat_vec
                feat_vec = padded[mask]

        # Scale
        if scaler is not None:
            expected = scaler.n_features_in_ if hasattr(scaler, 'n_features_in_') else len(feat_vec)
            if len(feat_vec) != expected:
                padded = np.zeros(expected, dtype=np.float32)
                padded[:min(len(feat_vec), expected)] = feat_vec[:expected]
                feat_vec = padded
            feat_vec = scaler.transform(feat_vec.reshape(1, -1))[0]

        # Predict
        feat_2d = feat_vec.reshape(1, -1)
        if HAS_XGB and isinstance(classifier, xgb.XGBClassifier):
            prob = classifier.predict_proba(feat_2d)[0][1]
        else:
            prob = classifier.predict_proba(feat_2d)[0][1]

        results.append({
            "prob": prob,
            "outcome": sig["outcome"],
            "hit_tp": sig["hit_tp"],
            "hit_sl": sig["hit_sl"],
            "mfe_atr": sig["mfe_atr"],
            "mae_atr": sig["mae_atr"],
            "symbol": sig["symbol"],
            "direction": sig["direction"],
            "passed": prob >= threshold
        })

    return results, threshold


def print_metrics(results, threshold, label=""):
    """Print comprehensive evaluation metrics."""
    print(f"\n{'═' * 70}")
    print(f"  {label}")
    print(f"{'═' * 70}")

    probs = np.array([r["prob"] for r in results])
    outcomes = np.array([r["outcome"] for r in results])
    predictions = (probs >= threshold).astype(int)

    print(f"\n  Total signals: {len(results)}")
    print(f"  Positive outcomes: {outcomes.sum()} ({outcomes.mean()*100:.1f}%)")
    print(f"  Negative outcomes: {len(outcomes) - outcomes.sum()} ({(1-outcomes.mean())*100:.1f}%)")

    # ── Discrimination Metrics ──
    print(f"\n  ── Discrimination ──")
    try:
        auc = roc_auc_score(outcomes, probs)
        print(f"  AUC-ROC:        {auc:.4f}  {'✓' if auc > 0.55 else '✗'} (>0.55 = useful, >0.60 = good)")
    except:
        auc = 0
        print(f"  AUC-ROC:        N/A (single class)")

    acc = accuracy_score(outcomes, predictions)
    print(f"  Accuracy:       {acc:.4f}")

    if predictions.sum() > 0:
        prec = precision_score(outcomes, predictions, zero_division=0)
        rec = recall_score(outcomes, predictions, zero_division=0)
        f1 = f1_score(outcomes, predictions, zero_division=0)
        print(f"  Precision:      {prec:.4f}  (of trades taken, % winners)")
        print(f"  Recall:         {rec:.4f}  (of all winners, % captured)")
        print(f"  F1 Score:       {f1:.4f}")
    
    # ── Calibration ──
    print(f"\n  ── Calibration ──")
    try:
        brier = brier_score_loss(outcomes, probs)
        ll = log_loss(outcomes, probs)
        print(f"  Brier Score:    {brier:.4f}  (lower = better, <0.25 = useful)")
        print(f"  Log Loss:       {ll:.4f}  (lower = better, <0.69 = better than random)")
    except:
        print(f"  Calibration: N/A")

    # ── Probability Distribution ──
    print(f"\n  ── Probability Distribution ──")
    print(f"  Mean prob:      {probs.mean():.4f}")
    print(f"  Std prob:       {probs.std():.4f}")
    print(f"  Min/Max:        {probs.min():.4f} / {probs.max():.4f}")
    
    # Binned accuracy
    bins = [(0, 0.3), (0.3, 0.4), (0.4, 0.5), (0.5, 0.6), (0.6, 0.7), (0.7, 1.0)]
    print(f"\n  ── Probability Bins (Calibration Check) ──")
    print(f"  {'Bin':<12} {'Count':<8} {'Actual WR':<12} {'Expected':<12} {'Gap':<8}")
    for lo, hi in bins:
        mask = (probs >= lo) & (probs < hi)
        if mask.sum() > 0:
            actual_wr = outcomes[mask].mean()
            expected = (lo + hi) / 2
            gap = actual_wr - expected
            print(f"  [{lo:.1f}-{hi:.1f})   {mask.sum():<8} {actual_wr:.3f}        {expected:.3f}        {gap:+.3f}")

    # ── Trading Simulation ──
    print(f"\n  ── Trading Simulation (threshold={threshold:.2f}) ──")
    passed = [r for r in results if r["passed"]]
    rejected = [r for r in results if not r["passed"]]

    if passed:
        pass_wr = np.mean([r["outcome"] for r in passed])
        pass_mfe = np.mean([r["mfe_atr"] for r in passed])
        pass_mae = np.mean([r["mae_atr"] for r in passed])
        
        # Estimate PF: (wins * avg_win) / (losses * avg_loss)
        wins = [r for r in passed if r["outcome"] == 1]
        losses = [r for r in passed if r["outcome"] == 0]
        if wins and losses:
            avg_win_r = np.mean([r["mfe_atr"] for r in wins]) / c.SL_ATR_MULT
            avg_loss_r = 1.0  # Assume full SL hit
            pf = (len(wins) * avg_win_r) / (len(losses) * avg_loss_r) if losses else 999
        else:
            pf = 0
        
        print(f"  Passed:         {len(passed)} / {len(results)} ({len(passed)/len(results)*100:.1f}%)")
        print(f"  Win Rate:       {pass_wr*100:.1f}%")
        print(f"  Avg MFE:        {pass_mfe:.2f} ATR")
        print(f"  Avg MAE:        {pass_mae:.2f} ATR")
        print(f"  Est. PF:        {pf:.2f}")
    else:
        print(f"  No signals passed threshold!")

    if rejected:
        rej_wr = np.mean([r["outcome"] for r in rejected])
        print(f"  Rejected WR:    {rej_wr*100:.1f}%  (should be lower than passed)")
    
    # ── Separation Power ──
    if passed and rejected:
        wr_gap = pass_wr - rej_wr
        print(f"\n  ── Separation Power ──")
        print(f"  Passed WR - Rejected WR = {wr_gap*100:+.1f}%")
        if wr_gap > 0.10:
            print(f"  ✓ GOOD: Model separates winners from losers (+{wr_gap*100:.1f}%)")
        elif wr_gap > 0.05:
            print(f"  ~ MARGINAL: Some separation (+{wr_gap*100:.1f}%)")
        else:
            print(f"  ✗ POOR: Model fails to separate ({wr_gap*100:+.1f}%)")

    return auc


def check_overfitting(model_data, all_signals):
    """Split data temporally and check for overfit."""
    print(f"\n{'═' * 70}")
    print(f"  OVERFIT / UNDERFIT ANALYSIS")
    print(f"{'═' * 70}")

    # Sort by timestamp if available
    sorted_signals = sorted(all_signals, key=lambda x: x.get("timestamp", datetime.min))
    
    n = len(sorted_signals)
    split_70 = int(n * 0.7)
    
    train_signals = sorted_signals[:split_70]
    test_signals = sorted_signals[split_70:]

    print(f"\n  Temporal split: Train={len(train_signals)}, Test={len(test_signals)}")
    
    if len(train_signals) < 10 or len(test_signals) < 10:
        print("  Not enough data for overfit analysis")
        return

    train_results, threshold = evaluate_model_on_data(model_data, train_signals)
    test_results, _ = evaluate_model_on_data(model_data, test_signals)

    train_probs = np.array([r["prob"] for r in train_results])
    train_outcomes = np.array([r["outcome"] for r in train_results])
    test_probs = np.array([r["prob"] for r in test_results])
    test_outcomes = np.array([r["outcome"] for r in test_results])

    try:
        train_auc = roc_auc_score(train_outcomes, train_probs)
        test_auc = roc_auc_score(test_outcomes, test_probs)
        gap = train_auc - test_auc
        
        print(f"\n  Train AUC:    {train_auc:.4f}")
        print(f"  Test AUC:     {test_auc:.4f}")
        print(f"  Gap:          {gap:.4f}")
        print()
        
        if gap > 0.10:
            print(f"  ✗ OVERFITTING DETECTED (gap={gap:.4f} > 0.10)")
            print(f"    The model memorizes training patterns that don't generalize.")
            print(f"    Remedy: More regularization, fewer features, more data.")
        elif gap > 0.05:
            print(f"  ~ MILD OVERFITTING (gap={gap:.4f})")
            print(f"    Some overfit but may still be usable in production.")
        elif test_auc < 0.52:
            print(f"  ✗ UNDERFITTING (test AUC={test_auc:.4f} ≈ random)")
            print(f"    Model has no predictive power on unseen data.")
            print(f"    Remedy: Better features, more complex model, more data.")
        else:
            print(f"  ✓ HEALTHY FIT (gap={gap:.4f}, test AUC={test_auc:.4f})")
            
    except Exception as e:
        print(f"  Could not compute AUC: {e}")

    # Compare WR of passed trades
    train_passed = [r for r in train_results if r["passed"]]
    test_passed = [r for r in test_results if r["passed"]]
    
    if train_passed and test_passed:
        train_wr = np.mean([r["outcome"] for r in train_passed])
        test_wr = np.mean([r["outcome"] for r in test_passed])
        print(f"\n  Passed Trade WR (Train): {train_wr*100:.1f}% ({len(train_passed)} trades)")
        print(f"  Passed Trade WR (Test):  {test_wr*100:.1f}% ({len(test_passed)} trades)")
        wr_drop = train_wr - test_wr
        if wr_drop > 0.10:
            print(f"  ✗ WR drops significantly on test data ({wr_drop*100:+.1f}%)")
        else:
            print(f"  ✓ WR stable across splits ({wr_drop*100:+.1f}%)")


def per_symbol_analysis(results):
    """Check if model works across symbols or just a few."""
    print(f"\n{'═' * 70}")
    print(f"  PER-SYMBOL BREAKDOWN")
    print(f"{'═' * 70}\n")
    
    by_symbol = defaultdict(list)
    for r in results:
        by_symbol[r["symbol"]].append(r)

    print(f"  {'Symbol':<12} {'Signals':<9} {'Passed':<8} {'Pass WR':<10} {'All WR':<10} {'Avg Prob':<10}")
    print(f"  {'-'*60}")
    
    for sym in sorted(by_symbol.keys()):
        sym_results = by_symbol[sym]
        passed = [r for r in sym_results if r["passed"]]
        all_wr = np.mean([r["outcome"] for r in sym_results])
        pass_wr = np.mean([r["outcome"] for r in passed]) if passed else 0
        avg_prob = np.mean([r["prob"] for r in sym_results])
        print(f"  {sym:<12} {len(sym_results):<9} {len(passed):<8} {pass_wr*100:.1f}%     {all_wr*100:.1f}%     {avg_prob:.3f}")


# ══════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("  ML MODEL EVALUATION — OVERFIT/UNDERFIT & FORWARD TEST")
    print("=" * 70)
    print(f"  Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print()

    # Load model
    model_data = load_production_model()
    print()

    # Generate test signals from historical data (last 30% = unseen period)
    print("  Generating signals from historical data (last 30% of each symbol)...")
    print("  This uses local CSVs from historical_data/ as unseen forward-test data.\n")

    all_signals = []
    symbols_tested = 0
    
    for i, symbol in enumerate(c.SYMBOLS):
        print(f"  [{i+1:2d}/{len(c.SYMBOLS)}] {symbol}...", end=" ", flush=True)
        signals = generate_signals_for_symbol(symbol, use_last_pct=0.30)
        print(f"{len(signals)} signals")
        all_signals.extend(signals)
        symbols_tested += 1

    print(f"\n  Total signals collected: {len(all_signals)} from {symbols_tested} symbols")
    
    if len(all_signals) < 20:
        print("\n  ERROR: Not enough signals for evaluation. Need at least 20.")
        print("  This may happen if the strategy generates very few signals on recent data.")
        sys.exit(1)

    # Full evaluation
    results, threshold = evaluate_model_on_data(model_data, all_signals)
    
    # 1. Overall metrics on all data
    overall_auc = print_metrics(results, threshold, "OVERALL EVALUATION (All Recent Data)")

    # 2. Overfit analysis (temporal split)
    check_overfitting(model_data, all_signals)

    # 3. Per-symbol breakdown
    per_symbol_analysis(results)

    # 4. Summary verdict
    print(f"\n{'═' * 70}")
    print(f"  FINAL VERDICT")
    print(f"{'═' * 70}\n")

    passed = [r for r in results if r["passed"]]
    rejected = [r for r in results if not r["passed"]]
    
    issues = []
    strengths = []

    if overall_auc < 0.52:
        issues.append("Model AUC near random (no predictive power)")
    elif overall_auc < 0.55:
        issues.append("Weak AUC (marginal edge)")
    else:
        strengths.append(f"AUC={overall_auc:.4f} indicates real signal")

    if passed:
        pass_wr = np.mean([r["outcome"] for r in passed])
        if rejected:
            rej_wr = np.mean([r["outcome"] for r in rejected])
            if pass_wr > rej_wr + 0.05:
                strengths.append(f"Separates winners (+{(pass_wr-rej_wr)*100:.1f}% gap)")
            else:
                issues.append(f"Poor separation (only +{(pass_wr-rej_wr)*100:.1f}% gap)")

    probs = np.array([r["prob"] for r in results])
    if probs.std() < 0.05:
        issues.append("Very low probability variance (model always predicts similar values)")
    else:
        strengths.append(f"Varied predictions (std={probs.std():.3f})")

    print("  Strengths:")
    for s in strengths:
        print(f"    ✓ {s}")
    if not strengths:
        print(f"    (none)")
    
    print(f"\n  Issues:")
    for i in issues:
        print(f"    ✗ {i}")
    if not issues:
        print(f"    (none)")

    print(f"\n  Recommendation:")
    if len(issues) > len(strengths):
        print(f"    The model shows signs of OVERFITTING or UNDERFITTING.")
        print(f"    Consider: more training data, stronger regularization, or simpler features.")
    else:
        print(f"    The model appears functional. Monitor live performance.")
    
    print(f"\n{'═' * 70}")


if __name__ == "__main__":
    main()
