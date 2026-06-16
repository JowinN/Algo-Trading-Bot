"""
ML Signal Filter + Dynamic TP/SL Optimizer
===========================================
Two-model approach:
1. CLASSIFIER: Should we take this trade? (win probability)
2. REGRESSOR: What's the optimal SL (ATR mult) and TP (R-multiple)?

Uses MFE/MAE (Maximum Favorable/Adverse Excursion) analysis
to learn what TP/SL values maximize expectancy for each setup.

Features: 50+ market-state indicators for robust generalization.
"""

import numpy as np
import pickle
import os
from sklearn.preprocessing import StandardScaler

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
    HAS_SKLEARN_GB = True
except ImportError:
    HAS_SKLEARN_GB = False


# ══════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════════════════

MIN_TRAINING_SAMPLES = 50
RETRAIN_INTERVAL = 15
CONFIDENCE_THRESHOLD = 0.15

# SL/TP bounds (in ATR multiples for SL, R-multiples for TP)
MIN_SL_ATR = 1.0
MAX_SL_ATR = 4.0
MIN_TP_R = 1.5
MAX_TP_R = 6.0


# ══════════════════════════════════════════════════════════════════════════
# EXPANDED FEATURE EXTRACTION (50+ features)
# ══════════════════════════════════════════════════════════════════════════

def extract_features(curr, prev, price, atr_val, direction):
    """
    Extract comprehensive feature vector from current market state.
    50+ features covering: trend, momentum, volatility, volume, structure.
    All features are normalized/scale-independent.
    """
    if atr_val <= 0:
        atr_val = 0.0001  # Safety

    dir_sign = 1.0 if direction == "LONG" else -1.0

    # ── Core Price Data ────────────────────────────────────────────────
    ema9 = float(curr["ema9"])
    ema21 = float(curr["ema21"])
    ema50 = float(curr["ema50"])
    prev_ema21 = float(prev["ema21"])
    prev_ema50 = float(prev["ema50"])

    # ── Trend Indicators ───────────────────────────────────────────────
    adx = float(curr["adx"])
    di_plus = float(curr["di_plus"])
    di_minus = float(curr["di_minus"])
    prev_adx = float(prev["adx"]) if "adx" in prev.index else adx
    prev_di_plus = float(prev["di_plus"]) if "di_plus" in prev.index else di_plus
    prev_di_minus = float(prev["di_minus"]) if "di_minus" in prev.index else di_minus

    # ── Momentum ──────────────────────────────────────────────────────
    rsi = float(curr["rsi"])
    prev_rsi = float(prev["rsi"]) if "rsi" in prev.index else rsi
    macd_val = float(curr["macd"])
    macd_hist = float(curr["macd_hist"])
    prev_macd_hist = float(prev["macd_hist"])
    macd_sig = float(curr["macd_sig"]) if "macd_sig" in curr.index else 0

    # ── Volatility ────────────────────────────────────────────────────
    bb_pct_b = float(curr["bb_pct_b"]) if "bb_pct_b" in curr.index else 0.5
    bb_width = float(curr["bb_width"]) if "bb_width" in curr.index else 0
    prev_bb_width = float(prev["bb_width"]) if "bb_width" in prev.index else bb_width

    # ── Volume ────────────────────────────────────────────────────────
    rel_vol = float(curr["rel_volume"]) if "rel_volume" in curr.index else 1.0
    cmf = float(curr["cmf"]) if "cmf" in curr.index else 0

    # ── VWAP ──────────────────────────────────────────────────────────
    vwap = float(curr["vwap"]) if "vwap" in curr.index else price

    # ── Structure ─────────────────────────────────────────────────────
    swing_high = float(curr["swing_high"]) if "swing_high" in curr.index else price
    swing_low = float(curr["swing_low"]) if "swing_low" in curr.index else price
    body_pct = float(curr["body_pct"]) if curr.get("body_pct", float('nan')) == curr.get("body_pct", float('nan')) else 0

    # ── Stochastic RSI ────────────────────────────────────────────────
    stoch_k = float(curr["stoch_k"]) if "stoch_k" in curr.index else 50
    stoch_d = float(curr["stoch_d"]) if "stoch_d" in curr.index else 50

    # ── Squeeze ──────────────────────────────────────────────────────
    squeeze_on = 1.0 if curr.get("squeeze_on", False) else 0.0
    squeeze_mom = float(curr["squeeze_mom"]) if "squeeze_mom" in curr.index else 0

    # ── Supertrend ───────────────────────────────────────────────────
    st_direction = float(curr["st_direction"]) if "st_direction" in curr.index else 0

    # ══════════════════════════════════════════════════════════════════
    # BUILD FEATURE VECTOR (50+ features)
    # ══════════════════════════════════════════════════════════════════

    features = {}

    # ── TREND (12 features) ─────────────────────────────────────────
    features["adx"] = adx
    features["adx_change"] = adx - prev_adx
    features["di_spread"] = (di_plus - di_minus) * dir_sign
    features["di_spread_change"] = ((di_plus - di_minus) - (prev_di_plus - prev_di_minus)) * dir_sign
    features["ema9_21_dist"] = (ema9 - ema21) / atr_val * dir_sign
    features["ema21_50_dist"] = (ema21 - ema50) / atr_val * dir_sign
    features["price_ema9_dist"] = (price - ema9) / atr_val * dir_sign
    features["price_ema21_dist"] = (price - ema21) / atr_val * dir_sign
    features["price_ema50_dist"] = (price - ema50) / atr_val * dir_sign
    features["ema21_slope"] = (ema21 - prev_ema21) / atr_val * dir_sign
    features["ema50_slope"] = (ema50 - prev_ema50) / atr_val * dir_sign
    features["supertrend_align"] = st_direction * dir_sign

    # ── MOMENTUM (12 features) ──────────────────────────────────────
    features["rsi"] = rsi
    features["rsi_change"] = rsi - prev_rsi
    features["rsi_dist_50"] = (rsi - 50) * dir_sign  # Distance from neutral
    features["macd_norm"] = macd_val / atr_val * dir_sign
    features["macd_hist_norm"] = macd_hist / atr_val * dir_sign
    features["macd_accel"] = (macd_hist - prev_macd_hist) / atr_val * dir_sign
    features["macd_hist_sign"] = 1.0 if (macd_hist * dir_sign > 0) else -1.0
    features["stoch_k"] = stoch_k
    features["stoch_d"] = stoch_d
    features["stoch_cross"] = (stoch_k - stoch_d) * dir_sign
    features["squeeze_on"] = squeeze_on
    features["squeeze_mom_dir"] = squeeze_mom / atr_val * dir_sign if atr_val > 0 else 0

    # ── VOLATILITY (8 features) ─────────────────────────────────────
    features["bb_pct_b"] = bb_pct_b
    features["bb_width"] = bb_width
    features["bb_width_change"] = bb_width - prev_bb_width
    features["bb_position"] = (bb_pct_b - 0.5) * 2 * dir_sign  # -1 to 1, positive = favorable
    features["atr_normalized"] = atr_val / price * 1000  # ATR as % of price (scaled)
    features["price_range_ratio"] = (float(curr["high"]) - float(curr["low"])) / atr_val
    features["body_pct"] = body_pct
    features["candle_direction"] = 1.0 if float(curr["close"]) > float(curr["open"]) else -1.0

    # ── VOLUME (6 features) ──────────────────────────────────────────
    features["rel_volume"] = rel_vol
    features["cmf"] = cmf * dir_sign
    features["vol_above_avg"] = 1.0 if rel_vol > 1.0 else 0.0
    features["vol_spike"] = 1.0 if rel_vol > 2.0 else 0.0
    features["price_vwap_dist"] = (price - vwap) / atr_val * dir_sign
    features["cmf_strong"] = 1.0 if (cmf * dir_sign > 0.1) else 0.0

    # ── STRUCTURE (8 features) ───────────────────────────────────────
    features["dist_swing_high"] = (swing_high - price) / atr_val
    features["dist_swing_low"] = (price - swing_low) / atr_val
    features["swing_range"] = (swing_high - swing_low) / atr_val
    features["price_in_range"] = (price - swing_low) / (swing_high - swing_low) if (swing_high - swing_low) > 0 else 0.5
    # Room to target (how much space in favorable direction)
    if direction == "LONG":
        features["room_to_target"] = (swing_high - price) / atr_val
        features["room_to_stop"] = (price - swing_low) / atr_val
    else:
        features["room_to_target"] = (price - swing_low) / atr_val
        features["room_to_stop"] = (swing_high - price) / atr_val
    features["structure_rr"] = features["room_to_target"] / max(features["room_to_stop"], 0.1)
    features["pullback_depth"] = features["price_ema21_dist"]  # How far from EMA21

    # ── REGIME (4 features) ──────────────────────────────────────────
    features["trend_strength"] = adx * abs(di_plus - di_minus) / 100  # Combined
    features["momentum_regime"] = 1.0 if (macd_hist * dir_sign > 0 and rsi * dir_sign > 50 * dir_sign) else 0.0
    features["vol_regime"] = min(rel_vol, 3.0) / 3.0  # Capped normalized
    features["squeeze_fire"] = 1.0 if (curr.get("squeeze_fire", False)) else 0.0

    return features


def extract_features_extended(curr, prev, price, atr_val, direction, df_slice=None):
    """
    Extended features that include lookback context.
    Uses recent history for pattern detection.
    """
    features = extract_features(curr, prev, price, atr_val, direction)

    if df_slice is not None and len(df_slice) >= 10:
        dir_sign = 1.0 if direction == "LONG" else -1.0

        # Last 5 candles RSI trend
        rsi_5 = df_slice["rsi"].iloc[-5:].values
        features["rsi_slope_5"] = (rsi_5[-1] - rsi_5[0]) / 5.0

        # ATR expansion/contraction (current vs 10-bar avg)
        atr_10 = df_slice["atr"].iloc[-10:].mean()
        features["atr_expansion"] = float(curr["atr"]) / atr_10 if atr_10 > 0 else 1.0

        # Consecutive candles in direction
        closes = df_slice["close"].iloc[-5:].values
        if direction == "LONG":
            consec = sum(1 for i in range(1, len(closes)) if closes[i] > closes[i-1])
        else:
            consec = sum(1 for i in range(1, len(closes)) if closes[i] < closes[i-1])
        features["consecutive_direction"] = consec / 4.0

        # Volatility percentile (where is current ATR vs last 50)
        if len(df_slice) >= 50:
            atr_50 = df_slice["atr"].iloc[-50:].values
            features["atr_percentile"] = np.searchsorted(np.sort(atr_50), float(curr["atr"])) / 50.0
        else:
            features["atr_percentile"] = 0.5

        # Volume trend (5-bar)
        vol_5 = df_slice["rel_volume"].iloc[-5:].values
        features["vol_trend"] = (vol_5[-1] - vol_5[0]) / max(vol_5.mean(), 0.1)

        # MACD histogram trend (accelerating or decelerating)
        hist_5 = df_slice["macd_hist"].iloc[-5:].values
        features["macd_hist_trend"] = (hist_5[-1] - hist_5[0]) / atr_val * dir_sign

    return features


# ══════════════════════════════════════════════════════════════════════════
# MFE/MAE CALCULATION (for TP/SL target generation)
# ══════════════════════════════════════════════════════════════════════════

def calculate_mfe_mae(df, entry_idx, direction, entry_price, atr_val, max_bars=30):
    """
    Calculate Maximum Favorable Excursion (MFE) and Maximum Adverse Excursion (MAE)
    in ATR units for a given entry.

    MFE = max profit reached before exit (in ATR)
    MAE = max drawdown before recovery or exit (in ATR)

    Returns: (mfe_atr, mae_atr) or None if insufficient data
    """
    if entry_idx + 2 >= len(df):
        return None

    end_idx = min(entry_idx + max_bars + 1, len(df))
    mfe = 0.0
    mae = 0.0

    for j in range(entry_idx + 1, end_idx):
        row = df.iloc[j]
        high = float(row["high"])
        low = float(row["low"])

        if direction == "LONG":
            favorable = (high - entry_price) / atr_val
            adverse = (entry_price - low) / atr_val
        else:
            favorable = (entry_price - low) / atr_val
            adverse = (high - entry_price) / atr_val

        mfe = max(mfe, favorable)
        mae = max(mae, adverse)

    return mfe, mae


def calculate_optimal_exit(df, entry_idx, direction, entry_price, atr_val, max_bars=30):
    """
    Fast optimal SL/TP using MFE/MAE heuristic.
    Instead of brute-force grid, uses the actual price excursions to derive optimal levels.
    Returns: (optimal_sl_atr, optimal_tp_r, actual_pnl_r)
    """
    if entry_idx + 2 >= len(df):
        return None

    end_idx = min(entry_idx + max_bars + 1, len(df))

    # Get highs and lows as numpy arrays for speed
    future_slice = df.iloc[entry_idx + 1:end_idx]
    if future_slice.empty:
        return None

    highs = future_slice["high"].values.astype(float)
    lows = future_slice["low"].values.astype(float)

    # Calculate running MFE and MAE bar by bar
    if direction == "LONG":
        favorable = (highs - entry_price) / atr_val  # How far price went up
        adverse = (entry_price - lows) / atr_val     # How far price went down
    else:
        favorable = (entry_price - lows) / atr_val
        adverse = (highs - entry_price) / atr_val

    max_favorable = np.maximum.accumulate(favorable)  # Running MFE
    max_adverse = np.maximum.accumulate(adverse)      # Running MAE

    total_mfe = max_favorable[-1] if len(max_favorable) > 0 else 0
    total_mae = max_adverse[-1] if len(max_adverse) > 0 else 0

    # Derive optimal SL: slightly beyond the MAE that was recovered from
    # Find the MAE at the point of max profit
    mfe_idx = np.argmax(max_favorable)
    mae_at_mfe = max_adverse[mfe_idx] if mfe_idx > 0 else max_adverse[0]

    # Optimal SL = MAE needed to survive + buffer
    optimal_sl = np.clip(mae_at_mfe + 0.3, MIN_SL_ATR, MAX_SL_ATR)

    # Optimal TP: based on achievable MFE in R-multiples of the SL
    if optimal_sl > 0:
        optimal_tp_r = np.clip(total_mfe / optimal_sl, MIN_TP_R, MAX_TP_R)
    else:
        optimal_tp_r = 3.0

    # Calculate actual PnL with these levels
    sl_dist = atr_val * optimal_sl
    tp_dist = sl_dist * optimal_tp_r

    if direction == "LONG":
        sl_price = entry_price - sl_dist
        tp_price = entry_price + tp_dist
    else:
        sl_price = entry_price + sl_dist
        tp_price = entry_price - tp_dist

    actual_pnl = -1.0  # Default: SL hit
    for j in range(len(highs)):
        if direction == "LONG":
            if lows[j] <= sl_price:
                actual_pnl = -1.0
                break
            if highs[j] >= tp_price:
                actual_pnl = optimal_tp_r
                break
        else:
            if highs[j] >= sl_price:
                actual_pnl = -1.0
                break
            if lows[j] <= tp_price:
                actual_pnl = optimal_tp_r
                break

    return optimal_sl, optimal_tp_r, actual_pnl


# ══════════════════════════════════════════════════════════════════════════
# ML FILTER CLASS (production use)
# ══════════════════════════════════════════════════════════════════════════

class MLFilter:
    """
    Production ML filter supporting both:
    - XGBoost models (from train_ml_v2.py with feature_mask)
    - sklearn GradientBoosting models (legacy online learning)
    """

    def __init__(self, confidence_threshold=CONFIDENCE_THRESHOLD):
        self.classifier = None
        self.sl_regressor = None
        self.tp_regressor = None
        self.scaler = StandardScaler()
        self.confidence_threshold = confidence_threshold
        self.feature_mask = None  # Boolean mask for feature selection
        self.training_features = []
        self.training_labels = []
        self.training_sl = []
        self.training_tp = []
        self.trades_since_retrain = 0
        self.is_trained = False
        self.feature_names = None

    def add_completed_trade(self, features: dict, won: bool, optimal_sl=None, optimal_tp=None):
        """Record a completed trade for future training."""
        self.training_features.append(list(features.values()))
        self.training_labels.append(1 if won else 0)
        if optimal_sl is not None:
            self.training_sl.append(optimal_sl)
        if optimal_tp is not None:
            self.training_tp.append(optimal_tp)
        self.trades_since_retrain += 1

        if self.feature_names is None:
            self.feature_names = list(features.keys())

        if (len(self.training_labels) >= MIN_TRAINING_SAMPLES and
                self.trades_since_retrain >= RETRAIN_INTERVAL):
            self._train()

    def _apply_feature_mask(self, feature_vals):
        """Apply feature mask to select relevant features from full vector."""
        if self.feature_mask is not None:
            # Ensure we have enough features for the mask
            if len(feature_vals) < len(self.feature_mask):
                feature_vals = feature_vals + [0.0] * (len(self.feature_mask) - len(feature_vals))
            return [v for v, m in zip(feature_vals, self.feature_mask) if m]
        return feature_vals

    def should_take_trade(self, features: dict) -> tuple:
        """
        Returns (should_take, confidence, suggested_sl_atr, suggested_tp_r).
        If not trained, returns defaults.
        """
        if not self.is_trained:
            return True, 0.5, 2.0, 3.0

        feature_vals = list(features.values())

        # Apply feature mask if present (XGBoost production model)
        feature_vals = self._apply_feature_mask(feature_vals)

        # Handle remaining size mismatch
        expected_n = self.scaler.n_features_in_
        if len(feature_vals) > expected_n:
            feature_vals = feature_vals[:expected_n]
        elif len(feature_vals) < expected_n:
            feature_vals.extend([0.0] * (expected_n - len(feature_vals)))

        X = np.array([feature_vals]).reshape(1, -1)
        X_scaled = self.scaler.transform(X)

        # Classifier
        prob = self.classifier.predict_proba(X_scaled)[0][1]
        should_take = prob >= self.confidence_threshold

        # SL/TP regression
        suggested_sl = 2.0
        suggested_tp = 3.0

        if self.sl_regressor is not None:
            suggested_sl = float(self.sl_regressor.predict(X_scaled)[0])
            suggested_sl = np.clip(suggested_sl, MIN_SL_ATR, MAX_SL_ATR)

        if self.tp_regressor is not None:
            suggested_tp = float(self.tp_regressor.predict(X_scaled)[0])
            suggested_tp = np.clip(suggested_tp, MIN_TP_R, MAX_TP_R)

        return should_take, prob, suggested_sl, suggested_tp

    def _train(self):
        """Train/retrain all models on accumulated online data."""
        if not HAS_SKLEARN_GB:
            return

        X = np.array(self.training_features)
        y = np.array(self.training_labels)

        if len(set(y)) < 2:
            return

        self.scaler.fit(X)
        X_scaled = self.scaler.transform(X)

        # Train classifier
        self.classifier = GradientBoostingClassifier(
            n_estimators=50,
            max_depth=2,
            learning_rate=0.1,
            min_samples_leaf=8,
            subsample=0.8,
            max_features=0.6,
            random_state=42,
        )
        self.classifier.fit(X_scaled, y)

        # Train SL regressor
        if len(self.training_sl) >= MIN_TRAINING_SAMPLES:
            y_sl = np.array(self.training_sl[:len(X)])
            self.sl_regressor = GradientBoostingRegressor(
                n_estimators=30,
                max_depth=2,
                learning_rate=0.1,
                min_samples_leaf=8,
                subsample=0.8,
                random_state=42,
            )
            self.sl_regressor.fit(X_scaled[:len(y_sl)], y_sl)

        # Train TP regressor
        if len(self.training_tp) >= MIN_TRAINING_SAMPLES:
            y_tp = np.array(self.training_tp[:len(X)])
            self.tp_regressor = GradientBoostingRegressor(
                n_estimators=30,
                max_depth=2,
                learning_rate=0.1,
                min_samples_leaf=8,
                subsample=0.8,
                random_state=42,
            )
            self.tp_regressor.fit(X_scaled[:len(y_tp)], y_tp)

        self.feature_mask = None  # Online-trained models don't use mask
        self.is_trained = True
        self.trades_since_retrain = 0

        train_acc = self.classifier.score(X_scaled, y)
        print(f"   ML retrained on {len(y)} trades | Acc: {train_acc:.1%}")

    def get_feature_importance(self):
        """Get feature importance from classifier."""
        if not self.is_trained or self.feature_names is None:
            return {}
        importances = self.classifier.feature_importances_
        return dict(sorted(
            zip(self.feature_names, importances),
            key=lambda x: x[1], reverse=True
        ))

    def save(self, filepath="ml_filter.pkl"):
        """Save all models to disk."""
        if not self.is_trained:
            return
        data = {
            "classifier": self.classifier,
            "sl_regressor": self.sl_regressor,
            "tp_regressor": self.tp_regressor,
            "scaler": self.scaler,
            "feature_names": self.feature_names,
            "feature_mask": self.feature_mask,
            "confidence_threshold": self.confidence_threshold,
            "training_features": self.training_features,
            "training_labels": self.training_labels,
            "training_sl": self.training_sl,
            "training_tp": self.training_tp,
        }
        with open(filepath, "wb") as f:
            pickle.dump(data, f)

    def load(self, filepath="ml_filter.pkl"):
        """Load trained models from disk. Supports both XGBoost and sklearn formats."""
        script_dir = os.path.dirname(os.path.abspath(__file__))
        full_path = os.path.join(script_dir, filepath) if not os.path.isabs(filepath) else filepath
        if not os.path.exists(full_path):
            # Try ml_models/ directory
            alt_path = os.path.join(script_dir, "ml_models", "ml_filter_production.pkl")
            if os.path.exists(alt_path):
                full_path = alt_path
            else:
                return False
        with open(full_path, "rb") as f:
            data = pickle.load(f)
        self.classifier = data.get("classifier")
        self.sl_regressor = data.get("sl_regressor")
        self.tp_regressor = data.get("tp_regressor")
        self.scaler = data.get("scaler", StandardScaler())
        self.feature_names = data.get("feature_names")
        self.feature_mask = data.get("feature_mask")
        self.confidence_threshold = data.get("confidence_threshold", CONFIDENCE_THRESHOLD)
        self.training_features = data.get("training_features", [])
        self.training_labels = data.get("training_labels", [])
        self.training_sl = data.get("training_sl", [])
        self.training_tp = data.get("training_tp", [])
        self.is_trained = self.classifier is not None
        return True
