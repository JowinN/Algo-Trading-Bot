from ml_model import MLFilter, extract_regime_features, compute_regimes
from data import get_ohlcv
from indicators import compute_all
from config import Config as c

# Test with AVAX (where L=58.7, S=57 - almost same)
df = get_ohlcv("AVAXUSDT", c.PRIMARY_TF, limit=100)
df = compute_all(df)
df_r = compute_regimes(df.copy())
idx = len(df_r) - 1

fl = extract_regime_features(df_r, idx, "LONG")
fs = extract_regime_features(df_r, idx, "SHORT")

# Count how many features actually differ
diff_count = 0
for k in fl:
    if abs(fl[k] - fs[k]) > 0.001:
        diff_count += 1
        print(f"  {k}: L={fl[k]:.4f} S={fs[k]:.4f} diff={fl[k]-fs[k]:.4f}")

print(f"\nTotal features: {len(fl)}")
print(f"Features that differ: {diff_count}")
print(f"Features same: {len(fl) - diff_count}")

# Now load model and check
ml = MLFilter()
ml.load()
_, conf_l, _, _ = ml.should_take_trade(fl)
_, conf_s, _, _ = ml.should_take_trade(fs)
print(f"\nModel scores: L={conf_l:.4f} S={conf_s:.4f} diff={abs(conf_l-conf_s):.4f}")
