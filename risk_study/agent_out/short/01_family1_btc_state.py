"""
Family 1: better BTC state variable for gating shorts.
Baseline to beat = gate OFF (no btc_short_gate at all), since that already
beats the shipped gate in-sample (see 00_baseline.py output).

For each state var and threshold, the rule is: skip SHORT entries where
state > threshold (for return/vol/dist-from-ema, "high" = risk of squeeze)
or state < threshold (for drawdown-from-high, "close to highs" = less negative
dd = risk of squeeze -> skip when dd > threshold, i.e. dd close to 0).
All state vars are already causal (see build_features.py).
"""
import sys
sys.path.insert(0, "/Users/lualakol/AutoTrading Bot/risk_study/agent_out/short")
from lib import *

df = load()
train, hold = split(df)

results = []

def eval_rule(sub, mask_skip, label):
    """mask_skip: boolean series, True = this short trade is skipped."""
    kept = sub[~(mask_skip)]
    d = summarize(kept, label)
    return d

base_train = summarize(train, "BASELINE gate-OFF (train)")
base_hold = summarize(hold, "BASELINE gate-OFF (holdout)")
results.append(base_train)
results.append(base_hold)
print(base_train)
print(base_hold)

state_specs = [
    # (col, direction) direction='high' means skip when state>thr, 'low' means skip when state<thr
    ("ret_7d_causal", "high"),
    ("ret_14d_causal", "high"),
    ("ret_30d_causal", "high"),
    ("ret_60d_causal", "high"),
    ("rvol_14d_causal", "high"),
    ("rvol_30d_causal", "high"),
    ("dist_ema200_causal", "high"),
    ("dd_from_high_all_causal", "high"),  # dd is negative; "high" (close to 0) = near highs = squeeze risk
    ("dd_from_high_90d_causal", "high"),
]

threshold_grids = {
    "ret_7d_causal": np.arange(0.02, 0.21, 0.02),
    "ret_14d_causal": np.arange(0.03, 0.31, 0.03),
    "ret_30d_causal": np.arange(0.05, 0.31, 0.025),
    "ret_60d_causal": np.arange(0.05, 0.51, 0.05),
    "rvol_14d_causal": np.quantile(train["rvol_14d_causal"].dropna(), np.arange(0.5, 1.0, 0.05)),
    "rvol_30d_causal": np.quantile(train["rvol_30d_causal"].dropna(), np.arange(0.5, 1.0, 0.05)),
    "dist_ema200_causal": np.arange(0.0, 0.31, 0.03),
    "dd_from_high_all_causal": np.arange(-0.30, 0.0, 0.03),
    "dd_from_high_90d_causal": np.arange(-0.30, 0.0, 0.03),
}

rows = []
for col, direction in state_specs:
    for thr in threshold_grids[col]:
        is_short = train.side == "short"
        if direction == "high":
            trigger = train[col] > thr
        else:
            trigger = train[col] < thr
        mask_skip = is_short & trigger
        kept = train[~mask_skip]
        n_skipped = mask_skip.sum()
        if n_skipped == 0 or n_skipped == is_short.sum():
            continue
        d = summarize(kept, f"{col} {direction} {thr:.4f}")
        w10, _ = worst_n_days(kept, 10)
        rows.append(dict(family="btc_state", var=col, direction=direction, threshold=thr,
                          n_skipped=int(n_skipped), n=d["n"], mean_r=d["mean_r"],
                          total_r=d["total_r"], worst10=d["worst10"]))

res = pd.DataFrame(rows)
res["delta_total_r"] = res["total_r"] - base_train["total_r"]
res["delta_worst10"] = res["worst10"] - base_train["worst10"]  # less negative = improvement
res = res.sort_values("delta_worst10")
pd.set_option("display.width", 200)
print("\n=== TOP 20 by improvement in worst10 (train) ===")
print(res.sort_values("delta_worst10", ascending=False).head(20).to_string(index=False))
print("\n=== TOP 20 by total_r (train), among those that also improve worst10 ===")
improve = res[res.delta_worst10 > 5]
print(improve.sort_values("total_r", ascending=False).head(20).to_string(index=False))

res.to_csv("/Users/lualakol/AutoTrading Bot/risk_study/agent_out/short/family1_sweep_train.csv", index=False)
print("\nsaved family1_sweep_train.csv, n_rows=", len(res))
