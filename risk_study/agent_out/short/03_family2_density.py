"""
Family 2: signal-density / crowding throttle.
Skip SHORT entries when the trailing count of short entries (24h or 72h window,
fully causal) exceeds a threshold.
"""
import sys
sys.path.insert(0, "/Users/lualakol/AutoTrading Bot/risk_study/agent_out/short")
from lib import *

df = load()
train, hold = split(df)

base_train = summarize(train, "BASELINE (train)")
print(base_train)

rows = []
for col in ["density_short_24h", "density_short_72h", "density_all_24h", "density_all_72h"]:
    qs = np.arange(0.3, 0.99, 0.05)
    thresholds = sorted(set(int(x) for x in train[col].quantile(qs).values))
    for thr in thresholds:
        is_short = train.side == "short"
        mask_skip = is_short & (train[col] > thr)
        kept = train[~mask_skip]
        n_skipped = mask_skip.sum()
        if n_skipped == 0:
            continue
        d = summarize(kept, f"{col}>{thr}")
        w10, _ = worst_n_days(kept, 10)
        rows.append(dict(family="density", var=col, threshold=thr, n_skipped=int(n_skipped),
                          n=d["n"], mean_r=d["mean_r"], total_r=d["total_r"], worst10=w10))

res = pd.DataFrame(rows)
res["delta_total_r"] = res["total_r"] - base_train["total_r"]
res["delta_worst10"] = res["worst10"] - base_train["worst10"]
pd.set_option("display.width", 200)
print("\n=== sorted by delta_worst10 desc (best tail improvement first) ===")
print(res.sort_values("delta_worst10", ascending=False).head(25).to_string(index=False))
res.to_csv("/Users/lualakol/AutoTrading Bot/risk_study/agent_out/short/family2_sweep_train.csv", index=False)
print("\nsaved family2_sweep_train.csv, n=", len(res))
