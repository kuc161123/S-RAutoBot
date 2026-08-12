"""
Family 4: cap the number of simultaneous open SHORT positions.
When a new short would push concurrency above the cap, skip it (longs uncapped).
"""
import sys
sys.path.insert(0, "/Users/lualakol/AutoTrading Bot/risk_study/agent_out/short")
from lib import *

df = load()
train, hold = split(df)

base_train = summarize(train, "BASELINE (train)")
print(base_train)

rows = []
for cap in [5, 8, 10, 15, 20, 25, 30, 40, 50, 60, 80, 100]:
    is_short = train.side == "short"
    mask_skip = is_short & (train["short_concurrency"] > cap)
    kept = train[~mask_skip]
    n_skipped = mask_skip.sum()
    d = summarize(kept, f"cap={cap}")
    w10, _ = worst_n_days(kept, 10)
    rows.append(dict(family="breadth_cap", var="short_concurrency", threshold=cap,
                      n_skipped=int(n_skipped), n=d["n"], mean_r=d["mean_r"],
                      total_r=d["total_r"], worst10=w10))

res = pd.DataFrame(rows)
res["delta_total_r"] = res["total_r"] - base_train["total_r"]
res["delta_worst10"] = res["worst10"] - base_train["worst10"]
pd.set_option("display.width", 200)
print(res.to_string(index=False))
res.to_csv("/Users/lualakol/AutoTrading Bot/risk_study/agent_out/short/family4_sweep_train.csv", index=False)
