import sys
sys.path.insert(0, "/Users/lualakol/AutoTrading Bot/risk_study/agent_out/short")
from lib import *

df = load()

# add more density windows (6h, 12h, 48h) for both all and short
all_entries = np.sort(df["entry_time"].values.astype("datetime64[ns]"))
short_entries = np.sort(df.loc[df.side == "short", "entry_time"].values.astype("datetime64[ns]"))

def trailing_count(query_times, event_times, window_hours):
    window = np.timedelta64(window_hours, "h")
    lo_idx = np.searchsorted(event_times, query_times - window, side="left")
    hi_idx = np.searchsorted(event_times, query_times, side="left")
    return hi_idx - lo_idx

qt = df["entry_time"].values.astype("datetime64[ns]")
for h in [6, 12, 48]:
    df[f"density_all_{h}h"] = trailing_count(qt, all_entries, h)
    df[f"density_short_{h}h"] = trailing_count(qt, short_entries, h)

train, hold = split(df)
base_train = summarize(train, "BASELINE (train)")
print(base_train)

rows = []
for col in ["density_short_72h", "density_all_72h", "density_short_48h", "density_all_48h",
            "density_short_24h", "density_all_24h", "density_short_12h", "density_all_12h",
            "density_short_6h", "density_all_6h"]:
    thresholds = sorted(set(int(x) for x in train[col].quantile(np.arange(0.85, 0.995, 0.01)).values))
    for thr in thresholds:
        is_short = train.side == "short"
        mask_skip = is_short & (train[col] > thr)
        kept = train[~mask_skip]
        n_skipped = mask_skip.sum()
        if n_skipped < 20:
            continue
        d = summarize(kept, f"{col}>{thr}")
        w10, _ = worst_n_days(kept, 10)
        rows.append(dict(family="density_fine", var=col, threshold=thr, n_skipped=int(n_skipped),
                          n=d["n"], mean_r=d["mean_r"], total_r=d["total_r"], worst10=w10))

res = pd.DataFrame(rows)
res["delta_total_r"] = res["total_r"] - base_train["total_r"]
res["delta_worst10"] = res["worst10"] - base_train["worst10"]
res["ratio"] = res["delta_worst10"] / (-res["delta_total_r"]).clip(lower=1)
res = res[res.delta_total_r < -1]
pd.set_option("display.width", 200)
print(res.sort_values("ratio", ascending=False).head(25).to_string(index=False))
res.to_csv("/Users/lualakol/AutoTrading Bot/risk_study/agent_out/short/family2_finesweep_train.csv", index=False)

df.to_parquet("/Users/lualakol/AutoTrading Bot/risk_study/agent_out/short/trades_enriched.parquet", index=False)
print("saved enriched parquet with extra windows")
