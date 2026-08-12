import sys
sys.path.insert(0, "/Users/lualakol/AutoTrading Bot/risk_study/agent_out/short")
from lib import *

df = load()
train, hold = split(df)

w10, worst_days = worst_n_days(train, 20)  # look at worst 20 for more signal
print("worst 20 days (train) and their BTC state at entry (mean across that day's short trades):")
cols = ["ret_7d_causal","ret_14d_causal","ret_30d_causal","ret_60d_causal",
        "rvol_14d_causal","rvol_30d_causal","dist_ema200_causal",
        "dd_from_high_all_causal","dd_from_high_90d_causal",
        "density_short_24h","density_short_72h","short_concurrency"]

rows = []
for d in worst_days.index:
    day_rows = train[(train.exit_date==d) & (train.side=="short")]
    entry_day_rows = train[(train.entry_date==d) & (train.side=="short")]
    r = {"exit_date": d, "n_short_exit": len(day_rows), "short_r": day_rows.net_r.sum()}
    for c in cols:
        r[c] = day_rows[c].mean() if len(day_rows) else np.nan
    rows.append(r)
wd = pd.DataFrame(rows)
pd.set_option("display.width", 220)
print(wd.to_string(index=False))

print("\n--- population stats for comparison (all short trades, train) ---")
short_train = train[train.side=="short"]
print(short_train[cols].describe().T[["mean","50%","75%","90%".replace("90%","75%") if False else "max"]])
for c in cols:
    print(c, "median=", short_train[c].median(), "p75=", short_train[c].quantile(.75), "p90=", short_train[c].quantile(.90))
