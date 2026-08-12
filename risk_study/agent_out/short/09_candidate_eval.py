import sys
sys.path.insert(0, "/Users/lualakol/AutoTrading Bot/risk_study/agent_out/short")
from lib import *

df = load()
train, hold = split(df)

def apply_density_rule(sub, col, thr):
    is_short = sub.side == "short"
    mask_skip = is_short & (sub[col] > thr)
    return sub[~mask_skip], mask_skip.sum()

candidates = [
    ("density_all_72h", 140), ("density_all_72h", 150), ("density_all_72h", 160),
    ("density_all_72h", 164), ("density_all_72h", 170), ("density_all_72h", 180),
    ("density_all_48h", 100), ("density_all_48h", 110), ("density_all_48h", 120), ("density_all_48h", 130),
    ("density_short_72h", 140), ("density_short_72h", 150), ("density_short_72h", 160),
]

print(f"{'rule':30s} {'period':8s} {'n':>7s} {'n_skip':>7s} {'mean_r':>8s} {'total_r':>10s} {'worst10':>10s}")
base_train = summarize(train, "base"); w10_bt,_ = worst_n_days(train,10)
base_hold = summarize(hold, "base"); w10_bh,_ = worst_n_days(hold,10)
print(f"{'BASELINE (gate-off)':30s} {'train':8s} {base_train['n']:7d} {0:7d} {base_train['mean_r']:8.4f} {base_train['total_r']:10.1f} {w10_bt:10.1f}")
print(f"{'BASELINE (gate-off)':30s} {'hold':8s} {base_hold['n']:7d} {0:7d} {base_hold['mean_r']:8.4f} {base_hold['total_r']:10.1f} {w10_bh:10.1f}")

rows = []
for col, thr in candidates:
    for pname, sub in [("train", train), ("hold", hold)]:
        kept, n_skip = apply_density_rule(sub, col, thr)
        d = summarize(kept)
        w10, _ = worst_n_days(kept, 10)
        print(f"{col+'>'+str(thr):30s} {pname:8s} {d['n']:7d} {n_skip:7d} {d['mean_r']:8.4f} {d['total_r']:10.1f} {w10:10.1f}")
        rows.append(dict(rule=f"{col}>{thr}", period=pname, n=d['n'], n_skipped=int(n_skip),
                          mean_r=d['mean_r'], total_r=d['total_r'], worst10=w10))
    print()

pd.DataFrame(rows).to_csv("/Users/lualakol/AutoTrading Bot/risk_study/agent_out/short/candidate_eval.csv", index=False)
