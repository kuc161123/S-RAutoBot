import sys
sys.path.insert(0, "/Users/lualakol/AutoTrading Bot/risk_study/agent_out/short")
from lib import *

df = load()
train, hold = split(df)

rows = []

def add(rule, period, sub, n_skipped=0, notes=""):
    d = summarize(sub)
    w10, _ = worst_n_days(sub, 10)
    rows.append(dict(rule=rule, period=period, n=d["n"], n_skipped=int(n_skipped),
                      mean_net_r=round(d["mean_r"], 4), total_net_r=round(d["total_r"], 1),
                      worst10_days_r=round(w10, 1), notes=notes))

# baselines
gate_on_train = train[~((train.side=="short") & (train.btc_impulse))]
add("baseline: gate ON (shipped btc_short_gate, ret30d>0.10)", "train", gate_on_train,
    n_skipped=len(train)-len(gate_on_train))
gate_on_hold = hold[~((hold.side=="short") & (hold.btc_impulse))]
add("baseline: gate ON (shipped btc_short_gate, ret30d>0.10)", "holdout", gate_on_hold,
    n_skipped=len(hold)-len(gate_on_hold))

add("baseline: gate OFF (no btc_short_gate)", "train", train, notes="current best-known baseline; beats shipped gate")
add("baseline: gate OFF (no btc_short_gate)", "holdout", hold)

# family 1 best few (fixed thresholds, evaluated both periods)
f1_specs = [
    ("family1: skip short if ret_30d_causal>0.10 (=shipped gate replicated)", "ret_30d_causal", 0.10, "high"),
    ("family1: skip short if ret_30d_causal>0.15", "ret_30d_causal", 0.15, "high"),
    ("family1: skip short if rvol_30d_causal>0.0298 (top~15pct realized vol)", "rvol_30d_causal", 0.0298, "high"),
    ("family1: skip short if rvol_14d_causal>0.0293", "rvol_14d_causal", 0.0293, "high"),
]
for label, col, thr, direction in f1_specs:
    for pname, sub in [("train", train), ("holdout", hold)]:
        is_short = sub.side == "short"
        trig = sub[col] > thr if direction == "high" else sub[col] < thr
        mask = is_short & trig
        kept = sub[~mask]
        add(label, pname, kept, n_skipped=int(mask.sum()), notes="BTC/vol state gate")

# family 2: crowding/density -- the winner
for thr in [140, 150, 160, 170, 180]:
    label = f"family2: skip short if density_all_72h>{thr} (universe-wide trailing-72h entry count)"
    for pname, sub in [("train", train), ("holdout", hold)]:
        is_short = sub.side == "short"
        mask = is_short & (sub["density_all_72h"] > thr)
        kept = sub[~mask]
        add(label, pname, kept, n_skipped=int(mask.sum()),
            notes="RECOMMENDED at thr=160" if thr == 160 else "sensitivity check")

# family 3: side-asymmetric sizing, always
for f in [0.5, 0.65, 0.8]:
    label = f"family3: scale ALL short net_r by {f} (always)"
    for pname, sub in [("train", train), ("holdout", hold)]:
        s = sub.copy()
        s.loc[s.side=="short", "net_r"] *= f
        add(label, pname, s, n_skipped=0, notes="linear de-risking, not selective")

# family 3b: conditional sizing on the density flag
for f in [0.0, 0.5, 0.65, 0.8]:
    label = f"family3b: scale short net_r by {f} only when density_all_72h>160"
    for pname, sub in [("train", train), ("holdout", hold)]:
        s = sub.copy()
        mask = (s.side == "short") & (s["density_all_72h"] > 160)
        s.loc[mask, "net_r"] *= f
        add(label, pname, s, n_skipped=int((mask & (f==0)).sum()) if f == 0 else 0,
            notes="dial between full skip (f=0) and no action (f=1)")

# family 4: breadth cap (best-performing cap, for completeness -- shown to be much worse trade-off)
for cap in [20, 50, 100]:
    label = f"family4: cap concurrent open shorts at {cap}"
    for pname, sub in [("train", train), ("holdout", hold)]:
        if "short_concurrency" not in sub.columns:
            continue
        is_short = sub.side == "short"
        mask = is_short & (sub["short_concurrency"] > cap)
        kept = sub[~mask]
        add(label, pname, kept, n_skipped=int(mask.sum()), notes="poor cost/benefit -- see report")

res = pd.DataFrame(rows)
out = "/Users/lualakol/AutoTrading Bot/risk_study/agent_out/short/short_results.csv"
res.to_csv(out, index=False)
print(f"wrote {out}, {len(res)} rows")
print(res.to_string(index=False))
