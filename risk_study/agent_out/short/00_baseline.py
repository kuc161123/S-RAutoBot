import sys
sys.path.insert(0, "/Users/lualakol/AutoTrading Bot/risk_study/agent_out/short")
from lib import *

df = load()
train, hold = split(df)

print("="*80)
print("PART 0: confirm the diagnosis -- worst 10 days, side breakdown")
print("="*80)

for name, sub in [("FULL", df), ("TRAIN (<2026-05-25)", train), ("HOLDOUT (>=2026-05-25)", hold)]:
    print(f"\n--- {name} --- n={len(sub)}")
    w10, worst_days = worst_n_days(sub, 10)
    print(f"worst 10 days total net_r: {w10:.1f}")
    # for each of those worst days, side split
    if len(worst_days):
        rows = []
        for d in worst_days.index:
            day_rows = sub[sub.exit_date == d]
            s = day_rows[day_rows.side=="short"]["net_r"].sum()
            l = day_rows[day_rows.side=="long"]["net_r"].sum()
            rows.append((d, len(day_rows), s, l, s+l))
        wd = pd.DataFrame(rows, columns=["date","n_trades","short_r","long_r","total_r"])
        print(wd.to_string(index=False))
        print(f"sum short_r across worst10: {wd.short_r.sum():.1f}, sum long_r: {wd.long_r.sum():.1f}")

print()
print("="*80)
print("PART 1: current control (btc_short_gate) ON vs OFF -- verify claim")
print("="*80)
# gate ON: drop short rows where btc_impulse True (these would never have been taken live)
def gate_on(sub):
    drop = (sub.side == "short") & (sub.btc_impulse)
    return sub[~drop]

for name, sub in [("TRAIN", train), ("HOLDOUT", hold)]:
    on = gate_on(sub)
    off = sub
    print(f"\n--- {name} ---")
    for label, s in [("gate ON (current live)", on), ("gate OFF", off)]:
        d = summarize(s, label)
        w10short, _ = worst_n_days(s[s.side=="short"], 10)
        print(f"  {label:28s} n={d['n']:6d} mean_r={d['mean_r']:.4f} total_r={d['total_r']:8.1f} "
              f"worst10(all)={d['worst10']:8.1f} worst10(short-only-book)={w10short:8.1f}")
