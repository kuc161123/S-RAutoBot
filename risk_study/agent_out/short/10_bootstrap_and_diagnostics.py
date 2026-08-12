import sys
sys.path.insert(0, "/Users/lualakol/AutoTrading Bot/risk_study/agent_out/short")
from lib import *

df = load()
train, hold = split(df)

RULE_COL = "density_all_72h"
RULE_THR = 160

def apply_rule(sub):
    is_short = sub.side == "short"
    mask_skip = is_short & (sub[RULE_COL] > RULE_THR)
    return sub[~mask_skip]

# --- diagnostic: how many of the ORIGINAL worst-10 train days' SHORT R does this rule remove? ---
w10, worst_days = worst_n_days(train, 10)
orig_short_r_on_worst = train[(train.exit_date.isin(worst_days.index)) & (train.side=="short")]["net_r"].sum()
kept_train = apply_rule(train)
kept_short_r_on_worst = kept_train[(kept_train.exit_date.isin(worst_days.index)) & (kept_train.side=="short")]["net_r"].sum()
print(f"Original worst-10-days short R (train): {orig_short_r_on_worst:.1f}")
print(f"After rule, short R still remaining on those SAME 10 days: {kept_short_r_on_worst:.1f}")
print(f"-> rule removed {orig_short_r_on_worst - kept_short_r_on_worst:.1f} R specifically from the original worst days")
print(f"-> total worst10 change (recomputed, days can shift): {worst_n_days(kept_train,10)[0] - w10:.1f}")

n_short_train = (train.side=="short").sum()
n_skipped_train = n_short_train - (apply_rule(train).side=="short").sum()
print(f"\nfraction of ALL short trades skipped (train): {n_skipped_train}/{n_short_train} = {n_skipped_train/n_short_train:.2%}")

n_short_hold = (hold.side=="short").sum()
n_skipped_hold = n_short_hold - (apply_rule(hold).side=="short").sum()
print(f"fraction of ALL short trades skipped (holdout): {n_skipped_hold}/{n_short_hold} = {n_skipped_hold/n_short_hold:.2%}")

# --- weekly block bootstrap CIs ---
print("\n=== weekly block bootstrap (1000 resamples), rule vs baseline ===")
for pname, sub in [("TRAIN", train), ("HOLDOUT", hold)]:
    kept = apply_rule(sub)
    for label, s in [("baseline", sub), ("rule-applied", kept)]:
        m_total, lo_total, hi_total = week_block_bootstrap(s, stat_total_r, n_boot=1000, seed=42)
        m_w10, lo_w10, hi_w10 = week_block_bootstrap(s, stat_worst10, n_boot=1000, seed=43)
        print(f"{pname:8s} {label:14s} total_r: mean={m_total:8.1f} CI=[{lo_total:8.1f},{hi_total:8.1f}]   "
              f"worst10: mean={m_w10:8.1f} CI=[{lo_w10:8.1f},{hi_w10:8.1f}]")
