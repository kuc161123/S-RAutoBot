import sys
sys.path.insert(0, "/Users/lualakol/AutoTrading Bot/risk_study/agent_out/short")
from lib import *
from scipy import stats as sstats

df = load()
train, hold = split(df)
short_train = train[train.side == "short"].copy()

w10, worst10_days = worst_n_days(train, 10)
w30, worst30_days = worst_n_days(train, 30)
short_train["is_tail10"] = short_train["exit_date"].isin(worst10_days.index)
short_train["is_tail30"] = short_train["exit_date"].isin(worst30_days.index)

cols = ["ret_7d_causal","ret_14d_causal","ret_30d_causal","ret_60d_causal",
        "rvol_14d_causal","rvol_30d_causal","dist_ema200_causal",
        "dd_from_high_all_causal","dd_from_high_90d_causal",
        "density_short_24h","density_short_72h","short_concurrency"]

for tail_col in ["is_tail10", "is_tail30"]:
    print(f"\n=== {tail_col} (n_tail={short_train[tail_col].sum()}, n_other={(~short_train[tail_col]).sum()}) ===")
    for c in cols:
        tail_vals = short_train.loc[short_train[tail_col], c].dropna()
        other_vals = short_train.loc[~short_train[tail_col], c].dropna()
        t, p = sstats.mannwhitneyu(tail_vals, other_vals, alternative="two-sided")
        print(f"  {c:28s} tail_mean={tail_vals.mean():+.4f} other_mean={other_vals.mean():+.4f} "
              f"tail_med={tail_vals.median():+.4f} other_med={other_vals.median():+.4f}  MWU p={p:.2e}")

# --- combo: deep drawdown + recent bounce already underway ---
print("\n=== combo feature: dd_from_high_all_causal AND ret_7d_causal (bounce-off-lows) ===")
short_train["combo_bounce"] = (short_train["dd_from_high_all_causal"] < -0.15) & (short_train["ret_7d_causal"] > 0.0)
for tail_col in ["is_tail10", "is_tail30"]:
    ct = pd.crosstab(short_train["combo_bounce"], short_train[tail_col])
    print(tail_col)
    print(ct)
    rate_flag = short_train.loc[short_train.combo_bounce, tail_col].mean()
    rate_noflag = short_train.loc[~short_train.combo_bounce, tail_col].mean()
    print(f"  P({tail_col}|flag)={rate_flag:.4f}  P({tail_col}|~flag)={rate_noflag:.4f}  n_flag={short_train.combo_bounce.sum()}")
