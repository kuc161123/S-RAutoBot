import pandas as pd
import numpy as np
import pickle

OUT = "/Users/lualakol/AutoTrading Bot/risk_study/agent_out/cadence"

with open(f"{OUT}/oos_trades.pkl", "rb") as f:
    oos_trades = pickle.load(f)
with open(f"{OUT}/selection_log.pkl", "rb") as f:
    selection_log = pickle.load(f)
with open(f"{OUT}/meta.pkl", "rb") as f:
    meta = pickle.load(f)

T0, END = meta['T0'], meta['END']

rng = np.random.default_rng(12345)
B = 2000

# ---------------------------------------------------------------
# S4 static benchmark -- per-week sum/count over the common OOS window
# ---------------------------------------------------------------
s4_df = oos_trades[('static', 'na', 'S4_static')]

def week_sum_count(df, week_grid):
    g = df.groupby('week')['net_r'].agg(['sum', 'count'])
    g = g.reindex(week_grid, fill_value=0.0)
    return g['sum'].to_numpy(), g['count'].to_numpy()

week_grid = sorted(s4_df['week'].unique())
s4_sum, s4_cnt = week_sum_count(s4_df, week_grid)
s4_mean = s4_sum.sum() / s4_cnt.sum()

def block_bootstrap_diff(rule_sum, rule_cnt, base_sum, base_cnt, n_weeks, B=2000):
    idx = rng.integers(0, n_weeks, size=(B, n_weeks))
    r_sum = rule_sum[idx].sum(axis=1)
    r_cnt = rule_cnt[idx].sum(axis=1)
    b_sum = base_sum[idx].sum(axis=1)
    b_cnt = base_cnt[idx].sum(axis=1)
    with np.errstate(invalid='ignore', divide='ignore'):
        r_mean = np.where(r_cnt > 0, r_sum / np.maximum(r_cnt, 1), np.nan)
        b_mean = np.where(b_cnt > 0, b_sum / np.maximum(b_cnt, 1), np.nan)
    diff = r_mean - b_mean
    diff = diff[~np.isnan(diff)]
    return diff

# ---------------------------------------------------------------
# churn helpers
# ---------------------------------------------------------------
def churn_s1(hist):
    pairs = [p for (_, p) in hist]
    if len(pairs) < 2:
        return np.nan
    changes = sum(1 for a, b in zip(pairs, pairs[1:]) if a != b)
    return changes / (len(pairs) - 1)

def churn_grouped(hist):
    # hist: list of (date, {grp: pair}) -- fraction of COMMON groups whose pair changed,
    # averaged across consecutive refit transitions
    if len(hist) < 2:
        return np.nan
    fracs = []
    for (_, m0), (_, m1) in zip(hist, hist[1:]):
        common = set(m0.keys()) & set(m1.keys())
        if not common:
            continue
        changed = sum(1 for g in common if m0[g] != m1[g])
        fracs.append(changed / len(common))
    return np.mean(fracs) if fracs else np.nan

# ---------------------------------------------------------------
# main loop over every (cadence, lookback, rule)
# ---------------------------------------------------------------
rows = []
year_rows = []

for key, tdf in oos_trades.items():
    cad, lookback, rule = key
    if rule == 'S4_static':
        continue
    n = len(tdf)
    if n == 0:
        rows.append(dict(cadence=cad, lookback=lookback, rule=rule, n_trades=0,
                          mean_net_r=np.nan, churn_frac=np.nan,
                          diff_vs_s4=np.nan, ci_lo=np.nan, ci_hi=np.nan, beats_s4=False))
        continue
    mean_r = tdf['net_r'].mean()

    hist = selection_log[key]
    if rule == 'S1_global':
        churn = churn_s1(hist)
    else:
        churn = churn_grouped(hist)

    rule_sum, rule_cnt = week_sum_count(tdf, week_grid)
    diffs = block_bootstrap_diff(rule_sum, rule_cnt, s4_sum, s4_cnt, len(week_grid), B=B)
    point_diff = mean_r - s4_mean
    if len(diffs):
        ci_lo, ci_hi = np.percentile(diffs, [2.5, 97.5])
    else:
        ci_lo, ci_hi = np.nan, np.nan
    beats_s4 = bool(ci_lo > 0)

    rows.append(dict(cadence=cad, lookback=lookback, rule=rule, n_trades=n,
                      mean_net_r=mean_r, churn_frac=churn,
                      diff_vs_s4=point_diff, ci_lo=ci_lo, ci_hi=ci_hi, beats_s4=beats_s4))

    yg = tdf.groupby('year')['net_r'].agg(['mean', 'count'])
    for yr, r in yg.iterrows():
        year_rows.append(dict(cadence=cad, lookback=lookback, rule=rule, year=int(yr),
                               n_trades=int(r['count']), mean_net_r=r['mean']))

# S4 itself
s4_year = s4_df.groupby('year')['net_r'].agg(['mean', 'count'])
for yr, r in s4_year.iterrows():
    year_rows.append(dict(cadence='static', lookback='na', rule='S4_static', year=int(yr),
                           n_trades=int(r['count']), mean_net_r=r['mean']))
rows.append(dict(cadence='static', lookback='na', rule='S4_static', n_trades=len(s4_df),
                  mean_net_r=s4_mean, churn_frac=np.nan,
                  diff_vs_s4=0.0, ci_lo=0.0, ci_hi=0.0, beats_s4=False))

overall = pd.DataFrame(rows).sort_values(['rule', 'cadence', 'lookback'])
by_year = pd.DataFrame(year_rows).sort_values(['rule', 'cadence', 'lookback', 'year'])

# combine into one long CSV: 'year' == 'ALL' rows carry churn/bootstrap; per-year rows carry only mean/count
overall['year'] = 'ALL'
by_year['churn_frac'] = np.nan
by_year['diff_vs_s4'] = np.nan
by_year['ci_lo'] = np.nan
by_year['ci_hi'] = np.nan
by_year['beats_s4'] = np.nan

combined = pd.concat([overall, by_year], ignore_index=True, sort=False)
combined = combined[['cadence', 'lookback', 'rule', 'year', 'n_trades', 'mean_net_r',
                      'churn_frac', 'diff_vs_s4', 'ci_lo', 'ci_hi', 'beats_s4']]
combined.to_csv(f"{OUT}/cadence_results.csv", index=False)

print("wrote", f"{OUT}/cadence_results.csv", "rows:", len(combined))
print()
print("=== overall (ALL years) table, sorted by mean_net_r desc within rule ===")
with pd.option_context('display.max_rows', 200):
    print(overall.sort_values('mean_net_r', ascending=False).to_string(index=False))

print()
print("S4 static mean_net_r:", s4_mean, "n:", len(s4_df))
