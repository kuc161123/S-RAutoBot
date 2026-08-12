import pandas as pd
import numpy as np
import pickle, time, os

pd.set_option('display.width', 160)

DATA = "/Users/lualakol/AutoTrading Bot/risk_study/grid_inuni.parquet"
OUT = "/Users/lualakol/AutoTrading Bot/risk_study/agent_out/cadence"

RRS = [2, 3, 5, 8, 10]
ATRS = [1.0, 1.5, 2.0, 3.0]
PAIRS = [(a, r) for a in ATRS for r in RRS]

# ---------------------------------------------------------------
# 1. Load + filter  (verified against spec: CHOP<52 gate, drop NaN chop,
#    trim last 21 days of ENTRIES for right-censoring, cost = 0.00242/stop_frac)
# ---------------------------------------------------------------
df = pd.read_parquet(DATA)
df = df[df['chop_bos'] < 52].copy()
df = df.dropna(subset=['chop_bos'])

max_entry = df['entry_time'].max()
cutoff = max_entry - pd.Timedelta(days=21)
df = df[df['entry_time'] < cutoff].copy()
print("after chop+trim:", len(df), "date range", df['entry_time'].min(), df['entry_time'].max())

df['cost'] = 0.00242 / df['stop_frac']
df['grp'] = df['symbol'].astype(str) + '|' + df['div_type'].astype(str)

# long format: one row per (signal-row, rr) -- atr_mult already varies per source row
frames = []
for rr in RRS:
    tmp = df[['grp', 'symbol', 'div_type', 'entry_time', 'atr_mult', 'cost']].copy()
    tmp['rr'] = rr
    tmp['net_r'] = df[f'r_{rr}'] - df['cost']
    frames.append(tmp)
long = pd.concat(frames, ignore_index=True)
long = long.dropna(subset=['net_r'])
long['pair'] = list(zip(long['atr_mult'], long['rr']))
print("long rows (resolved trades x pair):", len(long))

long = long.sort_values('entry_time').reset_index(drop=True)
long['week'] = long['entry_time'].dt.to_period('W').astype(str)
long['year'] = long['entry_time'].dt.year

START = df['entry_time'].min()
END = df['entry_time'].max()
BURNIN_MONTHS = 6
T0 = START + pd.DateOffset(months=BURNIN_MONTHS)
print("T0 (first refit / OOS start):", T0, " END:", END)

# ---------------------------------------------------------------
# helper: refit date schedules
# ---------------------------------------------------------------
def refit_dates(cadence_months):
    if cadence_months is None:  # 'never'
        return [T0]
    dates = [T0]
    d = T0
    while True:
        d = d + pd.DateOffset(months=cadence_months)
        if d >= END:
            break
        dates.append(d)
    return dates

CADENCES = {
    'never': None,
    '12m': 12,
    '6m': 6,
    '3m': 3,
    '1m': 1,   # compute is cheap here (~0.1s per groupby) -- kept in scope
}

MIN_TRADES_GLOBAL = 100

def train_slice(t, lookback):
    if lookback == 'anchored':
        mask = long['entry_time'] < t
    else:  # rolling12
        mask = (long['entry_time'] < t) & (long['entry_time'] >= t - pd.DateOffset(months=12))
    return long[mask]

def best_global_pair(train):
    stats = train.groupby('pair')['net_r'].agg(['mean', 'count'])
    qual = stats[stats['count'] >= MIN_TRADES_GLOBAL]
    if qual.empty:
        if stats.empty:
            return None
        return stats['mean'].idxmax()
    return qual['mean'].idxmax()

def best_per_group_pairs(train, threshold):
    stats = train.groupby(['grp', 'pair'])['net_r'].agg(['mean', 'count']).reset_index()
    stats = stats[stats['count'] >= threshold]
    if stats.empty:
        return {}
    idx = stats.groupby('grp')['mean'].idxmax()
    best = stats.loc[idx].set_index('grp')['pair'].to_dict()
    return best

# ---------------------------------------------------------------
# core simulation for one (cadence, lookback) -> produces per-rule OOS trade tables
# Saves incrementally to disk after each cadence so nothing is lost.
# ---------------------------------------------------------------
oos_trades = {}      # key: (cadence, lookback, rule) -> DataFrame(sig cols: entry_time, net_r, week, year)
selection_log = {}   # key: (cadence, lookback, rule) -> list of (refit_date, selection)

KEEP_COLS = ['entry_time', 'net_r', 'week', 'year']

t_all = time.time()
for cad_name, cad_months in CADENCES.items():
    t_cad = time.time()
    r_dates = refit_dates(cad_months)
    seg_bounds = list(zip(r_dates, r_dates[1:] + [END + pd.Timedelta(days=1)]))
    for lookback in ['anchored', 'rolling12']:
        # --- S1 global ---
        global_sel_hist = []
        s1_trades = []
        for (t_start, t_end) in seg_bounds:
            train = train_slice(t_start, lookback)
            pair = best_global_pair(train)
            global_sel_hist.append((t_start, pair))
            if pair is None:
                continue
            seg = long[(long['entry_time'] >= t_start) & (long['entry_time'] < t_end) &
                       (long['atr_mult'] == pair[0]) & (long['rr'] == pair[1])]
            s1_trades.append(seg[KEEP_COLS])
        s1_df = pd.concat(s1_trades, ignore_index=True) if s1_trades else pd.DataFrame(columns=KEEP_COLS)
        oos_trades[(cad_name, lookback, 'S1_global')] = s1_df
        selection_log[(cad_name, lookback, 'S1_global')] = global_sel_hist

        # --- S2 / S3 per group, for thresholds 20 and 50 ---
        # Vectorised: map each row's grp -> its assigned (atr_mult, rr) via a merge,
        # then keep rows whose own (atr_mult, rr) equals the assignment. This avoids
        # looping over ~1100 groups per refit (that loop is what made the previous
        # attempt too slow).
        for thr in [20, 50]:
            s2_hist = []
            s3_hist = []
            s2_trades = []
            s3_trades = []
            for (t_start, t_end) in seg_bounds:
                train = train_slice(t_start, lookback)
                grp_pairs = best_per_group_pairs(train, thr)
                gpair = best_global_pair(train)
                s2_hist.append((t_start, dict(grp_pairs)))
                all_groups = train['grp'].unique()
                s3_map = {g: grp_pairs.get(g, gpair) for g in all_groups}
                s3_hist.append((t_start, dict(s3_map)))

                seg = long[(long['entry_time'] >= t_start) & (long['entry_time'] < t_end)]
                if len(seg) == 0:
                    continue

                # S2: only groups with a qualifying own pair
                if grp_pairs:
                    map2 = pd.DataFrame(
                        [(g, p[0], p[1]) for g, p in grp_pairs.items()],
                        columns=['grp', 'atr_want', 'rr_want'])
                    merged = seg.merge(map2, on='grp', how='inner')
                    sub = merged[(merged['atr_mult'] == merged['atr_want']) &
                                 (merged['rr'] == merged['rr_want'])]
                    if len(sub):
                        s2_trades.append(sub[KEEP_COLS])

                # S3: every group appearing in this OOS segment gets a pair
                #     (own qualifying pair if available, else global; even if the
                #     group never appeared in training at all)
                if gpair is not None:
                    map3 = pd.DataFrame(
                        [(g, p[0], p[1]) for g, p in s3_map.items() if p is not None],
                        columns=['grp', 'atr_want', 'rr_want'])
                    merged3 = seg.merge(map3, on='grp', how='left')
                    # groups unseen in training (not in s3_map) fall back to global
                    merged3['atr_want'] = merged3['atr_want'].fillna(gpair[0])
                    merged3['rr_want'] = merged3['rr_want'].fillna(gpair[1])
                    sub3 = merged3[(merged3['atr_mult'] == merged3['atr_want']) &
                                   (merged3['rr'] == merged3['rr_want'])]
                    if len(sub3):
                        s3_trades.append(sub3[KEEP_COLS])

            s2_df = pd.concat(s2_trades, ignore_index=True) if s2_trades else pd.DataFrame(columns=KEEP_COLS)
            s3_df = pd.concat(s3_trades, ignore_index=True) if s3_trades else pd.DataFrame(columns=KEEP_COLS)
            oos_trades[(cad_name, lookback, f'S2_persym_thr{thr}')] = s2_df
            oos_trades[(cad_name, lookback, f'S3_shrunk_thr{thr}')] = s3_df
            selection_log[(cad_name, lookback, f'S2_persym_thr{thr}')] = s2_hist
            selection_log[(cad_name, lookback, f'S3_shrunk_thr{thr}')] = s3_hist

    print(f"cadence {cad_name} done in {time.time()-t_cad:.1f}s (n_refits={len(r_dates)})")
    # incremental save -- nothing is lost if the run is interrupted
    with open(f"{OUT}/oos_trades.pkl", "wb") as f:
        pickle.dump(oos_trades, f)
    with open(f"{OUT}/selection_log.pkl", "wb") as f:
        pickle.dump(selection_log, f)

print(f"done simulating cadence x lookback grid in {time.time()-t_all:.1f}s")

# ---------------------------------------------------------------
# S4 static control: rr=10, atr_mult=3.0 fixed for ALL history, never refit.
# Restricted to entry_time >= T0 so the OOS window matches every other rule.
# ---------------------------------------------------------------
s4_all = long[(long['atr_mult'] == 3.0) & (long['rr'] == 10)][KEEP_COLS]
s4_oos = s4_all[s4_all['entry_time'] >= T0].reset_index(drop=True)
oos_trades[('static', 'na', 'S4_static')] = s4_oos

with open(f"{OUT}/oos_trades.pkl", "wb") as f:
    pickle.dump(oos_trades, f)
with open(f"{OUT}/selection_log.pkl", "wb") as f:
    pickle.dump(selection_log, f)
with open(f"{OUT}/meta.pkl", "wb") as f:
    pickle.dump({'T0': T0, 'START': START, 'END': END, 'CADENCES': CADENCES}, f)

print("S4 static trades:", len(s4_oos), "mean net_r:", s4_oos['net_r'].mean())
for k, v in oos_trades.items():
    if len(v):
        print(k, len(v), round(v['net_r'].mean(), 4))
    else:
        print(k, 0, 'EMPTY')
