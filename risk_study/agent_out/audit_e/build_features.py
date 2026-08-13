"""
AUDIT-E: build a feature-augmented copy of the current-config trade set
(uni_glob_rr10_am3_same.parquet) for testing additive ideas.

All features are computed CAUSALLY (only information available at/ before entry_time).
Output: audit_e/trades_features.parquet
"""
import pandas as pd
import numpy as np
import glob, os

ROOT = "/Users/lualakol/AutoTrading Bot"
OUT = f"{ROOT}/risk_study/agent_out/audit_e"

print("Loading base trade set...")
uni = pd.read_parquet(f"{ROOT}/risk_study/uni_glob_rr10_am3_same.parquet")
uni = uni.sort_values("entry_time").reset_index(drop=True)
print("base rows", len(uni))

COST_BPS = 0.00242
uni["net_r"] = uni["r_result"] - COST_BPS / uni["stop_frac"]

# ---- trim last 21 days of entries ----
cutoff = uni.entry_time.max() - pd.Timedelta(days=21)
print("trim cutoff (entries after this dropped):", cutoff)
uni = uni[uni.entry_time <= cutoff].reset_index(drop=True)
print("rows after trim", len(uni))

SPLIT = pd.Timestamp("2026-05-25")
uni["period"] = np.where(uni.entry_time < SPLIT, "pre", "post")
print(uni.period.value_counts())

# ================= 1. liq_30d, chop_bos (already have), from grid_inuni =================
print("Merging liq_30d from grid_inuni...")
grid = pd.read_parquet(f"{ROOT}/risk_study/grid_inuni.parquet")
g3 = grid[grid.atr_mult == 3.0].copy()
key_cols = ["symbol", "div_type", "side", "entry_time"]
g3 = g3.drop_duplicates(subset=key_cols)
uni = uni.merge(g3[key_cols + ["liq_30d"]], on=key_cols, how="left")
print("liq_30d matched:", uni.liq_30d.notna().sum(), "/", len(uni))
del grid, g3

# ================= 2. calendar features =================
uni["hour"] = uni.entry_time.dt.hour
uni["dow"] = uni.entry_time.dt.dayofweek  # 0=Mon

# ================= 3. alphabetical rank (symbol position in config.yaml symbols block) =================
import yaml
cfg = yaml.safe_load(open(f"{ROOT}/config.yaml"))
sym_keys = list(cfg["symbols"].keys())
enabled_keys = [k for k, v in cfg["symbols"].items() if isinstance(v, dict) and v.get("enabled", True)]
rank_map = {s: i for i, s in enumerate(enabled_keys)}
n_enabled = len(enabled_keys)
uni["alpha_rank"] = uni.symbol.map(rank_map)
uni["alpha_rank_frac"] = uni["alpha_rank"] / n_enabled
print("alpha_rank matched:", uni.alpha_rank.notna().sum(), "/", len(uni), "of", n_enabled, "enabled symbols")

# ================= 4. concurrency at entry (causal: trades opened before t, still open at t) =================
print("Computing concurrency...")
uni_sorted = uni.sort_values("entry_time").reset_index(drop=True)
entries = uni_sorted["entry_time"].values.astype("datetime64[ns]")
exits = uni_sorted["exit_time"].values.astype("datetime64[ns]")
n = len(uni_sorted)
# event-based sweep: at each entry, count trades with entry < t and exit > t
order_entry_idx = np.argsort(entries)
# Use a sweep-line: sort all entry/exit events
starts = entries
ends = exits
concurrency = np.zeros(n, dtype=int)
# sort trades by entry time (already sorted), use a heap of active exit times
import heapq
heap = []
sorted_idx = np.argsort(entries, kind="mergesort")
for pos, idx in enumerate(sorted_idx):
    t = entries[idx]
    # pop all trades whose exit <= t (no longer open at time t; entry counts as open at own entry, exit_time marks close)
    while heap and heap[0] <= t:
        heapq.heappop(heap)
    concurrency[idx] = len(heap)  # trades open BEFORE this one that are still open (excludes itself)
    heapq.heappush(heap, ends[idx])
uni_sorted["concurrency"] = concurrency
uni = uni_sorted
print("concurrency describe:\n", uni.concurrency.describe())

# ================= 5. EMA200 distance (1H) and 4H trend agreement, realized vol =================
print("Computing per-symbol EMA/vol features from cache_3yr_1h (this takes a while)...")
cache_dir = f"{ROOT}/cache_3yr_1h"

feat_rows = []
symbols_needed = uni.symbol.unique()
print(f"{len(symbols_needed)} symbols needed")

results = {}
for i, sym in enumerate(symbols_needed):
    path = f"{cache_dir}/{sym}.parquet"
    if not os.path.exists(path):
        continue
    df = pd.read_parquet(path)
    df = df.sort_values("start").reset_index(drop=True)
    df["ema200_1h"] = df["close"].ewm(span=200, adjust=False).mean()
    # 4H resample: only use CLOSED 4h bars
    df4 = df.set_index("start").resample("4h", label="left", closed="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last"}
    ).dropna()
    df4["ema200_4h"] = df4["close"].ewm(span=200, adjust=False).mean()
    df4["trend_4h_up"] = df4["close"] > df4["ema200_4h"]
    # realized vol: rolling 30d (720h) std of hourly log returns, causal (uses past only)
    df["logret"] = np.log(df["close"] / df["close"].shift(1))
    df["rvol_30d"] = df["logret"].rolling(720, min_periods=168).std()
    results[sym] = (df[["start", "ema200_1h", "rvol_30d", "close"]],
                     df4[["trend_4h_up"]].reset_index())
    if (i + 1) % 50 == 0:
        print(f"  {i+1}/{len(symbols_needed)}")

print("Merging EMA/vol features via merge_asof per symbol...")
out_frames = []
for sym, grp in uni.groupby("symbol"):
    if sym not in results:
        grp = grp.copy()
        grp["ema200_1h"] = np.nan
        grp["rvol_30d"] = np.nan
        grp["trend_4h_up"] = np.nan
        out_frames.append(grp)
        continue
    df1h, df4h = results[sym]
    grp = grp.sort_values("entry_time").copy()
    # use last CLOSED 1h bar strictly before entry_time (entry executes at open of new candle)
    lookup_time = grp["entry_time"] - pd.Timedelta(hours=1)
    tmp = grp[["entry_time"]].copy()
    tmp["lookup_time"] = lookup_time
    tmp = tmp.sort_values("lookup_time")
    m1 = pd.merge_asof(tmp, df1h.rename(columns={"start": "lookup_time"}), on="lookup_time", direction="backward")
    m1 = m1.sort_values("entry_time")
    grp = grp.sort_values("entry_time")
    grp["ema200_1h"] = m1["ema200_1h"].values
    grp["rvol_30d"] = m1["rvol_30d"].values
    px_at_lookup = m1["close"].values
    grp["ema_dist_1h"] = (px_at_lookup - grp["ema200_1h"].values) / grp["ema200_1h"].values

    tmp4 = grp[["entry_time"]].copy()
    tmp4["lookup_time"] = lookup_time.values
    tmp4 = tmp4.sort_values("lookup_time")
    m4 = pd.merge_asof(tmp4, df4h.rename(columns={"start": "lookup_time"}), on="lookup_time", direction="backward")
    m4 = m4.sort_values("entry_time")
    grp["trend_4h_up"] = m4["trend_4h_up"].values
    out_frames.append(grp)

uni = pd.concat(out_frames).sort_values("entry_time").reset_index(drop=True)
print("ema_dist_1h coverage:", uni.ema_dist_1h.notna().sum(), "/", len(uni))
print("trend_4h_up coverage:", uni.trend_4h_up.notna().sum(), "/", len(uni))
print("rvol_30d coverage:", uni.rvol_30d.notna().sum(), "/", len(uni))

# direction-aligned versions: for longs, being ABOVE ema / uptrend is "aligned"; for shorts, below/downtrend is aligned
uni["side_long"] = uni.side == "long"
uni["ema_dist_1h_aligned"] = np.where(uni.side_long, uni.ema_dist_1h, -uni.ema_dist_1h)
uni["trend_4h_aligned"] = np.where(uni.side_long, uni.trend_4h_up == True, uni.trend_4h_up == False)

# ================= 6. funding at entry =================
print("Merging funding rate at entry (trailing value known before entry)...")
fund_frames = []
for sym, grp in uni.groupby("symbol"):
    fpath = f"{ROOT}/funding_cache/{sym}.parquet"
    grp = grp.sort_values("entry_time").copy()
    if not os.path.exists(fpath):
        grp["funding_last"] = np.nan
        grp["funding_avg_7d"] = np.nan
        fund_frames.append(grp)
        continue
    fdf = pd.read_parquet(fpath).sort_values("time")
    fdf["funding_avg_7d"] = fdf["funding_rate"].rolling(21, min_periods=5).mean()  # ~7 days at 8h cadence
    tmp = grp[["entry_time"]].copy().sort_values("entry_time")
    m = pd.merge_asof(tmp, fdf[["time", "funding_rate", "funding_avg_7d"]].rename(columns={"time": "entry_time"}),
                       on="entry_time", direction="backward")
    grp["funding_last"] = m["funding_rate"].values
    grp["funding_avg_7d"] = m["funding_avg_7d"].values
    fund_frames.append(grp)

uni = pd.concat(fund_frames).sort_values("entry_time").reset_index(drop=True)
print("funding coverage:", uni.funding_last.notna().sum(), "/", len(uni))
# funding earned by a SHORT holder is +funding_rate (shorts receive when funding>0); LONG holder pays when funding>0
uni["funding_for_side"] = np.where(uni.side == "short", uni.funding_last, -uni.funding_last)
uni["funding_avg7d_for_side"] = np.where(uni.side == "short", uni.funding_avg_7d, -uni.funding_avg_7d)

out_path = f"{OUT}/trades_features.parquet"
uni.to_parquet(out_path)
print("Saved", out_path, uni.shape)
print(uni.dtypes)
