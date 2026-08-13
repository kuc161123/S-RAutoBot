#!/usr/bin/env python3
"""AUDIT-A item 1: CHOP threshold sweep at the CURRENT config (rr=10, atr_mult=3.0).

Pure pandas, no engine needed -- flat thresholds don't depend on portfolio state.
Population: grid_inuni.parquet restricted to the 728 (symbol,div_type) pairs that are
actually enabled in the live config (grid_inuni has 1,100 pairs; the live universe file
uni_glob_rr10_am3_same.parquet has the 728 that are walk-forward-selected AND enabled).

Cost: net R = r_10 - 0.00242/stop_frac.
Split: IS < 2026-05-25, holdout >= 2026-05-25. Trim last 21 days of entries (global,
off the dataset's true max entry_time, before any split) -- unresolved trades near the
end are silently dropped by the grid builder and slow winners resolve last.
CI: weekly-block bootstrap (resample by ISO week with replacement), 1000 draws.
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent.parent  # AutoTrading Bot/
RS = ROOT / "risk_study"

COST = 0.00242
IS_END = pd.Timestamp("2026-05-25")
THRESHOLDS = [None, 38, 42, 45, 48, 52, 56, 60, 65]

rng = np.random.default_rng(20260812)


def weekly_block_bootstrap(sub, n=1000):
    """Bootstrap CI on mean net R by resampling ISO weeks with replacement."""
    if len(sub) < 20:
        return np.nan, np.nan
    wk = sub.entry_time.dt.isocalendar().year.astype(str) + "-" + \
         sub.entry_time.dt.isocalendar().week.astype(str)
    groups = sub.groupby(wk)["net_r"].apply(lambda s: s.to_numpy())
    keys = groups.index.to_list()
    vals = groups.to_list()
    if len(keys) < 4:
        return np.nan, np.nan
    means = np.empty(n)
    k = len(keys)
    for i in range(n):
        idx = rng.integers(0, k, size=k)
        arr = np.concatenate([vals[j] for j in idx])
        means[i] = arr.mean()
    lo, hi = np.percentile(means, [2.5, 97.5])
    return lo, hi


def summarize(sub, label):
    rows = []
    n_all = len(sub)
    for th in THRESHOLDS:
        if th is None:
            ent = sub
            th_label = "none"
        else:
            ent = sub[sub.chop_bos < th]
            th_label = th
        n = len(ent)
        if n == 0:
            continue
        block_rate = 1 - n / n_all if n_all else np.nan
        avg_r = ent.net_r.mean()
        sum_r = ent.net_r.sum()
        wr = (ent.r_10 > 0).mean()
        lo, hi = weekly_block_bootstrap(ent)
        rows.append(dict(window=label, threshold=th_label, n=n, n_all=n_all,
                          block_rate=block_rate, avg_net_r=avg_r, sum_net_r=sum_r,
                          wr=wr, ci_lo=lo, ci_hi=hi))
    return rows


def main():
    grid = pd.read_parquet(RS / "grid_inuni.parquet")
    uni = pd.read_parquet(RS / "uni_glob_rr10_am3_same.parquet")
    pairs = set(zip(uni.symbol, uni.div_type))

    sub = grid[grid.atr_mult == 3.0].copy()
    sub = sub[[(s, t) in pairs for s, t in zip(sub.symbol, sub.div_type)]]
    sub = sub.dropna(subset=["r_10", "exit_10", "chop_bos", "stop_frac"])
    sub["net_r"] = sub.r_10 - COST / sub.stop_frac

    max_entry = sub.entry_time.max()
    cutoff = max_entry - pd.Timedelta(days=21)
    print(f"rows before trim: {len(sub)}, max_entry={max_entry}, cutoff={cutoff}")
    sub = sub[sub.entry_time <= cutoff].reset_index(drop=True)
    print(f"rows after 21d trim: {len(sub)}")

    all_rows = []
    all_rows += summarize(sub, "FULL")
    all_rows += summarize(sub[sub.entry_time < IS_END], "IS")
    all_rows += summarize(sub[sub.entry_time >= IS_END], "HOLDOUT")

    # rolling 6-month windows
    start = sub.entry_time.min().normalize().replace(day=1)
    end = sub.entry_time.max()
    edges = pd.date_range(start, end + pd.Timedelta(days=180), freq="6MS")
    win_rows = []
    for i in range(len(edges) - 1):
        w0, w1 = edges[i], edges[i + 1]
        wsub = sub[(sub.entry_time >= w0) & (sub.entry_time < w1)]
        if len(wsub) < 200:
            continue
        lbl = f"{w0.date()}..{w1.date()}"
        all_rows += summarize(wsub, lbl)
        win_rows.append(lbl)

    df = pd.DataFrame(all_rows)
    df.to_csv(HERE / "item1_chop_sweep_results.csv", index=False)

    print("\n=== IS vs HOLDOUT summary ===")
    piv = df[df.window.isin(["IS", "HOLDOUT"])]
    print(piv.to_string(index=False, float_format=lambda x: f"{x:10.4f}"))

    # rolling window win count: for each threshold, how many of the 6mo windows beat
    # 'none' (no gate) and how many beat the live '52' threshold, on avg_net_r
    print(f"\n=== rolling-window record (n windows = {len(win_rows)}) ===")
    wdf = df[df.window.isin(win_rows)]
    piv2 = wdf.pivot(index="window", columns="threshold", values="avg_net_r")
    print(piv2.to_string(float_format=lambda x: f"{x:8.4f}"))
    base_col = 52
    none_col = "none"
    print("\nwins vs NO-GATE (avg_net_r higher) / wins vs live-52 / n windows:")
    for th in THRESHOLDS:
        col = "none" if th is None else th
        if col not in piv2.columns:
            continue
        beat_none = (piv2[col] > piv2[none_col]).sum()
        beat_52 = (piv2[col] > piv2[base_col]).sum() if base_col in piv2.columns else np.nan
        print(f"  threshold={col!s:>6}  beat_no_gate={beat_none}/{len(piv2)}  "
              f"beat_52={beat_52}/{len(piv2)}")

    df.to_csv(HERE / "item1_chop_sweep_results.csv", index=False)
    print("\nwrote item1_chop_sweep_results.csv")


if __name__ == "__main__":
    main()
