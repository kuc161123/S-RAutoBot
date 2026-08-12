#!/usr/bin/env python3
"""AGENT-EXIT / LEAD 2 analysis — split-sample evaluation of the trigger_r x trail_atr
sweep for global rr=10/atr_mult=3.0 (same-pairs). Split at 2026-05-25: fit/explore
strictly BEFORE it, judge strictly ON/AFTER it (untouched holdout).

For every one of the 24 cells: mean net R/trade in both periods, weekly block
bootstrap 95% CI on the net R mean, top-1%-of-trades profit concentration, trade count.
Then the top-5 in-sample cells are printed against their holdout numbers so
overfitting is visible at a glance, and the benchmark cell (trigger=3, trail=1 --
today's live constants) is printed as a fixed reference row in both periods.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

HERE_UNI = "trigger_sweep_universe.parquet"
SPLIT = pd.Timestamp("2026-05-25")
COST = 0.00242

TRIGGERS = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0]
TRAILS = [0.5, 1.0, 1.5, 2.0]


def colname(t, a):
    return f"r_{t:g}_{a:g}"


def block_bootstrap_ci(entry_time, net_r, n_boot=2000, seed=0):
    """Weekly block bootstrap on the mean. Returns (lo, hi) 95% CI."""
    if len(net_r) < 10:
        return (np.nan, np.nan)
    week = entry_time.dt.to_period("W").astype(str)
    groups = pd.Series(net_r.values, index=week.values)
    weeks = groups.index.unique()
    by_week = {w: groups.loc[[w]].values if isinstance(groups.loc[w], pd.Series)
               else np.array([groups.loc[w]]) for w in weeks}
    # more robust groupby
    gb = pd.DataFrame({"week": week.values, "r": net_r.values}).groupby("week")["r"] \
        .apply(lambda s: s.values).to_dict()
    wk_list = list(gb.keys())
    nwk = len(wk_list)
    if nwk < 5:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot)
    for b in range(n_boot):
        sample_weeks = rng.choice(wk_list, size=nwk, replace=True)
        vals = np.concatenate([gb[w] for w in sample_weeks])
        means[b] = vals.mean()
    return (np.percentile(means, 2.5), np.percentile(means, 97.5))


def top1pct_share(net_r):
    n = len(net_r)
    if n < 100:
        return np.nan
    k = max(1, int(round(n * 0.01)))
    total = net_r.sum()
    if total == 0:
        return np.nan
    top = net_r.sort_values(ascending=False).head(k).sum()
    return top / net_r.sum() if net_r.sum() != 0 else np.nan


def main():
    d = pd.read_parquet(HERE_UNI)
    d["net_cost"] = COST / d.stop_frac
    ins = d[d.entry_time < SPLIT].copy()
    oos = d[d.entry_time >= SPLIT].copy()
    print(f"Total signals: {len(d):,}  |  in-sample (<{SPLIT.date()}): {len(ins):,}  "
          f"|  holdout (>={SPLIT.date()}): {len(oos):,}")

    rows = []
    for t in TRIGGERS:
        for a in TRAILS:
            col = colname(t, a)
            ins_net = ins[col] - ins.net_cost
            oos_net = oos[col] - oos.net_cost
            ins_lo, ins_hi = block_bootstrap_ci(ins.entry_time, ins_net)
            oos_lo, oos_hi = block_bootstrap_ci(oos.entry_time, oos_net)
            rows.append(dict(
                trigger_r=t, trail_atr=a,
                n_ins=len(ins_net), mean_ins=ins_net.mean(),
                ins_ci_lo=ins_lo, ins_ci_hi=ins_hi,
                wr_ins=(ins_net > 0).mean(),
                top1pct_ins=top1pct_share(ins_net),
                n_oos=len(oos_net), mean_oos=oos_net.mean(),
                oos_ci_lo=oos_lo, oos_ci_hi=oos_hi,
                wr_oos=(oos_net > 0).mean(),
                top1pct_oos=top1pct_share(oos_net),
            ))
    res = pd.DataFrame(rows).sort_values("mean_ins", ascending=False).reset_index(drop=True)
    res.to_csv("trigger_sweep_results.csv", index=False)

    print("\n=== ALL 24 CELLS, ranked by in-sample mean net R/trade ===")
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(res[["trigger_r", "trail_atr", "n_ins", "mean_ins", "ins_ci_lo", "ins_ci_hi",
                    "n_oos", "mean_oos", "oos_ci_lo", "oos_ci_hi", "top1pct_ins", "top1pct_oos"]]
              .round(4).to_string(index=False))

    bench = res[(res.trigger_r == 3.0) & (res.trail_atr == 1.0)]
    print("\n=== BENCHMARK CELL (trigger=3.0, trail=1.0 -- current live constants) ===")
    print(bench.round(4).to_string(index=False))

    print("\n=== TOP 5 BY IN-SAMPLE MEAN, WITH HOLDOUT ===")
    top5 = res.head(5)
    print(top5[["trigger_r", "trail_atr", "mean_ins", "ins_ci_lo", "ins_ci_hi",
                "mean_oos", "oos_ci_lo", "oos_ci_hi"]].round(4).to_string(index=False))

    bench_oos = bench.mean_oos.iloc[0]
    beats_both = res[(res.mean_ins > bench.mean_ins.iloc[0]) & (res.mean_oos > bench_oos)]
    print(f"\nCells beating the benchmark cell (3.0/1.0) in BOTH periods: {len(beats_both)}")
    if len(beats_both):
        print(beats_both[["trigger_r", "trail_atr", "mean_ins", "mean_oos"]]
              .round(4).to_string(index=False))


if __name__ == "__main__":
    main()
