#!/usr/bin/env python3
"""AGENT-EXIT / LEAD 1 analysis — scale-out (4 equal TPs, breakeven after TP1) vs the
live s3_a1 trail vs the plain fixed-TP base, at the R level, split at 2026-05-25.

Reads scaleout_universe.parquet (root of repo), which is ALREADY on the fixed CHOP gate
(reads ch[bos], not ch[e] -- verified by inspecting build_scaleout_universe.py lines
256-260, which contains an explicit comment citing the lookahead fix). Uses the
per-symbol LIVE-FITTED (rr, atr_mult) pairs from config.yaml -- NOT the global
rr=10/atr_mult=3 arm used for LEAD 2. Entries/signals/CHOP gate/EMA gate are identical
across all four arms scored below (base fixed-TP, live s3_a1 trail, scale-out+BE,
scale-out no-BE control) because build_scaleout_universe.py resolves all four in one
walk over the same bars.

Cost model, per the task spec: net R = r - 0.00242/stop_frac, applied flat to the whole
trade for every arm. This is CONSERVATIVE for scale-out specifically: three of its four
exits are resting limit orders that in reality pay no slippage (only the stopped
remainder is a market fill) -- see the note atop the repo's own report_scaleout.py. A
flat per-trade charge therefore under-states scale-out's edge, not over-states it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

UNI = "/Users/lualakol/AutoTrading Bot/scaleout_universe.parquet"
SPLIT = pd.Timestamp("2026-05-25")
COST = 0.00242
LEVELS = np.array([0.25, 0.50, 0.75, 1.00])  # fraction of rr_cfg, matches build script


def block_bootstrap_ci(entry_time, net_r, n_boot=2000, seed=0):
    if len(net_r) < 10:
        return (np.nan, np.nan)
    week = entry_time.dt.to_period("W").astype(str)
    gb = pd.DataFrame({"week": week.values, "r": np.asarray(net_r)}).groupby("week")["r"] \
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
    top = net_r.sort_values(ascending=False).head(k).sum()
    return top / total if total != 0 else np.nan


def scaleout_r(row, be):
    n_hit = row.n_tp_hit if be else row.n_tp_hit_nobe
    n_hit = int(n_hit)
    rr = row.rr_cfg
    booked = (LEVELS[:n_hit] * rr).sum() * 0.25
    remain = 1.0 - 0.25 * n_hit
    if be:
        final_r = row.so_stop_r  # 0.0 if BE stop caught it, -1.0 if original stop caught it
    else:
        # no-BE control: stop (if hit) is always the ORIGINAL stop => -1.0. If all 4
        # TPs filled first, remain is 0 so this value is never used.
        final_r = -1.0
    return booked + remain * final_r


def base_r(row):
    if pd.notna(row.base_stop_t):
        return -1.0
    if pd.notna(row.base_tp_t):
        return row.rr_cfg
    return np.nan


def trail_r(row):
    if pd.notna(row.tr_stop_t):
        return row.tr_stop_r
    if pd.notna(row.tr_tp_t):
        return row.rr_cfg
    return np.nan


def main():
    d = pd.read_parquet(UNI)
    print(f"Loaded {len(d):,} signals, {d.symbol.nunique()} symbols, "
          f"{d.entry_time.min()} .. {d.entry_time.max()}")
    print(f"rr/atr_mult combos: {sorted(d[['rr_cfg','atr_mult']].drop_duplicates().apply(tuple, axis=1))}")

    d["stop_frac"] = d.sl_dist / d.entry_price
    d["r_base"] = d.apply(base_r, axis=1)
    d["r_trail"] = d.apply(trail_r, axis=1)
    d["r_scaleout_be"] = d.apply(lambda r: scaleout_r(r, be=True), axis=1)
    d["r_scaleout_nobe"] = d.apply(lambda r: scaleout_r(r, be=False), axis=1)

    n_unresolved = d[["r_base", "r_trail"]].isna().any(axis=1).sum()
    print(f"Unresolved (dropped) rows: {n_unresolved}")
    d = d.dropna(subset=["r_base", "r_trail", "r_scaleout_be", "r_scaleout_nobe"]).reset_index(drop=True)

    d["net_cost"] = COST / d.stop_frac
    for arm in ["base", "trail", "scaleout_be", "scaleout_nobe"]:
        d[f"net_{arm}"] = d[f"r_{arm}"] - d.net_cost

    ins = d[d.entry_time < SPLIT].copy()
    oos = d[d.entry_time >= SPLIT].copy()
    print(f"\nSplit at {SPLIT.date()}: in-sample {len(ins):,} trades, "
          f"holdout {len(oos):,} trades")
    if len(oos) < 200:
        print("WARNING: holdout sample is thin (< 200 trades) -- CIs will be wide, "
              "treat point estimates cautiously.")

    rows = []
    for arm in ["base", "trail", "scaleout_be", "scaleout_nobe"]:
        ins_net = ins[f"net_{arm}"]
        oos_net = oos[f"net_{arm}"]
        ins_lo, ins_hi = block_bootstrap_ci(ins.entry_time, ins_net)
        oos_lo, oos_hi = block_bootstrap_ci(oos.entry_time, oos_net)
        rows.append(dict(
            arm=arm,
            n_ins=len(ins_net), mean_ins=ins_net.mean(), ins_ci_lo=ins_lo, ins_ci_hi=ins_hi,
            wr_ins=(ins_net > 0).mean(), top1pct_ins=top1pct_share(ins_net),
            n_oos=len(oos_net), mean_oos=oos_net.mean(), oos_ci_lo=oos_lo, oos_ci_hi=oos_hi,
            wr_oos=(oos_net > 0).mean(), top1pct_oos=top1pct_share(oos_net),
        ))
    res = pd.DataFrame(rows)
    res.to_csv("scaleout_arm_results.csv", index=False)
    print("\n=== ARM COMPARISON (net of cost, R/trade) ===")
    with pd.option_context("display.width", 200):
        print(res.round(4).to_string(index=False))

    # gross (no cost) for comparison to the config.yaml-quoted +0.367 vs +0.208 claim
    print("\n=== GROSS (no cost) mean R, full sample, for reference to the config.yaml claim ===")
    for arm in ["base", "trail", "scaleout_be", "scaleout_nobe"]:
        print(f"  {arm:14s}  gross mean R = {d[f'r_{arm}'].mean():+.4f}  (n={len(d):,})")

    d.to_parquet("scaleout_arm_trades.parquet", index=False)


if __name__ == "__main__":
    main()
