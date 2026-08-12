#!/usr/bin/env python3
"""AGENT-RED probe 1: block-bootstrap CI on HOLDOUT mean R and DEV-vs-HOLDOUT gap.

Naive per-trade t-tests on this universe are invalid: signals cluster in bursts
(a single market move fires divergence signals on many correlated alts within the
same hour/day), so trades are not i.i.d. draws. A weekly block bootstrap resamples
whole ISO weeks with replacement, preserving within-week correlation structure, and
gives an honest CI on the mean.

Read-only against risk_study/universe_chopBOS.parquet. Writes only to
risk_study/agent_out/.
"""
import numpy as np
import pandas as pd

np.random.seed(20260811)

HERE_UNI = "/Users/lualakol/AutoTrading Bot/risk_study/universe_chopBOS.parquet"
OUT = "/Users/lualakol/AutoTrading Bot/risk_study/agent_out"

PERIODS = {
    "DEV": ("2023-06-01", "2025-07-01"),
    "VAL": ("2025-07-01", "2026-05-25"),
    "HOLDOUT": ("2026-05-25", "2026-07-26"),
}
COST = 0.00341  # 34.1 bps, the study's headline figure


def net_r(d, cost):
    return d.r_result - cost / d.stop_frac


def block_bootstrap_mean(df, rcol, n_boot=20000, block_key="week"):
    """Resample whole blocks (weeks) with replacement; return array of bootstrap means."""
    blocks = df[block_key].unique()
    nblk = len(blocks)
    # pre-group for speed
    grouped = {b: df.loc[df[block_key] == b, rcol].to_numpy() for b in blocks}
    sizes = {b: len(v) for b, v in grouped.items()}
    means = np.empty(n_boot)
    blk_arr = np.array(blocks)
    for i in range(n_boot):
        chosen = np.random.choice(blk_arr, size=nblk, replace=True)
        vals = np.concatenate([grouped[b] for b in chosen])
        means[i] = vals.mean()
    return means


def main():
    d = pd.read_parquet(HERE_UNI)
    d["week"] = d.entry_time.dt.to_period("W").astype(str)
    d["net_r"] = net_r(d, COST)

    results = {}
    for name, (t0, t1) in PERIODS.items():
        sub = d[(d.entry_time >= t0) & (d.entry_time < t1)].copy()
        results[name] = sub
        nweeks = sub.week.nunique()
        print(f"{name}: n={len(sub):,}  weeks={nweeks}  "
              f"gross_mean={sub.r_result.mean():.4f}  net_mean={sub.net_r.mean():.4f}")

    print("\n" + "=" * 90)
    print("BLOCK BOOTSTRAP (weekly blocks, 20000 resamples) -- 95% CI on the MEAN")
    print("=" * 90)
    boot_means = {}
    for name, sub in results.items():
        for col, label in [("r_result", "gross"), ("net_r", "net@34.1bps")]:
            bm = block_bootstrap_mean(sub, col)
            boot_means[(name, label)] = bm
            lo, hi = np.percentile(bm, [2.5, 97.5])
            point = sub[col].mean()
            print(f"{name:<9}{label:<14} point={point:+.4f}   95% CI [{lo:+.4f}, {hi:+.4f}]   "
                  f"P(mean>0)={np.mean(bm > 0)*100:.1f}%")

    print("\n" + "=" * 90)
    print("DEV-minus-HOLDOUT and VAL-minus-HOLDOUT gap, block-bootstrapped independently per side")
    print("=" * 90)
    n_boot = 20000
    for base in ("DEV", "VAL"):
        for col, label in [("r_result", "gross"), ("net_r", "net@34.1bps")]:
            bm_base = block_bootstrap_mean(results[base], col, n_boot=n_boot)
            bm_hold = block_bootstrap_mean(results["HOLDOUT"], col, n_boot=n_boot)
            diff = bm_base - bm_hold
            lo, hi = np.percentile(diff, [2.5, 97.5])
            point = results[base][col].mean() - results["HOLDOUT"][col].mean()
            frac_le0 = np.mean(diff <= 0) * 100
            print(f"{base}-HOLDOUT  {label:<14} point={point:+.4f}   95% CI [{lo:+.4f}, {hi:+.4f}]   "
                  f"P(gap<=0)={frac_le0:.2f}%")

    # Does HOLDOUT's CI ever touch DEV/VAL's point estimate?
    print("\n" + "=" * 90)
    print("Does HOLDOUT's own CI upper bound reach DEV/VAL's point estimate?")
    print("=" * 90)
    for col, label in [("r_result", "gross"), ("net_r", "net@34.1bps")]:
        hold_bm = boot_means[("HOLDOUT", label)]
        hold_hi = np.percentile(hold_bm, 97.5)
        hold_hi99 = np.percentile(hold_bm, 99.5)
        dev_point = results["DEV"][col].mean()
        val_point = results["VAL"][col].mean()
        print(f"{label}: HOLDOUT 97.5%ile={hold_hi:+.4f}  99.5%ile={hold_hi99:+.4f}  "
              f"DEV point={dev_point:+.4f}  VAL point={val_point:+.4f}  "
              f"-> HOLDOUT CI reaches DEV point: {hold_hi >= dev_point}")

    # save summary csv
    rows = []
    for name, sub in results.items():
        for col, label in [("r_result", "gross"), ("net_r", "net@34.1bps")]:
            bm = boot_means[(name, label)]
            lo, hi = np.percentile(bm, [2.5, 97.5])
            rows.append(dict(period=name, cost=label, n=len(sub), weeks=sub.week.nunique(),
                              mean=sub[col].mean(), ci_lo=lo, ci_hi=hi,
                              p_mean_gt0=np.mean(bm > 0)))
    pd.DataFrame(rows).to_csv(f"{OUT}/bootstrap_ci.csv", index=False)
    print(f"\nsaved -> {OUT}/bootstrap_ci.csv")


if __name__ == "__main__":
    main()
