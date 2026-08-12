#!/usr/bin/env python3
"""Analytic anchor: the growth-optimal risk fraction, computed WITHOUT the simulator.

The sweep answers "what happened". This answers "what had to happen", from the net-R
distribution alone, so the two can be cross-checked. If the simulator's optimum and the
Kelly optimum disagree badly, one of them is wrong.

Two quantities:
  f*_single   maximises E[log(1 + f*R)] treating trades as sequential. This is the number
              people quote, and for this bot it is MISLEADING -- it ignores that ~45
              positions are open simultaneously, so many bets are settled against the same
              equity before any of them compounds.
  f*_conc     the same, but on the distribution of CONCURRENT-CLUSTER outcomes: aggregate
              the net R of all trades that were open at the same time, so correlated
              stop-outs are counted as the single joint bet they actually are.

Also reports the breakeven cost: the round-trip bps at which mean net R crosses zero.
Below that the strategy has an edge to size; above it, every f > 0 loses money and the
whole optimisation is moot.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PERIODS = {
    "DEV": ("2023-06-01", "2025-07-01"),
    "VAL": ("2025-07-01", "2026-05-25"),
    "HOLDOUT": ("2026-05-25", "2026-07-26"),
    "FULL": ("2023-06-01", "2026-07-26"),
}
COSTS = [0.0011, 0.0018, 0.0025, 0.00341, 0.0045]


def net_r(d, cost):
    return d.r_result - cost / d.stop_frac


def kelly_f(r, lo=1e-5, hi=0.5):
    """Argmax of E[log(1+f R)] by golden section. Returns 0 if E[R] <= 0."""
    r = np.asarray(r, dtype=float)
    if r.mean() <= 0:
        return 0.0
    # f must keep 1 + f*min(R) > 0
    worst = r.min()
    cap = (0.999 / -worst) if worst < 0 else hi
    hi = min(hi, cap)

    def g(f):
        v = 1.0 + f * r
        if np.any(v <= 0):
            return -np.inf
        return np.mean(np.log(v))

    gr = (np.sqrt(5) - 1) / 2
    a, b = lo, hi
    for _ in range(200):
        c, dd = b - gr * (b - a), a + gr * (b - a)
        if g(c) < g(dd):
            a = c
        else:
            b = dd
    return (a + b) / 2


def cluster_returns(d, cost, hours=24):
    """Aggregate net R over non-overlapping calendar buckets of `hours`.

    Trades are assigned to the bucket of their EXIT. A bucket's total R is the sum of the
    net R of everything that settled in it -- which is what actually hits equity at once.
    """
    x = d.copy()
    x["nr"] = net_r(x, cost)
    b = x.set_index("exit_time")["nr"].resample(f"{hours}h").sum()
    return b[b.index.isin(x.exit_time.dt.floor(f"{hours}h").unique())].to_numpy()


def main():
    d = pd.read_parquet(HERE / "universe_chopBOS.parquet")
    print(f"universe {len(d):,} trades  {d.entry_time.min().date()}..{d.entry_time.max().date()}\n")

    # ── breakeven cost per period ───────────────────────────────────────────────
    print("=" * 92)
    print("MEAN NET R PER TRADE vs ROUND-TRIP COST   (positive = an edge exists to size)")
    print("=" * 92)
    hdr = f"{'period':<9}{'n':>7}  " + "".join(f"{c*1e4:>9.1f}bps" for c in COSTS) + f"{'breakeven':>12}"
    print(hdr)
    rows = []
    for name, (t0, t1) in PERIODS.items():
        s = d[(d.entry_time >= t0) & (d.entry_time < t1)]
        line = f"{name:<9}{len(s):>7,}  "
        for c in COSTS:
            line += f"{net_r(s, c).mean():>12.4f}"
        # breakeven: mean(r) = cost * mean(1/stop_frac)  ->  cost = mean(r)/mean(1/sf)
        be = s.r_result.mean() / (1.0 / s.stop_frac).mean()
        line += f"{be*1e4:>12.1f}"
        print(line)
        rows.append(dict(period=name, n=len(s), breakeven_bps=be * 1e4,
                         gross_r=s.r_result.mean(),
                         **{f"net_r_{int(c*1e4*10)/10}bps": net_r(s, c).mean() for c in COSTS}))

    # ── growth-optimal f ───────────────────────────────────────────────────────
    print()
    print("=" * 92)
    print("GROWTH-OPTIMAL RISK FRACTION  (Kelly, on NET R)   cost = 34.1 bps")
    print("=" * 92)
    print(f"{'period':<9}{'mean netR':>11}{'sd':>8}{'f*_single':>11}{'f*_24h_cluster':>16}"
          f"{'f*/4 (quarter)':>16}")
    krows = []
    for name, (t0, t1) in PERIODS.items():
        s = d[(d.entry_time >= t0) & (d.entry_time < t1)]
        r = net_r(s, 0.00341).to_numpy()
        fs = kelly_f(r)
        cl = cluster_returns(s, 0.00341, 24)
        fc = kelly_f(cl) if len(cl) > 30 else np.nan
        print(f"{name:<9}{r.mean():>11.4f}{r.std():>8.2f}{fs*100:>10.2f}%"
              f"{(fc*100 if fc == fc else float('nan')):>15.3f}%{(fc*100/4 if fc == fc else float('nan')):>15.3f}%")
        krows.append(dict(period=name, mean_net_r=r.mean(), sd_net_r=r.std(),
                          kelly_single_pct=fs * 100,
                          kelly_cluster24h_pct=fc * 100 if fc == fc else np.nan))

    pd.DataFrame(rows).to_csv(HERE / "results" / "breakeven_cost.csv", index=False)
    pd.DataFrame(krows).to_csv(HERE / "results" / "kelly.csv", index=False)

    # ── how much of the edge is a handful of trades ─────────────────────────────
    print()
    print("=" * 92)
    print("CONCENTRATION — profit share of the top trades (net R @ 34.1 bps)")
    print("=" * 92)
    print(f"{'period':<9}{'top 1%':>10}{'top 5%':>10}{'top 10%':>10}   (share of total POSITIVE net R)")
    for name, (t0, t1) in PERIODS.items():
        s = d[(d.entry_time >= t0) & (d.entry_time < t1)]
        r = np.sort(net_r(s, 0.00341).to_numpy())[::-1]
        tot = r.sum()
        if tot <= 0:
            print(f"{name:<9}{'—':>10}{'—':>10}{'—':>10}   total net R is negative ({tot:,.0f})")
            continue
        out = f"{name:<9}"
        for q in (0.01, 0.05, 0.10):
            k = max(1, int(len(r) * q))
            out += f"{r[:k].sum()/tot*100:>9.0f}%"
        print(out + f"   total {tot:,.0f} R")


if __name__ == "__main__":
    main()
