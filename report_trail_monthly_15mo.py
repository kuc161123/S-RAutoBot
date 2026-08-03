#!/usr/bin/env python3
"""Base vs trailing, month by month, 15 months from $1,500 — 5m-resolved.

Uses the paired protocol rather than a single run. Intra-hour processing order moves a
single result by 10-15% (verify_ordering_sensitivity.py), so one run's month-by-month
path is not a fact about the strategies — it is one draw. This runs both arms under the
SAME ordering across many seeds, then shows the path of the seed whose base-vs-trailing
gap is the MEDIAN, and reports the spread around it.

Balances are cumulative realised P&L from $1,500, sorted by exit time. Drawdown is
running peak-to-trough on that curve, so the figure printed against a month is the worst
the account had been down at any point up to the end of that month.
"""
from __future__ import annotations

import multiprocessing as mp
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

START = 1500.0
BASE_RT = 2 * (0.0006 + 0.0003)
END = pd.Timestamp("2026-07-25")
W15 = END - pd.DateOffset(months=15)
UNI = ROOT / "trail_5m_universe.parquet"
ARMS = [("base", "fixed TP (original)"), ("s3_a1", "trailing s3_a1")]
SEEDS = list(range(13))

_P = None
_CHOP = None
_D = None


def _init():
    global _P, _CHOP, _D
    import backtest_production_correct as P
    from backtest_shadow_gate import LIVE as LIVE_KW
    _P = (P, LIVE_KW)
    d = pd.read_parquet(UNI)
    d["entry_time"] = pd.to_datetime(d["entry_time"])
    _D = d[(d.entry_time >= W15) & (d.entry_time < END)].reset_index(drop=True)
    _CHOP = P.load_chop_data(sorted(_D.symbol.unique()))


def run(args):
    nm, seed = args
    P, LIVE_KW = _P
    d = _D
    out = d[["entry_time", "entry_price", "sl_price", "side", "symbol",
             "btc_bull", "btc_impulse"]].copy()
    out["exit_time"] = pd.to_datetime(d[f"x_{nm}"])
    out["r_result"] = d[f"r_{nm}"]
    out = out.dropna(subset=["exit_time", "r_result"])
    out = out.sample(frac=1.0, random_state=seed)
    out = out.sort_values("entry_time", kind="mergesort").reset_index(drop=True)
    P.ROUND_TRIP_COST = BASE_RT
    P.STARTING_BALANCE = START
    kw = dict(LIVE_KW, starting_balance=START,
              btc_bull_col="btc_bull", btc_short_col="btc_impulse")
    r = P.run_simulation(out, _CHOP, **kw)
    t = r["entered_trades"]
    if not t:
        return nm, seed, None

    ex = pd.DataFrame({"exit_time": pd.to_datetime([x["exit_time"] for x in t]),
                       "pnl": [x["pnl"] for x in t]}).sort_values("exit_time")
    ex["bal"] = START + ex["pnl"].cumsum()
    ex["peak"] = ex["bal"].cummax()
    ex["dd"] = (ex["peak"] - ex["bal"]) / ex["peak"] * 100
    ex["m"] = ex["exit_time"].dt.to_period("M").astype(str)
    g = ex.groupby("m").agg(pnl=("pnl", "sum"), bal=("bal", "last"),
                            dd=("dd", "max")).reset_index()
    g["ddrun"] = g["dd"].cummax()
    return nm, seed, {"final": float(ex["bal"].iloc[-1]), "maxdd": float(ex["dd"].max()),
                      "n": len(t), "months": g}


def main():
    jobs = [(nm, s) for nm, _ in ARMS for s in SEEDS]
    res = {}
    with mp.Pool(max(1, mp.cpu_count() - 1), initializer=_init) as pool:
        for nm, s, r in pool.imap_unordered(run, jobs):
            if r:
                res[(nm, s)] = r

    diffs = {s: res[("s3_a1", s)]["final"] - res[("base", s)]["final"]
             for s in SEEDS if ("s3_a1", s) in res and ("base", s) in res}
    med_seed = sorted(diffs, key=lambda s: diffs[s])[len(diffs) // 2]
    b, t = res[("base", med_seed)], res[("s3_a1", med_seed)]

    print("=" * 96)
    print(f"MONTH BY MONTH — 15 months from ${START:,.0f}  ({W15:%Y-%m-%d} → {END:%Y-%m-%d})")
    print("5m-resolved · live config, live protections · median-gap ordering "
          f"(seed {med_seed} of {len(diffs)})")
    print("=" * 96)
    print(f"  {'month':<9}{'BASE P&L':>12}{'BASE bal':>11}{'BASE DD':>9}"
          f"{'TRAIL P&L':>12}{'TRAIL bal':>11}{'TRAIL DD':>10}{'gap':>11}")
    months = sorted(set(b["months"].m) | set(t["months"].m))
    bm = b["months"].set_index("m")
    tm = t["months"].set_index("m")
    for m in months:
        br = bm.loc[m] if m in bm.index else None
        tr = tm.loc[m] if m in tm.index else None
        bp = br.pnl if br is not None else 0.0
        bb = br.bal if br is not None else float("nan")
        bd = br.ddrun if br is not None else float("nan")
        tp = tr.pnl if tr is not None else 0.0
        tb = tr.bal if tr is not None else float("nan")
        td = tr.ddrun if tr is not None else float("nan")
        print(f"  {m:<9}{bp:>+12,.0f}{bb:>11,.0f}{bd:>8.1f}%"
              f"{tp:>+12,.0f}{tb:>11,.0f}{td:>9.1f}%{tb - bb:>+11,.0f}")

    print("\n" + "=" * 96)
    print("TOTALS AFTER 15 MONTHS")
    print("=" * 96)
    print(f"  {'arm':<24}{'final $':>12}{'profit $':>13}{'ROI':>10}"
          f"{'max DD':>9}{'ROI/DD':>9}{'trades':>9}")
    for nm, label in ARMS:
        r = res[(nm, med_seed)]
        roi = r["final"] / START - 1
        print(f"  {label:<24}{r['final']:>12,.0f}{r['final'] - START:>+13,.0f}"
              f"{roi:>10.0%}{r['maxdd']:>8.1f}%"
              f"{roi * 100 / r['maxdd']:>9.1f}{r['n']:>9,}")
    print(f"  {'never trade':<24}{START:>12,.0f}{0:>+13,.0f}{0.0:>10.0%}"
          f"{0.0:>8.1f}%{0.0:>9}{0:>9}")
    print(f"\n  trailing advantage: {t['final'] - b['final']:+,.0f} "
          f"({t['final'] / b['final'] - 1:+.0%})   "
          f"drawdown {t['maxdd'] - b['maxdd']:+.1f}pp")

    print("\n" + "=" * 96)
    print(f"HOW MUCH OF THAT IS THE ORDERING DRAW? — {len(diffs)} paired runs")
    print("=" * 96)
    print(f"  {'arm':<24}{'min $':>12}{'median $':>12}{'max $':>12}"
          f"{'min DD':>9}{'max DD':>9}")
    for nm, label in ARMS:
        f = [res[(nm, s)]["final"] for s in diffs]
        dd = [res[(nm, s)]["maxdd"] for s in diffs]
        print(f"  {label:<24}{min(f):>12,.0f}{np.median(f):>12,.0f}{max(f):>12,.0f}"
              f"{min(dd):>8.1f}%{max(dd):>8.1f}%")
    dv = list(diffs.values())
    print(f"\n  trailing beat base in {sum(1 for x in dv if x > 0)}/{len(dv)} paired runs")
    print(f"  gap: median {np.median(dv):+,.0f}   range {min(dv):+,.0f} .. {max(dv):+,.0f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
