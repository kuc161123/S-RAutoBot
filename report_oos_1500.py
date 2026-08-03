#!/usr/bin/env python3
"""$1,500 through the CLEAN out-of-sample window only. 5m-resolved, paired.

Window: 2026-05-25 → 2026-07-25. This is the only slice the config's walk-forward fit
never touched, so it is the one place these numbers mean what they appear to mean —
and the one place the strategy has no built-in advantage.

Paired across orderings, because a single run moves 10-15% on intra-hour processing
order alone and that is larger than most of the differences here.
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
OOS = pd.Timestamp("2026-05-25")
END = pd.Timestamp("2026-07-25")
UNI = ROOT / "trail_5m_universe.parquet"
ARMS = [("base", "fixed TP (original)"), ("s3_a1", "trailing s3_a1 (live)"),
        ("s2_a1", "trailing s2_a1")]
SEEDS = list(range(11))

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
    _D = d[(d.entry_time >= OOS) & (d.entry_time < END)].reset_index(drop=True)
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
    wk = ex.set_index("exit_time").resample("W")
    weekly = wk.agg(pnl=("pnl", "sum"), bal=("bal", "last"), dd=("dd", "max")).dropna()
    gp = sum(x["pnl"] for x in t if x["pnl"] > 0)
    gl = abs(sum(x["pnl"] for x in t if x["pnl"] < 0))
    wins = sum(1 for x in t if x["pnl"] > 0)
    return nm, seed, dict(final=float(ex["bal"].iloc[-1]), maxdd=float(ex["dd"].max()),
                          low=float(ex["bal"].min()), n=len(t), wr=wins / len(t),
                          pf=(gp / gl) if gl else float("inf"), weekly=weekly)


def main():
    jobs = [(nm, s) for nm, _ in ARMS for s in SEEDS]
    res = {}
    with mp.Pool(max(1, mp.cpu_count() - 1), initializer=_init) as pool:
        for nm, s, r in pool.imap_unordered(run, jobs):
            if r:
                res[(nm, s)] = r

    print("=" * 100)
    print(f"CLEAN OUT-OF-SAMPLE — ${START:,.0f} start, {OOS:%Y-%m-%d} → {END:%Y-%m-%d} "
          f"(2 months)")
    print(f"5m-resolved · live config + all protections · {len(SEEDS)} paired orderings")
    print("=" * 100)
    print(f"  {'arm':<24}{'median $':>11}{'P&L':>10}{'ROI':>8}"
          f"{'range $':>20}{'maxDD':>8}{'worst bal':>11}{'PF':>7}{'WR':>7}")
    med = {}
    for nm, label in ARMS:
        rs = [res[(nm, s)] for s in SEEDS if (nm, s) in res]
        f = [x["final"] for x in rs]
        m = float(np.median(f))
        med[nm] = m
        print(f"  {label:<24}{m:>11,.0f}{m - START:>+10,.0f}{m / START - 1:>8.0%}"
              f"{min(f):>10,.0f} ..{max(f):>8,.0f}"
              f"{np.median([x['maxdd'] for x in rs]):>7.1f}%"
              f"{np.median([x['low'] for x in rs]):>11,.0f}"
              f"{np.median([x['pf'] for x in rs]):>7.2f}"
              f"{np.median([x['wr'] for x in rs]):>7.1%}")
    print(f"  {'NEVER TRADE':<24}{START:>11,.0f}{0:>+10,.0f}{0.0:>8.0%}"
          f"{'—':>20}{0.0:>7.1f}%{START:>11,.0f}{'—':>7}{'—':>7}")

    # week-by-week for the median-final ordering of each arm
    print("\n" + "=" * 100)
    print("WEEK BY WEEK — balance path (median-final ordering per arm)")
    print("=" * 100)
    paths = {}
    for nm, _ in ARMS:
        rs = sorted(((res[(nm, s)]["final"], s) for s in SEEDS if (nm, s) in res))
        paths[nm] = res[(nm, rs[len(rs) // 2][1])]["weekly"]
    weeks = sorted({w for p in paths.values() for w in p.index})
    hdr = f"  {'week ending':<14}"
    for _, label in ARMS:
        hdr += f"{label.split(' (')[0][:16]:>18}"
    print(hdr)
    for w in weeks:
        row = f"  {w:%Y-%m-%d}    "
        for nm, _ in ARMS:
            p = paths[nm]
            row += f"{p.loc[w, 'bal']:>18,.0f}" if w in p.index else f"{'—':>18}"
        print(row)

    print("\n" + "=" * 100)
    print("HEAD TO HEAD (same ordering both arms)")
    print("=" * 100)
    for nm, label in ARMS[1:]:
        d = [res[(nm, s)]["final"] - res[("base", s)]["final"]
             for s in SEEDS if (nm, s) in res and ("base", s) in res]
        w = sum(1 for x in d if x > 0)
        print(f"  {label:<24} beat fixed TP in {w}/{len(d)} orderings · "
              f"median {np.median(d):+,.0f} · range {min(d):+,.0f} .. {max(d):+,.0f}")
    for nm, label in ARMS:
        d = [res[(nm, s)]["final"] - START for s in SEEDS if (nm, s) in res]
        w = sum(1 for x in d if x > 0)
        print(f"  {label:<24} beat NOT TRADING in {w}/{len(d)} orderings")
    return 0


if __name__ == "__main__":
    sys.exit(main())
