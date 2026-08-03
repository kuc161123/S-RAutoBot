#!/usr/bin/env python3
"""Should each protection stay on? Leave-one-out ablation, 5m-resolved, paired.

Every row removes exactly ONE protection from the live stack and changes nothing else,
so the difference is attributable to that protection alone. ALL OFF removes everything
at once, which is not the sum of the individual rows — they interact.

Things that make this comparison honest and are easy to get wrong:

  * CHOP is filtered at BUILD time. run_simulation's own CHOP scenarios cannot undo a
    filter that already removed the signals from the universe, so the "-CHOP" rows read
    a separately built no-CHOP universe (49,103 signals vs 25,507 — CHOP removes 48%).
  * The balance taper and the regime MULTIPLIER are separate knobs. Passing
    custom_taper=None does NOT give flat risk — it falls back to the engine's default
    taper. The no-taper row pins a single rung instead, so it removes the taper without
    also removing regime sizing.
  * Paired across orderings: a single run moves 10-15% on intra-hour processing order
    alone, which is larger than several of these effects.
  * The clean out-of-sample window decides it. In-sample the edge is fitted, so anything
    that lets more trades through looks good there.
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
OOS = pd.Timestamp("2026-05-25")
UNI_CHOP = ROOT / "trail_5m_universe.parquet"
UNI_NOCHOP = ROOT / "trail_5m_universe_nochop.parquet"
ARMS = ["s3_a1", "base"]
SEEDS = list(range(4))

# label -> (universe, scenario, net_dir_cap, open_risk_cap, short_gate, long_boost, taper)
# taper: 'live' = the deployed schedule, 'flat' = single rung (removes taper only)
CONFIGS = [
    ("ALL ON (live)",     "chop",   "production",        0.10, 0.30, True,  1.3, "live"),
    ("- CHOP filter",     "nochop", "production_nochop", 0.10, 0.30, True,  1.3, "live"),
    ("- regime sizing",   "chop",   "chop_only",         0.10, 0.30, True,  1.3, "live"),
    ("- net-dir cap",     "chop",   "production",        None, 0.30, True,  1.3, "live"),
    ("- gross risk cap",  "chop",   "production",        0.10, None, True,  1.3, "live"),
    ("- BTC short gate",  "chop",   "production",        0.10, 0.30, False, 1.3, "live"),
    ("- long boost",      "chop",   "production",        0.10, 0.30, True,  1.0, "live"),
    ("- balance taper",   "chop",   "production",        0.10, 0.30, True,  1.3, "flat"),
    ("ALL OFF",           "nochop", "no_filters",        None, None, False, 1.0, "flat"),
]

_P = None
_CHOP = None
_U = {}


def _init():
    global _P, _CHOP, _U
    import backtest_production_correct as P
    from backtest_shadow_gate import LIVE as LIVE_KW
    _P = (P, LIVE_KW)
    for k, path in (("chop", UNI_CHOP), ("nochop", UNI_NOCHOP)):
        d = pd.read_parquet(path)
        d["entry_time"] = pd.to_datetime(d["entry_time"])
        _U[k] = d
    _CHOP = P.load_chop_data(sorted(_U["nochop"].symbol.unique()))


def run(args):
    win, arm, idx, seed = args
    label, uni, scen, ndc, orc, sg, lb, taper = CONFIGS[idx]
    P, LIVE_KW = _P
    t0 = W15 if win == "15mo" else OOS
    d = _U[uni]
    d = d[(d.entry_time >= t0) & (d.entry_time < END)]
    out = d[["entry_time", "entry_price", "sl_price", "side", "symbol",
             "btc_bull", "btc_impulse"]].copy()
    out["exit_time"] = pd.to_datetime(d[f"x_{arm}"])
    out["r_result"] = d[f"r_{arm}"]
    out = out.dropna(subset=["exit_time", "r_result"])
    if len(out) < 50:
        return args, None
    out = out.sample(frac=1.0, random_state=seed)
    out = out.sort_values("entry_time", kind="mergesort").reset_index(drop=True)

    P.ROUND_TRIP_COST = BASE_RT
    P.STARTING_BALANCE = START
    kw = dict(starting_balance=START, scenario=scen,
              base_risk=LIVE_KW["base_risk"],
              taper_basis=LIVE_KW["taper_basis"], size_basis=LIVE_KW["size_basis"],
              net_dir_cap=ndc, open_risk_cap=orc,
              short_gate=sg, long_boost=lb,
              overlay_min_balance=0.0,
              btc_bull_col="btc_bull", btc_short_col="btc_impulse")
    kw["custom_taper"] = (LIVE_KW["custom_taper"] if taper == "live"
                          else [[0, LIVE_KW["base_risk"]]])
    r = P.run_simulation(out, _CHOP, **kw)
    t = r["entered_trades"]
    if not t:
        return args, None
    gp = sum(x["pnl"] for x in t if x["pnl"] > 0)
    gl = abs(sum(x["pnl"] for x in t if x["pnl"] < 0))
    return args, dict(final=r["final_effective"], dd=r["max_dd_pct"], n=len(t),
                      pf=(gp / gl) if gl else float("inf"))


def agg(res, win, arm, idx):
    rows = [res[(win, arm, idx, s)] for s in SEEDS if (win, arm, idx, s) in res]
    if not rows:
        return None
    f = [x["final"] for x in rows]
    return dict(med=float(np.median(f)), lo=min(f), hi=max(f),
                dd=float(np.median([x["dd"] for x in rows])),
                n=int(np.median([x["n"] for x in rows])),
                pf=float(np.median([x["pf"] for x in rows])))


def table(res, win, arm, title, note=""):
    print("\n" + "=" * 106)
    print(title)
    if note:
        print(note)
    print("=" * 106)
    ref = agg(res, win, arm, 0)
    print(f"  {'configuration':<20}{'median $':>12}{'vs ALL ON':>12}{'%':>8}"
          f"{'maxDD':>9}{'PF':>7}{'trades':>9}{'verdict':>20}")
    for i, (label, *_ ) in enumerate(CONFIGS):
        a = agg(res, win, arm, i)
        if not a:
            continue
        if i == 0:
            print(f"  {label:<20}{a['med']:>12,.0f}{'—':>12}{'—':>8}"
                  f"{a['dd']:>8.1f}%{a['pf']:>7.2f}{a['n']:>9,}{'baseline':>20}")
            continue
        dv = a["med"] - ref["med"]
        pc = a["med"] / ref["med"] - 1
        ddv = a["dd"] - ref["dd"]
        if dv > 0 and ddv <= 2:
            v = "REMOVING HELPS"
        elif dv > 0:
            v = "more $, more DD"
        elif ddv < -2:
            v = "less $, less DD"
        else:
            v = "KEEP IT"
        print(f"  {label:<20}{a['med']:>12,.0f}{dv:>+12,.0f}{pc:>+8.0%}"
              f"{a['dd']:>8.1f}%{a['pf']:>7.2f}{a['n']:>9,}{v:>20}")
    if win == "oos":
        print(f"  {'never trade':<20}{START:>12,.0f}{START - ref['med']:>+12,.0f}"
              f"{'':>8}{0.0:>8.1f}%{'—':>7}{0:>9}{'':>20}")


def main():
    jobs = [(w, a, i, s) for w in ("15mo", "oos") for a in ARMS
            for i in range(len(CONFIGS)) for s in SEEDS]
    print(f"[ABLATE] {len(jobs)} simulations "
          f"({len(CONFIGS)} configs x {len(ARMS)} exit rules x {len(SEEDS)} orderings "
          f"x 2 windows)\n", flush=True)
    res = {}
    done = 0
    with mp.Pool(max(1, mp.cpu_count() - 1), initializer=_init) as pool:
        for key, r in pool.imap_unordered(run, jobs, chunksize=1):
            done += 1
            if r:
                res[key] = r
            if done % 40 == 0:
                print(f"  {done}/{len(jobs)}", flush=True)

    for arm, alabel in (("s3_a1", "TRAILING s3_a1 (deployed)"), ("base", "FIXED TP")):
        table(res, "15mo", arm,
              f"PROTECTION ABLATION — {alabel} — LAST 15 MONTHS, $1,500",
              "  In-sample-ish: the edge is fitted here, so loosening filters flatters.")
        table(res, "oos", arm,
              f"PROTECTION ABLATION — {alabel} — CLEAN OUT-OF-SAMPLE ({OOS:%Y-%m-%d} →)",
              "  The decisive window. never-trade = $1,500.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
