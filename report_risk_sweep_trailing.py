#!/usr/bin/env python3
"""Should base risk go up? Re-asked now that trailing has changed the drawdown profile.

Raising risk was tested and rejected before (GROWTH_VALIDATION_REPORT; base risk was cut
1.2% -> 0.3% in ea730b4 because it halved cold-period drawdown). What is new is that the
trailing stop cuts max drawdown ~40% -> ~29%, which in principle buys headroom. This asks
whether that headroom is real or whether it just gets spent.

Method notes that matter:
  * "Risk" here scales the WHOLE schedule, not just risk_per_trade. The taper rungs are
    what actually govern sizing above $1,500, so multiplying only the base would change
    almost nothing once the account grows.
  * Paired across orderings. A single run moves 10-15% on intra-hour processing order
    alone (verify_ordering_sensitivity.py), which is larger than several of the steps in
    this sweep — so an unpaired comparison here would be meaningless.
  * The clean out-of-sample window is reported separately and is the one that decides it.
    The edge is negative there; higher risk on a negative edge just loses faster.
  * risk-capped counts are reported because gross_open_risk_cap (30%) and
    net_directional_cap (10%) are hard ceilings — past some point extra risk-per-trade
    is simply refused, and returns saturate while drawdown does not.
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
UNI = ROOT / "trail_5m_universe.parquet"
ARMS = ["base", "s3_a1"]
SEEDS = list(range(5))
# multiplier on the entire risk schedule -> effective risk at $1,500
KS = [0.5, 0.67, 1.0, 1.33, 1.67, 2.0, 2.67, 4.0]

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
    _D = d
    _CHOP = P.load_chop_data(sorted(d.symbol.unique()))


def run(args):
    win, nm, k, seed = args
    P, LIVE_KW = _P
    t0 = W15 if win == "15mo" else OOS
    d = _D[(_D.entry_time >= t0) & (_D.entry_time < END)]
    out = d[["entry_time", "entry_price", "sl_price", "side", "symbol",
             "btc_bull", "btc_impulse"]].copy()
    out["exit_time"] = pd.to_datetime(d[f"x_{nm}"])
    out["r_result"] = d[f"r_{nm}"]
    out = out.dropna(subset=["exit_time", "r_result"])
    if len(out) < 50:
        return args, None
    out = out.sample(frac=1.0, random_state=seed)
    out = out.sort_values("entry_time", kind="mergesort").reset_index(drop=True)

    P.ROUND_TRIP_COST = BASE_RT
    P.STARTING_BALANCE = START
    kw = dict(LIVE_KW, starting_balance=START,
              btc_bull_col="btc_bull", btc_short_col="btc_impulse")
    kw["base_risk"] = LIVE_KW["base_risk"] * k
    kw["custom_taper"] = [[b, p * k] for b, p in LIVE_KW["custom_taper"]]
    r = P.run_simulation(out, _CHOP, **kw)
    t = r["entered_trades"]
    if not t:
        return args, None
    return args, dict(final=r["final_effective"], dd=r["max_dd_pct"], n=len(t),
                      capped=int(r.get("risk_capped") or 0))


def agg(res, win, nm, k):
    rows = [res[(win, nm, k, s)] for s in SEEDS if (win, nm, k, s) in res]
    if not rows:
        return None
    f = [x["final"] for x in rows]
    dd = [x["dd"] for x in rows]
    return dict(med=float(np.median(f)), lo=min(f), hi=max(f),
                dd=float(np.median(dd)), ddhi=max(dd),
                n=int(np.median([x["n"] for x in rows])),
                capped=int(np.median([x["capped"] for x in rows])))


def table(res, win, title, note=""):
    print("\n" + "=" * 104)
    print(title)
    if note:
        print(note)
    print("=" * 104)
    for nm in ARMS:
        label = "FIXED TP (base)" if nm == "base" else "TRAILING s3_a1"
        print(f"\n  --- {label} ---")
        print(f"  {'risk @$1.5k':<13}{'median $':>12}{'range $':>24}"
              f"{'med DD':>9}{'worst DD':>10}{'ROI/DD':>9}{'risk-capped':>13}")
        for k in KS:
            a = agg(res, win, nm, k)
            if not a:
                continue
            pct = 0.003 * k * 100
            roi = a["med"] / START - 1
            mark = "  <-- live" if abs(k - 1.0) < 1e-9 else ""
            print(f"  {pct:>6.2f}%      {a['med']:>12,.0f}"
                  f"{a['lo']:>11,.0f} .. {a['hi']:>10,.0f}"
                  f"{a['dd']:>8.1f}%{a['ddhi']:>9.1f}%"
                  f"{roi * 100 / max(a['dd'], 1e-9):>9.1f}{a['capped']:>13,}{mark}")


def main():
    jobs = [(w, nm, k, s) for w in ("15mo", "oos") for nm in ARMS
            for k in KS for s in SEEDS]
    print(f"[RISK] {len(jobs)} simulations "
          f"({len(KS)} risk levels x {len(ARMS)} exit rules x {len(SEEDS)} orderings "
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

    table(res, "15mo", "RISK SWEEP — LAST 15 MONTHS, $1,500 start",
          "  ~13 of 15 months sit inside the config's fitting window; magnitudes flatter "
          "reality.\n  Compare the SHAPE of return vs drawdown, not the levels.")
    table(res, "oos", f"RISK SWEEP — CLEAN OUT-OF-SAMPLE ({OOS:%Y-%m-%d} →)",
          "  The decisive window: nothing was fitted here. never-trade = $1,500.")

    # marginal efficiency: what does each extra unit of risk actually buy?
    print("\n" + "=" * 104)
    print("MARGINAL EFFICIENCY — what each step up in risk buys, 15 months")
    print("=" * 104)
    for nm in ARMS:
        label = "FIXED TP" if nm == "base" else "TRAILING"
        print(f"\n  --- {label} ---")
        print(f"  {'step':<20}{'return x':>11}{'drawdown x':>13}{'verdict':>28}")
        prev = None
        for k in KS:
            a = agg(res, "15mo", nm, k)
            if not a:
                continue
            if prev:
                rx = (a["med"] - START) / max(prev[0] - START, 1e-9)
                dx = a["dd"] / max(prev[1], 1e-9)
                v = "worth it" if rx > dx * 1.05 else (
                    "break-even" if rx > dx * 0.95 else "costs more than it pays")
                print(f"  {prev[2]:.2f}% -> {0.003*k*100:.2f}%".ljust(22)
                      + f"{rx:>10.2f}x{dx:>12.2f}x{v:>28}")
            prev = (a["med"], a["dd"], 0.003 * k * 100)
    return 0


if __name__ == "__main__":
    sys.exit(main())
