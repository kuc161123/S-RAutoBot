#!/usr/bin/env python3
"""Regime OFF wins on return. Can its drawdown be fixed by something other than the regime?

Regime-off on GLOBAL rr10/atr3 wins ROI in 7 of 11 rolling 6-month windows but has HIGHER
drawdown in 11 of 11, and turns the untouched holdout from +7.9% to -14.0%. The question is
whether a different brake recovers the drawdown without giving back the return.

Candidate brakes, each tested alone and stacked:
  HALT 7d / 21d   the shadow-R kill switch already shipped (advisory in live config)
  DD-THROTTLE     cut risk as the account falls from its peak -- the engine's `dd_taper`,
                  which the live bot does NOT currently use. Three schedules, mild to hard.
  CONCURRENCY     cap simultaneous open positions (the bot has no cap today)
  LOWER RISK      simply size down -- the null hypothesis for every other brake here. If a
                  brake cannot beat "just use less risk", it is not earning its complexity.

The comparison that matters is not raw return, it is return per unit of drawdown AND the
untouched holdout, because the whole reason regime-off is tempting is in-sample return.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import backtest_production_correct as P  # noqa: E402
from backtest_shadow_gate import LIVE as LIVE_SIM, LIVE_TAPER  # noqa: E402
import compare_bots as C  # noqa: E402
from variations import with_halt  # noqa: E402

START = 1500.0
UNI = "uni_glob_rr10_am3_same.parquet"

DD_MILD = [(20, 0.75), (30, 0.5), (40, 0.35)]
DD_MED = [(15, 0.6), (25, 0.4), (35, 0.25)]
DD_HARD = [(10, 0.5), (20, 0.3), (30, 0.15), (40, 0.07)]

ARMS = [
    ("regime ON  (recommended)", dict(), None, 0.003),
    ("regime OFF (raw)", dict(scenario="chop_only"), None, 0.003),
    ("regime OFF + halt 21d", dict(scenario="chop_only"), 21, 0.003),
    ("regime OFF + halt 7d", dict(scenario="chop_only"), 7, 0.003),
    ("regime OFF + dd-throttle mild", dict(scenario="chop_only", dd_taper=DD_MILD), None, 0.003),
    ("regime OFF + dd-throttle med", dict(scenario="chop_only", dd_taper=DD_MED), None, 0.003),
    ("regime OFF + dd-throttle hard", dict(scenario="chop_only", dd_taper=DD_HARD), None, 0.003),
    ("regime OFF + cap 20 open", dict(scenario="chop_only", max_concurrent=20), None, 0.003),
    ("regime OFF + cap 10 open", dict(scenario="chop_only", max_concurrent=10), None, 0.003),
    ("regime OFF @ 0.15% risk", dict(scenario="chop_only"), None, 0.0015),
    ("regime OFF @ 0.20% risk", dict(scenario="chop_only"), None, 0.002),
    ("regime OFF + dd-med + halt21", dict(scenario="chop_only", dd_taper=DD_MED), 21, 0.003),
    ("regime OFF + dd-hard + halt21", dict(scenario="chop_only", dd_taper=DD_HARD), 21, 0.003),
    ("regime ON  + dd-med", dict(dd_taper=DD_MED), None, 0.003),
    ("regime ON  + halt21 (rec+)", dict(), 21, 0.003),
]


def sim(d, chop, over, base, t0=None, t1=None):
    sub = d
    if t0 is not None:
        sub = sub[sub.entry_time >= t0]
    if t1 is not None:
        sub = sub[sub.entry_time < t1]
    if len(sub) < 100:
        return None
    sc = base / 0.003
    kw = dict(LIVE_SIM)
    kw.update(starting_balance=START, base_risk=base,
              custom_taper=[(t, r * sc) for t, r in LIVE_TAPER],
              btc_bull_col="btc_bull", btc_short_col="btc_impulse",
              taper_basis="wallet", size_basis="wallet")
    kw.update(over)
    P.STARTING_BALANCE = START
    cols = ["entry_time", "exit_time", "entry_price", "sl_price", "r_result",
            "side", "symbol", "btc_bull", "btc_impulse"]
    r = P.run_simulation(sub[cols].reset_index(drop=True), chop, **kw)
    t = pd.DataFrame(r["entered_trades"])
    if t.empty:
        return None
    t = t.sort_values("exit_time")
    c = pd.Series((START + t.pnl.cumsum()).values, index=pd.to_datetime(t.exit_time.values))
    pk = c.cummax()
    dd = float(((pk - c) / pk).max() * 100)
    h = c[c.index >= pd.Timestamp("2026-05-31")]
    ho = (h.iloc[-1] / h.iloc[0] - 1) * 100 if len(h) > 2 else np.nan
    return dict(final=r["final_effective"], dd=dd, n=len(t), hold=ho,
                roi=(r["final_effective"] / START - 1) * 100,
                ratio=r["final_effective"] / dd if dd > 0 else np.nan)


def main():
    P.ROUND_TRIP_COST = 0.00242
    uni = pd.read_parquet(HERE / "universe_trail.parquet")
    chop = P.load_chop_data(sorted(uni.symbol.unique()))
    base_d = C.load(UNI, "r_result", "exit_time")

    print("=" * 104)
    print("CAN REGIME-OFF'S DRAWDOWN BE FIXED?  GLOBAL rr10/atr3 + trail · $1,500 · full history")
    print("=" * 104)
    print(f"{'arm':<32}{'trades':>8}{'final $':>11}{'maxDD%':>9}{'$/DDpt':>9}{'holdout%':>10}")
    rows = []
    cache = {}
    for nm, over, halt, base in ARMS:
        d = base_d if halt is None else with_halt(base_d, halt, -300, 200 if halt == 21 else 400)
        r = sim(d, chop, over, base)
        if r is None:
            continue
        cache[nm] = (d, over, base)
        rows.append(dict(arm=nm, **r))
        print(f"{nm:<32}{r['n']:>8,}{r['final']:>11,.0f}{r['dd']:>9.1f}"
              f"{r['ratio']:>9.0f}{r['hold']:>10.1f}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(HERE / "results" / "regime_dd.csv", index=False)

    # ---- the honest test: rolling windows, not one path ----
    print()
    print("=" * 104)
    print("ROLLING 6-MONTH WINDOWS — each a fresh $1,500, scored independently")
    print("=" * 104)
    picks = ["regime ON  (recommended)", "regime OFF (raw)", "regime OFF + dd-throttle med",
             "regime OFF + halt 21d", "regime OFF @ 0.15% risk", "regime ON  + halt21 (rec+)"]
    starts = pd.date_range("2023-06-01", "2026-02-01", freq="3MS")
    agg = {p: {"roi": [], "dd": []} for p in picks}
    print(f"{'window':<12}" + "".join(f"{p.split('(')[0][:13]:>15}" for p in picks))
    for s in starts:
        e = s + pd.DateOffset(months=6)
        line = f"{str(s.date()):<12}"
        ok = True
        vals = {}
        for p in picks:
            d, over, base = cache[p]
            r = sim(d, chop, over, base, t0=s, t1=e)
            if r is None:
                ok = False
                break
            vals[p] = r
        if not ok:
            continue
        for p in picks:
            agg[p]["roi"].append(vals[p]["roi"])
            agg[p]["dd"].append(vals[p]["dd"])
            line += f"{vals[p]['roi']:>9.0f}/{vals[p]['dd']:>4.0f}"
        print(line, flush=True)

    print()
    print(f"{'arm':<32}{'median ROI%':>13}{'median DD%':>12}{'ROI/DD':>9}{'win vs ON':>11}")
    onr = agg["regime ON  (recommended)"]["roi"]
    for p in picks:
        r = np.array(agg[p]["roi"]); dd = np.array(agg[p]["dd"])
        w = int((r > np.array(onr)).sum())
        print(f"{p:<32}{np.median(r):>13.0f}{np.median(dd):>12.0f}"
              f"{np.median(r)/np.median(dd):>9.2f}{w:>8}/{len(r)}")


if __name__ == "__main__":
    main()
