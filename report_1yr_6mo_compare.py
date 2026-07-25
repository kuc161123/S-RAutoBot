#!/usr/bin/env python3
"""BASE vs GATED-21d vs GATED-7d over the last 1 year and last 6 months, from $1,500.

CONTAMINATION MAP — read before the numbers. The live config was walk-forward fitted with
train < 2025-11-01 and test through 2026-05-25, then deployed 2026-05-25. So:

  2025-07-25 .. 2025-11-01   TRAINING data          — heaviest contamination
  2025-11-01 .. 2026-05-25   walk-forward TEST      — selection contamination
  2026-05-25 .. today        genuine out-of-sample  — clean

The 1-year window therefore spans all three; the 6-month window spans the last two. Both
flatter the config. Only the OOS sub-period is a clean read, and it is reported alongside
so the three can be compared directly.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
import backtest_production_correct as P
from backtest_shadow_gate import LIVE, gate_series
from report_monthly_1500 import prep

ROOT = Path(__file__).parent
START_BAL = 1500.0
ARMS = [("BASE (no gate)", None),
        ("GATED 21d", (21, -300, 200)),
        ("GATED 7d", (7, -300, 400))]

WINDOWS = [
    ("LAST 1 YEAR", "2025-07-25", "train + WF-test + OOS — most contaminated"),
    ("LAST 6 MONTHS", "2026-01-25", "WF-test + OOS — partly contaminated"),
    ("GENUINE OOS", "2026-05-25", "clean — config had never seen this data"),
]


def run(full, win, gate, chop):
    sub = win
    halted = 0.0
    if gate is not None:
        g = gate_series(full, *gate)
        allow = win.entry_time.dt.floor("h").map(g.to_dict()).fillna(True).astype(bool)
        sub = win[allow]
        gw = g[(g.index >= win.entry_time.min()) & (g.index <= win.entry_time.max())]
        halted = 1.0 - float(gw.mean()) if len(gw) else 0.0
    if sub.empty:
        return dict(monthly={}, final=START_BAL, dd=0.0, trades=0, pf=0.0, halted=halted)
    kw = dict(LIVE, starting_balance=START_BAL, btc_bull_col="btc_bull",
              btc_short_col="btc_impulse")
    P.STARTING_BALANCE = START_BAL
    r = P.run_simulation(sub, chop, **kw)
    t = r["entered_trades"]
    gp = sum(x["pnl"] for x in t if x["pnl"] > 0)
    gl = abs(sum(x["pnl"] for x in t if x["pnl"] < 0))
    return dict(monthly=r["monthly_pnl"], final=r["final_effective"],
                dd=r["max_dd_pct"], trades=len(t),
                pf=(gp / gl) if gl else float("inf"), halted=halted)


def main():
    full = prep("universe_live_1yr.csv")
    summary = []

    for title, t0, note in WINDOWS:
        win = full[full.entry_time >= pd.Timestamp(t0)].sort_values("entry_time")
        chop = P.load_chop_data(sorted(win.symbol.unique()))
        res = {n: run(full, win, g, chop) for n, g in ARMS}
        names = [n for n, _ in ARMS]

        print(f"\n{'=' * 88}\n{title}  (from {t0}, $1,500 start)\n  {note}\n{'=' * 88}")
        print(f"  {'month':<10}" + "".join(f"{n:>25}" for n in names))
        print(f"  {'':<10}" + "".join(f"{'P&L      balance':>25}" for _ in names))
        print("  " + "-" * 84)
        months = sorted({k for r in res.values() for k in r["monthly"]})
        bal = {n: START_BAL for n in names}
        for mo in months:
            row = f"  {mo:<10}"
            for n in names:
                p = res[n]["monthly"].get(mo, 0.0)
                bal[n] += p
                row += f"{p:>+13,.0f}{bal[n]:>12,.0f}"
            print(row)
        print("  " + "-" * 84)
        print(f"  {'FINAL':<10}" + "".join(
            f"{res[n]['final'] - START_BAL:>+13,.0f}{res[n]['final']:>12,.0f}" for n in names))
        print(f"  {'ROI':<10}" + "".join(
            f"{'':>13}{res[n]['final'] / START_BAL - 1:>11.1%}" for n in names))
        print(f"  {'max DD':<10}" + "".join(f"{'':>13}{res[n]['dd']:>10.1f}%" for n in names))
        print(f"  {'PF':<10}" + "".join(f"{'':>13}{res[n]['pf']:>11.2f}" for n in names))
        print(f"  {'trades':<10}" + "".join(f"{'':>13}{res[n]['trades']:>11,}" for n in names))
        print(f"  {'halted':<10}" + "".join(f"{'':>13}{res[n]['halted']:>10.1%}" for n in names))
        print(f"\n  never trading = ${START_BAL:,.0f}")

        for n in names:
            summary.append(dict(window=title, arm=n, final=res[n]["final"],
                                roi=res[n]["final"] / START_BAL - 1, dd=res[n]["dd"],
                                pf=res[n]["pf"], trades=res[n]["trades"]))

    s = pd.DataFrame(summary)
    s.to_csv(ROOT / "compare_1yr_6mo.csv", index=False)

    print(f"\n{'=' * 88}\nSUMMARY — final $ from $1,500\n{'=' * 88}")
    print(f"  {'window':<16}" + "".join(f"{n:>20}" for n, _ in ARMS))
    for w, _, _ in WINDOWS:
        r = s[s.window == w].set_index("arm")
        print(f"  {w:<16}" + "".join(
            f"{r.loc[n, 'final']:>13,.0f}{r.loc[n, 'roi']:>7.0%}" for n, _ in ARMS))
    print(f"\n  {'window':<16}" + "".join(f"{n + ' DD':>20}" for n, _ in ARMS))
    for w, _, _ in WINDOWS:
        r = s[s.window == w].set_index("arm")
        print(f"  {w:<16}" + "".join(f"{r.loc[n, 'dd']:>19.1f}%" for n, _ in ARMS))
    print(f"\n  -> {ROOT / 'compare_1yr_6mo.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
