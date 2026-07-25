#!/usr/bin/env python3
"""Base-risk sweep from $1,500, ungated vs shadow-gated, OOS and full-8-month.

The taper LADDER is scaled proportionally with the base so its shape is preserved —
otherwise a low base would be raised again by the first rung and the comparison would be
meaningless.

The point of crossing risk with the gate: on a NEGATIVE edge, less risk simply means less
loss (risk has no optimum, only "smaller is better"). On a POSITIVE edge, risk has a real
optimum. So the two arms should disagree about the best size, and that disagreement is
itself the evidence for whether the gate is doing real work.
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
BASE_TAPER = [(1500, 0.003), (3000, 0.0028), (5000, 0.0025), (8000, 0.0022),
              (12000, 0.002), (20000, 0.0017), (40000, 0.0014)]
RISKS = [0.0010, 0.0015, 0.003, 0.005, 0.008, 0.012, 0.020]
GATE = (21, -300, 200)


def scaled_taper(base):
    k = base / 0.003
    return [(thr, r * k) for thr, r in BASE_TAPER]


def run(full, win, chop, base_risk, gated):
    sub = win
    if gated:
        g = gate_series(full, *GATE)
        allow = win.entry_time.dt.floor("h").map(g.to_dict()).fillna(True).astype(bool)
        sub = win[allow]
    if sub.empty:
        return START_BAL, 0.0, 0
    kw = dict(LIVE, starting_balance=START_BAL, base_risk=base_risk,
              custom_taper=scaled_taper(base_risk),
              btc_bull_col="btc_bull", btc_short_col="btc_impulse")
    P.STARTING_BALANCE = START_BAL
    r = P.run_simulation(sub, chop, **kw)
    return r["final_effective"], r["max_dd_pct"], len(r["entered_trades"])


def view(full, t0, title, note):
    win = full[full.entry_time >= pd.Timestamp(t0)].sort_values("entry_time")
    chop = P.load_chop_data(sorted(win.symbol.unique()))
    print(f"\n{'=' * 96}\n{title}\n{note}\n{'=' * 96}")
    print(f"  start ${START_BAL:,.0f} · never-trade reference ${START_BAL:,.0f}\n")
    print(f"  {'base risk':<12}{'UNGATED $':>12}{'ROI':>9}{'DD':>8}{'trades':>9}"
          f"{'':>4}{'GATED $':>12}{'ROI':>9}{'DD':>8}{'trades':>9}")
    print("  " + "-" * 92)
    rows = []
    for br in RISKS:
        fu, du, nu = run(full, win, chop, br, False)
        fg, dg, ng = run(full, win, chop, br, True)
        rows.append((br, fu, du, nu, fg, dg, ng))
        print(f"  {br*100:>8.2f}%   {fu:>12,.0f}{fu/START_BAL-1:>9.1%}{du:>7.1f}%{nu:>9,}"
              f"{'':>4}{fg:>12,.0f}{fg/START_BAL-1:>9.1%}{dg:>7.1f}%{ng:>9,}", flush=True)
    print("  " + "-" * 92)
    bu = max(rows, key=lambda r: r[1])
    bg = max(rows, key=lambda r: r[4])
    print(f"  best UNGATED: {bu[0]*100:.2f}%  -> ${bu[1]:,.0f}")
    print(f"  best GATED  : {bg[0]*100:.2f}%  -> ${bg[4]:,.0f}")
    return rows


def main():
    full = prep()
    view(full, "2026-05-25",
         "GENUINE OUT-OF-SAMPLE (2026-05-25 -> today)",
         "   The config had never seen this data. This is the decision-relevant table.")
    view(full, "2025-11-01",
         "FULL 8 MONTHS (2025-11-01 -> today)",
         "   Contains the config's own fitting window — optimistic, shown for shape only.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
