#!/usr/bin/env python3
"""Why is the drawdown ~50%? Decompose it rather than accept it.

Candidate causes, each testable:
  1. No concurrency cap    — the bot holds 45+ positions at once and has no limit.
  2. Correlated book       — those positions are overwhelmingly on the same side (short),
                             so they are effectively ONE trade with 45x the size.
  3. Structural low WR     — RR 3-10 means 15-20% WR by design, so long loss runs are
                             normal and unavoidable.
  4. Clustered stop-outs   — when the market moves against a one-sided book, every stop
                             fires in the same few hours.

Reports the actual mechanism at the worst moment of the equity curve.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
import backtest_production_correct as P
from backtest_shadow_gate import LIVE
from report_monthly_1500 import prep

ROOT = Path(__file__).parent
START = 1500.0


def sim(win, chop, **over):
    kw = dict(LIVE, starting_balance=START, btc_bull_col="btc_bull",
              btc_short_col="btc_impulse")
    kw.update(over)
    P.STARTING_BALANCE = START
    return P.run_simulation(win, chop, **kw)


def curve(res):
    t = pd.DataFrame(res["entered_trades"])
    if t.empty:
        return t
    t = t.sort_values("exit_time").reset_index(drop=True)
    t["equity"] = START + t.pnl.cumsum()
    t["peak"] = t.equity.cummax()
    t["dd_pct"] = (t.peak - t.equity) / t.peak * 100
    return t


def concurrency(t, at):
    """positions open at timestamp `at`, by side"""
    m = (t.entry_time <= at) & (t.exit_time > at)
    o = t[m]
    return len(o), int((o.side == "long").sum()), int((o.side == "short").sum())


def main():
    full = prep("universe_live_1yr.csv")
    win = full[full.entry_time >= pd.Timestamp("2025-07-25")].sort_values("entry_time")
    chop = P.load_chop_data(sorted(win.symbol.unique()))

    res = sim(win, chop)
    t = curve(res)
    print(f"{'=' * 84}\nBASE, last 1 year, $1,500 start\n{'=' * 84}")
    print(f"  final ${res['final_effective']:,.0f}  ·  closed-balance maxDD "
          f"{res['max_dd_pct']:.1f}%  ·  mark-to-market maxDD {res['max_dd_mtm_pct']:.1f}%")
    print("  (mark-to-market includes unrealised P&L of open positions — the DD you FEEL)")

    # ---- worst moment -------------------------------------------------------
    i = int(t.dd_pct.idxmax())
    worst = t.loc[i]
    print(f"\n  worst closed-equity drawdown: {worst.dd_pct:.1f}% at {worst.exit_time}")
    print(f"    peak ${worst.peak:,.0f} -> trough ${worst.equity:,.0f} "
          f"(-${worst.peak - worst.equity:,.0f})")
    n, nl, ns = concurrency(t, worst.exit_time)
    print(f"    positions open at that moment: {n}  ({nl} long / {ns} short)")

    # ---- concurrency profile ------------------------------------------------
    print(f"\n  CONCURRENCY over the year (sampled daily):")
    days = pd.date_range(t.entry_time.min(), t.exit_time.max(), freq="D")
    cc = [concurrency(t, d)[0] for d in days]
    cc = pd.Series(cc)
    print(f"    median {cc.median():.0f} · p90 {cc.quantile(.9):.0f} · "
          f"max {cc.max():.0f} simultaneous positions")

    # ---- directional concentration -----------------------------------------
    sides = [concurrency(t, d) for d in days]
    net = pd.Series([abs(l - s) / n if n else 0 for n, l, s in sides])
    print(f"    directional imbalance |long-short|/total: median {net.median():.0%} "
          f"· p90 {net.quantile(.9):.0%}")
    print(f"    overall book: {(t.side=='short').mean():.0%} short")

    # ---- clustering: how concentrated are the losses in time? --------------
    print(f"\n  LOSS CLUSTERING:")
    daily = t.set_index("exit_time").pnl.resample("D").sum()
    worst_days = daily.nsmallest(10)
    tot_loss = daily[daily < 0].sum()
    print(f"    worst 10 days lost ${worst_days.sum():,.0f} = "
          f"{worst_days.sum()/tot_loss:.0%} of ALL losing-day dollars")
    print(f"    worst single day: ${worst_days.iloc[0]:,.0f} on {worst_days.index[0].date()}")
    d0 = worst_days.index[0]
    same = t[(t.exit_time >= d0) & (t.exit_time < d0 + pd.Timedelta(days=1))]
    print(f"      that day closed {len(same)} positions, "
          f"{(same.pnl<0).sum()} losers ({(same.side=='short').mean():.0%} short)")

    # ---- does capping concurrency fix it? ----------------------------------
    print(f"\n{'=' * 84}\nTEST: is concurrency the cause?\n{'=' * 84}")
    print(f"  {'setting':<26}{'final $':>11}{'ROI':>9}{'closed DD':>11}{'MTM DD':>9}{'trades':>9}")
    for lbl, over in [("no cap (current)", {}),
                      ("max 30 concurrent", dict(max_concurrent=30)),
                      ("max 20 concurrent", dict(max_concurrent=20)),
                      ("max 10 concurrent", dict(max_concurrent=10)),
                      ("max 5 concurrent", dict(max_concurrent=5)),
                      ("net-dir cap 5%", dict(net_dir_cap=0.05)),
                      ("max 10 + net-dir 5%", dict(max_concurrent=10, net_dir_cap=0.05))]:
        r = sim(win, chop, **over)
        print(f"  {lbl:<26}{r['final_effective']:>11,.0f}"
              f"{r['final_effective']/START-1:>9.0%}{r['max_dd_pct']:>10.1f}%"
              f"{r['max_dd_mtm_pct']:>8.1f}%{len(r['entered_trades']):>9,}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
