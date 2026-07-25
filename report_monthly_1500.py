#!/usr/bin/env python3
"""Month-by-month dollar comparison from a $1,500 start: ungated bot vs shadow-gated.

Two views, because they answer different questions:
  A) full 8 months (2025-11-01 -> today) — the whole arc, but the months before
     2026-05-25 are inside the config's own walk-forward fitting window, so those
     dollars are optimistic and are marked as such.
  B) genuine OOS only (2026-05-25 -> today) — starting fresh at $1,500. This is the
     honest "what would have actually happened" number.
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

ROOT = Path(__file__).parent
START_BAL = 1500.0
SPLIT = pd.Timestamp("2026-05-25")

ARMS = [
    ("BASE — no gate (current bot)", None),
    ("GATED 21d stop -300R start +200R", (21, -300, 200)),
    ("GATED 7d stop -300R start +400R", (7, -300, 400)),
]


def prep(universe="universe_live_8mo.csv"):
    d = pd.read_csv(ROOT / universe, parse_dates=["entry_time", "exit_time"])
    d["fee_r"] = 0.0018 * d.entry_price / (d.entry_price - d.sl_price).abs()
    d["r_net"] = d.r_result - d.fee_r
    b = pd.read_parquet(ROOT / "cache_3yr_1h" / "BTCUSDT.parquet").sort_values("start")
    b["ema200"] = b["close"].ewm(span=200, adjust=False).mean()
    bull = (b["close"] > b["ema200"]).shift(1).fillna(False)
    bull.index = b["start"]
    d_ = b.set_index("start")["close"].resample("1D").last().dropna()
    imp = (d_ / d_.shift(30) - 1.0) > 0.10
    imp.index = imp.index + pd.Timedelta(days=1)
    imp_h = imp.reindex(pd.date_range(b["start"].min().floor("D"),
                                      b["start"].max().ceil("D") + pd.Timedelta(days=1),
                                      freq="h")).ffill().fillna(False)
    et = d.entry_time.dt.floor("h")
    d["btc_bull"] = et.map(bull.to_dict()).fillna(False).astype(bool)
    d["btc_impulse"] = et.map(imp_h.to_dict()).fillna(False).astype(bool)
    return d


def run(full, win, gate, chop, bal):
    sub = win
    if gate is not None:
        W, stop_r, start_r = gate
        g = gate_series(full, W, stop_r, start_r)
        allow = win.entry_time.dt.floor("h").map(g.to_dict()).fillna(True).astype(bool)
        sub = win[allow]
    if sub.empty:
        return {}, bal, 0.0, 0
    kw = dict(LIVE, starting_balance=bal, btc_bull_col="btc_bull",
              btc_short_col="btc_impulse")
    P.STARTING_BALANCE = bal
    res = P.run_simulation(sub, chop, **kw)
    return (res["monthly_pnl"], res["final_effective"], res["max_dd_pct"],
            len(res["entered_trades"]))


def view(full, t0, title, note):
    win = full[full.entry_time >= pd.Timestamp(t0)].sort_values("entry_time")
    chop = P.load_chop_data(sorted(win.symbol.unique()))
    print(f"\n{'=' * 92}\n{title}\n{note}\n{'=' * 92}")

    results = {}
    for name, gate in ARMS:
        m, final, dd, n = run(full, win, gate, chop, START_BAL)
        results[name] = dict(monthly=m, final=final, dd=dd, trades=n)

    months = sorted({k for r in results.values() for k in r["monthly"]})
    names = [n for n, _ in ARMS]

    hdr = f"  {'month':<10}"
    for n in names:
        hdr += f"{n.split(' — ')[0].split(' stop')[0]:>22}"
    print(hdr)
    print(f"  {'':<10}" + "".join(f"{'P&L      balance':>22}" for _ in names))
    print("  " + "-" * 88)

    bal = {n: START_BAL for n in names}
    for mo in months:
        row = f"  {mo:<10}"
        for n in names:
            p = results[n]["monthly"].get(mo, 0.0)
            bal[n] += p
            row += f"{p:>+11,.0f}{bal[n]:>11,.0f}"
        mark = "" if pd.Timestamp(mo + "-01") >= SPLIT.normalize().replace(day=1) else "  (fit window)"
        print(row + mark)

    print("  " + "-" * 88)
    print(f"  {'FINAL':<10}" + "".join(f"{results[n]['final'] - START_BAL:>+11,.0f}"
                                       f"{results[n]['final']:>11,.0f}" for n in names))
    print(f"  {'ROI':<10}" + "".join(
        f"{'':>11}{(results[n]['final'] / START_BAL - 1):>10.1%} " for n in names))
    print(f"  {'max DD':<10}" + "".join(f"{'':>11}{results[n]['dd']:>10.1f}% " for n in names))
    print(f"  {'trades':<10}" + "".join(f"{'':>11}{results[n]['trades']:>11,}" for n in names))
    print(f"\n  reference: never trading = ${START_BAL:,.0f} flat")


def main():
    full = prep()
    view(full, "2026-05-25",
         "B) GENUINE OUT-OF-SAMPLE — since the current config went live (2026-05-25)",
         "   This is the honest number: the config had never seen this data.")
    view(full, "2025-11-01",
         "A) FULL 8 MONTHS (2025-11-01 -> today)",
         "   Months before 2026-05-25 sit INSIDE the config's walk-forward fitting\n"
         "   window — those dollars are optimistic and are marked '(fit window)'.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
