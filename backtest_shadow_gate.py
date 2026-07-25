#!/usr/bin/env python3
"""Would a shadow-R kill-switch have helped? Sweep stop/restart thresholds.

The shadow layer already grades every evaluated signal (executed or blocked). This asks:
if we had used trailing shadow R to HALT new entries when it fell below `stop_r`, and
RESUME when it recovered above `start_r`, how would the account have done versus running
ungated?

CAUSALITY — the two things that make this honest:
  1. A signal's R only enters the shadow ledger when it RESOLVES, not when it is entered.
     So the trailing sum at time t uses exit_time <= t only. This is exactly how the live
     resolver works (shadow_logger.resolve_pending grades from klines after the fact).
  2. Because losses resolve far faster than wins (median 6h vs 38h in this universe), the
     real-time trailing sum is structurally biased NEGATIVE relative to the same window
     measured in hindsight. That bias is a genuine property of the live signal and is
     deliberately preserved here — a gate built on it must survive that handicap.

Gating only ever blocks NEW entries; open positions run to their stop/target, matching
how /stop behaves live (bot.py:1530).
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
import backtest_production_correct as P

ROOT = Path(__file__).parent
LIVE_TAPER = [(1500, 0.003), (3000, 0.0028), (5000, 0.0025), (8000, 0.0022),
              (12000, 0.002), (20000, 0.0017), (40000, 0.0014)]
LIVE = dict(scenario="production", base_risk=0.003, custom_taper=LIVE_TAPER,
            taper_basis="wallet", size_basis="equity",
            net_dir_cap=0.10, open_risk_cap=0.30,
            short_gate=True, long_boost=1.3, overlay_min_balance=0.0)


def gate_series(df, window_d, stop_r, start_r):
    """Hourly bool: is trading enabled? Hysteresis state machine on trailing shadow R."""
    grid = pd.date_range(df.entry_time.min().floor("h"),
                         df.entry_time.max().ceil("h"), freq="h")
    # trailing sum of RESOLVED R, evaluated on the hourly grid (causal)
    res = df.set_index("exit_time").sort_index()["r_net"]
    per_h = res.resample("h").sum().reindex(grid, fill_value=0.0)
    trail = per_h.rolling(f"{window_d}D").sum()

    on = True
    out = np.empty(len(grid), dtype=bool)
    v = trail.to_numpy()
    for i in range(len(grid)):
        r = v[i]
        if on and r <= stop_r:
            on = False
        elif (not on) and r >= start_r:
            on = True
        out[i] = on
    return pd.Series(out, index=grid)


def metrics(res, bal):
    t = res["entered_trades"]
    if not t:
        return dict(trades=0, roi=0.0, final=bal, dd=0.0, pf=0.0, avg_r=0.0)
    gp = sum(x["pnl"] for x in t if x["pnl"] > 0)
    gl = abs(sum(x["pnl"] for x in t if x["pnl"] < 0))
    return dict(trades=len(t), roi=(res["final_effective"] - bal) / bal,
                final=res["final_effective"], dd=res["max_dd_pct"],
                pf=(gp / gl) if gl else float("inf"),
                avg_r=float(np.mean([x["r_result"] for x in t])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--universe", default="universe_live_8mo.csv")
    ap.add_argument("--start", default="2026-05-25")
    ap.add_argument("--end", default=None)
    ap.add_argument("--balance", type=float, default=1876.0)
    ap.add_argument("--label", default="")
    ap.add_argument("--out", default="shadow_gate_results.csv")
    a = ap.parse_args()

    full = pd.read_csv(ROOT / a.universe, parse_dates=["entry_time", "exit_time"])
    full["fee_r"] = 0.0018 * full.entry_price / (full.entry_price - full.sl_price).abs()
    full["r_net"] = full.r_result - full.fee_r

    # the shadow ledger sees the WHOLE history (it never stops observing, even when
    # trading is halted — bot.py logs signals before the /stop gate)
    win = full[full.entry_time >= pd.Timestamp(a.start)]
    if a.end:
        win = win[win.entry_time <= pd.Timestamp(a.end)]
    win = win.sort_values("entry_time").reset_index(drop=True)

    print(f"[GATE] {a.label or a.universe}")
    print(f"[GATE] {len(win)} signals · {win.entry_time.min()} .. {win.entry_time.max()}")
    print(f"[GATE] start ${a.balance:,.0f}\n")

    P.STARTING_BALANCE = a.balance
    chop = P.load_chop_data(sorted(win.symbol.unique()))

    # BTC overlay state (causal — same construction as backtest_protection_matrix)
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
    et = win.entry_time.dt.floor("h")
    win["btc_bull"] = et.map(bull.to_dict()).fillna(False).astype(bool)
    win["btc_impulse"] = et.map(imp_h.to_dict()).fillna(False).astype(bool)

    kw = dict(LIVE, starting_balance=a.balance,
              btc_bull_col="btc_bull", btc_short_col="btc_impulse")

    rows = []
    base = metrics(P.run_simulation(win, chop, **kw), a.balance)
    rows.append(dict(name="BASE — no gate (current bot)", window_d=0, stop_r=0,
                     start_r=0, halted_pct=0.0, **base))
    print(f"  {'BASE — no gate (current bot)':<40}{base['trades']:>6}t  "
          f"${base['final']:>9,.0f}  ROI {base['roi']:>+8.1%}  DD {base['dd']:>5.1f}%  "
          f"PF {base['pf']:.2f}")
    rows.append(dict(name="FLAT — never trade", window_d=0, stop_r=0, start_r=0,
                     halted_pct=100.0, trades=0, roi=0.0, final=a.balance, dd=0.0,
                     pf=0.0, avg_r=0.0))
    print(f"  {'FLAT — never trade':<40}{0:>6}t  ${a.balance:>9,.0f}  "
          f"ROI {0.0:>+8.1%}  DD {0.0:>5.1f}%")
    print()

    for W in [7, 14, 21]:
        for stop_r in [-100, -200, -300, -400, -600]:
            for start_r in [0, 100, 200, 400]:
                if start_r < stop_r:
                    continue
                g = gate_series(full, W, stop_r, start_r)
                allow = win.entry_time.dt.floor("h").map(g.to_dict())
                sub = win[allow.fillna(True).astype(bool)]
                halted = 1.0 - float(g.mean())
                if sub.empty:
                    m = dict(trades=0, roi=0.0, final=a.balance, dd=0.0, pf=0.0, avg_r=0.0)
                else:
                    m = metrics(P.run_simulation(sub, chop, **kw), a.balance)
                nm = f"{W}d  stop {stop_r:>+5}R  start {start_r:>+4}R"
                rows.append(dict(name=nm, window_d=W, stop_r=stop_r, start_r=start_r,
                                 halted_pct=halted * 100, **m))
                print(f"  {nm:<40}{m['trades']:>6}t  ${m['final']:>9,.0f}  "
                      f"ROI {m['roi']:>+8.1%}  DD {m['dd']:>5.1f}%  "
                      f"PF {m['pf']:.2f}  halted {halted:>5.1%}", flush=True)

    out = pd.DataFrame(rows).sort_values("final", ascending=False)
    out.to_csv(ROOT / a.out, index=False)
    print(f"\n[GATE] -> {a.out}")
    print("\nTOP 10 BY FINAL BALANCE")
    for _, r in out.head(10).iterrows():
        print(f"  {r['name']:<40} ${r['final']:>9,.0f}  ROI {r['roi']:>+8.1%}  "
              f"DD {r['dd']:>5.1f}%  halted {r['halted_pct']:>5.1f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
