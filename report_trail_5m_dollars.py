#!/usr/bin/env python3
"""1H vs 5m exit resolution, in DOLLARS.

Both arms come from ONE universe file, where every signal carries both outcomes
(`r_*` = 5m-resolved, `r1h_*` = 1H-resolved) computed in the same pass. So the two
columns are paired by construction — no join, and therefore no risk of pairing a trade
against a different config's outcome. (An earlier version joined two separately built
universes on (symbol, entry_time, side); that key is NOT unique, because a symbol can
fire two configs on the same bar and side with different RR.)

`resolve_1h` here was verified equal to build_trail_universe_wide.resolve_all on
97,130/97,130 variant-outcomes, so the 1H column is the same convention that produced
the originally reported figures.

What differs between the columns, and only this:
  1H : a candle touching both stop and take-profit is scored as the STOP
  5m : that hour is walked in twelve pieces, so the real order is known
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

from build_trail_universe_5m import VARIANTS

START = 1500.0
BASE_RT = 2 * (0.0006 + 0.0003)
UNI = ROOT / "trail_5m_universe.parquet"
OOS = pd.Timestamp("2026-05-25")
END = pd.Timestamp("2026-07-25")
NAMES = [v[0] for v in VARIANTS]

_P = None
_CHOP = None
_D = None


def _init():
    global _P, _CHOP, _D
    import backtest_production_correct as P
    from backtest_shadow_gate import LIVE as LIVE_KW
    _P = (P, LIVE_KW)
    _D = pd.read_parquet(UNI)
    _D["entry_time"] = pd.to_datetime(_D["entry_time"])
    _CHOP = P.load_chop_data(sorted(_D.symbol.unique()))


def _frame(d, name, res):
    pre = "r_" if res == "5m" else "r1h_"
    xpre = "x_" if res == "5m" else "x1h_"
    out = d[["entry_time", "entry_price", "sl_price", "side", "symbol",
             "btc_bull", "btc_impulse"]].copy()
    out["exit_time"] = pd.to_datetime(d[f"{xpre}{name}"])
    out["r_result"] = d[f"{pre}{name}"]
    return out.dropna(subset=["exit_time", "r_result"])


def _run(res, name, t0, t1, cost):
    P, LIVE_KW = _P
    d = _D[(_D.entry_time >= t0) & (_D.entry_time < t1)]
    if len(d) < 20:
        return None
    sub = _frame(d, name, res)
    P.ROUND_TRIP_COST = BASE_RT * cost
    P.STARTING_BALANCE = START
    kw = dict(LIVE_KW, starting_balance=START,
              btc_bull_col="btc_bull", btc_short_col="btc_impulse")
    r = P.run_simulation(sub.sort_values("entry_time").reset_index(drop=True), _CHOP, **kw)
    t = r["entered_trades"]
    if not t:
        return None
    gp = sum(x["pnl"] for x in t if x["pnl"] > 0)
    gl = abs(sum(x["pnl"] for x in t if x["pnl"] < 0))
    wins = sum(1 for x in t if x["pnl"] > 0)
    return dict(final=r["final_effective"], dd=r["max_dd_pct"], n=len(t),
                wr=wins / len(t), pf=(gp / gl) if gl else float("inf"))


def job(a):
    k, res, name, t0, t1, cost = a
    try:
        return (k, res, name), _run(res, name, t0, t1, cost)
    except Exception:
        return (k, res, name), None


def table(res_d, k, title):
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)
    print(f"  {'exit rule':<12}{'1H  final $':>14}{'5m  final $':>14}{'diff $':>12}"
          f"{'diff %':>9}{'1H DD':>8}{'5m DD':>8}{'5m PF':>7}")
    rows = [(nm, res_d.get((k, "1h", nm)), res_d.get((k, "5m", nm))) for nm in NAMES]
    rows = [r for r in rows if r[1] and r[2]]
    rows.sort(key=lambda x: -x[2]["final"])
    for nm, a, b in rows:
        mark = " *" if nm == "base" else ""
        print(f"  {nm + mark:<12}{a['final']:>14,.0f}{b['final']:>14,.0f}"
              f"{b['final'] - a['final']:>+12,.0f}{b['final'] / a['final'] - 1:>+9.0%}"
              f"{a['dd']:>7.1f}%{b['dd']:>7.1f}%{b['pf']:>7.2f}")
    print(f"  {'never trade':<12}{START:>14,.0f}{START:>14,.0f}{0:>+12,.0f}"
          f"{0.0:>9.0%}{0.0:>7.1f}%{0.0:>7.1f}%{'—':>7}")


def main():
    d = pd.read_parquet(UNI)
    d["entry_time"] = pd.to_datetime(d["entry_time"])
    print(f"[5M-$] {len(d):,} signals · {d.symbol.nunique()} symbols · "
          f"{d.entry_time.min():%Y-%m-%d} .. {d.entry_time.max():%Y-%m-%d}")
    print("[5M-$] both resolutions computed per signal in one pass — no join\n",
          flush=True)

    # ---- integrity: 5m can never detect an exit LATER than 1H does ----
    late = 0
    for nm in NAMES:
        a = pd.to_datetime(d[f"x1h_{nm}"]).to_numpy()
        b = pd.to_datetime(d[f"x_{nm}"]).to_numpy()
        late += int((b > a).sum())
    print(f"[CHECK] 5m exits later than the 1H exit: {late}  (must be 0)\n")

    print("=" * 100)
    print("PER-TRADE EFFECT OF FINER RESOLUTION")
    print("=" * 100)
    print(f"  {'exit rule':<12}{'trades':>9}{'changed':>10}{'% chg':>8}"
          f"{'1H mean R':>11}{'5m mean R':>11}{'delta R':>10}{'1H WR':>8}{'5m WR':>8}")
    for nm in NAMES:
        x = d[f"r1h_{nm}"].to_numpy(float)
        y = d[f"r_{nm}"].to_numpy(float)
        m = np.isfinite(x) & np.isfinite(y)
        chg = int((np.abs(x[m] - y[m]) > 1e-9).sum())
        print(f"  {nm:<12}{m.sum():>9,}{chg:>10,}{chg / max(m.sum(),1):>8.2%}"
              f"{x[m].mean():>+11.4f}{y[m].mean():>+11.4f}"
              f"{y[m].mean() - x[m].mean():>+10.4f}"
              f"{(x[m] > 0).mean():>8.1%}{(y[m] > 0).mean():>8.1%}")

    periods = {
        "full": (d.entry_time.min(), END),
        "15mo": (END - pd.DateOffset(months=15), END),
        "6mo": (END - pd.DateOffset(months=6), END),
        "oos": (OOS, END),
    }
    jobs = [(k, res, nm, t0, t1, 1.0)
            for k, (t0, t1) in periods.items()
            for res in ("1h", "5m") for nm in NAMES]
    for nm in NAMES:
        jobs.append(("c15", "5m", nm, *periods["15mo"], 1.5))
        jobs.append(("c20", "5m", nm, *periods["15mo"], 2.0))

    print(f"\n[5M-$] {len(jobs)} simulations...", flush=True)
    out = {}
    with mp.Pool(max(1, mp.cpu_count() - 1), initializer=_init) as pool:
        for key, r in pool.imap_unordered(job, jobs, chunksize=1):
            if r:
                out[key] = r

    table(out, "full", f"FULL 5m WINDOW ({d.entry_time.min():%Y-%m} → 2026-07) "
                       f"— $1,500 start, live config + protections")
    table(out, "15mo", "LAST 15 MONTHS — $1,500 start")
    table(out, "6mo", "LAST 6 MONTHS — $1,500 start")
    table(out, "oos", "CLEAN OUT-OF-SAMPLE (2026-05-25 →) — no fitting touched this")

    print("\n" + "=" * 100)
    print("COST SENSITIVITY at 5m resolution — 15 months, $1,500")
    print("=" * 100)
    print(f"  {'exit rule':<12}{'1.0x $':>14}{'1.5x $':>14}{'2.0x $':>14}")
    order = sorted(NAMES, key=lambda n: -(out.get(("15mo", "5m", n), {}).get("final", 0)))
    for nm in order:
        r1 = out.get(("15mo", "5m", nm))
        if not r1:
            continue
        mark = " *" if nm == "base" else ""
        print(f"  {nm + mark:<12}{r1['final']:>14,.0f}"
              f"{(out.get(('c15','5m',nm)) or {}).get('final', 0):>14,.0f}"
              f"{(out.get(('c20','5m',nm)) or {}).get('final', 0):>14,.0f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
