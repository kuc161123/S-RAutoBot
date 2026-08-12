#!/usr/bin/env python3
"""Six bot variants, $1,500 start, month by month.

The point is to compare the ACTUAL deployment histories against the candidate, not just
two abstract arms:

  OLD-BASE   fitted per-symbol (rr, atr_mult) + FIXED take-profit
             = what the bot did before 2026-08-02
  CURRENT    fitted per-symbol (rr, atr_mult) + s3_a1 TRAIL
             = what is deployed today
  GLOBAL     rr=10 atr_mult=3.0 for every pair + TRAIL          <- the candidate
  GLOBAL-FIX rr=10 atr_mult=3.0 + FIXED take-profit             <- is the trail load-bearing?
  GLB-10/2   rr=10 atr_mult=2.0 + TRAIL                          <- parameter robustness
  GLB-8/2    rr=8  atr_mult=2.0 + TRAIL                          <- parameter robustness

Identical in every other respect: same (symbol, div_type) pairs, same signals, same honest
ch[bos] CHOP gate, same 24.2 bps measured cost, same 0.3% base risk, live taper, regime
multiplier, wallet-taper/equity-sizing mismatch, net-directional cap, gross open-risk cap,
BTC short gate, long-bull boost, anti-pyramid, margin, funding, $50k withdrawal.

Also runs a START-DATE ROBUSTNESS pass: the same six arms begun at eight different months,
because a single continuous compounding run is path dependent and one lucky early month can
carry a three-year result.
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

import backtest_production_correct as P  # noqa: E402
from backtest_shadow_gate import LIVE as LIVE_SIM, LIVE_TAPER  # noqa: E402

START = 1500.0
COST = 0.00242
F = 0.003

ARMS = [
    ("OLD-BASE  fitted + fixed TP", "universe_trail.parquet", "r_fixed", "exit_fixed"),
    ("CURRENT   fitted + trail",    "universe_trail.parquet", "r_trail", "exit_trail"),
    ("GLOBAL    rr10/a3 + trail",   "uni_glob_rr10_am3_same.parquet", "r_result", "exit_time"),
    ("GLOBAL-FIX rr10/a3 + fixed",  "uni_globfix_rr10_am3_same.parquet", "r_result", "exit_time"),
    ("GLB-10/2  rr10/a2 + trail",   "uni_glob_rr10_am2_same.parquet", "r_result", "exit_time"),
    ("GLB-8/2   rr8/a2  + trail",   "uni_glob_rr8_am2_same.parquet", "r_result", "exit_time"),
]


def load(path, rcol, xcol):
    d = pd.read_parquet(HERE / path).copy()
    d["r_result"] = d[rcol]
    d["exit_time"] = d[xcol]
    return d.dropna(subset=["r_result", "exit_time"]).sort_values("entry_time").reset_index(drop=True)


def sim(d, chop, t0=None, t1=None, start=START):
    if t0 is not None:
        d = d[d.entry_time >= t0]
    if t1 is not None:
        d = d[d.entry_time < t1]
    if d.empty:
        return None
    cols = ["entry_time", "exit_time", "entry_price", "sl_price", "r_result",
            "side", "symbol", "btc_bull", "btc_impulse"]
    kw = dict(LIVE_SIM)
    kw.update(starting_balance=start, base_risk=F, custom_taper=LIVE_TAPER,
              btc_bull_col="btc_bull", btc_short_col="btc_impulse")
    P.STARTING_BALANCE = start
    r = P.run_simulation(d[cols].reset_index(drop=True), chop, **kw)
    t = pd.DataFrame(r["entered_trades"])
    if t.empty:
        return None
    t = t.sort_values("exit_time")
    c = pd.Series((start + t.pnl.cumsum()).values, index=pd.to_datetime(t.exit_time.values))
    pk = c.cummax()
    return dict(final=r["final_effective"], dd=float(((pk - c) / pk).max() * 100),
                n=len(t), curve=c,
                wr=float((t.r_result > 0).mean() * 100),
                meanR=float(t.r_result.mean()))


def main():
    P.ROUND_TRIP_COST = COST
    uni = pd.read_parquet(HERE / "universe_trail.parquet")
    chop = P.load_chop_data(sorted(uni.symbol.unique()))

    data = {nm: load(p, rc, xc) for nm, p, rc, xc in ARMS}

    # ---------------- full continuous run ----------------
    res, curves = {}, {}
    print("=" * 104)
    print(f"FULL RUN · ${START:,.0f} start · 2023-06-01 -> 2026-07-25 · 0.3% risk · 24.2 bps")
    print("=" * 104)
    print(f"{'arm':<30}{'trades':>8}{'final $':>12}{'ROI%':>10}{'maxDD%':>9}"
          f"{'$/DDpt':>9}{'WR%':>7}{'meanR':>8}")
    for nm, *_ in ARMS:
        r = sim(data[nm], chop)
        res[nm] = r
        curves[nm] = r["curve"]
        print(f"{nm:<30}{r['n']:>8,}{r['final']:>12,.0f}{(r['final']/START-1)*100:>10.0f}"
              f"{r['dd']:>9.1f}{r['final']/r['dd']:>9.0f}{r['wr']:>7.1f}{r['meanR']:>8.3f}")

    M = pd.DataFrame({k: v.resample("ME").last() for k, v in curves.items()})
    M = M.ffill()
    M.index = M.index.to_period("M")
    M.to_csv(HERE / "results" / "compare_monthly.csv")

    print()
    print("=" * 104)
    print("MONTH-END EQUITY")
    print("=" * 104)
    print(f"{'month':<9}" + "".join(f"{k.split()[0][:9]:>12}" for k in M.columns) + "  period")
    for idx, row in M.iterrows():
        tag = ("HOLDOUT" if idx >= pd.Period("2026-06") else
               "wf-test" if idx >= pd.Period("2025-07") else "wf-train")
        print(f"{str(idx):<9}" + "".join(f"{v:>12,.0f}" for v in row.values) + f"  {tag}")

    R = M.pct_change() * 100
    print()
    print("MONTHLY RETURN STATS")
    print(f"{'':<9}" + "".join(f"{k.split()[0][:9]:>12}" for k in M.columns))
    for lbl, fn in (("months +", lambda s: (s > 0).sum()), ("months -", lambda s: (s < 0).sum()),
                    ("median %", lambda s: s.median()), ("mean %", lambda s: s.mean()),
                    ("worst %", lambda s: s.min()), ("best %", lambda s: s.max())):
        print(f"{lbl:<9}" + "".join(f"{fn(R[c].dropna()):>12.1f}" for c in M.columns))

    print()
    print("HOLDOUT (2026-05-31 -> end, untouched)")
    for c in M.columns:
        v = M[c][M.index >= pd.Period("2026-05")].dropna()
        if len(v) >= 2:
            print(f"  {c:<30} ${v.iloc[0]:>10,.0f} -> ${v.iloc[-1]:>10,.0f}"
                  f"   {(v.iloc[-1]/v.iloc[0]-1)*100:>+8.1f}%")

    # ---------------- start-date robustness ----------------
    print()
    print("=" * 104)
    print("START-DATE ROBUSTNESS · each arm restarted at $1,500 on 8 different dates, run to end")
    print("=" * 104)
    starts = ["2023-06-01", "2023-12-01", "2024-06-01", "2024-12-01",
              "2025-01-01", "2025-06-01", "2025-12-01", "2026-01-01"]
    rows = []
    print(f"{'start':<12}" + "".join(f"{k.split()[0][:9]:>12}" for k in M.columns))
    for s in starts:
        line = f"{s:<12}"
        rec = {"start": s}
        for nm, *_ in ARMS:
            r = sim(data[nm], chop, t0=s)
            v = r["final"] if r else np.nan
            rec[nm] = v
            line += f"{v:>12,.0f}"
        rows.append(rec)
        print(line, flush=True)
    sd = pd.DataFrame(rows).set_index("start")
    sd.to_csv(HERE / "results" / "compare_startdates.csv")

    print()
    base = "CURRENT   fitted + trail"
    print(f"how often does each arm beat {base.split()[0]} across the 8 start dates?")
    for nm, *_ in ARMS:
        w = (sd[nm] > sd[base]).sum()
        print(f"  {nm:<30}{w}/8   median ${sd[nm].median():>10,.0f}")


if __name__ == "__main__":
    main()
