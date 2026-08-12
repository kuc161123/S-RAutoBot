#!/usr/bin/env python3
"""Month-by-month equity from $1,500 — live config vs a single global (rr, atr_mult).

All arms run the SAME live machinery: s3_a1 trailing exit, honest ch[bos] CHOP gate,
24.2 bps measured cost, 0.3% base risk, live taper, regime multiplier, wallet-taper /
equity-sizing mismatch, net-directional cap 0.10, gross open-risk cap 0.30, BTC short gate,
long-bull boost 1.3, anti-pyramid, margin, funding. Only the (rr, atr_mult) assignment and
the traded (symbol, div_type) set differ between arms.

One continuous compounding run per arm across the whole history — NOT a fresh $1,500 each
month — because the question is what the account would have done, and that is path
dependent.
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

START = 1500.0
COST = 0.00242
F = 0.003

ARMS = [
    ("A0-live   fitted per-symbol rr/atr", "universe_trail.parquet", "r_trail", "exit_trail"),
    ("A1-rr10a3 global rr=10 atr=3.0",     "uni_glob_rr10_am3_same.parquet", "r_result", "exit_time"),
    ("A1-rr10a2 global rr=10 atr=2.0",     "uni_glob_rr10_am2_same.parquet", "r_result", "exit_time"),
    ("A1-rr8a2  global rr=8  atr=2.0",     "uni_glob_rr8_am2_same.parquet",  "r_result", "exit_time"),
    ("A1-rr3a2  global rr=3  atr=2.0",     "uni_glob_rr3_am2_same.parquet",  "r_result", "exit_time"),
]


def curve(trades):
    t = pd.DataFrame(trades).sort_values("exit_time")
    return pd.Series((START + t["pnl"].cumsum()).values,
                     index=pd.to_datetime(t["exit_time"].values))


def run(path, rcol, xcol, chop):
    d = pd.read_parquet(HERE / path).copy()
    d["r_result"] = d[rcol]
    d["exit_time"] = d[xcol]
    d = d.dropna(subset=["r_result", "exit_time"]).sort_values("entry_time").reset_index(drop=True)
    cols = ["entry_time", "exit_time", "entry_price", "sl_price", "r_result",
            "side", "symbol", "btc_bull", "btc_impulse"]
    kw = dict(LIVE_SIM)
    kw.update(starting_balance=START, base_risk=F, custom_taper=LIVE_TAPER,
              btc_bull_col="btc_bull", btc_short_col="btc_impulse")
    P.STARTING_BALANCE = START
    res = P.run_simulation(d[cols], chop, **kw)
    return res, d


def main():
    P.ROUND_TRIP_COST = COST
    uni = pd.read_parquet(HERE / "universe_trail.parquet")
    chop = P.load_chop_data(sorted(uni.symbol.unique()))

    monthly = {}
    summary = []
    for name, path, rcol, xcol in ARMS:
        if not (HERE / path).exists():
            print(f"skip {name}: {path} missing")
            continue
        res, d = run(path, rcol, xcol, chop)
        tr = res["entered_trades"]
        c = curve(tr)
        m = c.resample("ME").last().ffill()
        m.loc[pd.Timestamp("2023-05-31")] = START
        m = m.sort_index()
        monthly[name] = m

        peak = c.cummax()
        dd = float(((peak - c) / peak).max() * 100)
        final = res["final_effective"]
        summary.append(dict(arm=name, n=len(tr), final=final,
                            roi=final / START - 1, maxdd=dd,
                            wr=np.mean([t["r_result"] > 0 for t in tr]) * 100,
                            meanR=np.mean([t["r_result"] for t in tr])))
        print(f"{name:<44} n={len(tr):>6,}  final ${final:>12,.0f}  maxDD {dd:5.1f}%")

    if not monthly:
        return
    M = pd.DataFrame(monthly)
    M.index = M.index.to_period("M")
    M = M[~M.index.duplicated(keep="last")]

    print()
    print("=" * 108)
    print(f"MONTH-END EQUITY FROM ${START:,.0f}  ·  0.3% risk  ·  live trail  ·  24.2 bps")
    print("=" * 108)
    hdr = f"{'month':<9}" + "".join(f"{c.split()[0]:>13}" for c in M.columns) + "   regime"
    print(hdr)
    for i, (idx, row) in enumerate(M.iterrows()):
        if idx < pd.Period("2023-06"):
            continue
        tag = ""
        if idx >= pd.Period("2026-06"):
            tag = "HOLDOUT (untouched)"
        elif idx >= pd.Period("2025-07"):
            tag = "walk-forward TEST"
        else:
            tag = "walk-forward TRAIN"
        print(f"{str(idx):<9}" + "".join(f"{v:>13,.0f}" for v in row.values) + f"   {tag}")

    M.to_csv(HERE / "results" / "monthly_equity.csv")
    pd.DataFrame(summary).to_csv(HERE / "results" / "monthly_summary.csv", index=False)

    print()
    print("=" * 108)
    print("MONTHLY RETURN %  (month-over-month on the compounding balance)")
    print("=" * 108)
    R = M.pct_change() * 100
    print(f"{'':<9}" + "".join(f"{c.split()[0]:>13}" for c in M.columns))
    for lbl, fn in (("months +", lambda s: (s > 0).sum()),
                    ("months -", lambda s: (s < 0).sum()),
                    ("best %", lambda s: s.max()),
                    ("worst %", lambda s: s.min()),
                    ("median %", lambda s: s.median())):
        print(f"{lbl:<9}" + "".join(f"{fn(R[c].dropna()):>13.1f}" for c in M.columns))

    print()
    print("HOLDOUT ONLY (2026-06 onward, genuinely untouched)")
    H = M[M.index >= pd.Period("2026-05")]
    for c in M.columns:
        v = H[c].dropna()
        if len(v) >= 2:
            print(f"  {c:<44} ${v.iloc[0]:>9,.0f} -> ${v.iloc[-1]:>9,.0f}   "
                  f"{(v.iloc[-1]/v.iloc[0]-1)*100:+7.1f}%")


if __name__ == "__main__":
    main()
