#!/usr/bin/env python3
"""RISK-PER-TRADE SWEEP — the core harness for this study.

Holds entries, exits, stops, filters and RR constant. Varies ONE thing: base risk per
trade. Everything else is the live config (`backtest_shadow_gate.LIVE`), driven through
the production engine `backtest_production_correct.run_simulation`.

Two corrections applied vs. the repo default, both from STRATEGY_VERDICT_2026-08-11.md:
  * cost:  ROUND_TRIP_COST 18 bps -> 34.1 bps (S2.3, calibrated to live /trailstats n=195)
  * CHOP:  universe built on ch[bos], engine's lookup_chop steps back one bar (S2.1)

The taper schedule is scaled proportionally with base risk (k = f / 0.003) so each arm is
the SAME risk POLICY at a different scale, not a policy change. `--flat-taper` keeps the
live schedule fixed and moves only the base, as a sensitivity.

Usage:
  python3 risk_study/sweep.py --window all --out risk_study/results
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import backtest_production_correct as P  # noqa: E402
from backtest_shadow_gate import LIVE as LIVE_SIM, LIVE_TAPER  # noqa: E402

HERE = Path(__file__).resolve().parent
START_BAL = 1_500.0             # the live account's own scale -- the taper rungs are in
                                # DOLLARS, so starting balance changes which rung is active
                                # and is NOT a scale-free choice. $1,500 sits exactly on the
                                # first rung; at $10,000 the taper has already cut base risk
                                # from 0.30% to 0.22% before the sweep does anything.
TRUE_COST = 0.00242             # 24.2 bps round trip, measured from 133 live stop-outs
BASE_F = 0.003                  # the live setting

RISK_GRID = [0.0025, 0.005, 0.0075, 0.010, 0.0125, 0.015, 0.0175, 0.020, 0.025, 0.030]
# the live setting is not on the brief's grid -- add it, plus two below it
RISK_GRID = sorted(set([0.001, 0.002, 0.003] + RISK_GRID))

WINDOWS = {
    # name:            (start, end, contamination status)
    "DEV":      ("2023-06-01", "2025-07-01", "in-sample (walk-forward TRAIN)"),
    "VAL":      ("2025-07-01", "2026-05-25", "walk-forward TEST — config accepted on it"),
    "HOLDOUT":  ("2026-05-25", "2026-07-26", "genuinely untouched"),
    "FULL":     ("2023-06-01", "2026-07-26", "mixed"),
}


# ── metrics ────────────────────────────────────────────────────────────────────

def equity_curve(trades, start_bal):
    """Realized equity curve in exit order. The engine books PnL at exit."""
    if not trades:
        return pd.Series(dtype=float)
    t = pd.DataFrame(trades).sort_values("exit_time")
    return pd.Series((start_bal + t["pnl"].cumsum()).values,
                     index=pd.to_datetime(t["exit_time"].values))


def max_dd(curve, start_bal):
    """Peak-to-trough on the realized curve, seeded at the starting balance."""
    if curve.empty:
        return 0.0, 0.0
    s = pd.concat([pd.Series([start_bal], index=[curve.index[0] - pd.Timedelta(hours=1)]),
                   curve])
    peak = s.cummax()
    dd = (peak - s) / peak
    return float(dd.max() * 100), float((peak - s).max())


def underwater(curve, start_bal):
    """Longest stretch below a prior peak, and total fraction of time underwater."""
    if curve.empty:
        return 0.0, 0.0
    s = pd.concat([pd.Series([start_bal], index=[curve.index[0] - pd.Timedelta(hours=1)]),
                   curve])
    peak = s.cummax()
    uw = s < peak
    if not uw.any():
        return 0.0, 0.0
    longest = cur = 0.0
    start = None
    for ts, flag in uw.items():
        if flag and start is None:
            start = ts
        elif not flag and start is not None:
            cur = (ts - start).total_seconds() / 86400
            longest = max(longest, cur)
            start = None
    if start is not None:
        longest = max(longest, (uw.index[-1] - start).total_seconds() / 86400)
    return longest, float(uw.mean())


def max_consec_losses(trades):
    best = cur = 0
    for t in sorted(trades, key=lambda x: x["exit_time"]):
        if t["pnl"] <= 0:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best


def ratios(curve, start_bal, years):
    """Sharpe / Sortino on DAILY returns of the realized curve. Rf = 0."""
    if curve.empty or years <= 0:
        return 0.0, 0.0
    daily = curve.resample("1D").last().ffill()
    daily = pd.concat([pd.Series([start_bal], index=[daily.index[0] - pd.Timedelta(days=1)]),
                       daily])
    r = daily.pct_change().dropna()
    if len(r) < 5 or r.std() == 0:
        return 0.0, 0.0
    sharpe = r.mean() / r.std() * np.sqrt(365)
    down = r[r < 0]
    sortino = (r.mean() / down.std() * np.sqrt(365)) if len(down) > 1 and down.std() > 0 else np.nan
    return float(sharpe), float(sortino)


def summarize(res, start_bal, t0, t1, f, label):
    tr = res["entered_trades"]
    n = len(tr)
    years = (pd.Timestamp(t1) - pd.Timestamp(t0)).days / 365.25
    final = res["final_effective"]
    roi = final / start_bal - 1.0
    curve = equity_curve(tr, start_bal)
    dd_pct, dd_usd = max_dd(curve, start_bal)
    uw_days, uw_frac = underwater(curve, start_bal)
    sharpe, sortino = ratios(curve, start_bal, years)

    # CAGR: guard the wipeout case -- a negative ending balance has no real CAGR.
    if final <= 0:
        cagr = -1.0
    else:
        cagr = (final / start_bal) ** (1 / years) - 1 if years > 0 else 0.0

    pnls = [t["pnl"] for t in tr]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    gp, gl = sum(wins), abs(sum(losses))

    return {
        "window": label, "risk_pct": f * 100, "n_trades": n,
        "start_balance": start_bal, "final_balance": final,
        "net_roi_pct": roi * 100, "cagr_pct": cagr * 100,
        "max_dd_pct": dd_pct, "max_dd_usd": dd_usd,
        "max_dd_mtm_pct": res["max_dd_mtm_pct"],
        "roi_dd_ratio": (roi * 100 / dd_pct) if dd_pct > 1e-9 else np.nan,
        "calmar": (cagr * 100 / dd_pct) if dd_pct > 1e-9 else np.nan,
        "sharpe": sharpe, "sortino": sortino,
        "win_rate_pct": (len(wins) / n * 100) if n else 0.0,
        "profit_factor": (gp / gl) if gl > 0 else np.nan,
        "expectancy_r": float(np.mean([t["r_result"] for t in tr])) if n else 0.0,
        "avg_win_usd": float(np.mean(wins)) if wins else 0.0,
        "avg_loss_usd": float(np.mean(losses)) if losses else 0.0,
        "largest_loss_usd": float(min(pnls)) if pnls else 0.0,
        "largest_loss_pct_of_start": (float(min(pnls)) / start_bal * 100) if pnls else 0.0,
        "max_consec_losses": max_consec_losses(tr),
        "longest_underwater_days": uw_days, "time_underwater_frac": uw_frac,
        "margin_blocked": res["margin_blocked"],
        "risk_capped": res["risk_capped"],
        "pyramid_blocked": res["pyramid_blocked"],
        "years": years,
    }


# ── the sweep ──────────────────────────────────────────────────────────────────

def scaled_taper(f, flat=False):
    if flat:
        return list(LIVE_TAPER)
    k = f / BASE_F
    return [(bal, rate * k) for bal, rate in LIVE_TAPER]


def run_one(uni, chop, f, t0, t1, label, start_bal=START_BAL, flat_taper=False,
            r_col="r_result", exit_col="exit_time", **over):
    sub = uni[(uni.entry_time >= t0) & (uni.entry_time < t1)].copy()
    # Let the caller pick which exit rule the run uses. The live bot has trailed since
    # 2026-08-02, so `r_trail`/`exit_trail` is the deployed arm; `r_fixed` is what the
    # strategy did before that date and what every older study in this repo measured.
    if r_col != "r_result":
        sub["r_result"] = sub[r_col]
    if exit_col != "exit_time":
        sub["exit_time"] = sub[exit_col]
    sub = sub.dropna(subset=["r_result", "exit_time"])
    sub = sub.sort_values("entry_time").reset_index(drop=True)
    if sub.empty:
        return None
    cols = ["entry_time", "exit_time", "entry_price", "sl_price", "r_result",
            "side", "symbol", "btc_bull", "btc_impulse"]
    kw = dict(LIVE_SIM)
    kw.update(starting_balance=start_bal, base_risk=f,
              custom_taper=scaled_taper(f, flat_taper),
              btc_bull_col="btc_bull", btc_short_col="btc_impulse")
    kw.update(over)
    P.STARTING_BALANCE = start_bal
    res = P.run_simulation(sub[cols], chop, **kw)
    return summarize(res, start_bal, t0, t1, f, label)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--universe", default=str(HERE / "universe_chopBOS.parquet"))
    ap.add_argument("--out", default=str(HERE / "results"))
    ap.add_argument("--cost", type=float, default=TRUE_COST)
    ap.add_argument("--flat-taper", action="store_true")
    ap.add_argument("--tag", default="main")
    ap.add_argument("--start-balance", type=float, default=START_BAL)
    ap.add_argument("--exit", dest="exit_rule", choices=["fixed", "trail", "legacy"],
                    default="legacy",
                    help="legacy = the r_result column; fixed/trail = the two arms of "
                         "universe_trail.parquet")
    a = ap.parse_args()
    rcol, xcol = {"legacy": ("r_result", "exit_time"),
                  "fixed": ("r_fixed", "exit_fixed"),
                  "trail": ("r_trail", "exit_trail")}[a.exit_rule]

    P.ROUND_TRIP_COST = a.cost
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    uni = pd.read_parquet(a.universe)
    print(f"[SWEEP] universe {Path(a.universe).name}  {len(uni):,} trades  "
          f"cost {a.cost*1e4:.1f} bps  exit={a.exit_rule}  "
          f"start=${a.start_balance:,.0f}  taper={'flat' if a.flat_taper else 'scaled'}")
    chop = P.load_chop_data(sorted(uni.symbol.unique()))

    rows = []
    for label, (t0, t1, note) in WINDOWS.items():
        print(f"\n--- {label}  {t0}..{t1}  [{note}] ---")
        print(f"  {'risk%':>6} {'n':>6} {'final$':>12} {'ROI%':>9} {'maxDD%':>7} "
              f"{'ROI/DD':>7} {'Calmar':>7} {'Sharpe':>7} {'PF':>6}")
        for f in RISK_GRID:
            r = run_one(uni, chop, f, t0, t1, label, start_bal=a.start_balance,
                        flat_taper=a.flat_taper, r_col=rcol, exit_col=xcol)
            if r is None:
                continue
            r["note"] = note
            rows.append(r)
            print(f"  {r['risk_pct']:>6.2f} {r['n_trades']:>6d} {r['final_balance']:>12,.0f} "
                  f"{r['net_roi_pct']:>9.1f} {r['max_dd_pct']:>7.1f} "
                  f"{r['roi_dd_ratio']:>7.2f} {r['calmar']:>7.2f} "
                  f"{r['sharpe']:>7.2f} {r['profit_factor']:>6.2f}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(out / f"sweep_{a.tag}.csv", index=False)
    print(f"\n[SWEEP] wrote {out / f'sweep_{a.tag}.csv'}  ({len(df)} rows)")


if __name__ == "__main__":
    main()
