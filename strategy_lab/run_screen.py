#!/usr/bin/env python3
"""Tier-1 screen: score every pre-registered candidate against its own placebo.

Protocol: STRATEGY_SEARCH_PROTOCOL.md (committed 2026-07-27, before any result existed).

DESIGN period only. The HOLDOUT (2026-02-01 onward) is not touched by this script at all —
it is filtered out before any strategy sees a bar.

Tier-1 kill rules (pre-registered):
  - fails to beat its own random-entry placebo at t > 2
  - fewer than 300 trades (untestable)
  - avg R net of costs below -0.05
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from strategy_lab.contract import assert_causal
from strategy_lab.execution import replay, placebo_signals
from strategy_lab.metrics import stats, welch_t, bootstrap_ci, breakeven_wr
from strategy_lab.strategies import (DivergenceBOS, Donchian, DoubleDivergence,
                                     BoomHunter, EMAStackPullback, OscillatorDivergence)

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "cache_3yr_1h"
DESIGN_START = pd.Timestamp("2023-06-01")
HOLDOUT_START = pd.Timestamp("2026-02-01")        # FIREWALL — nothing past this is read

# pre-registered grids (subset run in tier 1: one representative cell per strategy)
CANDIDATES = [
    ("C2_divergence_bos",     DivergenceBOS,          dict(atr_mult=1.5, rr=5.0)),
    ("S1_donchian_96",        Donchian,               dict(channel=96, atr_mult=1.5, rr=8.0)),
    ("S1_donchian_48",        Donchian,               dict(channel=48, atr_mult=1.5, rr=8.0)),
    ("S1_donchian_168",       Donchian,               dict(channel=168, atr_mult=1.5, rr=8.0)),
    ("S2_double_div_obv",     DoubleDivergence,       dict(atr_mult=1.5, rr=5.0, second="obv")),
    ("S2_double_div_macd",    DoubleDivergence,       dict(atr_mult=1.5, rr=5.0, second="macdh")),
    ("S3_boom_hunter_all",    BoomHunter,             dict(atr_mult=1.5, rr=3.0, tier="all")),
    ("S3_boom_hunter_lime",   BoomHunter,             dict(atr_mult=1.5, rr=3.0, tier="lime")),
    ("S4_ema_stack_pullback", EMAStackPullback,       dict(rsi_trigger=35, atr_mult=1.5, rr=4.5)),
    ("S8_macd_div",           lambda: OscillatorDivergence("macdh"), dict(atr_mult=1.5, rr=5.0)),
    ("S9_obv_div",            lambda: OscillatorDivergence("obv"),   dict(atr_mult=1.5, rr=5.0)),
    ("S10_stoch_div",         lambda: OscillatorDivergence("stoch"), dict(atr_mult=1.5, rr=5.0)),
]


def _load(sym):
    f = CACHE / f"{sym}.parquet"
    if not f.exists():
        return None
    try:
        df = pd.read_parquet(f)
    except Exception:
        return None
    if df.empty or "start" not in df.columns:
        return None
    df = df.sort_values("start").reset_index(drop=True)
    # FIREWALL: cut the holdout off before any strategy sees it
    df = df[(df.start >= DESIGN_START) & (df.start < HOLDOUT_START)].reset_index(drop=True)
    return df if len(df) >= 1500 else None


def work(args):
    sym, cost_mult = args
    df = _load(sym)
    if df is None:
        return {}
    out = {}
    for name, factory, params in CANDIDATES:
        try:
            strat = factory() if not isinstance(factory, type) else factory()
            prepared = strat.prepare(df)
            sigs = strat.detect(prepared, **params)
            if not sigs:
                continue
            real = replay(prepared, sigs, sym, cost_mult=cost_mult)
            am = float(params.get('atr_mult', 1.5))
            pl = replay(prepared, placebo_signals(prepared, sigs,
                        seed=abs(hash(sym)) % 10**6, atr_mult=am),
                        sym, cost_mult=cost_mult)
            out[name] = ([t["r_net"] for t in real], [t["r_net"] for t in pl])
        except Exception:
            continue
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=100, help="symbols (tier-1 uses 100)")
    ap.add_argument("--cost-mult", type=float, default=1.0)
    ap.add_argument("--workers", type=int, default=max(1, mp.cpu_count() - 1))
    ap.add_argument("--out", default="tier1_screen.csv")
    a = ap.parse_args()

    syms = sorted(p.stem for p in CACHE.glob("*.parquet"))
    syms = [s for s in syms if not s.endswith(("26JUN26", "03APR26", "10APR26", "17APR26"))]
    # stratified: every k-th symbol, so it isn't just alphabetical-A liquidity bias
    if a.limit and a.limit < len(syms):
        step = len(syms) / a.limit
        syms = [syms[int(i * step)] for i in range(a.limit)]

    print(f"[TIER1] {len(syms)} symbols · {len(CANDIDATES)} candidates · "
          f"cost x{a.cost_mult}")
    print(f"[TIER1] DESIGN {DESIGN_START.date()} .. {HOLDOUT_START.date()} "
          f"(holdout firewalled out at load time)\n", flush=True)

    agg = {name: ([], []) for name, _, _ in CANDIDATES}
    with mp.Pool(a.workers) as pool:
        for i, res in enumerate(pool.imap_unordered(
                work, [(s, a.cost_mult) for s in syms], chunksize=2), 1):
            for k, (r, p) in res.items():
                agg[k][0].extend(r)
                agg[k][1].extend(p)
            if i % 20 == 0:
                print(f"  {i}/{len(syms)} symbols", flush=True)

    rows = []
    print(f"\n{'=' * 108}")
    print(f"TIER-1 SCREEN — every candidate vs its OWN random-entry placebo")
    print(f"{'=' * 108}")
    print(f"  {'strategy':<24}{'trades':>8}{'WR':>8}{'avgR':>9}{'PF':>7}"
          f"{'placebo avgR':>14}{'delta':>9}{'t':>7}  verdict")
    for name, _, params in CANDIDATES:
        r, p = np.array(agg[name][0]), np.array(agg[name][1])
        s = stats(r)
        sp = stats(p)
        t = welch_t(r, p)
        delta = s["avg"] - sp["avg"]
        # pre-registered tier-1 kill rules
        fail = []
        if s["n"] < 300:
            fail.append("n<300")
        if t <= 2.0:
            fail.append("placebo t<=2")
        if s["avg"] < -0.05:
            fail.append("avgR<-0.05")
        verdict = "PASS -> tier2" if not fail else "kill: " + ",".join(fail)
        pf = "inf" if s["pf"] == float("inf") else f"{s['pf']:.2f}"
        print(f"  {name:<24}{s['n']:>8,}{s['wr']:>8.2%}{s['avg']:>+9.4f}{pf:>7}"
              f"{sp['avg']:>+14.4f}{delta:>+9.4f}{t:>7.2f}  {verdict}")
        rows.append(dict(strategy=name, n=s["n"], wr=s["wr"], avg_r=s["avg"],
                         pf=s["pf"], sharpe=s["sharpe"], max_dd=s["max_dd"],
                         placebo_avg=sp["avg"], placebo_n=sp["n"],
                         delta=delta, t=t, passed=not fail,
                         fail_reasons=";".join(fail), **params))
    pd.DataFrame(rows).to_csv(ROOT / a.out, index=False)
    print(f"\n[TIER1] -> {a.out}")
    n_pass = sum(1 for r in rows if r["passed"])
    print(f"[TIER1] {n_pass}/{len(rows)} candidates advance to Tier 2")
    return 0


if __name__ == "__main__":
    sys.exit(main())
