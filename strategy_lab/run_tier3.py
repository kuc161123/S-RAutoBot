#!/usr/bin/env python3
"""Tier-3: the full pre-registered gauntlet, including the ONE holdout scoring.

Runs, per surviving candidate:
  1. cost sensitivity at 1x / 2x / 3x        (DESIGN)
  2. per-month avg R, incl. the 16 regime-analogue months (DESIGN)
  3. ONE scoring on the untouched HOLDOUT
  4. dollars end-to-end through run_simulation, $1,500 start
  5. Deflated Sharpe with N = 147 trials, using the CROSS-SECTIONAL sd of trial Sharpes

The holdout is read exactly once, here, at the end. Everything before this point ran with
2026-02-01 onward filtered out at load time.
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
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from strategy_lab.execution import replay, placebo_signals
from strategy_lab.metrics import (stats, welch_t, bootstrap_ci, deflated_sharpe,
                                  expected_max_sharpe)
from strategy_lab.strategies import (DivergenceBOS, Donchian, DoubleDivergence,
                                     OscillatorDivergence)
import backtest_production_correct as P

CACHE = ROOT / "cache_3yr_1h"
DESIGN_START = pd.Timestamp("2023-06-01")
HOLDOUT_START = pd.Timestamp("2026-02-01")
N_TRIALS = 147                      # fixed in STRATEGY_SEARCH_PROTOCOL.md
START_BAL = 1500.0

# the 16 regime-analogue months, fixed in the protocol before results existed
ANALOGUE = {"2023-04", "2023-09", "2023-11", "2024-03", "2024-05", "2024-07", "2024-09",
            "2024-10", "2025-01", "2025-04", "2025-05", "2025-06", "2025-07", "2025-09",
            "2026-03", "2026-04"}

SURVIVORS = [
    ("C2_divergence_bos",  DivergenceBOS,  dict(atr_mult=1.5, rr=5.0)),
    ("S1_donchian_96",     Donchian,       dict(channel=96, atr_mult=1.5, rr=8.0)),
    ("S1_donchian_48",     Donchian,       dict(channel=48, atr_mult=1.5, rr=8.0)),
    ("S1_donchian_168",    Donchian,       dict(channel=168, atr_mult=1.5, rr=8.0)),
    ("S2_double_div_obv",  DoubleDivergence, dict(atr_mult=1.5, rr=5.0, second="obv")),
    ("S2_double_div_macd", DoubleDivergence, dict(atr_mult=1.5, rr=5.0, second="macdh")),
    ("S8_macd_div",  lambda: OscillatorDivergence("macdh"), dict(atr_mult=1.5, rr=5.0)),
    ("S9_obv_div",   lambda: OscillatorDivergence("obv"),   dict(atr_mult=1.5, rr=5.0)),
    ("S10_stoch_div", lambda: OscillatorDivergence("stoch"), dict(atr_mult=1.5, rr=5.0)),
]


def _load(sym, segment):
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
    if segment == "design":
        df = df[(df.start >= DESIGN_START) & (df.start < HOLDOUT_START)]
    else:                                   # holdout — warmup bars kept, entries filtered later
        df = df[df.start >= HOLDOUT_START - pd.Timedelta(days=60)]
    df = df.reset_index(drop=True)
    return df if len(df) >= 1200 else None


def work(args):
    sym, segment, cost_mult = args
    df = _load(sym, segment)
    if df is None:
        return {}
    out = {}
    for name, factory, params in SURVIVORS:
        try:
            s = factory()
            prep = s.prepare(df)
            sigs = s.detect(prep, **params)
            if not sigs:
                continue
            tr = replay(prep, sigs, sym, cost_mult=cost_mult)
            if segment == "holdout":
                tr = [t for t in tr if pd.Timestamp(t["entry_time"]) >= HOLDOUT_START]
            if not tr:
                continue
            am = float(params.get("atr_mult", 1.5))
            pl = replay(prep, placebo_signals(prep, sigs, seed=abs(hash(sym)) % 10**6,
                                              atr_mult=am), sym, cost_mult=cost_mult)
            if segment == "holdout":
                pl = [t for t in pl if pd.Timestamp(t["entry_time"]) >= HOLDOUT_START]
            out[name] = (tr, [t["r_net"] for t in pl])
        except Exception:
            continue
    return out


def gather(syms, segment, cost_mult, workers):
    agg = {n: ([], []) for n, _, _ in SURVIVORS}
    with mp.Pool(workers) as pool:
        for i, res in enumerate(pool.imap_unordered(
                work, [(s, segment, cost_mult) for s in syms], chunksize=4), 1):
            for k, (tr, pl) in res.items():
                agg[k][0].extend(tr)
                agg[k][1].extend(pl)
            if i % 100 == 0:
                print(f"    {i}/{len(syms)}", flush=True)
    return agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=max(1, mp.cpu_count() - 1))
    a = ap.parse_args()
    syms = sorted(p.stem for p in CACHE.glob("*.parquet"))
    syms = [s for s in syms if not s.endswith(("26JUN26", "03APR26", "10APR26", "17APR26"))]

    # ---------- 1+2. DESIGN: cost sensitivity and monthly breakdown ----------
    print(f"[T3] DESIGN pass, cost x1 ({len(syms)} symbols)", flush=True)
    d1 = gather(syms, "design", 1.0, a.workers)
    print(f"[T3] DESIGN pass, cost x2", flush=True)
    d2 = gather(syms, "design", 2.0, a.workers)
    print(f"[T3] DESIGN pass, cost x3", flush=True)
    d3 = gather(syms, "design", 3.0, a.workers)

    # ---------- 3. HOLDOUT — the single scoring ----------
    print(f"[T3] HOLDOUT pass (ONE shot)", flush=True)
    ho = gather(syms, "holdout", 1.0, a.workers)

    rows = []
    sharpes = []
    for name, _, _ in SURVIVORS:
        tr = d1[name][0]
        r = np.array([t["r_net"] for t in tr])
        p = np.array(d1[name][1])
        s = stats(r)
        sharpes.append(s["sharpe"])
    sd_trials = float(np.std(sharpes, ddof=1)) if len(sharpes) > 1 else 1.0

    print(f"\n{'=' * 118}")
    print(f"TIER-3 — full pre-registered gauntlet. N={N_TRIALS} trials, "
          f"sd(trial Sharpes)={sd_trials:.4f}")
    print(f"expected max Sharpe from luck alone = "
          f"{expected_max_sharpe(N_TRIALS, sd_trials):+.4f}")
    print("=" * 118)
    hdr = (f"  {'strategy':<22}{'design avgR':>12}{'x2 cost':>10}{'x3 cost':>10}"
           f"{'analogue+':>11}{'HOLDOUT':>10}{'holdout n':>11}{'DSR':>8}{'plc t':>7}")
    print(hdr)

    for name, _, _ in SURVIVORS:
        tr1 = d1[name][0]
        r1 = np.array([t["r_net"] for t in tr1])
        p1 = np.array(d1[name][1])
        s1 = stats(r1)
        t_pl = welch_t(r1, p1)
        r2 = np.array([t["r_net"] for t in d2[name][0]])
        r3 = np.array([t["r_net"] for t in d3[name][0]])

        # regime-analogue months
        months = {}
        for t in tr1:
            m = pd.Timestamp(t["entry_time"]).strftime("%Y-%m")
            if m in ANALOGUE:
                months.setdefault(m, []).append(t["r_net"])
        pos = sum(1 for v in months.values() if len(v) >= 20 and np.mean(v) > 0)
        tot_m = sum(1 for v in months.values() if len(v) >= 20)
        an_frac = pos / tot_m if tot_m else 0.0

        rh = np.array([t["r_net"] for t in ho[name][0]])
        sh = stats(rh)
        dsr = deflated_sharpe(s1["sharpe"], s1["n"], N_TRIALS, sd_trials)

        print(f"  {name:<22}{s1['avg']:>+12.4f}{np.mean(r2) if len(r2) else 0:>+10.4f}"
              f"{np.mean(r3) if len(r3) else 0:>+10.4f}"
              f"{an_frac:>10.0%} {sh['avg']:>+10.4f}{sh['n']:>11,}{dsr:>8.3f}{t_pl:>7.2f}",
              flush=True)

        rows.append(dict(strategy=name, design_n=s1["n"], design_avg=s1["avg"],
                         design_pf=s1["pf"], design_sharpe=s1["sharpe"],
                         cost2=float(np.mean(r2)) if len(r2) else 0.0,
                         cost3=float(np.mean(r3)) if len(r3) else 0.0,
                         analogue_pos=pos, analogue_tot=tot_m, analogue_frac=an_frac,
                         holdout_n=sh["n"], holdout_avg=sh["avg"], holdout_pf=sh["pf"],
                         dsr=dsr, placebo_t=t_pl))

    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "tier3_results.csv", index=False)

    print(f"\n{'=' * 118}")
    print("PRE-REGISTERED PASS CRITERIA (all seven must hold)")
    print("=" * 118)
    print(f"  {'strategy':<22}{'1 plc t>3':>11}{'2 design>0':>12}{'3 anlg>=60%':>13}"
          f"{'4 hold>0':>10}{'5 DSR>0':>9}{'6 2x cost>0':>13}  overall")
    for _, r in df.iterrows():
        c1 = r.placebo_t > 3
        c2 = r.design_avg > 0
        c3 = r.analogue_frac >= 0.60
        c4 = r.holdout_avg > 0
        c5 = r.dsr > 0.5
        c6 = r.cost2 > 0
        allp = all([c1, c2, c3, c4, c5, c6])
        f = lambda b: " PASS" if b else " fail"
        print(f"  {r.strategy:<22}{f(c1):>11}{f(c2):>12}{f(c3):>13}{f(c4):>10}"
              f"{f(c5):>9}{f(c6):>13}  {'*** CANDIDATE ***' if allp else '-'}")
    print(f"\n[T3] -> tier3_results.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
