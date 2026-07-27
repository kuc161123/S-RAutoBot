#!/usr/bin/env python3
"""Does the harness reproduce numbers we already know? If not, the harness is wrong.

Target (from analyze_strategy_design.py, 514 symbols, design period, rr=5 / atr=1.5):
    REAL     122,077 trades, WR 18.07%, avg R -0.0172
    PLACEBO  244,103 trades, WR 17.42%, avg R -0.0549
    delta    +0.0377 R/trade, t = +4.66

Any material deviation means strategy_lab's execution or placebo differs from the earlier
measurement, and every candidate score would inherit that error.
"""
from __future__ import annotations

import multiprocessing as mp
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from strategy_lab.execution import replay, placebo_signals
from strategy_lab.metrics import stats, welch_t
from strategy_lab.strategies import DivergenceBOS

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "cache_3yr_1h"
DESIGN_START = pd.Timestamp("2023-06-01")
HOLDOUT_START = pd.Timestamp("2026-02-01")


def work(sym):
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
    df = df[(df.start >= DESIGN_START) & (df.start < HOLDOUT_START)].reset_index(drop=True)
    if len(df) < 1500:
        return None
    s = DivergenceBOS()
    p = s.prepare(df)
    sigs = s.detect(p, atr_mult=1.5, rr=5.0)
    if not sigs:
        return None
    real = replay(p, sigs, sym)
    pl = replay(p, placebo_signals(p, sigs, seed=abs(hash(sym)) % 10**6, atr_mult=1.5), sym)
    return ([t["r_net"] for t in real], [t["r_net"] for t in pl])


def main():
    syms = sorted(p.stem for p in CACHE.glob("*.parquet"))
    syms = [s for s in syms if not s.endswith(("26JUN26", "03APR26", "10APR26", "17APR26"))]
    print(f"[VALIDATE] incumbent on {len(syms)} symbols, rr=5 atr=1.5, design period\n",
          flush=True)
    R, P = [], []
    with mp.Pool(max(1, mp.cpu_count() - 1)) as pool:
        for i, res in enumerate(pool.imap_unordered(work, syms, chunksize=4), 1):
            if res:
                R.extend(res[0]); P.extend(res[1])
            if i % 100 == 0:
                print(f"  {i}/{len(syms)}", flush=True)

    r, p = np.array(R), np.array(P)
    sr, sp = stats(r), stats(p)
    t = welch_t(r, p)
    d = sr["avg"] - sp["avg"]

    print(f"\n{'':<12}{'trades':>10}{'WR':>9}{'avg R':>10}")
    print(f"  {'REAL':<10}{sr['n']:>10,}{sr['wr']:>9.2%}{sr['avg']:>+10.4f}")
    print(f"  {'PLACEBO':<10}{sp['n']:>10,}{sp['wr']:>9.2%}{sp['avg']:>+10.4f}")
    print(f"  {'delta':<10}{'':>10}{'':>9}{d:>+10.4f}   t = {t:+.2f}")

    print(f"\n  KNOWN TARGET: REAL 122,077t WR 18.07% avg -0.0172 · "
          f"PLACEBO 244,103t WR 17.42% avg -0.0549 · delta +0.0377 t=+4.66")
    checks = [
        ("real trade count within 15%", abs(sr["n"] - 122077) / 122077 < 0.15),
        ("real WR within 1.5pp", abs(sr["wr"] - 0.1807) < 0.015),
        ("real avg R within 0.02", abs(sr["avg"] - (-0.0172)) < 0.02),
        ("placebo avg R within 0.03", abs(sp["avg"] - (-0.0549)) < 0.03),
        ("delta positive and t > 3", d > 0 and t > 3),
    ]
    print()
    for lbl, ok in checks:
        print(f"  [{'OK ' if ok else 'FAIL'}] {lbl}")
    print(f"\n{'HARNESS VALIDATED' if all(o for _, o in checks) else 'HARNESS MISMATCH — investigate before trusting any candidate score'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
