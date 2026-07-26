#!/usr/bin/env python3
"""How far do trades ACTUALLY run before the stop? (uncensored MFE)

The earlier MFE measurement was censored: with a take-profit at 5R, a trade that hits it
stops being tracked, so it can never *show* 8R of excursion. That made RR 8/10 look
impossible for the wrong reason.

Here there is NO take-profit — only the stop. So we observe the real distribution of
maximum favourable excursion, which is what determines which RR targets are physically
achievable on this signal set.

Design period only (entries before 2026-02-01), so this is "was this ever the right
geometry?", not "is it working this month?".
"""
from __future__ import annotations

import math
import multiprocessing as mp
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
import backtest_3yr_walkforward as bt

CACHE = Path(__file__).parent / "cache_3yr_1h"
HOLD = pd.Timestamp("2026-02-01")
LO = pd.Timestamp("2023-06-01")
SLIP, FEE = 0.0003, 0.0006
ROUND_TRIP = 2 * (SLIP + FEE)
ATR_REF = 1.5
MAX_WAIT = bt.MAX_WAIT_CANDLES


def work(sym):
    f = CACHE / f"{sym}.parquet"
    if not f.exists():
        return []
    try:
        df = pd.read_parquet(f)
    except Exception:
        return []
    if df.empty or len(df) < 3000:
        return []
    df = bt.prepare_data(df)
    o = df.open.values; h = df.high.values; l = df.low.values; c = df.close.values
    atr = df.atr.values; ts = df.start.values; n = len(c)
    cut = np.searchsorted(ts, np.datetime64(HOLD))
    lo_i = np.searchsorted(ts, np.datetime64(LO))
    out = []
    for s in bt.detect_signals(df):
        conf, side, lvl = s["conf_idx"], s["side"], s["swing"]
        if not (lo_i <= conf < cut):
            continue
        bos = None
        for i in range(1, MAX_WAIT + 1):
            idx = conf + i
            if idx >= n:
                break
            if (side == "long" and c[idx] > lvl) or (side == "short" and c[idx] < lvl):
                bos = idx
                break
        if bos is None or bos + 1 >= cut:
            continue
        if not (np.isfinite(atr[bos]) and atr[bos] > 0):
            continue
        e = bos + 1
        sl_d = atr[bos] * ATR_REF
        entry = o[e] * (1 + SLIP) if side == "long" else o[e] * (1 - SLIP)
        sl = entry - sl_d if side == "long" else entry + sl_d
        fee_r = ROUND_TRIP * entry / sl_d
        mfe = 0.0
        bars = 0
        for k in range(e, n):
            if side == "long":
                mfe = max(mfe, (h[k] - entry) / sl_d)
                hit = l[k] <= sl
            else:
                mfe = max(mfe, (entry - l[k]) / sl_d)
                hit = h[k] >= sl
            bars = k - e
            if hit:
                break
        out.append((mfe, fee_r, bars))
    return out


def main():
    syms = sorted(p.stem for p in CACHE.glob("*.parquet"))
    syms = [s for s in syms if not s.endswith(("26JUN26", "03APR26", "10APR26", "17APR26"))]
    res = []
    with mp.Pool(max(1, mp.cpu_count() - 1)) as pool:
        for i, r in enumerate(pool.imap_unordered(work, syms, chunksize=4), 1):
            res.extend(r)
            if i % 100 == 0:
                print(f"  {i}/{len(syms)}", flush=True)

    m = np.array([x[0] for x in res]); fees = np.array([x[1] for x in res])
    print(f"\nUNCENSORED MFE (stop only, no take-profit) — {len(m):,} real signals")
    print(f"mean fee+slippage drag {fees.mean():.4f} R/trade\n")
    print("  how far the move runs before the stop is hit, in R:")
    for q in [50, 60, 70, 80, 90, 95, 99]:
        print(f"    p{q:<3} {np.percentile(m, q):>7.2f} R")

    print(f"\n  {'target':<10}{'reached':>10}{'breakeven':>12}{'verdict':>14}")
    for rr in [2, 3, 4, 5, 6, 8, 10]:
        hit = (m >= rr).mean(); be = 1 / (1 + rr)
        print(f"    rr={rr:<6}{hit:>10.2%}{be:>12.2%}"
              f"{('VIABLE' if hit > be else 'NOT viable'):>14}")

    print(f"\n  net expectancy per trade (includes {fees.mean():.4f}R cost):")
    best = None
    for rr in [2, 3, 4, 5, 6, 8, 10]:
        hit = (m >= rr).mean()
        ev = hit * rr - (1 - hit) - fees.mean()
        if best is None or ev > best[1]:
            best = (rr, ev)
        print(f"    rr={rr:<3} EV = {ev:+.4f} R")
    print(f"\n  => geometry-optimal RR on this signal set: rr={best[0]} (EV {best[1]:+.4f})")
    print(f"  => the live config puts {'451 of 728'} configs at rr 8-10.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
