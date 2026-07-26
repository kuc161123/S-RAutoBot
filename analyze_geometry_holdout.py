#!/usr/bin/env python3
"""ONE pre-registered holdout test of the geometry conclusion.

The design period (pre-2026-02-01) says the best stop/target geometry is ~3.0x ATR with
rr=6 (net EV +0.0756 R/trade, 1.08pp headroom), versus the live config's 1.0-2.0x ATR /
rr 3-10 (best cell +0.0746, dominant cells far worse — every 1.0x ATR cell is negative).

That conclusion was reached WITHOUT touching the holdout. This script now scores a small
pre-committed set of geometries on the holdout, once, to see whether the ranking survives.
It is a confirmation, not a search: no new geometry is chosen here.
"""
from __future__ import annotations

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
SLIP, FEE = 0.0003, 0.0006
ROUND_TRIP = 2 * (SLIP + FEE)
MAX_WAIT = bt.MAX_WAIT_CANDLES
# pre-committed: the design winner, the live config's typical cells, and neighbours
GEOMS = [(3.0, 6), (3.0, 5), (2.0, 8), (2.0, 6), (1.5, 10), (1.0, 10), (1.0, 5)]


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
    lo_i = np.searchsorted(ts, np.datetime64(HOLD))
    out = []
    for s in bt.detect_signals(df):
        conf, side, lvl = s["conf_idx"], s["side"], s["swing"]
        if conf < lo_i:
            continue
        bos = None
        for i in range(1, MAX_WAIT + 1):
            idx = conf + i
            if idx >= n:
                break
            if (side == "long" and c[idx] > lvl) or (side == "short" and c[idx] < lvl):
                bos = idx
                break
        if bos is None or bos + 1 >= n:
            continue
        if not (np.isfinite(atr[bos]) and atr[bos] > 0):
            continue
        e = bos + 1
        entry = o[e] * (1 + SLIP) if side == "long" else o[e] * (1 - SLIP)
        for m, rr in GEOMS:
            sl_d = atr[bos] * m
            sl = entry - sl_d if side == "long" else entry + sl_d
            tp = entry + sl_d * rr if side == "long" else entry - sl_d * rr
            fee_r = ROUND_TRIP * entry / sl_d
            r = None
            for k in range(e, n):
                if side == "long":
                    hs, ht = l[k] <= sl, h[k] >= tp
                else:
                    hs, ht = h[k] >= sl, l[k] <= tp
                if hs or ht:
                    r = (-1.0 if hs else rr) - fee_r
                    break
            if r is not None:
                out.append((m, rr, r))
    return out


def main():
    syms = sorted(p.stem for p in CACHE.glob("*.parquet"))
    syms = [s for s in syms if not s.endswith(("26JUN26", "03APR26", "10APR26", "17APR26"))]
    rows = []
    with mp.Pool(max(1, mp.cpu_count() - 1)) as pool:
        for i, r in enumerate(pool.imap_unordered(work, syms, chunksize=4), 1):
            rows.extend(r)
            if i % 150 == 0:
                print(f"  {i}/{len(syms)}", flush=True)
    d = pd.DataFrame(rows, columns=["atr_mult", "rr", "r"])
    print(f"\nHOLDOUT 2026-02-01 -> today  ({len(d)//len(GEOMS):,} signals)\n")
    print(f"  {'geometry':<18}{'trades':>9}{'WR':>9}{'avgR':>10}{'PF':>8}{'note':>28}")
    notes = {(3.0, 6): "design-period WINNER",
             (1.0, 10): "live: 336 cfgs at 1.0x ATR",
             (1.5, 10): "live: most common rr"}
    for (m, rr) in GEOMS:
        g = d[(d.atr_mult == m) & (d.rr == rr)].r.values
        if len(g) == 0:
            continue
        gp = g[g > 0].sum(); gl = abs(g[g < 0].sum())
        print(f"  {m:.1f}x ATR / rr{rr:<8}{len(g):>9,}{(g>0).mean():>9.2%}"
              f"{g.mean():>+10.4f}{(gp/gl if gl else 9.99):>8.2f}{notes.get((m,rr),''):>28}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
