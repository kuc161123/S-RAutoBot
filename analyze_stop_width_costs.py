#!/usr/bin/env python3
"""Stop width vs cost drag — is the design fighting itself?

fee_r = round_trip_cost * entry / sl_distance. The stop distance is the DENOMINATOR, so
a tight stop makes every fee enormous in R terms. At 1.5x ATR the drag measured 0.1017
R/trade against a gross edge of 0.155 R — costs eat 66% of the edge.

A wider stop cuts the drag proportionally, but also re-scales R: the same price move is
worth fewer R, so the hit rate for a given RR target falls. Which effect wins is an
empirical question, and it decides whether the current 1.0-2.0x ATR range is the right
geometry at all.

Measures the full (atr_mult x rr) surface on net expectancy, design period only.
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
LO = pd.Timestamp("2023-06-01")
SLIP, FEE = 0.0003, 0.0006
ROUND_TRIP = 2 * (SLIP + FEE)
MAX_WAIT = bt.MAX_WAIT_CANDLES
ATRS = [1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
RRS = [2, 3, 4, 5, 6, 8, 10, 12]


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
        base_atr = atr[bos]
        entry = o[e] * (1 + SLIP) if side == "long" else o[e] * (1 - SLIP)
        for m in ATRS:
            sl_d = base_atr * m
            sl = entry - sl_d if side == "long" else entry + sl_d
            fee_r = ROUND_TRIP * entry / sl_d
            mfe = 0.0
            for k in range(e, n):
                if side == "long":
                    mfe = max(mfe, (h[k] - entry) / sl_d)
                    hit = l[k] <= sl
                else:
                    mfe = max(mfe, (entry - l[k]) / sl_d)
                    hit = h[k] >= sl
                if hit:
                    break
            out.append((m, mfe, fee_r))
    return out


def main():
    syms = sorted(p.stem for p in CACHE.glob("*.parquet"))
    syms = [s for s in syms if not s.endswith(("26JUN26", "03APR26", "10APR26", "17APR26"))]
    rows = []
    with mp.Pool(max(1, mp.cpu_count() - 1)) as pool:
        for i, r in enumerate(pool.imap_unordered(work, syms, chunksize=4), 1):
            rows.extend(r)
            if i % 100 == 0:
                print(f"  {i}/{len(syms)}", flush=True)

    d = pd.DataFrame(rows, columns=["atr_mult", "mfe", "fee_r"])
    print(f"\n{len(d)//len(ATRS):,} signals x {len(ATRS)} stop widths\n")
    print("  cost drag by stop width (this is the whole point):")
    for m in ATRS:
        g = d[d.atr_mult == m]
        print(f"    {m:>4.1f}x ATR   fee drag {g.fee_r.mean():>7.4f} R/trade")

    print(f"\n  NET EXPECTANCY SURFACE (R/trade, costs included)")
    print(f"  {'atr':>5}" + "".join(f"{'rr'+str(r):>9}" for r in RRS))
    best = None
    for m in ATRS:
        g = d[d.atr_mult == m]
        mfe = g.mfe.values; fee = g.fee_r.mean()
        line = f"  {m:>5.1f}"
        for rr in RRS:
            hit = (mfe >= rr).mean()
            ev = hit * rr - (1 - hit) - fee
            if best is None or ev > best[2]:
                best = (m, rr, ev)
            line += f"{ev:>+9.4f}"
        print(line)
    print(f"\n  => best geometry: {best[0]:.1f}x ATR with rr={best[1]}  "
          f"(net EV {best[2]:+.4f} R/trade)")
    print(f"  => live config uses 1.0-2.0x ATR, rr 3-10")

    # headroom at the best geometry
    g = d[d.atr_mult == best[0]]
    hit = (g.mfe.values >= best[1]).mean(); fee = g.fee_r.mean()
    be = (1 + fee) / (1 + best[1])
    print(f"\n  at the best geometry: hit {hit:.3%} vs breakeven-incl-costs {be:.3%} "
          f"-> headroom {100*(hit-be):.2f}pp")
    return 0


if __name__ == "__main__":
    sys.exit(main())
