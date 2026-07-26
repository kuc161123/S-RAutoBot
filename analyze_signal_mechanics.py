#!/usr/bin/env python3
"""Are the divergence and BOS mechanics set up correctly?

Tests the SIGNAL DEFINITION parameters (not the trade geometry), one at a time from the
live baseline, at a fixed sensible geometry so differences are attributable to the signal.

  pivot width      left/right bars needed to confirm a swing (live: 3/3)
  freshness        max bars between the triggering pivot and confirmation (live: 10)
  BOS wait         max bars to wait for the break (live: 12)
  BOS trigger      close beyond the level (live) vs wick beyond it
  RSI period       (live: 14)

This is a SENSITIVITY analysis, not a search for the best setting. The question is whether
the live values sit on a plateau (robust — small changes barely matter) or on a spike
(fragile — the settings were fitted). A plateau is what a sound design looks like.

Design period only (pre-2026-02-01), so the holdout stays clean.
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
ATR_M, RR = 3.0, 6.0            # design-period best geometry, held fixed throughout
MIN_PIV_DIST = 3
LOOKBACK = 50

# (label, pivot_lr, freshness, bos_wait, bos_on_wick, rsi_period)
BASE = ("LIVE baseline", 3, 10, 12, False, 14)
VARIANTS = [
    BASE,
    ("pivot width 2", 2, 10, 12, False, 14),
    ("pivot width 5", 5, 10, 12, False, 14),
    ("freshness 5", 3, 5, 12, False, 14),
    ("freshness 20", 3, 20, 12, False, 14),
    ("BOS wait 6", 3, 10, 6, False, 14),
    ("BOS wait 24", 3, 10, 24, False, 14),
    ("BOS on wick", 3, 10, 12, True, 14),
    ("RSI 7", 3, 10, 12, False, 7),
    ("RSI 21", 3, 10, 12, False, 21),
]


def prep(df, rsi_p):
    df = df.copy()
    delta = df["close"].diff()
    gain = delta.where(delta > 0, 0).rolling(rsi_p).mean()
    loss = -delta.where(delta < 0, 0).rolling(rsi_p).mean()
    df["rsi"] = 100 - (100 / (1 + gain / (loss + 1e-10)))
    hl = df["high"] - df["low"]
    hc = (df["high"] - df["close"].shift()).abs()
    lc = (df["low"] - df["close"].shift()).abs()
    df["atr"] = pd.concat([hl, hc, lc], axis=1).max(axis=1).rolling(14).mean()
    df["ema"] = df["close"].ewm(span=200, adjust=False).mean()
    return df


def detect(df, piv, fresh):
    """Mirror bt.detect_signals with configurable pivot width and freshness."""
    close = df["close"].values; high = df["high"].values; low = df["low"].values
    rsi = df["rsi"].values; ema = df["ema"].values
    ph, pl = bt.find_pivots(close, piv, piv)
    sig = []; used = set()
    for i in range(max(210, LOOKBACK + piv + 1), len(df) - piv):
        if np.isnan(rsi[i]) or np.isnan(ema[i]):
            continue
        if close[i] > ema[i]:
            p = []
            for j in range(i - piv, max(0, i - LOOKBACK), -1):
                if not np.isnan(pl[j]):
                    p.append((j, pl[j]))
                    if len(p) >= 2:
                        break
            if len(p) == 2:
                (ci, cv), (pi_, pv) = p
                k = (ci, pi_, "B")
                if (i - ci) <= fresh and k not in used and (ci - pi_) >= MIN_PIV_DIST:
                    if (cv < pv and rsi[ci] > rsi[pi_]) or (cv > pv and rsi[ci] < rsi[pi_]):
                        sig.append({"conf_idx": i, "side": "long",
                                    "swing": max(high[ci:i + 1])}); used.add(k)
        if close[i] < ema[i]:
            p = []
            for j in range(i - piv, max(0, i - LOOKBACK), -1):
                if not np.isnan(ph[j]):
                    p.append((j, ph[j]))
                    if len(p) >= 2:
                        break
            if len(p) == 2:
                (ci, cv), (pi_, pv) = p
                k = (ci, pi_, "S")
                if (i - ci) <= fresh and k not in used and (ci - pi_) >= MIN_PIV_DIST:
                    if (cv > pv and rsi[ci] < rsi[pi_]) or (cv < pv and rsi[ci] > rsi[pi_]):
                        sig.append({"conf_idx": i, "side": "short",
                                    "swing": min(low[ci:i + 1])}); used.add(k)
    return sig


def work(args):
    sym, cache_dir = args
    f = Path(cache_dir) / f"{sym}.parquet"
    if not f.exists():
        return {}
    try:
        raw = pd.read_parquet(f)
    except Exception:
        return {}
    if raw.empty or len(raw) < 3000:
        return {}
    out = {}
    cache_rsi = {}
    for (lbl, piv, fresh, wait, wick, rsi_p) in VARIANTS:
        if rsi_p not in cache_rsi:
            cache_rsi[rsi_p] = prep(raw, rsi_p)
        df = cache_rsi[rsi_p]
        o = df.open.values; h = df.high.values; l = df.low.values; c = df.close.values
        atr = df.atr.values; ts = df.start.values; n = len(c)
        cut = np.searchsorted(ts, np.datetime64(HOLD))
        lo_i = np.searchsorted(ts, np.datetime64(LO))
        rs = []
        for s in detect(df, piv, fresh):
            conf, side, lvl = s["conf_idx"], s["side"], s["swing"]
            if not (lo_i <= conf < cut):
                continue
            bos = None
            for i in range(1, wait + 1):
                idx = conf + i
                if idx >= n:
                    break
                if wick:
                    brk = (h[idx] > lvl) if side == "long" else (l[idx] < lvl)
                else:
                    brk = (c[idx] > lvl) if side == "long" else (c[idx] < lvl)
                if brk:
                    bos = idx
                    break
            if bos is None or bos + 1 >= cut:
                continue
            if not (np.isfinite(atr[bos]) and atr[bos] > 0):
                continue
            e = bos + 1
            sl_d = atr[bos] * ATR_M
            entry = o[e] * (1 + SLIP) if side == "long" else o[e] * (1 - SLIP)
            sl = entry - sl_d if side == "long" else entry + sl_d
            tp = entry + sl_d * RR if side == "long" else entry - sl_d * RR
            fee_r = ROUND_TRIP * entry / sl_d
            for k in range(e, n):
                if side == "long":
                    hs, ht = l[k] <= sl, h[k] >= tp
                else:
                    hs, ht = h[k] >= sl, l[k] <= tp
                if hs or ht:
                    rs.append((-1.0 if hs else RR) - fee_r)
                    break
        out[lbl] = rs
    return out


def main():
    syms = sorted(p.stem for p in CACHE.glob("*.parquet"))
    syms = [s for s in syms if not s.endswith(("26JUN26", "03APR26", "10APR26", "17APR26"))]
    agg = {v[0]: [] for v in VARIANTS}
    with mp.Pool(max(1, mp.cpu_count() - 1)) as pool:
        for i, r in enumerate(pool.imap_unordered(
                work, [(s, str(CACHE)) for s in syms], chunksize=4), 1):
            for k, v in r.items():
                agg[k].extend(v)
            if i % 100 == 0:
                print(f"  {i}/{len(syms)}", flush=True)

    print(f"\nSIGNAL MECHANICS SENSITIVITY — geometry fixed at {ATR_M}x ATR / rr{RR:.0f}")
    print("design period only; a robust design shows a PLATEAU, not a spike\n")
    base = np.asarray(agg[BASE[0]], float)
    print(f"  {'variant':<20}{'trades':>9}{'WR':>9}{'avgR':>10}{'PF':>8}{'vs base':>10}")
    for (lbl, *_rest) in VARIANTS:
        x = np.asarray(agg[lbl], float)
        if len(x) == 0:
            print(f"  {lbl:<20}   (none)"); continue
        gp = x[x > 0].sum(); gl = abs(x[x < 0].sum())
        delta = "" if lbl == BASE[0] else f"{x.mean()-base.mean():>+10.4f}"
        print(f"  {lbl:<20}{len(x):>9,}{(x>0).mean():>9.2%}{x.mean():>+10.4f}"
              f"{(gp/gl if gl else 9.99):>8.2f}{delta:>10}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
