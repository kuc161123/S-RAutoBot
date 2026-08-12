#!/usr/bin/env python3
"""Resolve the FULL (symbol x div_type x rr x atr_mult) grid, in and out of universe.

One row per (signal, atr_mult); the five RR outcomes ride along as columns, because RR only
moves the take-profit — the signal, the BOS bar, the entry and the stop are all identical
across RR. So a single forward walk per (signal, atr_mult) resolves all five at once:
record the first bar that touches the stop, and the first bar that touches each TP, then
whichever comes first wins per RR. Stop wins ties (repo-wide pessimistic convention).

CHOP is read at `bos`, never at the entry bar — see STRATEGY_VERDICT_2026-08-11.md 2.1.
The gate is NOT applied here; `chop_bos` rides on the row so arms can ablate it.

Output: risk_study/grid_<tag>.parquet
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import backtest_3yr_walkforward as bt  # noqa: E402
from backtest_halt_multiwindow import chop_series  # noqa: E402

HERE = Path(__file__).resolve().parent
GEN_FROM = pd.Timestamp("2023-06-01")
MAX_WAIT = bt.MAX_WAIT_CANDLES
RRS = (2.0, 3.0, 5.0, 8.0, 10.0)
AMS = (1.0, 1.5, 2.0, 3.0)
DIV_TYPES = ("REG_BULL", "REG_BEAR", "HID_BULL", "HID_BEAR")


def one(args):
    sym, cache_dir = args
    f = Path(cache_dir) / f"{sym}.parquet"
    if not f.exists():
        return []
    try:
        df = pd.read_parquet(f)
    except Exception:
        return []
    if df.empty or len(df) < 2000 or "start" not in df.columns:
        return []
    try:
        df = bt.prepare_data(df)
    except Exception:
        return []
    df["chop"] = chop_series(df)
    o = df.open.values; h = df.high.values; l = df.low.values; c = df.close.values
    atr = df.atr.values; ema = df.ema.values; ch = df.chop.values; ts = df.start.values
    turn = df["turnover"].values if "turnover" in df.columns else np.full(len(df), np.nan)
    n = len(c)

    sigs = {}
    for s in bt.detect_signals(df):
        sigs.setdefault(s["type"], []).append(s)

    out = []
    for dt in DIV_TYPES:
        for s in sigs.get(dt, []):
            conf, side, lvl = s["conf_idx"], s["side"], s["swing"]
            bos = None
            for i in range(1, MAX_WAIT + 1):
                idx = conf + i
                if idx >= n:
                    break
                if (side == "long" and c[idx] > lvl) or (side == "short" and c[idx] < lvl):
                    bos = idx
                    break
            if bos is None:
                continue
            e = bos + 1
            if e >= n or not np.isfinite(ema[bos]):
                continue
            if side == "long" and not c[bos] > ema[bos]:
                continue
            if side == "short" and not c[bos] < ema[bos]:
                continue
            if not (np.isfinite(atr[bos]) and atr[bos] > 0):
                continue

            entry = o[e]
            if not (np.isfinite(entry) and entry > 0):
                continue
            long_ = side == "long"
            # median hourly turnover over the 30 days before the signal -- a causal
            # liquidity measure for the A4 arm. Never touches bar e or later.
            lo_t = max(0, bos - 720)
            tw = turn[lo_t:bos]
            liq = float(np.nanmedian(tw)) if len(tw) and np.isfinite(tw).any() else np.nan

            for am in AMS:
                sl_d = atr[bos] * am
                if sl_d <= 0:
                    continue
                sl = entry - sl_d if long_ else entry + sl_d
                if sl <= 0:
                    continue
                tps = [entry + sl_d * rr if long_ else entry - sl_d * rr for rr in RRS]
                if not long_ and min(tps) <= 0:
                    continue

                k_stop = -1
                k_tp = [-1] * len(RRS)
                need = len(RRS)
                for k in range(e, n):
                    if long_:
                        if l[k] <= sl:
                            k_stop = k
                            break
                        for j, tp in enumerate(tps):
                            if k_tp[j] < 0 and h[k] >= tp:
                                k_tp[j] = k
                                need -= 1
                    else:
                        if h[k] >= sl:
                            k_stop = k
                            break
                        for j, tp in enumerate(tps):
                            if k_tp[j] < 0 and l[k] <= tp:
                                k_tp[j] = k
                                need -= 1
                    if need == 0:
                        break

                row = [sym, dt, side, ts[e], entry, sl, am,
                       ch[bos] if np.isfinite(ch[bos]) else np.nan, liq]
                unresolved = True
                for j, rr in enumerate(RRS):
                    if k_tp[j] >= 0 and (k_stop < 0 or k_tp[j] < k_stop):
                        row += [rr, ts[k_tp[j]]]
                        unresolved = False
                    elif k_stop >= 0:
                        row += [-1.0, ts[k_stop]]
                        unresolved = False
                    else:
                        # neither level reached before data ends -- right-censored.
                        row += [np.nan, np.datetime64("NaT")]
                if not unresolved:
                    out.append(tuple(row))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=str(ROOT / "cache_3yr_1h"))
    ap.add_argument("--tag", default="inuni")
    ap.add_argument("--only", choices=["live", "outside", "all"], default="live")
    a = ap.parse_args()

    cfg = yaml.safe_load(open(ROOT / "config.yaml"))["symbols"]
    live = {s for s, sc in (cfg or {}).items()
            if (sc or {}).get("enabled", True) and (sc or {}).get("configs")}

    syms = []
    for f in sorted(Path(a.cache).glob("*.parquet")):
        s = f.stem
        if "-" in s:
            continue
        if a.only == "live" and s not in live:
            continue
        if a.only == "outside" and s in live:
            continue
        syms.append(s)

    print(f"[GRID] {len(syms)} symbols from {Path(a.cache).name} "
          f"({len(RRS)}rr x {len(AMS)}am x {len(DIV_TYPES)}types)", flush=True)

    cols = ["symbol", "div_type", "side", "entry_time", "entry_price", "sl_price",
            "atr_mult", "chop_bos", "liq_30d"]
    for rr in RRS:
        cols += [f"r_{rr:g}", f"exit_{rr:g}"]

    rows = []
    jobs = [(s, a.cache) for s in syms]
    with mp.Pool(max(1, mp.cpu_count() - 1)) as pool:
        for i, r in enumerate(pool.imap_unordered(one, jobs, chunksize=2), 1):
            rows.extend(r)
            if i % 25 == 0:
                print(f"  {i}/{len(syms)} · {len(rows):,} rows", flush=True)

    d = pd.DataFrame(rows, columns=cols)
    if d.empty:
        print("[GRID] no rows"); return
    d["entry_time"] = pd.to_datetime(d["entry_time"])
    for rr in RRS:
        d[f"exit_{rr:g}"] = pd.to_datetime(d[f"exit_{rr:g}"])
    d = d[d.entry_time >= GEN_FROM].sort_values("entry_time").reset_index(drop=True)
    d["stop_frac"] = (d.entry_price - d.sl_price).abs() / d.entry_price
    d["in_universe"] = d.symbol.isin(live)

    out = HERE / f"grid_{a.tag}.parquet"
    d.to_parquet(out, index=False)
    print(f"[GRID] wrote {out.name}: {len(d):,} rows, {d.symbol.nunique()} symbols, "
          f"{d.entry_time.min().date()}..{d.entry_time.max().date()}")
    for rr in RRS:
        col = f"r_{rr:g}"
        v = d[col].dropna()
        print(f"  rr={rr:>4g}  n={len(v):>8,}  gross meanR {v.mean():+.4f}  "
              f"net@24.2bps {(v - 0.00242 / d.loc[v.index, 'stop_frac']).mean():+.4f}")


if __name__ == "__main__":
    main()
