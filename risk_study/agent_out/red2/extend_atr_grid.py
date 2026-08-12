#!/usr/bin/env python3
"""RED-TEAM attack #1: is the atr_mult=3.0 'advantage' just 'wider stops never get hit'?

Extends the grid beyond the study's own AMS=(1.0,1.5,2.0,3.0) cap to (2.0,3.0,4.0,5.0,7.0,10.0),
holding rr=10 fixed (the winning arm's rr), on the FIXED take-profit exit (same resolution
logic as build_param_grid.py -- stop wins ties, first bar touches stop or TP).
Only in-universe (config.yaml) 275 symbols, same as grid_inuni.parquet.

For each atr_mult, reports:
  - n trades resolved
  - stop-out fraction (share of trades whose first touch is the stop, not the TP)
  - unresolved (right-censored, data ran out before either level touched) fraction
  - mean holding time in bars (hours) for trades that DID resolve
  - mean net R at 24.2bps cost
"""
from __future__ import annotations
import multiprocessing as mp
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

warnings.filterwarnings("ignore")
ROOT = Path("/Users/lualakol/AutoTrading Bot")
sys.path.insert(0, str(ROOT))
import backtest_3yr_walkforward as bt
from backtest_halt_multiwindow import chop_series

HERE = Path(__file__).resolve().parent
GEN_FROM = pd.Timestamp("2023-06-01")
MAX_WAIT = bt.MAX_WAIT_CANDLES
RR = 10.0
AMS = (2.0, 3.0, 4.0, 5.0, 7.0, 10.0)
DIV_TYPES = ("REG_BULL", "REG_BEAR", "HID_BULL", "HID_BEAR")
COST = 0.00242


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
            cb = ch[bos] if np.isfinite(ch[bos]) else np.nan
            if not (np.isfinite(cb) and cb < 52):
                continue  # apply the live CHOP gate (52 threshold, same as headline result)

            for am in AMS:
                sl_d = atr[bos] * am
                if sl_d <= 0:
                    continue
                sl = entry - sl_d if long_ else entry + sl_d
                if sl <= 0:
                    continue
                tp = entry + sl_d * RR if long_ else entry - sl_d * RR
                if not long_ and tp <= 0:
                    continue

                k_stop = -1
                k_tp = -1
                for k in range(e, n):
                    if long_:
                        if l[k] <= sl:
                            k_stop = k; break
                        if h[k] >= tp:
                            k_tp = k; break
                    else:
                        if h[k] >= sl:
                            k_stop = k; break
                        if l[k] <= tp:
                            k_tp = k; break
                stop_frac = abs(entry - sl) / entry
                if k_stop >= 0:
                    out.append((sym, dt, am, e, k_stop, -1.0, stop_frac, True))
                elif k_tp >= 0:
                    out.append((sym, dt, am, e, k_tp, RR, stop_frac, False))
                else:
                    out.append((sym, dt, am, e, -1, np.nan, stop_frac, np.nan))  # unresolved
    return out


def main():
    cfg = yaml.safe_load(open(ROOT / "config.yaml"))["symbols"]
    live = {s for s, sc in (cfg or {}).items()
            if (sc or {}).get("enabled", True) and (sc or {}).get("configs")}
    cache = ROOT / "cache_3yr_1h"
    syms = sorted(f.stem for f in cache.glob("*.parquet") if "-" not in f.stem and f.stem in live)
    print(f"[EXT-GRID] {len(syms)} symbols, rr={RR}, ams={AMS}", flush=True)

    jobs = [(s, str(cache)) for s in syms]
    rows = []
    with mp.Pool(max(1, mp.cpu_count() - 1)) as pool:
        for i, r in enumerate(pool.imap_unordered(one, jobs, chunksize=2), 1):
            rows.extend(r)
            if i % 50 == 0:
                print(f"  {i}/{len(syms)} · {len(rows):,} rows", flush=True)

    cols = ["symbol", "div_type", "atr_mult", "entry_idx", "exit_idx", "r_gross",
            "stop_frac", "is_stop"]
    d = pd.DataFrame(rows, columns=cols)
    d.to_parquet(HERE / "extended_atr_grid_raw.parquet", index=False)
    print(f"[EXT-GRID] wrote {len(d):,} rows")

    print(f"\n{'atr_mult':>9}{'n':>9}{'stopfrac%':>11}{'unresolved%':>13}"
          f"{'meanhold(bars)':>16}{'meanhold(stop)':>16}{'meanhold(tp)':>14}{'net_R':>9}")
    for am in AMS:
        s = d[d.atr_mult == am]
        n = len(s)
        resolved = s.dropna(subset=["r_gross"])
        unresolved_frac = 1 - len(resolved) / n if n else np.nan
        is_stop = resolved.is_stop == True
        stopfrac = is_stop.mean() * 100 if len(resolved) else np.nan
        hold = (resolved.exit_idx - resolved.entry_idx)
        hold_stop = hold[is_stop]
        hold_tp = hold[~is_stop]
        net_r = (resolved.r_gross - COST / resolved.stop_frac).mean()
        print(f"{am:>9.1f}{n:>9,}{stopfrac:>11.2f}{unresolved_frac*100:>13.2f}"
              f"{hold.mean():>16.1f}{hold_stop.mean():>16.1f}{hold_tp.mean():>14.1f}{net_r:>9.4f}")


if __name__ == "__main__":
    main()
