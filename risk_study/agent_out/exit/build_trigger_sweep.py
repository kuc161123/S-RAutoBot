#!/usr/bin/env python3
"""AGENT-EXIT / LEAD 2 — sweep trigger_r x trail_atr for the global rr=10/atr_mult=3.0
"same-pairs" configuration (the current best-known arm per CLAUDE.md: $17,205 from
$1,500, 47.0% maxDD, ROI/DD ~22).

Signal generation, BOS confirmation, EMA gate and CHOP gate (read at bar `bos`, NOT the
entry bar `e` -- avoids the CHOP-lookahead bug documented in
STRATEGY_VERDICT_2026-08-11.md) are byte-for-byte the same as
risk_study/build_global_trail.py with --pairs same. Only the trail resolution differs:
instead of the single fixed TRIGGER_R=3.0/TRAIL_ATR=1.0 in
risk_study/build_universe_trail_helpers.py, this walks each signal ONCE and evaluates
all 24 (trigger_r, trail_atr) combinations in parallel on the same bar sequence, so
every cell is scored on the identical trade set (a combo that fails to resolve by the
end of the cache is dropped from ALL cells for that signal, not just its own -- keeps
the trade set aligned across cells).

Output: trigger_sweep_universe.parquet, wide format, one row per signal, one column per
combo: r_<trigger>_<trail> (net-of-cost R is computed later in analyze.py from
stop_frac, which does not depend on the combo).
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
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import backtest_3yr_walkforward as bt  # noqa: E402
from backtest_halt_multiwindow import chop_series  # noqa: E402

HERE = Path(__file__).resolve().parent
CACHE = ROOT / "cache_3yr_1h"
GEN_FROM = pd.Timestamp("2023-06-01")
MAX_WAIT = bt.MAX_WAIT_CANDLES
CHOP_T = 52.0
RR, AM = 10.0, 3.0

TRIGGERS = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0]
TRAILS = [0.5, 1.0, 1.5, 2.0]
COMBOS = [(t, a) for t in TRIGGERS for a in TRAILS]
NCOMBO = len(COMBOS)
COLS = [f"r_{t:g}_{a:g}" for t, a in COMBOS]


def resolve_multi(e, n, long_, entry, sl0, tp, risk_dist, h, l, atr):
    """Walk bars once from e; return a list of NCOMBO R values (np.nan if unresolved
    by end of data)."""
    stops = [sl0] * NCOMBO
    done = [False] * NCOMBO
    r = [np.nan] * NCOMBO
    n_done = 0
    for k in range(e, n):
        hk, lk = h[k], l[k]
        if long_:
            hit_tp = hk >= tp
            for ci in range(NCOMBO):
                if done[ci]:
                    continue
                if lk <= stops[ci]:
                    r[ci] = (stops[ci] - entry) / risk_dist
                    done[ci] = True
                    n_done += 1
                elif hit_tp:
                    r[ci] = rr_target
                    done[ci] = True
                    n_done += 1
        else:
            hit_tp = lk <= tp
            for ci in range(NCOMBO):
                if done[ci]:
                    continue
                if hk >= stops[ci]:
                    r[ci] = (entry - stops[ci]) / risk_dist
                    done[ci] = True
                    n_done += 1
                elif hit_tp:
                    r[ci] = rr_target
                    done[ci] = True
                    n_done += 1
        if n_done == NCOMBO:
            break
        a = atr[k]
        if not (np.isfinite(a) and a > 0):
            continue
        if long_:
            mfe = (hk - entry) / risk_dist
            for ci, (trig, ta) in enumerate(COMBOS):
                if done[ci] or mfe < trig:
                    continue
                cand = hk - ta * a
                if cand > stops[ci]:
                    stops[ci] = cand
        else:
            mfe = (entry - lk) / risk_dist
            for ci, (trig, ta) in enumerate(COMBOS):
                if done[ci] or mfe < trig:
                    continue
                cand = lk + ta * a
                if cand < stops[ci]:
                    stops[ci] = cand
    return r


# rr_target is module-global so resolve_multi doesn't need it passed every call
rr_target = RR


def one(args):
    sym, types, cache_dir = args
    f = Path(cache_dir) / f"{sym}.parquet"
    if not f.exists():
        return []
    try:
        df = pd.read_parquet(f)
    except Exception:
        return []
    if df.empty or len(df) < 2000:
        return []
    df = bt.prepare_data(df)
    df["chop"] = chop_series(df)
    o = df.open.values; h = df.high.values; l = df.low.values; c = df.close.values
    atr = df.atr.values; ema = df.ema.values; ch = df.chop.values; ts = df.start.values
    n = len(c)

    sigs = {}
    for s in bt.detect_signals(df):
        sigs.setdefault(s["type"], []).append(s)

    out = []
    for dt in types:
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
            if np.isfinite(ch[bos]) and ch[bos] >= CHOP_T:
                continue

            long_ = side == "long"
            entry = o[e]
            risk_dist = atr[bos] * AM
            sl = entry - risk_dist if long_ else entry + risk_dist
            tp = entry + risk_dist * RR if long_ else entry - risk_dist * RR
            if sl <= 0 or tp <= 0 or entry <= 0:
                continue

            rs = resolve_multi(e, n, long_, entry, sl, tp, risk_dist, h, l, atr)
            if any(np.isnan(rv) for rv in rs):
                continue  # keep the trade set identical across all 24 cells
            stop_frac = abs(entry - sl) / entry
            out.append((ts[e], sym, dt, side, entry, stop_frac, *rs))
    return out


def main():
    cfg = yaml.safe_load(open(ROOT / "config.yaml"))["symbols"]
    jobs = []
    for s, sc in cfg.items():
        if not (sc or {}).get("enabled", True):
            continue
        live_types = [c["divergence_type"] for c in (sc or {}).get("configs", []) or []]
        if not live_types or not (CACHE / f"{s}.parquet").exists():
            continue
        jobs.append((s, tuple(live_types), str(CACHE)))
    jobs.sort()
    print(f"[TRIGGER-SWEEP] rr={RR:g} atr={AM:g} pairs=same · {len(jobs)} symbols · "
          f"{NCOMBO} combos", flush=True)

    rows = []
    with mp.Pool(max(1, mp.cpu_count() - 1)) as pool:
        for i, r in enumerate(pool.imap_unordered(one, jobs, chunksize=2), 1):
            rows.extend(r)
            if i % 40 == 0:
                print(f"  {i}/{len(jobs)} symbols · {len(rows):,} signals", flush=True)

    cols = ["entry_time", "symbol", "div_type", "side", "entry_price", "stop_frac"] + COLS
    d = pd.DataFrame(rows, columns=cols)
    d["entry_time"] = pd.to_datetime(d["entry_time"])
    d = d[d.entry_time >= GEN_FROM].sort_values("entry_time").reset_index(drop=True)
    out = HERE / "trigger_sweep_universe.parquet"
    d.to_parquet(out, index=False)
    print(f"[TRIGGER-SWEEP] {out.name}: {len(d):,} signals (fully resolved across all "
          f"{NCOMBO} combos), {d.symbol.nunique()} symbols, "
          f"{d.entry_time.min()} .. {d.entry_time.max()}")


if __name__ == "__main__":
    main()
