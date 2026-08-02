#!/usr/bin/env python3
"""Wide trailing universe — the LIVE config, every exit variant against identical signals.

Same signal generation as the gauntlet (live config.yaml picks: per-symbol, per-divergence
rr + atr_mult, EMA filter, chop < 52, BOS wait, entry on the bar after BOS). Only the exit
rule differs between columns, so any difference is attributable to the exit and nothing else.

Causality note on the trailing rule: the stop is recomputed from bar k's HIGH/LOW at bar k's
close and only takes effect from bar k+1. That is exactly what a 1H bot can do live — update
the stop once per closed candle. No intrabar lookahead.

Single-pass resolver: one walk over bars per signal, carrying a stop per variant, exiting the
loop as soon as every variant has resolved. Lets ~20 variants cost barely more than 6.
"""
from __future__ import annotations

import math
import multiprocessing as mp
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import backtest_3yr_walkforward as bt

CACHE = ROOT / "cache_3yr_1h"
OUT = ROOT / "trail_wide_universe.parquet"
GEN_FROM = pd.Timestamp("2023-06-01")
MAX_WAIT = bt.MAX_WAIT_CANDLES
FEE, SLIP = 0.0006, 0.0003
CHOP_T = 52.0

# (label, breakeven_trigger_R, trail_start_R, trail_atr_mult)
# None trail_start = no trailing;  None be = no breakeven move
VARIANTS = [
    ("base",          None, None, None),
    # ---- breakeven only (no trail) ----
    ("be1",           1.0,  None, None),
    ("be1.5",         1.5,  None, None),
    ("be2",           2.0,  None, None),
    # ---- pure trail, no breakeven ----
    ("s1_a1",         None, 1.0,  1.0),
    ("s1_a2",         None, 1.0,  2.0),
    ("s1.5_a1.5",     None, 1.5,  1.5),
    ("s2_a1",         None, 2.0,  1.0),
    ("s2_a1.5",       None, 2.0,  1.5),
    ("s2_a2",         None, 2.0,  2.0),
    ("s2_a3",         None, 2.0,  3.0),
    ("s3_a1",         None, 3.0,  1.0),
    ("s3_a2",         None, 3.0,  2.0),
    ("s3_a3",         None, 3.0,  3.0),
    ("s4_a2",         None, 4.0,  2.0),
    # ---- breakeven + trail ----
    ("be1_s1_a1",     1.0,  1.0,  1.0),
    ("be1_s2_a1",     1.0,  2.0,  1.0),
    ("be1_s2_a2",     1.0,  2.0,  2.0),
    ("be1_s3_a2",     1.0,  3.0,  2.0),
    ("be1.5_s3_a2",   1.5,  3.0,  2.0),
    ("be2_s3_a1",     2.0,  3.0,  1.0),
    ("be2_s4_a2",     2.0,  4.0,  2.0),
]
NV = len(VARIANTS)
BE_A = np.array([np.inf if v[1] is None else v[1] for v in VARIANTS])
TS_A = np.array([np.inf if v[2] is None else v[2] for v in VARIANTS])
TA_A = np.array([0.0 if v[3] is None else v[3] for v in VARIANTS])


def chop_series(df, period=14):
    hl = df["high"] - df["low"]
    hc = (df["high"] - df["close"].shift()).abs()
    lc = (df["low"] - df["close"].shift()).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    hi = df["high"].rolling(period).max()
    lo = df["low"].rolling(period).min()
    return (100 * np.log10(tr.rolling(period).sum() / (hi - lo).replace(0, np.nan))
            / math.log10(period))


def resolve_all(o, h, l, atr, e, side, sl_d, rr, n):
    """One bar walk, all variants. Returns (r[NV], exit_idx[NV]) with NaN where unresolved."""
    entry = o[e] * (1 + SLIP) if side == "long" else o[e] * (1 - SLIP)
    long = side == "long"
    init_stop = entry - sl_d if long else entry + sl_d
    tp = entry + sl_d * rr if long else entry - sl_d * rr

    stops = np.full(NV, init_stop)
    moved = np.zeros(NV, dtype=bool)
    out_r = np.full(NV, np.nan)
    out_x = np.full(NV, -1, dtype=np.int64)
    live = np.ones(NV, dtype=bool)
    remaining = NV

    for k in range(e, n):
        hk, lk = h[k], l[k]
        if long:
            mfe = (hk - entry) / sl_d
            hit_tp = hk >= tp
            hit_sl = lk <= stops
        else:
            mfe = (entry - lk) / sl_d
            hit_tp = lk <= tp
            hit_sl = hk >= stops

        # stop first: within a bar we cannot know order, so assume the adverse one
        done = live & hit_sl
        if done.any():
            s = stops[done]
            out_r[done] = (s - entry) / sl_d if long else (entry - s) / sl_d
            out_x[done] = k
            live[done] = False
            remaining -= int(done.sum())
            if remaining == 0:
                break

        if hit_tp:
            done = live.copy()
            if done.any():
                out_r[done] = rr
                out_x[done] = k
                live[done] = False
                remaining = 0
                break

        # ---- ratchet, effective from bar k+1 ----
        ak = atr[k]
        mv = live & (~moved) & (mfe >= BE_A)
        if mv.any():
            if long:
                stops[mv] = np.maximum(stops[mv], entry)
            else:
                stops[mv] = np.minimum(stops[mv], entry)
            moved[mv] = True
        if np.isfinite(ak) and ak > 0:
            tr_ = live & (mfe >= TS_A)
            if tr_.any():
                if long:
                    cand = hk - ak * TA_A[tr_]
                    stops[tr_] = np.maximum(stops[tr_], cand)
                else:
                    cand = lk + ak * TA_A[tr_]
                    stops[tr_] = np.minimum(stops[tr_], cand)
    return out_r, out_x


def work(args):
    sym, picks, cache_dir = args
    f = Path(cache_dir) / f"{sym}.parquet"
    if not f.exists():
        return []
    try:
        df = pd.read_parquet(f)
    except Exception:
        return []
    if df.empty or len(df) < 2500:
        return []
    df = bt.prepare_data(df)
    df["chop"] = chop_series(df)
    o = df.open.values.astype(float)
    h = df.high.values.astype(float)
    l = df.low.values.astype(float)
    c = df.close.values.astype(float)
    atr = df.atr.values.astype(float)
    ema = df.ema.values.astype(float)
    ch = df.chop.values.astype(float)
    ts = df.start.values
    n = len(c)
    rng = np.random.default_rng(abs(hash(sym)) % (2 ** 31))
    sigs = {}
    for s in bt.detect_signals(df):
        sigs.setdefault(s["type"], []).append(s)

    rows = []

    def emit(e, side, sl_d, rr, placebo):
        rv, xv = resolve_all(o, h, l, atr, e, side, sl_d, rr, n)
        if not np.isfinite(rv).all():
            return
        rec = {"entry_time": ts[e], "symbol": sym, "side": side,
               "entry_price": o[e],
               "sl_price": (o[e] - sl_d) if side == "long" else (o[e] + sl_d),
               "placebo": placebo, "rr": rr}
        for i, (name, *_) in enumerate(VARIANTS):
            rec[f"r_{name}"] = float(rv[i])
            rec[f"x_{name}"] = ts[int(xv[i])]
        rows.append(rec)

    for dt, (rr, am) in picks.items():
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
            if np.isfinite(ch[e]) and ch[e] >= CHOP_T:
                continue
            if not (np.isfinite(atr[bos]) and atr[bos] > 0):
                continue
            emit(e, side, atr[bos] * am, rr, False)
            pe = int(rng.integers(250, max(251, n - 2)))
            if np.isfinite(atr[pe]) and atr[pe] > 0:
                emit(pe, side, atr[pe] * am, rr, True)
    return rows


def main():
    if OUT.exists():
        print(f"[WIDE] {OUT.name} already exists — delete to rebuild")
        return 0
    cfg = yaml.safe_load(open(ROOT / "config.yaml"))["symbols"]
    jobs = []
    for s, sc in cfg.items():
        if not (sc or {}).get("enabled", True):
            continue
        picks = {c_["divergence_type"]: (float(c_["rr"]), float(c_["atr_mult"]))
                 for c_ in (sc or {}).get("configs", []) or []}
        if picks and (CACHE / f"{s}.parquet").exists():
            jobs.append((s, picks, str(CACHE)))
    jobs.sort()
    print(f"[WIDE] {len(jobs)} live symbols · {NV} exit rules + placebo", flush=True)
    rows = []
    with mp.Pool(max(1, mp.cpu_count() - 1)) as pool:
        for i, r in enumerate(pool.imap_unordered(work, jobs, chunksize=4), 1):
            rows.extend(r)
            if i % 40 == 0:
                print(f"  {i}/{len(jobs)} · {len(rows):,} rows", flush=True)
    d = pd.DataFrame(rows)
    d["entry_time"] = pd.to_datetime(d["entry_time"])
    d = d[d.entry_time >= GEN_FROM].reset_index(drop=True)

    b = pd.read_parquet(CACHE / "BTCUSDT.parquet").sort_values("start")
    b["ema200"] = b["close"].ewm(span=200, adjust=False).mean()
    bull = (b["close"] > b["ema200"]).shift(1).fillna(False)
    bull.index = b["start"]
    dd_ = b.set_index("start")["close"].resample("1D").last().dropna()
    imp = (dd_ / dd_.shift(30) - 1.0) > 0.10
    imp.index = imp.index + pd.Timedelta(days=1)
    imp_h = imp.reindex(pd.date_range(b["start"].min().floor("D"),
                                      b["start"].max().ceil("D") + pd.Timedelta(days=1),
                                      freq="h")).ffill().fillna(False)
    et = d.entry_time.dt.floor("h")
    d["btc_bull"] = et.map(bull.to_dict()).fillna(False).astype(bool)
    d["btc_impulse"] = et.map(imp_h.to_dict()).fillna(False).astype(bool)
    for name, *_ in VARIANTS:
        d[f"x_{name}"] = pd.to_datetime(d[f"x_{name}"])
    d.to_parquet(OUT, index=False)
    real = int((~d.placebo).sum())
    print(f"\n[WIDE] {len(d):,} rows ({real:,} real) · {d.symbol.nunique()} symbols · "
          f"{d.entry_time.min():%Y-%m} .. {d.entry_time.max():%Y-%m} -> {OUT.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
