#!/usr/bin/env python3
"""Wide trailing universe resolved on 5-MINUTE bars.

Why this exists: with 1H candles, a bar that touches both the stop and the take-profit
has an unknowable order, so every resolver in this repo scores it as the STOP. That
convention is conservative, but it is not neutral between the two arms — a TRAILED stop
sits much closer to price than the original bracket stop, so it is touched far more
often, and therefore absorbs far more of the penalty. The 1H comparison is biased
AGAINST trailing by construction.

5m bars cut the ambiguous window by 12x, so this measures how much of the trailing
result was real and how much was that artefact.

What is modelled, and it matters that these differ:
  * SIGNALS come from the 1H frame — identical to build_trail_universe_wide.py, same
    config picks, same BOS/EMA/chop gates. The strategy is unchanged.
  * The TRAIL RATCHET is recomputed only at 1H boundaries, from the completed 1H
    candle's extreme and its 1H ATR. That is what the live bot can do (it polls hourly),
    so refreshing it every 5m would model a bot that does not exist.
  * Only EXIT DETECTION runs at 5m resolution, because stop and take-profit orders rest
    on the exchange and trigger intrabar regardless of how often the bot wakes up.

Residual ambiguity inside a single 5m bar still resolves SL-first.
"""
from __future__ import annotations

import math
import multiprocessing as mp
import os
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

CACHE_1H = ROOT / "cache_3yr_1h"
CACHE_5M = ROOT / "cache_5m"
# CHOP is filtered at BUILD time (a signal above the threshold never becomes a row), so
# ablating it requires a separate universe — run_simulation's own CHOP scenarios cannot
# undo a filter that already removed the signals. Env-overridable for exactly that.
OUT = ROOT / os.environ.get("TRAIL5M_OUT", "trail_5m_universe.parquet")
GEN_FROM = pd.Timestamp("2025-01-05")      # 5m cache starts 2025-01-01; leave ATR warmup
MAX_WAIT = bt.MAX_WAIT_CANDLES
FEE, SLIP = 0.0006, 0.0003
CHOP_T = float(os.environ.get("TRAIL5M_CHOP", "52.0"))

# (label, breakeven_trigger_R, trail_start_R, trail_atr_mult)
VARIANTS = [
    ("base",       None, None, None),
    ("s2_a1",      None, 2.0,  1.0),
    ("s2_a2",      None, 2.0,  2.0),
    ("s3_a1",      None, 3.0,  1.0),      # <-- the deployed rule
    ("s3_a2",      None, 3.0,  2.0),
    ("s4_a2",      None, 4.0,  2.0),
    ("be1",        1.0,  None, None),
    ("be1_s2_a1",  1.0,  2.0,  1.0),
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


def resolve_1h(h1_hi, h1_lo, h1_atr, e, side, sl_d, rr, entry, n1):
    """Same rule, resolved on 1H bars — the old convention, for a paired comparison.

    Emitted alongside the 5m result for EVERY signal so the two resolutions are matched
    by construction. Joining two separately-built universes on
    (symbol, entry_time, side) is not safe: a symbol can fire two configs on the same
    bar and side with different RR, so that key is not unique and the join silently
    pairs a trade against a different config's outcome.
    """
    long = side == "long"
    init_stop = entry - sl_d if long else entry + sl_d
    tp = entry + sl_d * rr if long else entry - sl_d * rr
    stops = np.full(NV, init_stop)
    moved = np.zeros(NV, dtype=bool)
    out_r = np.full(NV, np.nan)
    out_x = np.full(NV, -1, dtype=np.int64)
    live = np.ones(NV, dtype=bool)
    remaining = NV
    for j in range(e, n1):
        if long:
            bar_r = (h1_hi[j] - entry) / sl_d
            hit_sl, hit_tp = h1_lo[j] <= stops, h1_hi[j] >= tp
        else:
            bar_r = (entry - h1_lo[j]) / sl_d
            hit_sl, hit_tp = h1_hi[j] >= stops, h1_lo[j] <= tp
        done = live & hit_sl
        if done.any():
            sv = stops[done]
            out_r[done] = (sv - entry) / sl_d if long else (entry - sv) / sl_d
            out_x[done] = j
            live[done] = False
            remaining -= int(done.sum())
            if remaining == 0:
                return out_r, out_x
        if hit_tp and live.any():
            out_r[live] = rr
            out_x[live] = j
            live[:] = False
            return out_r, out_x
        mv = live & (~moved) & (bar_r >= BE_A)
        if mv.any():
            stops[mv] = np.maximum(stops[mv], entry) if long \
                else np.minimum(stops[mv], entry)
            moved[mv] = True
        a = h1_atr[j]
        if np.isfinite(a) and a > 0:
            tr_ = live & (bar_r >= TS_A)
            if tr_.any():
                if long:
                    stops[tr_] = np.maximum(stops[tr_], h1_hi[j] - a * TA_A[tr_])
                else:
                    stops[tr_] = np.minimum(stops[tr_], h1_lo[j] + a * TA_A[tr_])
    return out_r, out_x


def resolve_5m(h1_hi, h1_lo, h1_atr, m5_hi, m5_lo, m5_ts_h, h1_ts,
               e, side, sl_d, rr, entry):
    """Walk 5m bars for exits; ratchet the stop only on 1H boundaries.

    Returns (r[NV], exit_hour_idx[NV]); NaN where the trade never resolved.
    """
    n1 = len(h1_ts)
    long = side == "long"
    init_stop = entry - sl_d if long else entry + sl_d
    tp = entry + sl_d * rr if long else entry - sl_d * rr

    stops = np.full(NV, init_stop)
    moved = np.zeros(NV, dtype=bool)
    out_r = np.full(NV, np.nan)
    out_x = np.full(NV, -1, dtype=np.int64)
    live = np.ones(NV, dtype=bool)
    remaining = NV

    for j in range(e, n1):
        # --- 5m bars inside hour j: exits can trigger at any of them ---
        lo_i = np.searchsorted(m5_ts_h, h1_ts[j], side="left")
        hi_i = np.searchsorted(m5_ts_h, h1_ts[j], side="right")

        # Walk whatever 5m bars exist for this hour, then ALWAYS re-check the 1H bar as
        # a backstop. With complete coverage the backstop is a no-op, because the hour's
        # extremes are exactly the max/min of its 5m bars. With missing or partial
        # coverage (a later-listed symbol, a gap in the feed) it is what stops the trade
        # running on past an exit that really happened — which showed up as 5m exits
        # landing LATER than 1H ones, impossible if 5m only refines ordering.
        # The backstop keeps the conservative SL-first convention for that hour.
        for k in range(lo_i, hi_i + 1):
            if k < hi_i:
                bar_hi, bar_lo = m5_hi[k], m5_lo[k]
            else:
                bar_hi, bar_lo = h1_hi[j], h1_lo[j]          # backstop
            if long:
                hit_sl = bar_lo <= stops
                hit_tp = bar_hi >= tp
            else:
                hit_sl = bar_hi >= stops
                hit_tp = bar_lo <= tp
            done = live & hit_sl
            if done.any():
                s = stops[done]
                out_r[done] = (s - entry) / sl_d if long else (entry - s) / sl_d
                out_x[done] = j
                live[done] = False
                remaining -= int(done.sum())
                if remaining == 0:
                    return out_r, out_x
            if hit_tp and live.any():
                out_r[live] = rr
                out_x[live] = j
                live[:] = False
                return out_r, out_x

        # --- hour j has closed: ratchet from ITS extreme and ITS ATR (live semantics) ---
        if long:
            bar_r = (h1_hi[j] - entry) / sl_d
        else:
            bar_r = (entry - h1_lo[j]) / sl_d
        mv = live & (~moved) & (bar_r >= BE_A)
        if mv.any():
            stops[mv] = np.maximum(stops[mv], entry) if long \
                else np.minimum(stops[mv], entry)
            moved[mv] = True
        a = h1_atr[j]
        if np.isfinite(a) and a > 0:
            tr_ = live & (bar_r >= TS_A)
            if tr_.any():
                if long:
                    stops[tr_] = np.maximum(stops[tr_], h1_hi[j] - a * TA_A[tr_])
                else:
                    stops[tr_] = np.minimum(stops[tr_], h1_lo[j] + a * TA_A[tr_])
    return out_r, out_x


def work(args):
    sym, picks = args
    f1, f5 = CACHE_1H / f"{sym}.parquet", CACHE_5M / f"{sym}.parquet"
    if not (f1.exists() and f5.exists()):
        return []
    try:
        df = bt.prepare_data(pd.read_parquet(f1))
        d5 = pd.read_parquet(f5, columns=["start", "high", "low"])
    except Exception:
        return []
    if len(df) < 2500 or len(d5) < 5000:
        return []
    df["chop"] = chop_series(df)
    d5["start"] = pd.to_datetime(d5["start"])
    d5 = d5.sort_values("start")
    m5_ts_h = d5["start"].dt.floor("h").values.astype("datetime64[ns]")
    m5_hi = d5["high"].to_numpy(dtype=float)
    m5_lo = d5["low"].to_numpy(dtype=float)

    ts = df["start"].values.astype("datetime64[ns]")
    o = df.open.values.astype(float)
    h = df.high.values.astype(float)
    l = df.low.values.astype(float)
    c = df.close.values.astype(float)
    atr = df.atr.values.astype(float)
    ema = df.ema.values.astype(float)
    ch = df.chop.values.astype(float)
    n = len(df)

    # only signals whose entry hour is covered by the 5m cache
    first5 = m5_ts_h[0]
    last5 = m5_ts_h[-1]

    sigs = {}
    for s in bt.detect_signals(df):
        sigs.setdefault(s["type"], []).append(s)

    rows = []
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
            if ts[e] < first5 or ts[e] > last5:
                continue
            sl_d = atr[bos] * am
            entry = o[e] * (1 + SLIP) if side == "long" else o[e] * (1 - SLIP)
            rv, xv = resolve_5m(h, l, atr, m5_hi, m5_lo, m5_ts_h, ts,
                                e, side, sl_d, rr, entry)
            r1, x1 = resolve_1h(h, l, atr, e, side, sl_d, rr, entry, n)
            # require BOTH resolutions, so the paired comparison never silently
            # compares a resolved trade against an unresolved one
            if not (np.isfinite(rv).all() and np.isfinite(r1).all()):
                continue
            rec = {"entry_time": ts[e], "symbol": sym, "side": side,
                   "entry_price": o[e],
                   "sl_price": (o[e] - sl_d) if side == "long" else (o[e] + sl_d),
                   "rr": rr}
            for i, (name, *_) in enumerate(VARIANTS):
                rec[f"r_{name}"] = float(rv[i])
                rec[f"x_{name}"] = ts[int(xv[i])]
                rec[f"r1h_{name}"] = float(r1[i])
                rec[f"x1h_{name}"] = ts[int(x1[i])]
            rows.append(rec)
    return rows


def main():
    if OUT.exists():
        print(f"[5M] {OUT.name} exists — delete to rebuild")
        return 0
    cfg = yaml.safe_load(open(ROOT / "config.yaml"))["symbols"]
    jobs = []
    for s, sc in cfg.items():
        if not (sc or {}).get("enabled", True):
            continue
        picks = {c_["divergence_type"]: (float(c_["rr"]), float(c_["atr_mult"]))
                 for c_ in (sc or {}).get("configs", []) or []}
        if picks and (CACHE_1H / f"{s}.parquet").exists() \
                and (CACHE_5M / f"{s}.parquet").exists():
            jobs.append((s, picks))
    jobs.sort()
    print(f"[5M] {len(jobs)} symbols with both caches · {NV} exit rules", flush=True)
    rows = []
    with mp.Pool(max(1, mp.cpu_count() - 1)) as pool:
        for i, r in enumerate(pool.imap_unordered(work, jobs, chunksize=2), 1):
            rows.extend(r)
            if i % 25 == 0:
                print(f"  {i}/{len(jobs)} · {len(rows):,} rows", flush=True)
    d = pd.DataFrame(rows)
    d["entry_time"] = pd.to_datetime(d["entry_time"])
    d = d[d.entry_time >= GEN_FROM].reset_index(drop=True)

    b = pd.read_parquet(CACHE_1H / "BTCUSDT.parquet").sort_values("start")
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
        d[f"x1h_{name}"] = pd.to_datetime(d[f"x1h_{name}"])
    d.to_parquet(OUT, index=False)
    print(f"\n[5M] {len(d):,} signals · {d.symbol.nunique()} symbols · "
          f"{d.entry_time.min():%Y-%m-%d} .. {d.entry_time.max():%Y-%m-%d} -> {OUT.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
