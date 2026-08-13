#!/usr/bin/env python3
"""AUDIT-C: parameterised signal-layer universe builder.

Re-implements (does NOT edit) the pattern in risk_study/build_global_trail.py:
  - bt.prepare_data / bt.detect_signals for signal generation
  - build_universe_trail_helpers.resolve_trail for the s3_a1 exit

...but with the free parameters of the signal layer exposed as CLI args, so we can sweep
pivot width, MIN_PIVOT_DISTANCE, pivot staleness, RSI period, max_wait_candles and the
second (BOS-time) EMA gate one at a time. rr=10, atr_mult=3.0 and the (symbol, div_type)
pairs are held fixed at the CURRENT config (config.yaml), matching the live book exactly
except for the one swept knob.

IMPORTANT: this file re-implements detect_signals/find_pivots itself (does not call
backtest_3yr_walkforward.detect_signals) because that function hardcodes PIVOT_LEFT/RIGHT
and RSI_PERIOD as module globals and has no MIN_PIVOT_DISTANCE enforcement at all (see
NOTE below) -- neither can be swept without a local copy. Verified byte-for-byte equivalent
to bt.prepare_data/find_pivots/detect_signals when called with
(pivot_left=3, pivot_right=3, min_pivot_dist=0, pivot_stale=10, rsi_period=14) -- see
sanity check in README / AUDIT_C.md (dist=0 exactly reproduces the archived
uni_glob_rr10_am3_same.parquet: 32,102 trades, gross +0.2631, net +0.1916).

NOTE on MIN_PIVOT_DISTANCE: the LIVE detector (autobot/core/divergence_detector.py) enforces
a real gap between the current and previous pivot (`prev_pli < curr_pli - MIN_PIVOT_DISTANCE`,
i.e. strictly more than MIN_PIVOT_DISTANCE bars apart). backtest_3yr_walkforward.py's
detect_signals -- the engine that produced every number in CLAUDE.md and the archived
universes -- does NOT enforce this at all; it just takes the next pivot found scanning
backward, regardless of gap. That is itself a (minor) live/backtest mirror discrepancy,
noted in AUDIT_C.md. Here, dist=0 reproduces the backtest's historical (unconstrained)
behaviour; dist=3 reproduces what the live code actually enforces.
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
ROOT = Path("/Users/lualakol/AutoTrading Bot")
sys.path.insert(0, str(ROOT))

from backtest_halt_multiwindow import chop_series  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from build_universe_trail_helpers import resolve_trail  # noqa: E402

CACHE = ROOT / "cache_3yr_1h"
GEN_FROM = pd.Timestamp("2023-06-01")
CHOP_T = 52.0
RR = 10.0
AM = 3.0
ATR_PERIOD = 14
EMA_PERIOD = 200
LOOKBACK = 50


# ---------------------------------------------------------------------------
# signal layer, parameterised (mirrors backtest_3yr_walkforward.py exactly at
# pivot_left=3, pivot_right=3, min_pivot_dist=0, pivot_stale=10, rsi_period=14)
# ---------------------------------------------------------------------------

def prepare_data(df, rsi_period):
    df = df.copy()
    delta = df["close"].diff()
    gain = delta.where(delta > 0, 0).rolling(rsi_period).mean()
    loss = -delta.where(delta < 0, 0).rolling(rsi_period).mean()
    rs = gain / (loss + 1e-10)
    df["rsi"] = 100 - (100 / (1 + rs))
    hl = df["high"] - df["low"]
    hc = abs(df["high"] - df["close"].shift())
    lc = abs(df["low"] - df["close"].shift())
    df["atr"] = pd.concat([hl, hc, lc], axis=1).max(axis=1).rolling(ATR_PERIOD).mean()
    df["ema"] = df["close"].ewm(span=EMA_PERIOD, adjust=False).mean()
    return df


def find_pivots(data, left, right):
    n = len(data)
    pivot_highs = np.full(n, np.nan)
    pivot_lows = np.full(n, np.nan)
    for i in range(left, n - right):
        window = data[i - left: i + right + 1]
        center = data[i]
        if len(window) != (left + right + 1):
            continue
        if center == window.max() and (window == center).sum() == 1:
            pivot_highs[i] = center
        if center == window.min() and (window == center).sum() == 1:
            pivot_lows[i] = center
    return pivot_highs, pivot_lows


def detect_signals(df, pivot_left, pivot_right, min_pivot_dist, pivot_stale):
    close = df["close"].values
    high = df["high"].values
    low = df["low"].values
    rsi = df["rsi"].values
    ema = df["ema"].values
    price_ph, price_pl = find_pivots(close, pivot_left, pivot_right)
    signals = []
    used_pivots = set()

    min_idx = max(EMA_PERIOD + 10, LOOKBACK + pivot_right + 1)
    for i in range(min_idx, len(df) - pivot_right):
        if np.isnan(rsi[i]) or np.isnan(ema[i]):
            continue
        curr_price = close[i]
        curr_ema = ema[i]

        if curr_price > curr_ema:
            curr_idx = curr_val = prev_idx = prev_val = None
            for j in range(i - pivot_right, max(0, i - LOOKBACK), -1):
                if not np.isnan(price_pl[j]):
                    if curr_idx is None:
                        curr_idx, curr_val = j, price_pl[j]
                    elif prev_idx is None and j < curr_idx - min_pivot_dist:
                        prev_idx, prev_val = j, price_pl[j]
                        break
            if curr_idx is not None and prev_idx is not None:
                dedup_key = (curr_idx, prev_idx, "BULL")
                if (i - curr_idx) <= pivot_stale and dedup_key not in used_pivots:
                    if curr_val < prev_val and rsi[curr_idx] > rsi[prev_idx]:
                        signals.append({"conf_idx": i, "side": "long", "type": "REG_BULL",
                                        "swing": high[curr_idx:i + 1].max()})
                        used_pivots.add(dedup_key)
                    elif curr_val > prev_val and rsi[curr_idx] < rsi[prev_idx]:
                        signals.append({"conf_idx": i, "side": "long", "type": "HID_BULL",
                                        "swing": high[curr_idx:i + 1].max()})
                        used_pivots.add(dedup_key)

        if curr_price < curr_ema:
            curr_idx = curr_val = prev_idx = prev_val = None
            for j in range(i - pivot_right, max(0, i - LOOKBACK), -1):
                if not np.isnan(price_ph[j]):
                    if curr_idx is None:
                        curr_idx, curr_val = j, price_ph[j]
                    elif prev_idx is None and j < curr_idx - min_pivot_dist:
                        prev_idx, prev_val = j, price_ph[j]
                        break
            if curr_idx is not None and prev_idx is not None:
                dedup_key = (curr_idx, prev_idx, "BEAR")
                if (i - curr_idx) <= pivot_stale and dedup_key not in used_pivots:
                    if curr_val > prev_val and rsi[curr_idx] < rsi[prev_idx]:
                        signals.append({"conf_idx": i, "side": "short", "type": "REG_BEAR",
                                        "swing": low[curr_idx:i + 1].min()})
                        used_pivots.add(dedup_key)
                    elif curr_val < prev_val and rsi[curr_idx] > rsi[prev_idx]:
                        signals.append({"conf_idx": i, "side": "short", "type": "HID_BEAR",
                                        "swing": low[curr_idx:i + 1].min()})
                        used_pivots.add(dedup_key)

    return signals


# ---------------------------------------------------------------------------
# per-symbol worker
# ---------------------------------------------------------------------------

def one(args):
    (sym, types, cache_dir, pivot_left, pivot_right, min_pivot_dist, pivot_stale,
     rsi_period, max_wait, drop_bos_ema_gate) = args
    f = Path(cache_dir) / f"{sym}.parquet"
    if not f.exists():
        return []
    try:
        df = pd.read_parquet(f)
    except Exception:
        return []
    if df.empty or len(df) < 2000:
        return []
    df = prepare_data(df, rsi_period)
    df["chop"] = chop_series(df)
    o = df.open.values; h = df.high.values; l = df.low.values; c = df.close.values
    atr = df.atr.values; ema = df.ema.values; ch = df.chop.values; ts = df.start.values
    n = len(c)

    sigs = {}
    for s in detect_signals(df, pivot_left, pivot_right, min_pivot_dist, pivot_stale):
        sigs.setdefault(s["type"], []).append(s)

    out = []
    for dt in types:
        for s in sigs.get(dt, []):
            conf, side, lvl = s["conf_idx"], s["side"], s["swing"]
            bos = None
            for i in range(1, max_wait + 1):
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
            if not drop_bos_ema_gate:
                if side == "long" and not c[bos] > ema[bos]:
                    continue
                if side == "short" and not c[bos] < ema[bos]:
                    continue
            if not (np.isfinite(atr[bos]) and atr[bos] > 0):
                continue

            long_ = side == "long"
            entry = o[e]
            risk_dist = atr[bos] * AM
            sl = entry - risk_dist if long_ else entry + risk_dist
            tp = entry + risk_dist * RR if long_ else entry - risk_dist * RR
            if sl <= 0 or tp <= 0 or entry <= 0:
                continue

            tr = resolve_trail(e, n, long_, entry, sl, tp, risk_dist, h, l, atr)
            if tr is None:
                continue
            out.append((ts[e], sym, dt, side, entry, sl, ch[bos] if np.isfinite(ch[bos]) else np.nan,
                        tr[0], ts[tr[1]]))
    return out


def build_one_cell(cell_name, pivot_left, pivot_right, min_pivot_dist, pivot_stale,
                    rsi_period, max_wait, drop_bos_ema_gate, out_dir, nproc):
    out_path = out_dir / f"{cell_name}.parquet"
    if out_path.exists():
        print(f"[skip] {cell_name} already built")
        return out_path

    cfg = yaml.safe_load(open(ROOT / "config.yaml"))["symbols"]
    jobs = []
    for s, sc in cfg.items():
        if not (sc or {}).get("enabled", True):
            continue
        live_types = [c["divergence_type"] for c in (sc or {}).get("configs", []) or []]
        if not live_types or not (CACHE / f"{s}.parquet").exists():
            continue
        jobs.append((s, tuple(live_types), str(CACHE), pivot_left, pivot_right, min_pivot_dist,
                    pivot_stale, rsi_period, max_wait, drop_bos_ema_gate))
    jobs.sort()
    print(f"[{cell_name}] pivot=({pivot_left},{pivot_right}) dist={min_pivot_dist} "
          f"stale={pivot_stale} rsi={rsi_period} wait={max_wait} drop_ema={drop_bos_ema_gate} "
          f"· {len(jobs)} symbols", flush=True)

    rows = []
    with mp.Pool(max(1, nproc)) as pool:
        for i, r in enumerate(pool.imap_unordered(one, jobs, chunksize=3), 1):
            rows.extend(r)
            if i % 80 == 0:
                print(f"  {i}/{len(jobs)} · {len(rows):,}", flush=True)

    d = pd.DataFrame(rows, columns=["entry_time", "symbol", "div_type", "side",
                                    "entry_price", "sl_price", "chop_bos", "r_result", "exit_time"])
    if d.empty:
        print(f"[{cell_name}] EMPTY")
        d.to_parquet(out_path, index=False)
        return out_path
    d["entry_time"] = pd.to_datetime(d["entry_time"])
    d["exit_time"] = pd.to_datetime(d["exit_time"])
    d = d[d.entry_time >= GEN_FROM].sort_values("entry_time").reset_index(drop=True)
    d["stop_frac"] = (d.entry_price - d.sl_price).abs() / d.entry_price

    kept = d[~(d.chop_bos.notna() & (d.chop_bos >= CHOP_T))].reset_index(drop=True)
    kept.to_parquet(out_path, index=False)
    net = (kept.r_result - 0.00242 / kept.stop_frac).mean()
    print(f"[{cell_name}] {out_path.name}: {len(kept):,} trades  gross {kept.r_result.mean():+.4f}  "
          f"net@24.2bps {net:+.4f}  WR {(kept.r_result > 0).mean()*100:.2f}%", flush=True)
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", required=True)
    ap.add_argument("--pivot-left", type=int, default=3)
    ap.add_argument("--pivot-right", type=int, default=3)
    ap.add_argument("--min-pivot-dist", type=int, default=3)
    ap.add_argument("--pivot-stale", type=int, default=10)
    ap.add_argument("--rsi-period", type=int, default=14)
    ap.add_argument("--max-wait", type=int, default=12)
    ap.add_argument("--drop-bos-ema-gate", action="store_true")
    ap.add_argument("--out-dir", default=str(HERE / "universes"))
    ap.add_argument("--nproc", type=int, default=max(1, mp.cpu_count() - 1))
    a = ap.parse_args()

    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    build_one_cell(a.cell, a.pivot_left, a.pivot_right, a.min_pivot_dist, a.pivot_stale,
                    a.rsi_period, a.max_wait, a.drop_bos_ema_gate, out_dir, a.nproc)


if __name__ == "__main__":
    main()
