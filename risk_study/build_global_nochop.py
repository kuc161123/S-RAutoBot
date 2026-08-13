#!/usr/bin/env python3
"""Build the GLOBAL-parameter arms on the live exit rule (s3_a1 trail).

Two variants, because "use a global rr/atr_mult" is ambiguous and the two readings are very
different bots:

  A1b  SAME-PAIRS  -- keep exactly the (symbol, div_type) pairs the live config selected,
                     but replace their fitted (rr, atr_mult) with one global choice.
                     This isolates the PARAMETER change. Same signals, same count.
  A1a  ALL-PAIRS   -- trade all 4 divergence types on all 277 symbols with the global
                     parameters. This is the natural reading, but it also roughly doubles
                     the trade count, so concurrency and the risk caps bind differently.
                     The difference vs A1b is a UNIVERSE change, not a parameter change.

Reporting both keeps those two effects from being confused for each other.

Exit rule, cost, CHOP gate and tie handling are identical to build_universe_trail.py.
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
from build_universe_trail_helpers import resolve_trail  # noqa: E402

HERE = Path(__file__).resolve().parent
CACHE = ROOT / "cache_3yr_1h"
GEN_FROM = pd.Timestamp("2023-06-01")
MAX_WAIT = bt.MAX_WAIT_CANDLES
CHOP_T = 999.0  # AUDIT: no gate; chop_bos carried on the row for sweeping
ALL_TYPES = ("REG_BULL", "REG_BEAR", "HID_BULL", "HID_BEAR")


def one(args):
    sym, types, rr, am, cache_dir = args
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

            long_ = side == "long"
            entry = o[e]
            risk_dist = atr[bos] * am
            sl = entry - risk_dist if long_ else entry + risk_dist
            tp = entry + risk_dist * rr if long_ else entry - risk_dist * rr
            if sl <= 0 or tp <= 0 or entry <= 0:
                continue

            tr = resolve_trail(e, n, long_, entry, sl, tp, risk_dist, h, l, atr)
            if tr is None:
                continue
            out.append((ts[e], sym, dt, side, entry, sl, rr, am,
                        ch[bos] if np.isfinite(ch[bos]) else np.nan, tr[0], ts[tr[1]]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rr", type=float, required=True)
    ap.add_argument("--atr", type=float, required=True)
    ap.add_argument("--pairs", choices=["same", "all"], default="same")
    a = ap.parse_args()

    cfg = yaml.safe_load(open(ROOT / "config.yaml"))["symbols"]
    jobs = []
    for s, sc in cfg.items():
        if not (sc or {}).get("enabled", True):
            continue
        live_types = [c["divergence_type"] for c in (sc or {}).get("configs", []) or []]
        if not live_types or not (CACHE / f"{s}.parquet").exists():
            continue
        types = tuple(live_types) if a.pairs == "same" else ALL_TYPES
        jobs.append((s, types, a.rr, a.atr, str(CACHE)))
    jobs.sort()
    print(f"[GLOBAL] rr={a.rr:g} atr={a.atr:g} pairs={a.pairs} · {len(jobs)} symbols", flush=True)

    rows = []
    with mp.Pool(max(1, mp.cpu_count() - 1)) as pool:
        for i, r in enumerate(pool.imap_unordered(one, jobs, chunksize=4), 1):
            rows.extend(r)
            if i % 60 == 0:
                print(f"  {i}/{len(jobs)} · {len(rows):,}", flush=True)

    d = pd.DataFrame(rows, columns=["entry_time", "symbol", "div_type", "side",
                                    "entry_price", "sl_price", "rr", "atr_mult",
                                    "chop_bos", "r_result", "exit_time"])
    d["entry_time"] = pd.to_datetime(d["entry_time"])
    d["exit_time"] = pd.to_datetime(d["exit_time"])
    d = d[d.entry_time >= GEN_FROM].sort_values("entry_time").reset_index(drop=True)
    d["stop_frac"] = (d.entry_price - d.sl_price).abs() / d.entry_price

    b = pd.read_parquet(CACHE / "BTCUSDT.parquet").sort_values("start")
    b["ema200"] = b["close"].ewm(span=200, adjust=False).mean()
    bull = (b["close"] > b["ema200"]).shift(1).fillna(False); bull.index = b["start"]
    dd_ = b.set_index("start")["close"].resample("1D").last().dropna()
    imp = (dd_ / dd_.shift(30) - 1.0) > 0.10
    imp.index = imp.index + pd.Timedelta(days=1)
    imp_h = imp.reindex(pd.date_range(b["start"].min().floor("D"),
                                      b["start"].max().ceil("D") + pd.Timedelta(days=1),
                                      freq="h")).ffill().fillna(False)
    et = d.entry_time.dt.floor("h")
    d["btc_bull"] = et.map(bull.to_dict()).fillna(False).astype(bool)
    d["btc_impulse"] = et.map(imp_h.to_dict()).fillna(False).astype(bool)

    kept = d[~(d.chop_bos.notna() & (d.chop_bos >= CHOP_T))].reset_index(drop=True)
    out = HERE / f"uni_nochop_rr{a.rr:g}_am{a.atr:g}_{a.pairs}.parquet"
    kept.to_parquet(out, index=False)
    net = (kept.r_result - 0.00242 / kept.stop_frac).mean()
    print(f"[GLOBAL] {out.name}: {len(kept):,} trades  gross {kept.r_result.mean():+.4f}  "
          f"net@24.2bps {net:+.4f}  WR {(kept.r_result > 0).mean()*100:.2f}%")


if __name__ == "__main__":
    main()
