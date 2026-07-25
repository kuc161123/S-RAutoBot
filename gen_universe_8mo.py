#!/usr/bin/env python3
"""Generate a trade universe for backtest_production_correct.run_simulation.

Replays the live config's 728 divergence configs over cached 1H klines and emits the
exact column set the production engine expects:

    entry_time, exit_time, entry_price, sl_price, outcome, r_result, side,
    symbol, div_type, rr_setting, atr_mult

r_result is RAW (+rr on win, -1.0 on loss) and entry_price/sl_price carry NO slippage —
the production engine applies fees, slippage and funding itself (see its line 652).

The --arm flag selects BOS confirmation timing:
    fixed  BOS on close(idx), ATR from candle idx,   entry open(idx+1)   [proposed]
    live   BOS on close(idx), ATR from candle idx+1, entry open(idx+2)   [current bot]

No CHOP filtering here — the production engine applies the regime-aware CHOP gate itself,
so the universe must contain every signal including the ones CHOP would block.
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
import backtest_3yr_walkforward as bt

ROOT = Path(__file__).parent
MAX_WAIT = bt.MAX_WAIT_CANDLES

ARMS = {"fixed": (0, 0), "live": (1, 1)}


def run_symbol(args):
    symbol, configs, cache_dir, atr_off, entry_off, t0, t1 = args
    f = Path(cache_dir) / f"{symbol}.parquet"
    if not f.exists():
        return []
    try:
        df = pd.read_parquet(f)
    except Exception:
        return []
    if df.empty or len(df) < 600:
        return []

    df = bt.prepare_data(df)
    o = df["open"].to_numpy(); h = df["high"].to_numpy()
    lo = df["low"].to_numpy(); c = df["close"].to_numpy()
    atr_a = df["atr"].to_numpy(); ts = df["start"].to_numpy()
    n = len(c)

    signals = bt.detect_signals(df)
    by_type = {}
    for s in signals:
        by_type.setdefault(s["type"], []).append(s)

    out = []
    for cfg in configs:
        dt = cfg["divergence_type"]; rr = float(cfg["rr"]); am = float(cfg["atr_mult"])
        for s in by_type.get(dt, []):
            conf_idx = s["conf_idx"]; side = s["side"]; lvl = s["swing"]
            for i in range(1, MAX_WAIT + 1):
                idx = conf_idx + i
                if idx >= n:
                    break
                if not ((side == "long" and c[idx] > lvl) or
                        (side == "short" and c[idx] < lvl)):
                    continue
                a_idx = idx + atr_off
                e_idx = idx + 1 + entry_off
                if e_idx >= n or a_idx >= n:
                    break
                atr = atr_a[a_idx]
                if not np.isfinite(atr) or atr <= 0:
                    break

                entry = o[e_idx]                      # RAW — engine adds slippage
                sl_dist = atr * am
                if side == "long":
                    sl = entry - sl_dist
                    tp = entry + sl_dist * rr
                else:
                    sl = entry + sl_dist
                    tp = entry - sl_dist * rr

                for k in range(e_idx, n):
                    if side == "long":
                        hit_sl = lo[k] <= sl; hit_tp = h[k] >= tp
                    else:
                        hit_sl = h[k] >= sl; hit_tp = lo[k] <= tp
                    if not (hit_sl or hit_tp):
                        continue
                    outcome = "loss" if hit_sl else "win"   # SL wins ties
                    out.append({
                        "entry_time": ts[e_idx], "exit_time": ts[k],
                        "entry_price": entry, "sl_price": sl,
                        "outcome": outcome,
                        "r_result": rr if outcome == "win" else -1.0,
                        "side": side, "symbol": symbol, "div_type": dt,
                        "rr_setting": rr, "atr_mult": am,
                    })
                    break
                break
    if not out:
        return []
    d = pd.DataFrame(out)
    d["entry_time"] = pd.to_datetime(d["entry_time"])
    if t0:
        d = d[d.entry_time >= pd.Timestamp(t0)]
    if t1:
        d = d[d.entry_time <= pd.Timestamp(t1)]
    return d.to_dict("records")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="cache_3yr_1h")
    ap.add_argument("--arm", choices=list(ARMS), default="live")
    ap.add_argument("--start", default=None)
    ap.add_argument("--end", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=max(1, mp.cpu_count() - 1))
    a = ap.parse_args()

    atr_off, entry_off = ARMS[a.arm]
    cfg = yaml.safe_load(open(ROOT / "config.yaml"))
    jobs = []
    for sym, sc in (cfg.get("symbols") or {}).items():
        sc = sc or {}
        if not sc.get("enabled", True):
            continue
        cfgs = sc.get("configs") or []
        if cfgs and (ROOT / a.cache / f"{sym}.parquet").exists():
            jobs.append((sym, cfgs, str(ROOT / a.cache), atr_off, entry_off,
                         a.start, a.end))
    jobs.sort()
    print(f"[UNIV] arm={a.arm} cache={a.cache} symbols={len(jobs)} "
          f"window={a.start}..{a.end}", flush=True)

    rows = []
    with mp.Pool(a.workers) as pool:
        for i, res in enumerate(pool.imap_unordered(run_symbol, jobs, chunksize=4), 1):
            rows.extend(res)
            if i % 50 == 0:
                print(f"  {i}/{len(jobs)} symbols · {len(rows)} trades", flush=True)

    if not rows:
        print("[UNIV] no trades")
        return 1
    df = pd.DataFrame(rows).sort_values("entry_time").reset_index(drop=True)
    df.to_csv(a.out, index=False)
    wr = (df.r_result > 0).mean()
    print(f"[UNIV] {len(df)} trades -> {a.out}")
    print(f"[UNIV] window {df.entry_time.min()} .. {df.entry_time.max()}")
    print(f"[UNIV] raw WR {wr:.2%}  raw avgR {df.r_result.mean():+.4f}  "
          f"(pre-cost, pre-portfolio)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
