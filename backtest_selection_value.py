#!/usr/bin/env python3
"""Did the walk-forward RR/ATR selection add out-of-sample value, or was it curve-fitting?

For every (symbol, div_type) pair the live config trades, this replays the FULL
(rr x atr_mult) grid — 4 x 3 = 12 variants — over the same candles, then asks:

    does the variant config.yaml actually CHOSE beat the average of the 12
    alternatives it was chosen over?

If the chosen variant beats the grid average in the fitting window but NOT after the
config was frozen and deployed (2026-05-25), the selection was fitting noise.

That is a much sharper test than "is the strategy losing", because it controls for
market conditions: every variant trades the same symbol over the same period, so a
bad market hurts chosen and unchosen alike.
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
RR_GRID = [3.0, 5.0, 8.0, 10.0]
ATR_GRID = [1.0, 1.5, 2.0]
ROUND_TRIP = 0.0018


def run_symbol(args):
    symbol, dtypes, cache_dir, t0, t1 = args
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

    sigs = {}
    for s in bt.detect_signals(df):
        sigs.setdefault(s["type"], []).append(s)

    out = []
    for dt in dtypes:
        for s in sigs.get(dt, []):
            conf_idx = s["conf_idx"]; side = s["side"]; lvl = s["swing"]
            # BOS index is independent of (rr, atr) — find it once
            bos = None
            for i in range(1, MAX_WAIT + 1):
                idx = conf_idx + i
                if idx >= n:
                    break
                if ((side == "long" and c[idx] > lvl) or
                        (side == "short" and c[idx] < lvl)):
                    bos = idx
                    break
            if bos is None:
                continue
            e_idx = bos + 1
            if e_idx >= n:
                continue
            atr = atr_a[bos]
            if not np.isfinite(atr) or atr <= 0:
                continue
            entry = o[e_idx]
            et = ts[e_idx]

            for am in ATR_GRID:
                sl_dist = atr * am
                if sl_dist <= 0:
                    continue
                sl = entry - sl_dist if side == "long" else entry + sl_dist
                fee_r = ROUND_TRIP * entry / sl_dist
                for rr in RR_GRID:
                    tp = entry + sl_dist * rr if side == "long" else entry - sl_dist * rr
                    r = None; xt = None
                    for k in range(e_idx, n):
                        if side == "long":
                            hs = lo[k] <= sl; ht = h[k] >= tp
                        else:
                            hs = h[k] >= sl; ht = lo[k] <= tp
                        if not (hs or ht):
                            continue
                        r = -1.0 if hs else rr        # SL wins ties
                        xt = ts[k]
                        break
                    if r is None:
                        continue
                    out.append({"symbol": symbol, "div_type": dt, "rr": rr,
                                "atr_mult": am, "entry_time": et, "exit_time": xt,
                                "r_net": r - fee_r})
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
    ap.add_argument("--start", default="2025-11-01")
    ap.add_argument("--end", default=None)
    ap.add_argument("--split", default="2026-05-25",
                    help="config freeze/deploy date separating fit from true OOS")
    ap.add_argument("--out", default="selection_value_grid.csv")
    ap.add_argument("--workers", type=int, default=max(1, mp.cpu_count() - 1))
    a = ap.parse_args()

    cfg = yaml.safe_load(open(ROOT / "config.yaml"))
    chosen = {}   # (symbol, div_type) -> (rr, atr_mult)
    jobs = []
    for sym, sc in (cfg.get("symbols") or {}).items():
        sc = sc or {}
        if not sc.get("enabled", True):
            continue
        cfgs = sc.get("configs") or []
        if not cfgs or not (ROOT / a.cache / f"{sym}.parquet").exists():
            continue
        dts = []
        for c_ in cfgs:
            chosen[(sym, c_["divergence_type"])] = (float(c_["rr"]), float(c_["atr_mult"]))
            dts.append(c_["divergence_type"])
        jobs.append((sym, dts, str(ROOT / a.cache), a.start, a.end))
    jobs.sort()
    print(f"[SEL] {len(jobs)} symbols · {len(chosen)} (symbol,div_type) pairs · "
          f"{len(RR_GRID)*len(ATR_GRID)} variants each", flush=True)

    rows = []
    with mp.Pool(a.workers) as pool:
        for i, res in enumerate(pool.imap_unordered(run_symbol, jobs, chunksize=4), 1):
            rows.extend(res)
            if i % 50 == 0:
                print(f"  {i}/{len(jobs)} · {len(rows)} rows", flush=True)

    d = pd.DataFrame(rows)
    d.to_csv(a.out, index=False)
    d["chosen"] = [chosen.get((s, t)) == (r, m) for s, t, r, m
                   in zip(d.symbol, d.div_type, d.rr, d.atr_mult)]
    print(f"[SEL] {len(d)} grid trades -> {a.out}\n")

    split = pd.Timestamp(a.split)
    for lbl, sub in [("FIT window (config selected on this)", d[d.entry_time < split]),
                     ("TRUE OOS (after 2026-05-25 deploy)",   d[d.entry_time >= split])]:
        if sub.empty:
            continue
        ch = sub[sub.chosen].r_net
        un = sub[~sub.chosen].r_net
        # per-pair paired comparison: chosen vs mean of that pair's alternatives
        g = sub.groupby(["symbol", "div_type"])
        deltas = []
        for _, gg in g:
            cc = gg[gg.chosen].r_net
            aa = gg[~gg.chosen].r_net
            if len(cc) >= 3 and len(aa) >= 3:
                deltas.append(cc.mean() - aa.mean())
        deltas = np.array(deltas)
        print(f"=== {lbl} ===")
        print(f"  chosen variants   : {len(ch):>7} trades  avgR {ch.mean():+.4f}")
        print(f"  the 11 alternates : {len(un):>7} trades  avgR {un.mean():+.4f}")
        print(f"  selection edge    : {ch.mean()-un.mean():+.4f} R/trade")
        if len(deltas):
            wins = (deltas > 0).mean()
            se = deltas.std(ddof=1) / np.sqrt(len(deltas))
            print(f"  paired per-pair   : n={len(deltas)}  mean delta {deltas.mean():+.4f} "
                  f"(SE {se:.4f}, t={deltas.mean()/se if se else 0:+.2f})")
            print(f"  chosen beat their own alternatives in {wins:.1%} of pairs")
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
