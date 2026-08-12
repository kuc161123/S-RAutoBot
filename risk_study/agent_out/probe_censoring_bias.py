#!/usr/bin/env python3
"""Does right-censoring differentially bias the R of trades that DO resolve,
specifically among entries near the end of each symbol's data (last 21 days)?

Mechanism to test: near the data cutoff, slow-to-resolve winners (far TP,
RR up to 10) are more likely to still be open when data ends and get
silently dropped, while fast losers (stop is always closer than TP) still
resolve in time. So among entries in the last-21-days window, the resolved
subset should show excess losses vs the true (uncensored) population.
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
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import backtest_3yr_walkforward as bt  # noqa: E402
from backtest_halt_multiwindow import chop_series  # noqa: E402

CACHE = ROOT / "cache_3yr_1h"
MAX_WAIT = bt.MAX_WAIT_CANDLES
CHOP_T = 52.0
FEE = 0.00341  # honest round-trip per STRATEGY_VERDICT 2.3


def replay(args):
    sym, picks, cache_dir = args
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

    sym_last_ts = pd.Timestamp(ts[-1])
    cutoff21 = sym_last_ts - pd.Timedelta(days=21)

    out = []
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
            if np.isfinite(ch[bos]) and ch[bos] >= CHOP_T:
                continue
            if not (np.isfinite(atr[bos]) and atr[bos] > 0):
                continue

            entry = o[e]; sl_d = atr[bos] * am
            sl = entry - sl_d if side == "long" else entry + sl_d
            tp = entry + sl_d * rr if side == "long" else entry - sl_d * rr
            entry_ts = pd.Timestamp(ts[e])
            near_end = entry_ts >= cutoff21

            resolved = False
            for k in range(e, n):
                if side == "long":
                    hs, ht = l[k] <= sl, h[k] >= tp
                else:
                    hs, ht = h[k] >= sl, l[k] <= tp
                if hs or ht:
                    resolved = True
                    r_result = -1.0 if hs else rr
                    fee_r = FEE * entry / abs(entry - sl)
                    out.append((sym, entry_ts, near_end, True, r_result - fee_r))
                    break
            if not resolved:
                out.append((sym, entry_ts, near_end, False, np.nan))
    return out


def main():
    cfg = yaml.safe_load(open(ROOT / "config.yaml"))["symbols"]
    jobs = []
    for s, sc in cfg.items():
        if not (sc or {}).get("enabled", True):
            continue
        picks = {c["divergence_type"]: (float(c["rr"]), float(c["atr_mult"]))
                 for c in (sc or {}).get("configs", []) or []}
        if picks and (CACHE / f"{s}.parquet").exists():
            jobs.append((s, picks, str(CACHE)))
    jobs.sort()

    rows = []
    with mp.Pool(max(1, mp.cpu_count() - 1)) as pool:
        for i, r in enumerate(pool.imap_unordered(replay, jobs, chunksize=4), 1):
            rows.extend(r)

    d = pd.DataFrame(rows, columns=["symbol", "entry_time", "near_end21", "resolved", "r_net"])
    d.to_parquet(Path(__file__).parent / "censoring_bias_rows.parquet", index=False)

    print(f"total entered: {len(d):,}")
    for grp, label in [(~d.near_end21, "entries NOT in last-21d window"),
                        (d.near_end21, "entries IN last-21d window (per symbol)")]:
        sub = d[grp]
        n = len(sub)
        n_res = sub.resolved.sum()
        n_cens = n - n_res
        print(f"\n{label}: n={n:,}  resolved={n_res:,} ({n_res/max(1,n):.1%})  "
              f"censored={n_cens:,} ({n_cens/max(1,n):.1%})")
        if n_res:
            avg_r = sub.loc[sub.resolved, "r_net"].mean()
            print(f"  avg net R OF THE RESOLVED SUBSET: {avg_r:+.4f}")


if __name__ == "__main__":
    main()
