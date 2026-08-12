#!/usr/bin/env python3
import sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
import pandas as pd, numpy as np, yaml, multiprocessing as mp
import backtest_3yr_walkforward as bt
from backtest_halt_multiwindow import chop_series

CACHE = ROOT / "cache_3yr_1h"
MAX_WAIT = bt.MAX_WAIT_CANDLES
CHOP_T = 52.0


def replay(args):
    sym, picks = args
    f = CACHE / f"{sym}.parquet"
    if not f.exists():
        return []
    df = pd.read_parquet(f)
    if df.empty or len(df) < 2000:
        return []
    df = bt.prepare_data(df)
    df["chop"] = chop_series(df)
    o = df.open.values; h = df.high.values; l = df.low.values; c = df.close.values
    atr = df.atr.values; ema = df.ema.values; ch = df.chop.values
    n = len(c)
    sigs = {}
    for s in bt.detect_signals(df):
        sigs.setdefault(s["type"], []).append(s)
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
            for k in range(e, n):
                if side == "long":
                    hs, ht = l[k] <= sl, h[k] >= tp
                else:
                    hs, ht = h[k] >= sl, l[k] <= tp
                if hs or ht:
                    win = ht and not hs
                    out.append((k == e, win))
                    break
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
            jobs.append((s, picks))
    jobs.sort()
    rows = []
    with mp.Pool(max(1, mp.cpu_count() - 1)) as p:
        for r in p.imap_unordered(replay, jobs, chunksize=4):
            rows.extend(r)
    d = pd.DataFrame(rows, columns=["same_bar", "win"])
    print("total resolved:", len(d))
    print("same-bar resolved:", d.same_bar.sum(), f"{d.same_bar.mean():.2%}")
    print("overall win rate (all resolved):", round(d.win.mean(), 4))
    print("win rate WITHIN same-bar resolutions:", round(d.loc[d.same_bar, "win"].mean(), 4))
    print("win rate WITHIN non-same-bar resolutions:", round(d.loc[~d.same_bar, "win"].mean(), 4))


if __name__ == "__main__":
    main()
