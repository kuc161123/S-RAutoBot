#!/usr/bin/env python3
"""Build the trade universe this risk study runs on — with the HONEST CHOP gate.

`halt_universe_live.parquet` on disk was built 2026-07-26 with `ch[e]` (the entry bar's
own CHOP), the lookahead documented in STRATEGY_VERDICT_2026-08-11.md 2.1. The builder in
`backtest_halt_multiwindow.py` has since been corrected to `ch[bos]`; this script drives
that corrected `replay()` and writes to a SEPARATE file so nothing existing is clobbered.

Also emits a no-CHOP arm so the gate itself can be ablated inside the risk sweep.

Outputs (risk_study/):
  universe_chopBOS.parquet   -- honest gate, ch[bos] < 52   <- the study basis
  universe_nochop.parquet    -- no CHOP gate at all
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
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import backtest_3yr_walkforward as bt  # noqa: E402
from backtest_halt_multiwindow import chop_series  # noqa: E402

CACHE = ROOT / "cache_3yr_1h"
OUT = Path(__file__).resolve().parent
GEN_FROM = pd.Timestamp("2023-06-01")
MAX_WAIT = bt.MAX_WAIT_CANDLES
CHOP_T = 52.0


def replay(args):
    """One symbol. Emits rows for BOTH arms, tagged with the honest CHOP value.

    Identical to backtest_halt_multiwindow.replay() except that the CHOP gate is not
    applied here — ch[bos] is carried on the row so the caller can filter. That keeps the
    two arms bit-identical in every other respect (same signals, same BOS, same ATR).
    """
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
                    # SL wins ties -- the pessimistic convention used repo-wide.
                    out.append((ts[e], ts[k], entry, sl, -1.0 if hs else rr,
                                side, sym, rr, am,
                                ch[bos] if np.isfinite(ch[bos]) else np.nan,
                                ch[e] if (e < n and np.isfinite(ch[e])) else np.nan))
                    break
    return out


def annotate_btc(d):
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
    return d


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
    print(f"[BUILD] {len(jobs)} symbols, live config, full history", flush=True)
    rows = []
    with mp.Pool(max(1, mp.cpu_count() - 1)) as pool:
        for i, r in enumerate(pool.imap_unordered(replay, jobs, chunksize=4), 1):
            rows.extend(r)
            if i % 40 == 0:
                print(f"  {i}/{len(jobs)} · {len(rows):,} rows", flush=True)

    d = pd.DataFrame(rows, columns=["entry_time", "exit_time", "entry_price", "sl_price",
                                    "r_result", "side", "symbol", "rr", "atr_mult",
                                    "chop_bos", "chop_entry"])
    d["entry_time"] = pd.to_datetime(d["entry_time"])
    d["exit_time"] = pd.to_datetime(d["exit_time"])
    d = d[d.entry_time >= GEN_FROM].sort_values("entry_time").reset_index(drop=True)
    d["stop_frac"] = (d.entry_price - d.sl_price).abs() / d.entry_price
    d = annotate_btc(d)

    d.to_parquet(OUT / "universe_nochop.parquet", index=False)
    kept = d[~((d.chop_bos.notna()) & (d.chop_bos >= CHOP_T))].reset_index(drop=True)
    kept.to_parquet(OUT / "universe_chopBOS.parquet", index=False)

    print(f"[BUILD] no-CHOP  {len(d):,} trades  {d.entry_time.min().date()}..{d.entry_time.max().date()}")
    print(f"[BUILD] ch[bos]  {len(kept):,} trades  ({len(kept)/len(d):.1%} kept)")
    for nm, x in (("nochop", d), ("chopBOS", kept)):
        gross = x.r_result.mean()
        fee18 = (0.0018 / x.stop_frac).mean()
        fee34 = (0.00341 / x.stop_frac).mean()
        print(f"  {nm:>8}  gross {gross:+.4f} R  fee@18bps {fee18:.4f}  "
              f"fee@34.1bps {fee34:.4f}  net@34.1 {gross-fee34:+.4f}")


if __name__ == "__main__":
    main()
