#!/usr/bin/env python3
"""Would a 3x ATR stop help — for ALL symbols? Single-variable test vs base.

Each symbol KEEPS its live RR pick; only atr_mult changes. That isolates stop width from
every other decision (unlike the earlier test, which changed stop width, RR and the
per-symbol layer all at once and was therefore uninterpretable).

Last 10 months, $1,500 start, through the production engine so the answer is in dollars
with real compounding, margin and the live overlay stack.

Rationale for testing this at all: fee_r = round_trip_cost x entry / sl_distance, so the
stop distance is the DENOMINATOR of the cost drag. At 1.0x ATR the drag is 0.1526 R/trade
against a ~0.155 R gross edge; at 3.0x it is 0.0509. 336 of the 756 live configs sit at
1.0x ATR. Whether that translates into dollars is what this measures.
"""
from __future__ import annotations

import argparse
import math
import multiprocessing as mp
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

warnings.filterwarnings("ignore")
import backtest_3yr_walkforward as bt
import backtest_production_correct as P
from backtest_shadow_gate import LIVE as LIVE_SIM

ROOT = Path(__file__).parent
CACHE = ROOT / "cache_3yr_1h"
START_BAL = 1500.0
GEN_FROM = pd.Timestamp("2025-08-01")
W10 = pd.Timestamp("2025-09-26")                 # last 10 months
ATR_VARIANTS = [None, 1.5, 2.0, 3.0, 4.0]        # None = keep each symbol's live atr_mult
MAX_WAIT = bt.MAX_WAIT_CANDLES
CHOP_T = 52.0
OUT = ROOT / "atr3_universe.parquet"


def chop_series(df, period=14):
    hl = df["high"] - df["low"]
    hc = (df["high"] - df["close"].shift()).abs()
    lc = (df["low"] - df["close"].shift()).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    hi = df["high"].rolling(period).max(); lo = df["low"].rolling(period).min()
    return (100 * np.log10(tr.rolling(period).sum() / (hi - lo).replace(0, np.nan))
            / math.log10(period))


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
    out = []
    for dt, (rr, live_am) in picks.items():
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
            entry = o[e]
            for variant in ATR_VARIANTS:
                am = live_am if variant is None else variant
                sl_d = atr[bos] * am
                sl = entry - sl_d if side == "long" else entry + sl_d
                tp = entry + sl_d * rr if side == "long" else entry - sl_d * rr
                for k in range(e, n):
                    if side == "long":
                        hs, ht = l[k] <= sl, h[k] >= tp
                    else:
                        hs, ht = h[k] >= sl, l[k] <= tp
                    if hs or ht:
                        out.append((ts[e], ts[k], entry, sl, -1.0 if hs else rr,
                                    side, sym, "LIVE" if variant is None else str(variant)))
                        break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--workers", type=int, default=max(1, mp.cpu_count() - 1))
    a = ap.parse_args()

    if a.rebuild or not OUT.exists():
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
        print(f"[ATR3] {len(jobs)} symbols · variants {ATR_VARIANTS}", flush=True)
        rows = []
        with mp.Pool(a.workers) as pool:
            for i, r in enumerate(pool.imap_unordered(replay, jobs, chunksize=4), 1):
                rows.extend(r)
                if i % 50 == 0:
                    print(f"  {i}/{len(jobs)} · {len(rows):,}", flush=True)
        d = pd.DataFrame(rows, columns=["entry_time", "exit_time", "entry_price",
                                        "sl_price", "r_result", "side", "symbol", "arm"])
        d["entry_time"] = pd.to_datetime(d["entry_time"])
        d["exit_time"] = pd.to_datetime(d["exit_time"])
        d = d[d.entry_time >= GEN_FROM].reset_index(drop=True)
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
        d.to_parquet(OUT, index=False)
    else:
        d = pd.read_parquet(OUT)

    chop = P.load_chop_data(sorted(d.symbol.unique()))
    cols = ["entry_time", "exit_time", "entry_price", "sl_price", "r_result",
            "side", "symbol", "btc_bull", "btc_impulse"]

    for wlbl, lo, note in [
        ("LAST 10 MONTHS (2025-09-26 -> today)", W10, "as requested"),
        ("GENUINELY OOS (2026-05-25 -> today)", pd.Timestamp("2026-05-25"),
         "after the live config was deployed")]:
        seg = d[d.entry_time >= lo]
        print(f"\n{'=' * 96}\n{wlbl}   ${START_BAL:,.0f} start   ({note})\n{'=' * 96}")
        print(f"  {'stop width':<26}{'final $':>11}{'growth $':>12}{'ROI':>10}"
              f"{'maxDD':>8}{'ROI/DD':>8}{'PF':>7}{'trades':>8}")
        for variant in ATR_VARIANTS:
            arm = "LIVE" if variant is None else str(variant)
            sub = seg[seg.arm == arm]
            if sub.empty:
                continue
            kw = dict(LIVE_SIM, starting_balance=START_BAL,
                      btc_bull_col="btc_bull", btc_short_col="btc_impulse")
            P.STARTING_BALANCE = START_BAL
            r = P.run_simulation(sub[cols].sort_values("entry_time").reset_index(drop=True),
                                 chop, **kw)
            t = r["entered_trades"]
            gp = sum(x["pnl"] for x in t if x["pnl"] > 0)
            gl = abs(sum(x["pnl"] for x in t if x["pnl"] < 0))
            fin = r["final_effective"]; roi = fin / START_BAL - 1; ddp = r["max_dd_pct"]
            lbl = "BASE (live per-symbol atr)" if variant is None else f"ALL symbols @ {variant}x ATR"
            print(f"  {lbl:<26}{fin:>11,.0f}{fin-START_BAL:>+12,.0f}{roi:>10.1%}"
                  f"{ddp:>7.1f}%{(roi*100/ddp if ddp else 0):>8.2f}"
                  f"{(gp/gl if gl else 9.99):>7.2f}{len(t):>8,}", flush=True)
        print(f"  {'never trade':<26}{START_BAL:>11,.0f}{0:>+12,.0f}{0.0:>10.1%}"
              f"{0.0:>7.1f}%{0.0:>8.2f}{'—':>7}{0:>8}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
