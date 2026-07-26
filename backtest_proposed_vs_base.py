#!/usr/bin/env python3
"""Current bot vs proposed rebuild — head to head, $1,500, last 10 months.

Ablation so each change can be attributed:

  BASE            live config exactly as deployed: 728 per-symbol (rr, atr) picks,
                  20-trade regime sizing, no concurrency cap
  + cap 20        BASE with a max-concurrent-position cap
  + flat sizing   BASE with the 20-trade regime multiplier removed (taper kept)
  + global geom   BASE's per-symbol picks replaced by ONE global (atr, rr)
  PROPOSED        all three together

CONTAMINATION — stated up front, because both sides are partly fitted here:
  * the LIVE config was walk-forward fitted on data through 2026-05-25
  * the PROPOSED geometry was chosen on data before 2026-02-01
So the 10-month window flatters BASE more than PROPOSED. Three windows are reported:
  A 10 months  — both partly in-sample (BASE favoured)
  B from 02-01 — clean for PROPOSED, still in-sample for BASE (BASE favoured)
  C from 05-25 — clean for BOTH; this is the only fair one
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
GEN_FROM = pd.Timestamp("2025-08-01")          # warm-up before the 10-month window
W10 = pd.Timestamp("2025-09-26")               # last 10 months
RR_GRID = [3.0, 5.0, 6.0, 8.0, 10.0]
ATR_GRID = [1.0, 1.5, 2.0, 3.0]
DIVS = ["REG_BULL", "REG_BEAR", "HID_BULL", "HID_BEAR"]
MAX_WAIT = bt.MAX_WAIT_CANDLES
SLIP = 0.0003
CHOP_T = 52.0
PROPOSED_GEOM = (2.0, 6.0)      # holdout-best of the pre-committed set; design-derived


def chop_series(df, period=14):
    hl = df["high"] - df["low"]
    hc = (df["high"] - df["close"].shift()).abs()
    lc = (df["low"] - df["close"].shift()).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    hi = df["high"].rolling(period).max(); lo = df["low"].rolling(period).min()
    return (100 * np.log10(tr.rolling(period).sum() / (hi - lo).replace(0, np.nan))
            / math.log10(period))


def replay(args):
    sym, cache_dir = args
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
    lo_i = np.searchsorted(ts, np.datetime64(GEN_FROM))
    sigs = {}
    for s in bt.detect_signals(df):
        sigs.setdefault(s["type"], []).append(s)
    out = []
    for dt in DIVS:
        for s in sigs.get(dt, []):
            conf, side, lvl = s["conf_idx"], s["side"], s["swing"]
            if conf < lo_i - 400:
                continue
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
            for am in ATR_GRID:
                sl_d = atr[bos] * am
                sl = entry - sl_d if side == "long" else entry + sl_d
                for rr in RR_GRID:
                    tp = entry + sl_d * rr if side == "long" else entry - sl_d * rr
                    r = None; xi = None
                    for k in range(e, n):
                        if side == "long":
                            hs, ht = l[k] <= sl, h[k] >= tp
                        else:
                            hs, ht = h[k] >= sl, l[k] <= tp
                        if hs or ht:
                            r = -1.0 if hs else rr; xi = k
                            break
                    if r is None:
                        continue
                    out.append((ts[e], ts[xi], entry, sl, r, side, sym, dt, rr, am))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=max(1, mp.cpu_count() - 1))
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    cfg = yaml.safe_load(open(ROOT / "config.yaml"))["symbols"]
    live_map, syms = {}, []
    for s, sc in cfg.items():
        if not (sc or {}).get("enabled", True):
            continue
        for c_ in (sc or {}).get("configs", []) or []:
            live_map[(s, c_["divergence_type"])] = (float(c_["rr"]), float(c_["atr_mult"]))
        if (CACHE / f"{s}.parquet").exists():
            syms.append(s)
    syms.sort()
    if a.limit:
        syms = syms[: a.limit]
    print(f"[VS] {len(syms)} symbols · proposed geometry "
          f"{PROPOSED_GEOM[0]}x ATR / rr{PROPOSED_GEOM[1]:.0f}", flush=True)

    rows = []
    with mp.Pool(a.workers) as pool:
        for i, r in enumerate(pool.imap_unordered(
                replay, [(s, str(CACHE)) for s in syms], chunksize=4), 1):
            rows.extend(r)
            if i % 50 == 0:
                print(f"  {i}/{len(syms)} · {len(rows):,}", flush=True)

    d = pd.DataFrame(rows, columns=["entry_time", "exit_time", "entry_price", "sl_price",
                                    "r_result", "side", "symbol", "div_type", "rr",
                                    "atr_mult"])
    d["entry_time"] = pd.to_datetime(d["entry_time"]); d["exit_time"] = pd.to_datetime(d["exit_time"])

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

    key = list(zip(d.symbol, d.div_type))
    d["is_live"] = [live_map.get(k) == (rr, am) for k, rr, am in zip(key, d.rr, d.atr_mult)]
    pg_rr, pg_am = PROPOSED_GEOM[1], PROPOSED_GEOM[0]
    d["is_global"] = (d.rr == pg_rr) & (d.atr_mult == pg_am)

    chop = P.load_chop_data(sorted(d.symbol.unique()))

    ARMS = [
        ("BASE (live, as deployed)",      "live",   {}),
        ("BASE + concurrency cap 20",     "live",   dict(max_concurrent=20)),
        ("BASE + flat sizing (no regime)", "live",  dict(scenario="chop_only")),
        ("BASE + global geometry",        "global", {}),
        ("PROPOSED (all three)",          "global", dict(max_concurrent=20,
                                                        scenario="chop_only")),
    ]

    for wlbl, lo, note in [
        ("A) LAST 10 MONTHS (2025-09-26 -> today)", W10,
         "both partly in-sample — favours BASE"),
        ("B) FROM 2026-02-01", pd.Timestamp("2026-02-01"),
         "clean for PROPOSED, still in-sample for BASE — favours BASE"),
        ("C) FROM 2026-05-25", pd.Timestamp("2026-05-25"),
         "clean for BOTH — the only fair comparison")]:
        seg = d[d.entry_time >= lo]
        print(f"\n{'=' * 100}\n{wlbl}   ${START_BAL:,.0f} start   ({note})\n{'=' * 100}")
        print(f"  {'variant':<32}{'final $':>10}{'growth $':>11}{'ROI':>9}"
              f"{'maxDD':>8}{'ROI/DD':>8}{'PF':>7}{'trades':>8}")
        for lbl, sel, over in ARMS:
            sub = seg[seg.is_live] if sel == "live" else seg[seg.is_global]
            if sub.empty:
                continue
            cols = ["entry_time", "exit_time", "entry_price", "sl_price", "r_result",
                    "side", "symbol", "btc_bull", "btc_impulse"]
            kw = dict(LIVE_SIM, starting_balance=START_BAL,
                      btc_bull_col="btc_bull", btc_short_col="btc_impulse")
            kw.update(over)
            P.STARTING_BALANCE = START_BAL
            r = P.run_simulation(sub[cols].sort_values("entry_time").reset_index(drop=True),
                                 chop, **kw)
            t = r["entered_trades"]
            gp = sum(x["pnl"] for x in t if x["pnl"] > 0)
            gl = abs(sum(x["pnl"] for x in t if x["pnl"] < 0))
            fin = r["final_effective"]; roi = fin / START_BAL - 1; ddp = r["max_dd_pct"]
            print(f"  {lbl:<32}{fin:>10,.0f}{fin-START_BAL:>+11,.0f}{roi:>9.1%}"
                  f"{ddp:>7.1f}%{(roi*100/ddp if ddp else 0):>8.2f}"
                  f"{(gp/gl if gl else 9.99):>7.2f}{len(t):>8,}", flush=True)
        print(f"  {'never trade':<32}{START_BAL:>10,.0f}{0:>+11,.0f}{0.0:>9.1%}"
              f"{0.0:>7.1f}%{0.0:>8.2f}{'—':>7}{0:>8}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
