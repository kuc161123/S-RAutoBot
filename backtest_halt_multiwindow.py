#!/usr/bin/env python3
"""Is the shadow halt genuinely the best feature, or did it just fit the recent slump?

The halt was calibrated on 2026-05-25 -> today, a period where the edge was negative. A
feature that only helps when everything is losing is not a feature, it is hindsight. This
runs it across rolling 6-month windows covering bull, bear and chop, and compares it
against the other surviving change (the concurrency cap) and the two combined.

Each window starts fresh at $1,500 and is scored independently, so a single spectacular
period cannot carry the average.

The shadow ledger is the full set of BOS-confirmed signals the live config evaluates —
the same population the live shadow layer logs. A signal only enters the ledger when it
RESOLVES, preserving the real-time negative bias (losses resolve fast, wins slowly).
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
from backtest_shadow_gate import LIVE as LIVE_SIM, gate_series

ROOT = Path(__file__).parent
CACHE = ROOT / "cache_3yr_1h"
START_BAL = 1500.0
GEN_FROM = pd.Timestamp("2023-06-01")
MAX_WAIT = bt.MAX_WAIT_CANDLES
SLIP = 0.0003
CHOP_T = 52.0
UNIVERSE = ROOT / "halt_universe_live.parquet"


def chop_series(df, period=14):
    hl = df["high"] - df["low"]
    hc = (df["high"] - df["close"].shift()).abs()
    lc = (df["low"] - df["close"].shift()).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    hi = df["high"].rolling(period).max(); lo = df["low"].rolling(period).min()
    return (100 * np.log10(tr.rolling(period).sum() / (hi - lo).replace(0, np.nan))
            / math.log10(period))


def replay(args):
    """Only the LIVE config's own picks — this is the deployed bot, full history."""
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
            if np.isfinite(ch[e]) and ch[e] >= CHOP_T:
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
                    out.append((ts[e], ts[k], entry, sl, -1.0 if hs else rr,
                                side, sym))
                    break
    return out


def build():
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
    print(f"[HALT] replaying live config over full history · {len(jobs)} symbols", flush=True)
    rows = []
    with mp.Pool(max(1, mp.cpu_count() - 1)) as pool:
        for i, r in enumerate(pool.imap_unordered(replay, jobs, chunksize=4), 1):
            rows.extend(r)
            if i % 50 == 0:
                print(f"  {i}/{len(jobs)} · {len(rows):,}", flush=True)
    d = pd.DataFrame(rows, columns=["entry_time", "exit_time", "entry_price", "sl_price",
                                    "r_result", "side", "symbol"])
    d["entry_time"] = pd.to_datetime(d["entry_time"])
    d["exit_time"] = pd.to_datetime(d["exit_time"])
    d = d[d.entry_time >= GEN_FROM].sort_values("entry_time").reset_index(drop=True)
    d["fee_r"] = 0.0018 * d.entry_price / (d.entry_price - d.sl_price).abs()
    d["r_net"] = d.r_result - d.fee_r
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
    d.to_parquet(UNIVERSE, index=False)
    return d


def run(sub, chop, **over):
    if sub.empty:
        return dict(final=START_BAL, roi=0.0, dd=0.0, n=0)
    cols = ["entry_time", "exit_time", "entry_price", "sl_price", "r_result",
            "side", "symbol", "btc_bull", "btc_impulse"]
    kw = dict(LIVE_SIM, starting_balance=START_BAL,
              btc_bull_col="btc_bull", btc_short_col="btc_impulse")
    kw.update(over)
    P.STARTING_BALANCE = START_BAL
    r = P.run_simulation(sub[cols].sort_values("entry_time").reset_index(drop=True),
                         chop, **kw)
    return dict(final=r["final_effective"], roi=r["final_effective"] / START_BAL - 1,
                dd=r["max_dd_pct"], n=len(r["entered_trades"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild", action="store_true")
    a = ap.parse_args()
    d = build() if (a.rebuild or not UNIVERSE.exists()) else pd.read_parquet(UNIVERSE)
    print(f"[HALT] {len(d):,} live-config trades "
          f"{d.entry_time.min().date()} .. {d.entry_time.max().date()}")

    chop = P.load_chop_data(sorted(d.symbol.unique()))
    g7 = gate_series(d, 7, -300, 400)
    g21 = gate_series(d, 21, -300, 200)

    ARMS = [
        ("BASE", None, {}),
        ("HALT 7d", g7, {}),
        ("HALT 21d", g21, {}),
        ("CAP 20", None, dict(max_concurrent=20)),
        ("HALT 21d + CAP 20", g21, dict(max_concurrent=20)),
    ]

    starts = pd.date_range("2023-10-01", "2026-02-01", freq="3ME")
    rows = []
    print(f"\n{'=' * 108}\nROLLING 6-MONTH WINDOWS · $1,500 each · ROI / maxDD / "
          f"ROI-DD ratio\n{'=' * 108}")
    hdr = f"  {'window':<20}"
    for nm, _, _ in ARMS:
        hdr += f"{nm:>17}"
    print(hdr)
    print(f"  {'':<20}" + "".join(f"{'ROI    DD  R/DD':>17}" for _ in ARMS))
    for s0 in starts:
        s1 = s0 + pd.DateOffset(months=6)
        seg = d[(d.entry_time >= s0) & (d.entry_time < s1)]
        if len(seg) < 400:
            continue
        line = f"  {s0.date()} +6mo "
        rec = {"window": str(s0.date())}
        for nm, g, over in ARMS:
            sub = seg
            if g is not None:
                allow = seg.entry_time.dt.floor("h").map(g.to_dict()).fillna(True).astype(bool)
                sub = seg[allow]
            r = run(sub, chop, **over)
            ratio = r["roi"] * 100 / r["dd"] if r["dd"] else 0.0
            rec[nm] = r
            line += f"{r['roi']:>7.0%}{r['dd']:>6.0f}%{ratio:>6.1f}"
        rows.append(rec)
        print(line, flush=True)

    print(f"\n{'=' * 108}\nSUMMARY across {len(rows)} independent windows\n{'=' * 108}")
    base = [r["BASE"] for r in rows]
    print(f"  {'arm':<20}{'median ROI':>12}{'median DD':>12}{'median R/DD':>13}"
          f"{'beat BASE $':>13}{'better DD':>11}")
    for nm, _, _ in ARMS:
        xs = [r[nm] for r in rows]
        med_roi = np.median([x["roi"] for x in xs])
        med_dd = np.median([x["dd"] for x in xs])
        ratios = [x["roi"] * 100 / x["dd"] if x["dd"] else 0 for x in xs]
        winr = np.mean([x["final"] > b["final"] for x, b in zip(xs, base)])
        ddw = np.mean([x["dd"] < b["dd"] for x, b in zip(xs, base)])
        print(f"  {nm:<20}{med_roi:>12.0%}{med_dd:>11.0f}%{np.median(ratios):>13.1f}"
              f"{winr:>13.0%}{ddw:>11.0%}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
