#!/usr/bin/env python3
"""Test the hypothesis: the strategy is losing because the market turned extremely choppy.

Two questions, and they have different implications:

  Q1  Is the out-of-sample period actually choppier than the fitting window?
      -> establishes whether the premise is even true.

  Q2  CONDITIONAL on the same choppiness, does the strategy still earn?
      -> this is the decisive one. If low-CHOP trades still earn in the OOS window
         the same way they did in the fitting window, then the edge is intact and the
         problem is market MIX -> the fix is to wait / gate harder on CHOP.
         If low-CHOP trades ALSO lost, choppiness is not the explanation and the edge
         itself has decayed.

Uses the symbol's own CHOP at entry (what the live gate actually reads) plus BTC CHOP
and BTC ADX as market-wide regime context.
"""
from __future__ import annotations

import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent
CACHE = ROOT / "cache_3yr_1h"
SPLIT = pd.Timestamp("2026-05-25")
H = 336  # cohort-complete horizon (14d; 98.4% of trades resolve)


def chop(df, period=14):
    hl = df["high"] - df["low"]
    hc = (df["high"] - df["close"].shift()).abs()
    lc = (df["low"] - df["close"].shift()).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return (100 * np.log10(tr.rolling(period).sum() /
            (df["high"].rolling(period).max() - df["low"].rolling(period).min()).replace(0, np.nan))
            / math.log10(period))


def adx(df, period=14):
    up = df["high"].diff()
    dn = -df["low"].diff()
    plus = np.where((up > dn) & (up > 0), up, 0.0)
    minus = np.where((dn > up) & (dn > 0), dn, 0.0)
    hl = df["high"] - df["low"]
    hc = (df["high"] - df["close"].shift()).abs()
    lc = (df["low"] - df["close"].shift()).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / period, adjust=False).mean()
    pdi = 100 * pd.Series(plus, index=df.index).ewm(alpha=1 / period, adjust=False).mean() / atr
    mdi = 100 * pd.Series(minus, index=df.index).ewm(alpha=1 / period, adjust=False).mean() / atr
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    return dx.ewm(alpha=1 / period, adjust=False).mean()


def main():
    d = pd.read_csv(ROOT / "universe_live_8mo.csv", parse_dates=["entry_time", "exit_time"])
    d["fee_r"] = 0.0018 * d.entry_price / (d.entry_price - d.sl_price).abs()
    d["r_net"] = d.r_result - d.fee_r
    d["hold_h"] = (d.exit_time - d.entry_time).dt.total_seconds() / 3600
    END = d.exit_time.max()
    d = d[(d.entry_time <= END - pd.Timedelta(hours=H)) & (d.hold_h <= H)].copy()

    # --- symbol's own CHOP at entry (what the live gate reads) ---
    print("annotating per-symbol CHOP at entry ...", flush=True)
    vals = {}
    for sym in d.symbol.unique():
        f = CACHE / f"{sym}.parquet"
        if not f.exists():
            continue
        df = pd.read_parquet(f).set_index("start").sort_index()
        vals[sym] = chop(df)
    d["chop"] = [
        (vals[s].asof(t) if s in vals else np.nan)
        for s, t in zip(d.symbol, d.entry_time.dt.floor("h"))
    ]

    # --- BTC market-wide regime ---
    b = pd.read_parquet(CACHE / "BTCUSDT.parquet").set_index("start").sort_index()
    b["chop"] = chop(b)
    b["adx"] = adx(b)
    d["btc_chop"] = [b["chop"].asof(t) for t in d.entry_time.dt.floor("h")]
    d["btc_adx"] = [b["adx"].asof(t) for t in d.entry_time.dt.floor("h")]

    d["era"] = np.where(d.entry_time < SPLIT, "FIT", "OOS")
    d = d.dropna(subset=["chop"])

    # =====================================================================
    print("\n" + "=" * 78)
    print("Q1 — is the OOS period actually choppier?")
    print("=" * 78)
    print(f"  {'era':<6}{'trades':>8}{'sym CHOP mean':>15}{'median':>9}"
          f"{'BTC CHOP':>10}{'BTC ADX':>9}")
    for era, g in d.groupby("era"):
        print(f"  {era:<6}{len(g):>8}{g.chop.mean():>15.2f}{g.chop.median():>9.2f}"
              f"{g.btc_chop.mean():>10.2f}{g.btc_adx.mean():>9.2f}")

    print("\n  BTC monthly regime (hourly candles, whole month):")
    bm = b.loc["2025-11-01":].resample("ME").agg(chop=("chop", "mean"), adx=("adx", "mean"))
    for ts, row in bm.iterrows():
        print(f"    {ts:%Y-%m}   BTC CHOP {row.chop:5.1f}   BTC ADX {row.adx:5.1f}")

    # =====================================================================
    print("\n" + "=" * 78)
    print("Q2 — DECISIVE: conditional on the same CHOP, does the edge survive?")
    print("=" * 78)
    edges = [0, 38, 44, 48, 52, 56, 100]
    d["bucket"] = pd.cut(d.chop, edges)
    print(f"  {'CHOP bucket':<16}{'FIT n':>8}{'FIT avgR':>11}{'FIT PF':>8}"
          f"{'OOS n':>8}{'OOS avgR':>11}{'OOS PF':>8}{'delta':>9}")
    for bkt, g in d.groupby("bucket", observed=True):
        f_, o_ = g[g.era == "FIT"], g[g.era == "OOS"]
        if len(f_) < 100 or len(o_) < 60:
            continue

        def pf(x):
            gp = x[x > 0].sum(); gl = abs(x[x < 0].sum())
            return gp / gl if gl else float("inf")
        fa, oa = f_.r_net.mean(), o_.r_net.mean()
        print(f"  {str(bkt):<16}{len(f_):>8}{fa:>+11.4f}{pf(f_.r_net):>8.2f}"
              f"{len(o_):>8}{oa:>+11.4f}{pf(o_.r_net):>8.2f}{oa - fa:>+9.4f}")

    # decomposition: how much of the OOS drop is MIX vs WITHIN-bucket performance?
    print("\n" + "=" * 78)
    print("Decomposition of the OOS edge drop (Oaxaca-style)")
    print("=" * 78)
    f_, o_ = d[d.era == "FIT"], d[d.era == "OOS"]
    wf = f_.groupby("bucket", observed=True).size() / len(f_)
    wo = o_.groupby("bucket", observed=True).size() / len(o_)
    mf = f_.groupby("bucket", observed=True).r_net.mean()
    mo = o_.groupby("bucket", observed=True).r_net.mean()
    idx = mf.index.intersection(mo.index)
    total = o_.r_net.mean() - f_.r_net.mean()
    mix = float(((wo[idx] - wf[idx]) * mf[idx]).sum())      # same edge, new CHOP mix
    within = float((wo[idx] * (mo[idx] - mf[idx])).sum())   # same mix, worse edge
    print(f"  total OOS - FIT avg R          : {total:+.4f}")
    print(f"  explained by CHOP MIX shift    : {mix:+.4f}  ({mix/total*100 if total else 0:5.1f}%)")
    print(f"  explained by WITHIN-bucket drop: {within:+.4f}  ({within/total*100 if total else 0:5.1f}%)")
    print("\n  MIX  = market got choppier, strategy unchanged  -> wait it out")
    print("  WITHIN = same conditions, strategy now loses      -> edge decayed")

    # would a stricter CHOP gate have saved the OOS window?
    print("\n" + "=" * 78)
    print("Would a STRICTER CHOP gate have rescued the OOS window?")
    print("=" * 78)
    print(f"  {'gate':<18}{'OOS trades':>12}{'OOS avgR':>11}{'OOS PF':>9}{'kept':>8}")
    for thr in [56, 52, 48, 44, 40, 36, 32]:
        s = o_[o_.chop < thr]
        if len(s) < 30:
            print(f"  CHOP < {thr:<11}{len(s):>12}   (too few)")
            continue
        gp = s.r_net[s.r_net > 0].sum(); gl = abs(s.r_net[s.r_net < 0].sum())
        print(f"  CHOP < {thr:<11}{len(s):>12}{s.r_net.mean():>+11.4f}"
              f"{(gp/gl if gl else 9.99):>9.2f}{len(s)/len(o_):>8.1%}")

    d.to_csv(ROOT / "chop_hypothesis_annotated.csv", index=False)
    print("\nwrote chop_hypothesis_annotated.csv")


if __name__ == "__main__":
    main()
