#!/usr/bin/env python3
"""Criteria 2-5 of PROTOCOL_symbols.md, for the global-parameter change.

A0 = live config, each (symbol, div_type) with its own fitted (rr, atr_mult).
A1 = same pairs, one global (rr=10, atr_mult=3.0).
Both resolved under the live s3_a1 trailing exit, which passed a 4-way parity check
(live engine / shadow resolver / repo reference / this study's resolver, 100% agreement).

Cost 24.2 bps, measured from 133 live stop-outs.
"""
from __future__ import annotations

import glob
import math
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
HERE = Path(__file__).resolve().parent
COST = 0.00242
RNG = np.random.default_rng(42)
NB = 2000

FOLDS = [("F1", "2024-06-01", "2024-12-01"), ("F2", "2024-12-01", "2025-06-01"),
         ("F3", "2025-06-01", "2025-12-01"), ("F4", "2025-12-01", "2026-05-25"),
         ("HOLDOUT", "2026-05-25", "2026-07-05")]


def load(path, rcol, xcol):
    d = pd.read_parquet(HERE / path)
    d = d.rename(columns={rcol: "r", xcol: "x"})
    d = d.dropna(subset=["r", "x"]).copy()
    d["nr"] = d.r - COST / d.stop_frac
    return d[["symbol", "div_type", "entry_time", "x", "r", "nr", "stop_frac", "side"]]


KEY = ["symbol", "div_type", "entry_time"]


def dedup(d):
    """A (symbol, div_type) can produce two signals whose BOS lands on the same entry bar,
    so the natural key is not unique. Keep the first; doing this BEFORE any merge stops the
    join from multiplying rows (an inner join on a duplicated key was inflating the paired
    set to 101.9% of the smaller arm)."""
    return d.drop_duplicates(subset=KEY, keep="first")


def paired(a0, a1):
    return dedup(a0).merge(dedup(a1), on=KEY, suffixes=("_0", "_1"), how="inner")


def wblocks(times, n):
    """Weekly block bootstrap index draw."""
    wk = pd.to_datetime(times).dt.to_period("W")
    groups = [g.to_numpy() for _, g in pd.Series(np.arange(n)).groupby(wk.to_numpy())]
    return groups


def bstrap(diff, groups, nb=NB, rng=None):
    rng = rng or np.random.default_rng(42)
    k = len(groups)
    out = np.empty(nb)
    for b in range(nb):
        pick = rng.integers(0, k, k)
        idx = np.concatenate([groups[i] for i in pick])
        out[b] = diff[idx].mean()
    return out


def main():
    a0 = load("universe_trail.parquet", "r_trail", "exit_trail")
    a1 = load("uni_glob_rr10_am3_same.parquet", "r_result", "exit_time")
    m = paired(a0, a1)
    print(f"A0 {len(a0):,}  A1 {len(a1):,}  paired {len(m):,} "
          f"({len(m)/min(len(a0),len(a1))*100:.1f}% of the smaller arm)")
    m = m.sort_values("entry_time").reset_index(drop=True)
    diff = (m.nr_1 - m.nr_0).to_numpy()
    groups = wblocks(m.entry_time, len(m))

    # ---------------- CRITERION 2: walk-forward folds ----------------
    print("\n" + "=" * 84)
    print("CRITERION 2 — beat A0 in >=3 of 4 walk-forward test folds")
    print("=" * 84)
    print(f"{'fold':<9}{'n':>7}{'A0 netR':>10}{'A1 netR':>10}{'diff':>9}{'95% CI':>20}{'':>6}")
    wins = 0
    for nm, a, b in FOLDS:
        s = m[(m.entry_time >= a) & (m.entry_time < b)]
        if len(s) < 50:
            print(f"{nm:<9}{len(s):>7}   too few")
            continue
        dd = (s.nr_1 - s.nr_0).to_numpy()
        g = wblocks(s.entry_time, len(s))
        bs = bstrap(dd, g, 800, np.random.default_rng(42))
        lo, hi = np.percentile(bs, [2.5, 97.5])
        w = dd.mean() > 0
        if nm != "HOLDOUT" and w:
            wins += 1
        print(f"{nm:<9}{len(s):>7}{s.nr_0.mean():>10.4f}{s.nr_1.mean():>10.4f}"
              f"{dd.mean():>9.4f}   [{lo:+.4f},{hi:+.4f}]{'  WIN' if w else '  loss':>6}")
    print(f"\n  A1 wins {wins}/4 walk-forward folds  ->  "
          f"{'PASS' if wins >= 3 else 'FAIL'} (protocol needs >=3)")

    # ---------------- CRITERION 3: random null ----------------
    print("\n" + "=" * 84)
    print("CRITERION 3 — beat a size-matched RANDOM assignment, and survive best-of-8")
    print("=" * 84)
    arms = {}
    for p in sorted(glob.glob(str(HERE / "uni_glob_*_same.parquet"))):
        mm = re.search(r"rr([\d.]+)_am([\d.]+)", p)
        arms[(float(mm.group(1)), float(mm.group(2)))] = load(
            Path(p).name, "r_result", "exit_time")
    print(f"  {len(arms)} built global arms")

    base = m[KEY].copy()
    mat = {}
    for k, d in arms.items():
        j = base.merge(dedup(d)[KEY + ["nr"]], on=KEY, how="left")
        assert len(j) == len(base), f"join inflated rows for {k}"
        mat[k] = j.nr.to_numpy()
    A = np.vstack([mat[k] for k in arms])              # (8, n)
    keys = list(arms)
    ok = ~np.isnan(A).any(axis=0)
    A = A[:, ok]
    a0v = m.nr_0.to_numpy()[ok]
    a1v = m.nr_1.to_numpy()[ok]
    pairid = (m.symbol + "|" + m.div_type).to_numpy()[ok]
    print(f"  usable rows across all 8 arms: {ok.sum():,}")

    # (a) random per-pair assignment
    upair, inv = np.unique(pairid, return_inverse=True)
    rng = np.random.default_rng(42)
    rnd = np.empty(NB)
    for b in range(NB):
        pick = rng.integers(0, len(keys), len(upair))
        rnd[b] = A[pick[inv], np.arange(A.shape[1])].mean()
    pct_a1 = (rnd < a1v.mean()).mean() * 100
    pct_a0 = (rnd < a0v.mean()).mean() * 100
    print(f"\n  (a) RANDOM per-pair (rr,atr) assignment, {NB} draws:")
    print(f"      random mean netR   {rnd.mean():+.4f}  sd {rnd.std():.4f}  "
          f"[{np.percentile(rnd,2.5):+.4f},{np.percentile(rnd,97.5):+.4f}]")
    print(f"      A1 global rr10/a3  {a1v.mean():+.4f}   percentile {pct_a1:6.2f}%  "
          f"p={max(1-pct_a1/100,1/NB):.4f}")
    print(f"      A0 LIVE FITTED     {a0v.mean():+.4f}   percentile {pct_a0:6.2f}%")
    print(f"      -> the bot's fitted assignment is {'BETTER' if pct_a0>50 else 'WORSE'} "
          f"than a coin-flip assignment")

    # (b) best-of-8 selection correction
    g2 = wblocks(m.entry_time[ok], ok.sum())
    rng = np.random.default_rng(42)
    k = len(g2)
    beat = 0
    margins = np.empty(NB)
    for b in range(NB):
        pick = rng.integers(0, k, k)
        idx = np.concatenate([g2[i] for i in pick])
        best = A[:, idx].mean(axis=1).max()
        margins[b] = best - a0v[idx].mean()
        beat += best > a0v[idx].mean()
    print(f"\n  (b) BEST-OF-8 under resampling ({NB} weekly block bootstraps):")
    print(f"      best-of-8 beats A0 in {beat/NB*100:.1f}% of resamples")
    print(f"      margin mean {margins.mean():+.4f}  "
          f"95% CI [{np.percentile(margins,2.5):+.4f},{np.percentile(margins,97.5):+.4f}]")
    c3 = (pct_a1 >= 95) and (np.percentile(margins, 2.5) > 0)
    print(f"      -> CRITERION 3 {'PASS' if c3 else 'FAIL'}")

    # ---------------- CRITERION 5: concentration ----------------
    print("\n" + "=" * 84)
    print("CRITERION 5 — advantage not concentrated in a handful of trades")
    print("=" * 84)
    print(f"{'removed':<12}{'A0 netR':>10}{'A1 netR':>10}{'diff':>9}"
          f"{'hostile diff':>14}   (hostile = strip A1's best only)")
    n = len(m)
    o0 = np.sort(a0v)[::-1]
    o1 = np.sort(a1v)[::-1]
    for q in (0.0, 0.005, 0.01, 0.02, 0.05):
        c = int(len(a0v) * q)
        d0, d1 = o0[c:].mean(), o1[c:].mean()
        hostile = o1[c:].mean() - o0.mean()
        print(f"{q*100:>6.1f}%      {d0:>10.4f}{d1:>10.4f}{d1-d0:>9.4f}{hostile:>14.4f}")
    for lbl, v in (("A0", a0v), ("A1", v1 := a1v)):
        s = np.sort(v)[::-1]
        tot = s.sum()
        if tot > 0:
            print(f"  {lbl}: top 1% = {s[:max(1,int(len(s)*.01))].sum()/tot*100:5.1f}% of "
                  f"total net R · top 5% = {s[:max(1,int(len(s)*.05))].sum()/tot*100:5.1f}%")

    # ---------------- CRITERION 4: market controls ----------------
    print("\n" + "=" * 84)
    print("CRITERION 4 — market controls (unlevered)")
    print("=" * 84)
    cache = ROOT / "cache_3yr_1h"
    btc = pd.read_parquet(cache / "BTCUSDT.parquet").sort_values("start")
    btc = btc[(btc.start >= "2023-06-01") & (btc.start <= "2026-07-25")].reset_index(drop=True)

    def stats(c, lbl):
        c = c.dropna()
        pk = c.cummax(); dd = float(((pk - c) / pk).max() * 100)
        roi = (c.iloc[-1] / c.iloc[0] - 1) * 100
        yrs = (c.index[-1] - c.index[0]).days / 365.25
        cagr = ((c.iloc[-1] / c.iloc[0]) ** (1 / yrs) - 1) * 100 if yrs > 0 else 0
        dl = c.resample("1D").last().ffill().pct_change().dropna()
        sh = dl.mean() / dl.std() * math.sqrt(365) if dl.std() > 0 else 0
        h = c[c.index >= pd.Timestamp("2026-05-25")]
        ho = (h.iloc[-1] / h.iloc[0] - 1) * 100 if len(h) > 2 else float("nan")
        print(f"  {lbl:<34}{roi:>10.1f}{dd:>9.1f}{roi/dd if dd>0 else float('nan'):>9.2f}"
              f"{cagr:>9.1f}{sh:>8.2f}{ho:>10.1f}")
        return roi, dd, ho

    print(f"  {'benchmark':<34}{'ROI%':>10}{'maxDD%':>9}{'ROI/DD':>9}{'CAGR%':>9}"
          f"{'Sharpe':>8}{'holdout%':>10}")
    s = btc.set_index("start")["close"]
    stats(1500 * s / s.iloc[0], "C2  BTC buy & hold")

    ema = s.ewm(span=200, adjust=False).mean()
    sig = (s > ema).shift(1).fillna(False)
    ret = s.pct_change().fillna(0) * sig
    sw = sig.astype(int).diff().abs().fillna(0)
    ret = ret - sw * COST
    stats(1500 * (1 + ret).cumprod(), "C4  BTC 200-EMA trend follow")

    cfg = yaml.safe_load(open(ROOT / "config.yaml"))["symbols"]
    live = [k for k, v in cfg.items()
            if (v or {}).get("enabled", True) and (v or {}).get("configs")]
    px = {}
    for sym in live:
        f = cache / f"{sym}.parquet"
        if not f.exists():
            continue
        d = pd.read_parquet(f, columns=["start", "close"]).sort_values("start")
        d = d[(d.start >= "2023-06-01") & (d.start <= "2026-07-25")]
        if len(d) > 500:
            px[sym] = d.set_index("start")["close"]
    P = pd.DataFrame(px).resample("1D").last()
    r = P.pct_change()
    basket = r.mean(axis=1, skipna=True).fillna(0)
    stats(1500 * (1 + basket).cumprod(), f"C3  long-only basket ({len(px)} syms)")

    print(f"\n  {'STRATEGY (levered, 0.3% risk)':<34}{'ROI%':>10}{'maxDD%':>9}{'ROI/DD':>9}")
    print(f"  {'A0 live fitted config':<34}{472.5:>10.1f}{64.9:>9.1f}{472.5/64.9:>9.2f}"
          f"{'':>9}{'':>8}{-44.8:>10.1f}")
    print(f"  {'A1 global rr10/atr3':<34}{1285.0:>10.1f}{47.5:>9.1f}{1285.0/47.5:>9.2f}"
          f"{'':>9}{'':>8}{6.9:>10.1f}")
    print(f"\n  BTC return over the window: {(s.iloc[-1]/s.iloc[0]-1)*100:+.1f}%")


if __name__ == "__main__":
    main()
