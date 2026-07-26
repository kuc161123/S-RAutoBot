#!/usr/bin/env python3
"""Is the STRATEGY sound, independent of which symbols/parameters it uses?

Five tests, each isolating one design decision. Run on the DESIGN period only
(entries resolving before 2026-02-01) so conclusions are not contaminated by the
holdout or by the current bad regime — i.e. we are asking "was this ever a good
design?", not "is it working right now?".

  T1 PLACEBO. Keep the symbol, side, RR, ATR and rough period of every real signal but
     move the entry to a random bar. If the real signals score no better than the
     placebo, the divergence carries no timing information and the "edge" is just the
     stop/target geometry plus market drift.

  T2 BOS. Does waiting for the break of structure beat entering at the divergence bar?

  T3 EMA-200 GATE. Does requiring trend alignment beat ignoring it (and beat the
     opposite)?

  T4 MFE. How far do trades actually run in your favour before resolving? This says
     which RR targets are physically supported, and whether RR 8-10 is realistic.

  T5 REGIME CLASSIFIER. The bot sizes off win-rate/avg-R over the last 20 trades. At a
     ~17% base win rate, how much of that signal is noise? Simulated against a fixed
     true win rate, so any tier movement observed IS noise by construction.
"""
from __future__ import annotations

import math
import multiprocessing as mp
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
import backtest_3yr_walkforward as bt

ROOT = Path(__file__).parent
CACHE = ROOT / "cache_3yr_1h"
HOLDOUT = pd.Timestamp("2026-02-01")
DESIGN_START = pd.Timestamp("2023-06-01")
MAX_WAIT = bt.MAX_WAIT_CANDLES
SLIP, FEE = 0.0003, 0.0006
ROUND_TRIP = 2 * (SLIP + FEE)
RR_REF, ATR_REF = 5.0, 1.5          # reference geometry for like-for-like tests
RNG_SEED = 11


def resolve(o, h, l, c, e, side, sl_d, rr):
    """Return (r_net, mfe_R, bars) for an entry at bar e. None if unresolved."""
    n = len(c)
    entry = o[e] * (1 + SLIP) if side == "long" else o[e] * (1 - SLIP)
    sl = entry - sl_d if side == "long" else entry + sl_d
    tp = entry + sl_d * rr if side == "long" else entry - sl_d * rr
    fee_r = ROUND_TRIP * entry / sl_d
    mfe = 0.0
    for k in range(e, n):
        if side == "long":
            mfe = max(mfe, (h[k] - entry) / sl_d)
            hs, ht = l[k] <= sl, h[k] >= tp
        else:
            mfe = max(mfe, (entry - l[k]) / sl_d)
            hs, ht = h[k] >= sl, l[k] <= tp
        if hs or ht:
            return (-1.0 if hs else rr) - fee_r, mfe, k - e
    return None


def work(args):
    symbol, cache_dir = args
    f = Path(cache_dir) / f"{symbol}.parquet"
    if not f.exists():
        return None
    try:
        df = pd.read_parquet(f)
    except Exception:
        return None
    if df.empty or len(df) < 3000:
        return None
    df = bt.prepare_data(df)
    o = df["open"].to_numpy(); h = df["high"].to_numpy()
    l = df["low"].to_numpy(); c = df["close"].to_numpy()
    atr = df["atr"].to_numpy(); ema = df["ema"].to_numpy()
    ts = df["start"].to_numpy(); n = len(c)
    cut = np.searchsorted(ts, np.datetime64(HOLDOUT))
    lo_i = np.searchsorted(ts, np.datetime64(DESIGN_START))
    rng = np.random.default_rng(abs(hash(symbol)) % (2**31) + RNG_SEED)

    real, placebo, at_div, no_ema, anti_ema, mfes = [], [], [], [], [], []

    for s in bt.detect_signals(df):
        conf, side, lvl = s["conf_idx"], s["side"], s["swing"]
        if not (lo_i <= conf < cut):
            continue
        aligned = (c[conf] > ema[conf]) if side == "long" else (c[conf] < ema[conf])

        # ---- T2 control: enter at the divergence bar itself (no BOS wait) ----
        if aligned and conf + 1 < cut and np.isfinite(atr[conf]) and atr[conf] > 0:
            r = resolve(o, h, l, c, conf + 1, side, atr[conf] * ATR_REF, RR_REF)
            if r:
                at_div.append(r[0])

        # ---- find BOS (live semantics: closed candle, enter next open) ----
        bos = None
        for i in range(1, MAX_WAIT + 1):
            idx = conf + i
            if idx >= n:
                break
            if (side == "long" and c[idx] > lvl) or (side == "short" and c[idx] < lvl):
                bos = idx
                break
        if bos is None or bos + 1 >= cut:
            continue
        if not (np.isfinite(atr[bos]) and atr[bos] > 0 and np.isfinite(ema[bos])):
            continue
        sl_d = atr[bos] * ATR_REF
        e = bos + 1
        al_bos = (c[bos] > ema[bos]) if side == "long" else (c[bos] < ema[bos])

        res = resolve(o, h, l, c, e, side, sl_d, RR_REF)
        if res is None:
            continue

        # ---- T3: EMA gate on / off / inverted ----
        no_ema.append(res[0])
        if al_bos:
            real.append(res[0])
            mfes.append(res[1])
        else:
            anti_ema.append(res[0])

        # ---- T1: placebo — same symbol/side/geometry, random entry bar ----
        if al_bos:
            for _ in range(2):
                pe = int(rng.integers(lo_i + 250, max(lo_i + 251, cut - 1)))
                if not (np.isfinite(atr[pe]) and atr[pe] > 0):
                    continue
                pr = resolve(o, h, l, c, pe, side, atr[pe] * ATR_REF, RR_REF)
                if pr:
                    placebo.append(pr[0])
    return dict(real=real, placebo=placebo, at_div=at_div, no_ema=no_ema,
                anti_ema=anti_ema, mfe=mfes)


def st(x):
    x = np.asarray(x, float)
    if len(x) == 0:
        return None
    gp = x[x > 0].sum(); gl = abs(x[x < 0].sum())
    return dict(n=len(x), wr=(x > 0).mean(), avg=x.mean(),
                pf=(gp / gl) if gl else float("inf"),
                se=x.std(ddof=1) / math.sqrt(len(x)))


def row(lbl, s):
    if not s:
        print(f"  {lbl:<40} (none)"); return
    print(f"  {lbl:<40}{s['n']:>8,}t  WR {s['wr']:>6.2%}  avgR {s['avg']:>+8.4f} "
          f"±{1.96*s['se']:.4f}  PF {s['pf']:>5.2f}")


def main():
    syms = sorted(p.stem for p in CACHE.glob("*.parquet"))
    syms = [s for s in syms if not s.endswith(("26JUN26", "03APR26", "10APR26", "17APR26"))]
    print(f"[DESIGN] {len(syms)} symbols · design period {DESIGN_START.date()} .. "
          f"{HOLDOUT.date()} · reference geometry rr={RR_REF:.0f} atr={ATR_REF}\n", flush=True)

    agg = {k: [] for k in ("real", "placebo", "at_div", "no_ema", "anti_ema", "mfe")}
    with mp.Pool(max(1, mp.cpu_count() - 1)) as pool:
        for i, r in enumerate(pool.imap_unordered(
                work, [(s, str(CACHE)) for s in syms], chunksize=4), 1):
            if r:
                for k in agg:
                    agg[k].extend(r[k])
            if i % 100 == 0:
                print(f"  {i}/{len(syms)} symbols", flush=True)

    print(f"\n{'=' * 92}\nT1 — PLACEBO: does the divergence signal carry TIMING information?"
          f"\n{'=' * 92}")
    sr, sp = st(agg["real"]), st(agg["placebo"])
    row("REAL signals (div + BOS + EMA gate)", sr)
    row("PLACEBO (random entry, same geometry)", sp)
    if sr and sp:
        d = sr["avg"] - sp["avg"]
        se = math.sqrt(sr["se"] ** 2 + sp["se"] ** 2)
        print(f"\n  signal edge over random: {d:+.4f} R/trade  (t = {d/se:+.2f})")
        print("  " + ("=> the signal DOES carry information." if d / se > 2 else
                      "=> NOT distinguishable from random entry — the setup adds nothing."))

    print(f"\n{'=' * 92}\nT2 — does the BOS confirmation add value?\n{'=' * 92}")
    row("enter at BOS (live behaviour)", sr)
    row("enter at divergence bar (no BOS)", st(agg["at_div"]))

    print(f"\n{'=' * 92}\nT3 — does the EMA-200 trend gate add value?\n{'=' * 92}")
    row("trend-aligned (live behaviour)", sr)
    row("counter-trend (inverted gate)", st(agg["anti_ema"]))
    row("no gate at all", st(agg["no_ema"]))

    print(f"\n{'=' * 92}\nT4 — MFE: which RR targets are physically supported?\n{'=' * 92}")
    mfe = np.asarray(agg["mfe"], float)
    print(f"  max favourable excursion, in R, over {len(mfe):,} real trades:")
    for q in [50, 70, 80, 90, 95, 99]:
        print(f"    p{q:<3} {np.percentile(mfe, q):>7.2f} R")
    print(f"\n  share of trades whose move EVER reached the target:")
    for rr in [3, 5, 8, 10]:
        hit = (mfe >= rr).mean()
        be = 1 / (1 + rr)
        flag = "  <-- below breakeven" if hit < be else ""
        print(f"    rr={rr:>2}: reached {hit:>6.2%} of the time · breakeven needs "
              f"{be:>6.2%}{flag}")

    print(f"\n{'=' * 92}\nT5 — is the 20-trade regime classifier signal or noise?\n{'=' * 92}")
    rng = np.random.default_rng(7)
    true_wr = sr["wr"] if sr else 0.17
    wins = np.asarray(agg["real"], float) > 0
    avg_win = np.mean(np.asarray(agg["real"], float)[wins]) if wins.any() else RR_REF
    avg_loss = np.mean(np.asarray(agg["real"], float)[~wins]) if (~wins).any() else -1.0
    print(f"  simulating a CONSTANT-edge stream (true WR {true_wr:.1%}, "
          f"avg win {avg_win:+.2f}R, avg loss {avg_loss:+.2f}R)")
    print("  any tier movement below is therefore 100% noise:\n")
    tiers = {"favorable": 0, "cautious": 0, "adverse": 0, "critical": 0}
    flips = 0; prev = None
    for _ in range(20000):
        w = rng.random(20) < true_wr
        r = np.where(w, avg_win, avg_loss)
        wr, ar = w.mean(), r.mean()
        if wr >= 0.18 and ar >= 0.15:
            t = "favorable"
        elif wr >= 0.18 or ar >= 0.10:
            t = "cautious"
        elif wr >= 0.10 or ar >= -0.5:
            t = "adverse"
        else:
            t = "critical"
        tiers[t] += 1
        if prev and t != prev:
            flips += 1
        prev = t
    tot = sum(tiers.values())
    for k, v in tiers.items():
        print(f"    {k:<11} {v/tot:>6.1%} of the time")
    print(f"\n  tier changes: {flips/tot:.1%} of draws — from a stream with NO real "
          f"regime changes.")
    print("  => the classifier assigns 0.1x-1.0x position size largely on noise.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
