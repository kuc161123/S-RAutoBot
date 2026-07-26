#!/usr/bin/env python3
"""Rigorous symbol/config selection — with an actual out-of-sample firewall.

WHY THIS EXISTS
---------------
The live config came from backtest_3yr_walkforward.py, whose selection is compromised:

  line 544  best_total_test_r  -> the config chosen is the one that MAXIMISES the
                                  "test" window R. The test windows are therefore a
                                  second training set, not a test set.
  line 517  configs are also FILTERED on test-window avg-R / PF / drawdown.
  line 563  the Monte Carlo runs on those same test trades, so it validates nothing.
  line 57   MIN_TEST_TRADES = 3 — best-of-30 chosen on three trades.
  GRID      4 div x 3 atr x 10 rr x ~500 symbols ~= 60,000 tests, uncorrected.
  windows   overlap heavily, so "passes all 4" is not 4 independent confirmations.
  no purge/embargo; no untouched holdout anywhere.

Measured consequence (backtest_selection_value.py): the selection is worth
+0.2197 R/trade in-sample (t=+15.66) and -0.0437 out-of-sample (t=-1.09), with a
fit->OOS rank correlation of +0.036. It transfers nothing.

WHAT THIS DOES DIFFERENTLY
--------------------------
1. HOLDOUT FIREWALL. Everything from HOLDOUT_START onward is untouched until a single
   final evaluation. No peeking, no tuning against it, one shot.
2. SELECTION USES TRAIN ONLY. Inside the design period we run rolling folds; a config
   is chosen using data strictly BEFORE the fold and scored on the fold. The score is
   therefore genuinely out-of-fold.
3. EMBARGO. A gap between train end and fold start so trades that span the boundary
   cannot leak. Trades are assigned by ENTRY time and must also RESOLVE before the
   segment ends, or they are dropped — no look-past-the-edge.
4. HONEST MINIMUMS. >= MIN_TRADES per config in the selection window (not 8, not 3).
5. MULTIPLE TESTING IS PRICED IN. Selection uses a shrunk estimate (James-Stein style
   pull toward the global mean) so a config with few trades cannot win on noise.
6. THE NULL IS TESTED. Every selection rule is compared against "one global config for
   everything" and "trade everything equally". If selection cannot beat those on the
   holdout, the honest answer is that selection does not work — and this script will
   say so rather than shipping a config.

LIVE FIDELITY
-------------
Mirrors the bot as it now runs (post BOS-timing fix): BOS judged on the last CLOSED
candle, entry at the NEXT candle's open, ATR from the BOS candle, EMA-200 gate at
confirmation, 12-candle BOS wait, SL-wins-ties, fees+slippage per side, and the
regime-aware CHOP gate applied at the entry candle.
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

ROOT = Path(__file__).parent
CACHE = ROOT / "cache_3yr_1h"

# ---- the firewall -----------------------------------------------------------
HOLDOUT_START = pd.Timestamp("2026-02-01")   # nothing at/after this informs selection
DESIGN_START = pd.Timestamp("2023-06-01")

# ---- search space (deliberately SMALL — every extra option costs OOS power) --
RR_GRID = [3.0, 5.0, 8.0, 10.0]
ATR_GRID = [1.0, 1.5, 2.0]
DIV_TYPES = ["REG_BULL", "REG_BEAR", "HID_BULL", "HID_BEAR"]

MAX_WAIT = bt.MAX_WAIT_CANDLES
FEE_PER_SIDE = 0.0006
SLIP_PER_SIDE = 0.0003
ROUND_TRIP = 2 * (FEE_PER_SIDE + SLIP_PER_SIDE)
CHOP_THRESHOLD = 52.0
CHOP_PERIOD = 14

MIN_TRADES_SELECT = 40      # per config, in the selection window
EMBARGO_DAYS = 14
SHRINK_K = 60.0             # trades needed before a config's own mean is trusted ~50%


def chop_series(df, period=CHOP_PERIOD):
    hl = df["high"] - df["low"]
    hc = (df["high"] - df["close"].shift()).abs()
    lc = (df["low"] - df["close"].shift()).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    hi = df["high"].rolling(period).max()
    lo = df["low"].rolling(period).min()
    return (100 * np.log10(tr.rolling(period).sum() / (hi - lo).replace(0, np.nan))
            / math.log10(period))


def replay_symbol(args):
    """All (div,rr,atr) trades for one symbol, with live semantics. Returns rows."""
    symbol, cache_dir = args
    f = Path(cache_dir) / f"{symbol}.parquet"
    if not f.exists():
        return []
    try:
        df = pd.read_parquet(f)
    except Exception:
        return []
    if df.empty or len(df) < 3000:      # need real history to be a candidate at all
        return []

    df = bt.prepare_data(df)
    df["chop"] = chop_series(df)
    o = df["open"].to_numpy(); h = df["high"].to_numpy()
    lo_ = df["low"].to_numpy(); c = df["close"].to_numpy()
    atr_a = df["atr"].to_numpy(); ema = df["ema"].to_numpy()
    chop_a = df["chop"].to_numpy(); ts = df["start"].to_numpy()
    n = len(c)

    sigs = {}
    for s in bt.detect_signals(df):
        sigs.setdefault(s["type"], []).append(s)

    rows = []
    for dt in DIV_TYPES:
        for s in sigs.get(dt, []):
            conf, side, lvl = s["conf_idx"], s["side"], s["swing"]
            # --- BOS on the last CLOSED candle (live, post-fix) ---
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
            e = bos + 1                      # entry at NEXT candle open
            if e >= n:
                continue
            # --- EMA-200 gate at confirmation (live behaviour) ---
            if not np.isfinite(ema[bos]):
                continue
            if side == "long" and not c[bos] > ema[bos]:
                continue
            if side == "short" and not c[bos] < ema[bos]:
                continue
            # --- CHOP gate at the entry candle ---
            cv = chop_a[e]
            if np.isfinite(cv) and cv >= CHOP_THRESHOLD:
                continue
            atr = atr_a[bos]
            if not np.isfinite(atr) or atr <= 0:
                continue

            for am in ATR_GRID:
                sl_d = atr * am
                if sl_d <= 0:
                    continue
                raw = o[e]
                entry = raw * (1 + SLIP_PER_SIDE) if side == "long" else raw * (1 - SLIP_PER_SIDE)
                sl = entry - sl_d if side == "long" else entry + sl_d
                fee_r = ROUND_TRIP * entry / sl_d
                for rr in RR_GRID:
                    tp = entry + sl_d * rr if side == "long" else entry - sl_d * rr
                    r = None; xi = None
                    for k in range(e, n):
                        if side == "long":
                            hs, ht = lo_[k] <= sl, h[k] >= tp
                        else:
                            hs, ht = h[k] >= sl, lo_[k] <= tp
                        if not (hs or ht):
                            continue
                        r = -1.0 if hs else rr       # SL wins ties
                        xi = k
                        break
                    if r is None:
                        continue                     # unresolved — never counted
                    rows.append((symbol, dt, side, rr, am,
                                 ts[e], ts[xi], r - fee_r))
    return rows


# ----------------------------------------------------------------------------
# selection rules — each takes a TRAIN slice and returns {(sym,div): (rr,atr)}
# ----------------------------------------------------------------------------
def rule_global(train):
    """One (rr, atr) for every symbol and div type. Zero per-symbol freedom."""
    g = train.groupby(["rr", "atr_mult"]).r.agg(["mean", "count"])
    g = g[g["count"] >= 500]
    if g.empty:
        return None, None
    best = g["mean"].idxmax()
    return best, None


def rule_per_symbol_naive(train):
    """The OLD way: per (symbol,div) pick the max-mean variant. No guards."""
    g = train.groupby(["symbol", "div_type", "rr", "atr_mult"]).r.mean()
    out = {}
    for (sym, dv, rr, am), v in g.items():
        k = (sym, dv)
        if k not in out or v > out[k][1]:
            out[k] = ((rr, am), v)
    return {k: v[0] for k, v in out.items()}


def rule_per_symbol_shrunk(train, gmean):
    """Per (symbol,div) but with a minimum sample and shrinkage toward the global
    mean, so a config cannot win on a small lucky sample."""
    g = train.groupby(["symbol", "div_type", "rr", "atr_mult"]).r.agg(["mean", "count"])
    g = g[g["count"] >= MIN_TRADES_SELECT]
    if g.empty:
        return {}
    # James-Stein style: pull each estimate toward the global mean by sample size
    w = g["count"] / (g["count"] + SHRINK_K)
    g["score"] = w * g["mean"] + (1 - w) * gmean
    out = {}
    for (sym, dv, rr, am), row in g.iterrows():
        k = (sym, dv)
        if k not in out or row["score"] > out[k][1]:
            out[k] = ((rr, am), row["score"])
    return {k: v[0] for k, v in out.items()}


def apply_rule(seg, mapping, global_choice=None):
    """Score a segment under a rule. Returns the selected trades."""
    if mapping is None:
        rr, am = global_choice
        return seg[(seg.rr == rr) & (seg.atr_mult == am)]
    if not mapping:
        return seg.iloc[0:0]
    key = list(zip(seg.symbol, seg.div_type))
    want_rr = np.array([mapping.get(k, (None, None))[0] for k in key], dtype=object)
    want_am = np.array([mapping.get(k, (None, None))[1] for k in key], dtype=object)
    m = (seg.rr.values == want_rr) & (seg.atr_mult.values == want_am)
    return seg[m]


def stats(r):
    r = np.asarray(r, dtype=float)
    if len(r) == 0:
        return dict(n=0, avg=0.0, wr=0.0, pf=0.0, tot=0.0)
    gp = r[r > 0].sum(); gl = abs(r[r < 0].sum())
    return dict(n=len(r), avg=r.mean(), wr=(r > 0).mean(),
                pf=(gp / gl) if gl else float("inf"), tot=r.sum())


def line(lbl, s):
    pf = "inf" if s["pf"] == float("inf") else f"{s['pf']:.3f}"
    return (f"  {lbl:<34}{s['n']:>8}t  WR {s['wr']:>6.2%}  avgR {s['avg']:>+8.4f}  "
            f"totR {s['tot']:>+9.1f}  PF {pf:>6}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=max(1, mp.cpu_count() - 1))
    ap.add_argument("--out", default="rigorous_selection")
    a = ap.parse_args()

    syms = sorted(p.stem for p in CACHE.glob("*.parquet"))
    syms = [s for s in syms if not s.endswith(("26JUN26", "03APR26", "10APR26", "17APR26"))]
    if a.limit:
        syms = syms[: a.limit]
    print(f"[RIG] candidate universe: {len(syms)} cached symbols "
          f"(live config trades 277)", flush=True)
    print(f"[RIG] grid {len(DIV_TYPES)}div x {len(RR_GRID)}rr x {len(ATR_GRID)}atr "
          f"= {len(DIV_TYPES)*len(RR_GRID)*len(ATR_GRID)} per symbol")
    print(f"[RIG] HOLDOUT (untouched): {HOLDOUT_START.date()} onward\n", flush=True)

    rows = []
    with mp.Pool(a.workers) as pool:
        for i, res in enumerate(pool.imap_unordered(
                replay_symbol, [(s, str(CACHE)) for s in syms], chunksize=4), 1):
            rows.extend(res)
            if i % 50 == 0:
                print(f"  {i}/{len(syms)} symbols · {len(rows):,} trades", flush=True)

    df = pd.DataFrame(rows, columns=["symbol", "div_type", "side", "rr", "atr_mult",
                                     "entry_time", "exit_time", "r"])
    df["entry_time"] = pd.to_datetime(df["entry_time"])
    df["exit_time"] = pd.to_datetime(df["exit_time"])
    df = df[df.entry_time >= DESIGN_START]
    df.to_parquet(ROOT / f"{a.out}_trades.parquet", index=False)
    print(f"\n[RIG] {len(df):,} candidate trades over {df.symbol.nunique()} symbols")

    design = df[df.exit_time < HOLDOUT_START]        # must RESOLVE before the firewall
    holdout = df[df.entry_time >= HOLDOUT_START]
    print(f"[RIG] design {len(design):,} trades ({design.entry_time.min().date()} .. "
          f"{design.entry_time.max().date()})")
    print(f"[RIG] holdout {len(holdout):,} trades ({holdout.entry_time.min().date()} .. "
          f"{holdout.entry_time.max().date()})")

    gmean = design.r.mean()

    # ---------------- rolling out-of-fold validation inside DESIGN -------------
    print(f"\n{'=' * 92}\nOUT-OF-FOLD VALIDATION (inside design period; embargo "
          f"{EMBARGO_DAYS}d)\n{'=' * 92}")
    folds = pd.date_range(pd.Timestamp("2024-09-01"), HOLDOUT_START, freq="3ME")
    oof = {k: [] for k in ("global", "naive", "shrunk", "all")}
    for f0 in folds:
        f1 = f0 + pd.DateOffset(months=3)
        tr = design[design.exit_time < f0 - pd.Timedelta(days=EMBARGO_DAYS)]
        te = design[(design.entry_time >= f0) & (design.entry_time < f1)]
        if len(tr) < 20000 or len(te) < 500:
            continue
        gsel, _ = rule_global(tr)
        oof["global"].extend(apply_rule(te, None, gsel).r.tolist())
        oof["naive"].extend(apply_rule(te, rule_per_symbol_naive(tr)).r.tolist())
        oof["shrunk"].extend(apply_rule(te, rule_per_symbol_shrunk(tr, tr.r.mean())).r.tolist())
        oof["all"].extend(te.r.tolist())
        print(f"  fold {f0.date()}..{f1.date()}  train {len(tr):>7,}  test {len(te):>6,}"
              f"  global {gsel}", flush=True)

    print()
    for k, lbl in [("all", "trade everything (no selection)"),
                   ("global", "ONE global (rr,atr)"),
                   ("naive", "per-symbol best (the OLD way)"),
                   ("shrunk", "per-symbol + min-N + shrinkage")]:
        print(line(lbl, stats(oof[k])))

    # ---------------- final: fit on ALL design, evaluate ONCE on holdout -------
    print(f"\n{'=' * 92}\nFINAL — rules fitted on ALL design data, scored ONCE on the "
          f"untouched holdout\n{'=' * 92}")
    gsel, _ = rule_global(design)
    naive = rule_per_symbol_naive(design)
    shrunk = rule_per_symbol_shrunk(design, gmean)
    print(f"  global choice: rr={gsel[0]:.0f} atr={gsel[1]:.1f}   ·   "
          f"per-symbol configs selected: naive {len(naive):,}, shrunk {len(shrunk):,}\n")

    res = {}
    for k, lbl, sel in [
            ("all", "trade everything (no selection)", holdout),
            ("global", f"ONE global rr={gsel[0]:.0f}/atr={gsel[1]:.1f}",
             apply_rule(holdout, None, gsel)),
            ("naive", "per-symbol best (the OLD way)", apply_rule(holdout, naive)),
            ("shrunk", "per-symbol + min-N + shrinkage", apply_rule(holdout, shrunk))]:
        s = stats(sel.r.values)
        res[k] = s
        print(line(lbl, s))

    print(f"\n  DOES SELECTION ADD ANYTHING ON THE HOLDOUT?")
    base = res["global"]["avg"]
    for k, lbl in [("naive", "per-symbol best"), ("shrunk", "per-symbol shrunk")]:
        d = res[k]["avg"] - base
        print(f"    {lbl:<24} vs global: {d:+.4f} R/trade "
              f"{'(better)' if d > 0 else '(WORSE — selection destroys value)'}")

    # stability of per-symbol choices across time — if selection were real, a symbol's
    # best config should persist
    print(f"\n  STABILITY CHECK — does a symbol's best config persist across halves?")
    mid = design.entry_time.quantile(0.5)
    h1 = rule_per_symbol_shrunk(design[design.entry_time < mid], gmean)
    h2 = rule_per_symbol_shrunk(design[design.entry_time >= mid], gmean)
    both = set(h1) & set(h2)
    if both:
        agree = sum(1 for k in both if h1[k] == h2[k]) / len(both)
        print(f"    same (rr,atr) chosen in both halves: {agree:.1%} of {len(both)} pairs "
              f"(random = {1/(len(RR_GRID)*len(ATR_GRID)):.1%})")

    pd.DataFrame([dict(rule=k, **v) for k, v in res.items()]).to_csv(
        ROOT / f"{a.out}_holdout.csv", index=False)
    print(f"\n[RIG] -> {a.out}_holdout.csv, {a.out}_trades.parquet")
    return 0


if __name__ == "__main__":
    sys.exit(main())
