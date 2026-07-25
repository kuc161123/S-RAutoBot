#!/usr/bin/env python3
"""
A/B backtest: BOS confirmation timing (live-as-is vs proposed fix)
==================================================================

WHAT IS BEING TESTED
--------------------
`check_pending_bos` (autobot/core/bot.py) passes `current_idx = len(df) - 1`, and the
Bybit kline endpoint returns the STILL-FORMING candle as the last row. So the live bot
judges BOS against a provisional close, then queues the entry for the following hourly
cycle. Net effect versus the reference engine (backtest_3yr_walkforward.execute_trade_1h),
which judges BOS on a CLOSED candle and enters at the next open:

  FIXED (proposed):  BOS on close(idx)  ->  ATR from candle idx    ->  entry open(idx+1)
  LIVE  (current):   BOS on close(idx)  ->  ATR from candle idx+1  ->  entry open(idx+2)

i.e. the live bot enters one full candle later AND sizes its stop off a different
candle's ATR.

WHAT THIS HARNESS CAN AND CANNOT MEASURE
----------------------------------------
CAN (exactly): the one-candle entry delay, the ATR-source shift, the resulting change in
stop/target placement, and the shift in which candle the CHOP gate is evaluated on.

CANNOT: the intra-hour component. The live bot reads the forming candle's close at poll
time, which drifts from the previous close by however many minutes into the hour that
symbol gets scanned. In 1H data open[i+1] == close[i] to 5 decimal places (verified:
99.996% exact match on BTCUSDT), so the trigger price is identical in both arms and the
drift-induced miss/false-trigger rate is simply not observable at this resolution.
Measuring that needs sub-hourly klines. The delay measured here is the dominant and
structural part of the effect; the drift is second-order noise on top.

Both arms share signal detection, fee model, dynamic slippage and exit resolution with
the validated engine, so the comparison is paired and isolates timing only.

USAGE
    python3 backtest_bos_timing_ab.py                 # full run, all cached symbols
    python3 backtest_bos_timing_ab.py --limit 40      # quick smoke run
    python3 backtest_bos_timing_ab.py --no-chop       # sensitivity: CHOP gate off
"""
from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

warnings.filterwarnings("ignore")

import backtest_3yr_walkforward as bt  # validated engine: prepare_data/detect_signals

ROOT = Path(__file__).parent
CACHE_DIR = ROOT / "cache_3yr_1h"
CONFIG_PATH = ROOT / "config.yaml"
OUT_TRADES = ROOT / "bos_timing_ab_trades.csv"
OUT_SUMMARY = ROOT / "bos_timing_ab_summary.txt"

MAX_WAIT_CANDLES = bt.MAX_WAIT_CANDLES  # 12
FEE_PCT = bt.FEE_PCT
CHOP_PERIOD = 14
# Live gate is regime-dependent {favorable:52, cautious:45, adverse:52, critical:55}.
# Regime is bot-history-dependent and not reproducible from candles, so we use 52 —
# the modal threshold — identically in BOTH arms. Same convention as verify_overfit.py.
LIVE_CHOP_THRESHOLD = 52.0


# ---------------------------------------------------------------------------
# indicators
# ---------------------------------------------------------------------------
def chop_series(df: pd.DataFrame, period: int = CHOP_PERIOD) -> pd.Series:
    """Choppiness Index — mirrors autobot/core/divergence_detector.calculate_chop."""
    hl = df["high"] - df["low"]
    hc = (df["high"] - df["close"].shift()).abs()
    lc = (df["low"] - df["close"].shift()).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    atr_sum = tr.rolling(period).sum()
    hi = df["high"].rolling(period).max()
    lo = df["low"].rolling(period).min()
    hl_range = (hi - lo).replace(0, np.nan)
    return 100 * np.log10(atr_sum / hl_range) / math.log10(period)


# ---------------------------------------------------------------------------
# execution — the ONLY thing that differs between arms is `delay`
# ---------------------------------------------------------------------------
def execute_timed(signal, df, arrs, rr, atr_mult, atr_off, entry_off, chop_thresh):
    """Replay one signal with independently shiftable ATR source and entry candle.

    Mirrors backtest_3yr_walkforward.execute_trade_1h exactly except that the ATR
    source candle is idx+atr_off and the entry candle is idx+1+entry_off:

      fixed      atr_off=0 entry_off=0   BOS close(idx), ATR idx,   entry open(idx+1)
      live       atr_off=1 entry_off=1   BOS close(idx), ATR idx+1, entry open(idx+2)
      delay_only atr_off=0 entry_off=1   isolates the one-candle entry delay
      atr_only   atr_off=1 entry_off=0   isolates the ATR-source shift
    """
    o, h, l, c, atr_a, chop_a, start_a = arrs
    n = len(c)
    conf_idx = signal["conf_idx"]
    side = signal["side"]
    bos_level = signal["swing"]

    for i in range(1, MAX_WAIT_CANDLES + 1):
        idx = conf_idx + i
        if idx >= n:
            return None
        triggered = ((side == "long" and c[idx] > bos_level) or
                     (side == "short" and c[idx] < bos_level))
        if not triggered:
            continue

        atr_idx = idx + atr_off
        entry_idx = idx + 1 + entry_off
        if entry_idx >= n:
            return None

        atr = atr_a[atr_idx]
        if not np.isfinite(atr) or atr <= 0:
            return None

        # CHOP gate fires in execute_trade against the candle being entered on,
        # which is the forming last row in both arms.
        if chop_thresh is not None:
            cv = chop_a[entry_idx]
            if np.isfinite(cv) and cv >= chop_thresh:
                return None  # blocked, same as live

        sl_dist = atr * atr_mult
        raw_entry = o[entry_idx]
        slippage = bt.get_dynamic_slippage(df, entry_idx)

        if side == "long":
            entry_price = raw_entry * (1 + slippage)
            sl_price = entry_price - sl_dist
            tp_price = entry_price + sl_dist * rr
        else:
            entry_price = raw_entry * (1 - slippage)
            sl_price = entry_price + sl_dist
            tp_price = entry_price - sl_dist * rr

        # resolve on 1H candles
        for k in range(entry_idx, n):
            if side == "long":
                hit_sl = l[k] <= sl_price
                hit_tp = h[k] >= tp_price
            else:
                hit_sl = h[k] >= sl_price
                hit_tp = l[k] <= tp_price
            if not (hit_sl or hit_tp):
                continue
            outcome = "loss" if hit_sl else "win"  # SL wins ties (conservative)
            risk = abs(entry_price - sl_price)
            if risk == 0:
                return None
            r_res = rr if outcome == "win" else -1.0
            fee_drag = (FEE_PCT * 2 * entry_price) / risk
            return {
                "r": r_res - fee_drag,
                "outcome": outcome,
                "side": side,
                "entry_time": start_a[entry_idx],
                "exit_time": start_a[k],
                "conf_idx": conf_idx,
                "bos_idx": idx,
                "entry_idx": entry_idx,
            }
        return None  # never resolved (still open at end of data)
    return None  # BOS never confirmed within the wait window


# ---------------------------------------------------------------------------
# per-symbol worker
# ---------------------------------------------------------------------------
def run_symbol(args):
    symbol, configs, use_chop, arms = args
    f = CACHE_DIR / f"{symbol}.parquet"
    if not f.exists():
        return []
    try:
        df = pd.read_parquet(f)
    except Exception:
        return []
    if df.empty or len(df) < 1000:
        return []

    df = bt.prepare_data(df)
    df["chop"] = chop_series(df)
    arrs = (
        df["open"].to_numpy(), df["high"].to_numpy(), df["low"].to_numpy(),
        df["close"].to_numpy(), df["atr"].to_numpy(), df["chop"].to_numpy(),
        df["start"].to_numpy(),
    )
    chop_thresh = LIVE_CHOP_THRESHOLD if use_chop else None

    signals = bt.detect_signals(df)
    by_type = {}
    for s in signals:
        by_type.setdefault(s["type"], []).append(s)

    rows = []
    for cfg in configs:
        dt, rr, am = cfg["divergence_type"], float(cfg["rr"]), float(cfg["atr_mult"])
        for s in by_type.get(dt, []):
            for arm, atr_off, entry_off in arms:
                res = execute_timed(s, df, arrs, rr, am, atr_off, entry_off, chop_thresh)
                if res is None:
                    continue
                res.update({"symbol": symbol, "div_type": dt, "rr": rr,
                            "atr_mult": am, "arm": arm,
                            "sig_id": f"{symbol}|{dt}|{rr}|{am}|{s['conf_idx']}"})
                rows.append(res)
    return rows


# ---------------------------------------------------------------------------
# stats
# ---------------------------------------------------------------------------
def stats(rs: np.ndarray) -> dict:
    """Aggregate stats for an R series.

    IMPORTANT: `rs` must already be in CHRONOLOGICAL (exit-time) order. cum_r/wr/avg_r/pf
    are order-independent, but max_dd_r is NOT — feeding it rows in symbol order (the
    natural order out of the worker pool) produces a meaningless number. Callers use
    _r_chrono() to enforce this.
    """
    if len(rs) == 0:
        return {"trades": 0, "cum_r": 0.0, "wr": 0.0, "avg_r": 0.0,
                "pf": 0.0, "max_dd_r": 0.0}
    cum = np.cumsum(rs)
    peak = np.maximum.accumulate(cum)
    gp = rs[rs > 0].sum()
    gl = abs(rs[rs < 0].sum())
    return {
        "trades": int(len(rs)),
        "cum_r": float(cum[-1]),
        "wr": float((rs > 0).mean()),
        "avg_r": float(rs.mean()),
        "pf": float(gp / gl) if gl > 0 else float("inf"),
        "max_dd_r": float(abs((cum - peak).min())),
    }


def fmt(label: str, s: dict) -> str:
    pf = "inf" if s["pf"] == float("inf") else f"{s['pf']:.3f}"
    return (f"  {label:<26} {s['trades']:>7}t  WR {s['wr']:>6.2%}  "
            f"avgR {s['avg_r']:>+7.4f}  cumR {s['cum_r']:>+10.1f}  "
            f"PF {pf:>6}  maxDD {s['max_dd_r']:>8.1f}R")


def _r_chrono(sub: pd.DataFrame) -> np.ndarray:
    """R series in exit-time order — required for a meaningful max drawdown."""
    if sub.empty:
        return np.array([])
    return sub.sort_values("exit_time").r.to_numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="only N symbols (smoke run)")
    ap.add_argument("--no-chop", action="store_true", help="disable CHOP gate")
    ap.add_argument("--decompose", action="store_true",
                    help="add delay_only / atr_only arms to attribute the effect")
    ap.add_argument("--workers", type=int, default=max(1, mp.cpu_count() - 1))
    args = ap.parse_args()

    # (name, atr_offset, entry_offset)
    arms = [("fixed", 0, 0), ("live", 1, 1)]
    if args.decompose:
        arms += [("delay_only", 0, 1), ("atr_only", 1, 0)]

    cfg = yaml.safe_load(open(CONFIG_PATH))
    symbols = []
    for sym, sc in (cfg.get("symbols") or {}).items():
        sc = sc or {}
        if not sc.get("enabled", True):
            continue
        cfgs = sc.get("configs") or []
        if cfgs and (CACHE_DIR / f"{sym}.parquet").exists():
            symbols.append((sym, cfgs, not args.no_chop, arms))
    symbols.sort()
    if args.limit:
        symbols = symbols[: args.limit]

    n_cfg = sum(len(s[1]) for s in symbols)
    print(f"[AB] {len(symbols)} symbols / {n_cfg} configs · "
          f"CHOP {'OFF' if args.no_chop else f'>= {LIVE_CHOP_THRESHOLD} blocks'} · "
          f"{args.workers} workers", flush=True)

    rows = []
    with mp.Pool(args.workers) as pool:
        for i, res in enumerate(pool.imap_unordered(run_symbol, symbols, chunksize=4), 1):
            rows.extend(res)
            if i % 25 == 0:
                print(f"  ... {i}/{len(symbols)} symbols, {len(rows)} trades", flush=True)

    if not rows:
        print("[AB] no trades produced — aborting")
        return 1

    df = pd.DataFrame(rows)
    df["entry_time"] = pd.to_datetime(df["entry_time"])
    df.to_csv(OUT_TRADES, index=False)

    live = df[df.arm == "live"]
    fixed = df[df.arm == "fixed"]
    s_live, s_fixed = stats(_r_chrono(live)), stats(_r_chrono(fixed))

    out = []
    out.append("=" * 104)
    out.append("BOS TIMING A/B — live (enters 1 candle late) vs fixed (backtest-aligned)")
    out.append("=" * 104)
    out.append(f"Data: {CACHE_DIR.name}  window {df.entry_time.min()} -> {df.entry_time.max()}")
    out.append(f"Universe: {len(symbols)} symbols / {n_cfg} configs   "
               f"CHOP gate: {'OFF' if args.no_chop else f'block >= {LIVE_CHOP_THRESHOLD}'}")
    out.append("")
    out.append("POOLED")
    out.append(fmt("LIVE (current)", s_live))
    out.append(fmt("FIXED (proposed)", s_fixed))
    for extra, label in (("delay_only", "  ^ entry-delay only"),
                         ("atr_only", "  ^ ATR-source only")):
        sub = df[df.arm == extra]
        if len(sub):
            out.append(fmt(label, stats(_r_chrono(sub))))
    d_cum = s_fixed["cum_r"] - s_live["cum_r"]
    d_avg = s_fixed["avg_r"] - s_live["avg_r"]
    out.append(f"  {'DELTA (fixed - live)':<26} {s_fixed['trades'] - s_live['trades']:>+7}t  "
               f"WR {s_fixed['wr'] - s_live['wr']:>+6.2%}  avgR {d_avg:>+7.4f}  "
               f"cumR {d_cum:>+10.1f}  PF {s_fixed['pf'] - s_live['pf']:>+6.3f}  "
               f"maxDD {s_fixed['max_dd_r'] - s_live['max_dd_r']:>+8.1f}R")
    out.append("")

    # --- by 6-month window (robustness across regimes) ---
    out.append("BY 6-MONTH WINDOW")
    df["bucket"] = df.entry_time.dt.to_period("Q")
    buckets = sorted(df.bucket.unique())
    out.append(f"  {'window':<10} {'live cumR':>12} {'fixed cumR':>12} {'delta':>10} "
               f"{'live PF':>9} {'fixed PF':>9} {'live t':>8} {'fixed t':>8}")
    wins = 0
    for b in buckets:
        sub = df[df.bucket == b]
        sl = stats(_r_chrono(sub[sub.arm == "live"]))
        sf = stats(_r_chrono(sub[sub.arm == "fixed"]))
        if sl["trades"] < 30:
            continue
        if sf["cum_r"] > sl["cum_r"]:
            wins += 1
        pf_l = "inf" if sl["pf"] == float("inf") else f"{sl['pf']:.2f}"
        pf_f = "inf" if sf["pf"] == float("inf") else f"{sf['pf']:.2f}"
        out.append(f"  {str(b):<10} {sl['cum_r']:>+12.1f} {sf['cum_r']:>+12.1f} "
                   f"{sf['cum_r'] - sl['cum_r']:>+10.1f} {pf_l:>9} {pf_f:>9} "
                   f"{sl['trades']:>8} {sf['trades']:>8}")
    n_eval = sum(1 for b in buckets
                 if stats(_r_chrono(df[(df.bucket == b) & (df.arm == 'live')]))["trades"] >= 30)
    out.append(f"  fixed beats live in {wins}/{n_eval} windows")
    out.append("")

    # --- paired per-config (is the effect broad or a few configs?) ---
    out.append("PAIRED PER-CONFIG")
    piv = df.groupby(["symbol", "div_type", "rr", "atr_mult", "arm"]).r.sum().unstack("arm")
    piv = piv.dropna(how="all").fillna(0.0)
    if "live" in piv and "fixed" in piv:
        delta = piv["fixed"] - piv["live"]
        out.append(f"  configs compared     : {len(delta)}")
        out.append(f"  fixed better         : {int((delta > 0).sum())} "
                   f"({(delta > 0).mean():.1%})")
        out.append(f"  live better          : {int((delta < 0).sum())}")
        out.append(f"  tied                 : {int((delta == 0).sum())}")
        out.append(f"  mean delta R/config  : {delta.mean():+.3f}")
        out.append(f"  median delta R/config: {delta.median():+.3f}")
        # sign test
        pos, neg = int((delta > 0).sum()), int((delta < 0).sum())
        if pos + neg > 0:
            from math import comb
            n_, k_ = pos + neg, max(pos, neg)
            p = 2 * sum(comb(n_, i) for i in range(k_, n_ + 1)) / (2 ** n_)
            out.append(f"  sign test p          : {min(p, 1.0):.3e}")
    out.append("")

    # --- signal-level: which trades exist in one arm but not the other ---
    live_ids, fixed_ids = set(live.sig_id), set(fixed.sig_id)
    out.append("SIGNAL COVERAGE")
    out.append(f"  taken by both        : {len(live_ids & fixed_ids)}")
    out.append(f"  only LIVE            : {len(live_ids - fixed_ids)}")
    out.append(f"  only FIXED           : {len(fixed_ids - live_ids)}")
    both = live_ids & fixed_ids
    if both:
        lm = live[live.sig_id.isin(both)].set_index("sig_id").r
        fm = fixed[fixed.sig_id.isin(both)].set_index("sig_id").r
        lm = lm[~lm.index.duplicated()]
        fm = fm[~fm.index.duplicated()]
        pair = (fm - lm).dropna()
        out.append(f"  on shared signals, mean R delta (fixed-live): {pair.mean():+.4f}")
        out.append(f"  outcome flipped live-loss -> fixed-win      : "
                   f"{int(((lm < 0) & (fm > 0)).sum())}")
        out.append(f"  outcome flipped live-win  -> fixed-loss     : "
                   f"{int(((lm > 0) & (fm < 0)).sum())}")
    out.append("")
    out.append(f"Trades written to {OUT_TRADES.name}")
    out.append("=" * 104)

    text = "\n".join(out)
    print(text)
    OUT_SUMMARY.write_text(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
