"""
AGENT-REP2 -- independent replication, PART 1.

Re-resolves the trade set from raw 1H klines for the 277 live symbols, WITHOUT
importing backtest_production_correct.py, risk_study/sweep.py, risk_study/monthly.py
or build_trail_universe_wide.py. Written from scratch against the spec in the task
brief. I did read autobot/core/divergence_detector.py (the live detector, not one of
the four barred files) to pin down exact tie-break/dedup semantics that the brief left
implicit (e.g. which pivot pair "wins" when two scan positions land on the same pivots,
the scan_start floor, the swing-level slice bounds). The pivot-finding and signal-scan
code below is my own re-derivation (vectorized differently -- rolling max/min instead
of the live code's O(n*7) brute-force neighbour loop; the divergence scan uses bisect
on precomputed sorted pivot-index arrays instead of a raw backward Python loop), not a
copy of that file. Everything past signal detection (BOS, gates, entry construction,
the s3_a1 trailing exit) is implemented directly from the task brief with no reference
to any bot module.

Produces two trade sets (candidates identical between arms; only rr/atr_mult and
therefore the trailing-exit path differ):
  A0 -- live per-symbol (divergence_type, rr, atr_mult) from config.yaml
  A1 -- same (symbol, divergence_type) pairs, global rr=10.0, atr_mult=3.0

Output: trades_A0.parquet, trades_A1.parquet in this directory.
"""
import bisect
import os
import sys
import time

import numpy as np
import pandas as pd
import yaml

REPO = "/Users/lualakol/AutoTrading Bot"
CACHE_DIR = os.path.join(REPO, "cache_3yr_1h")
OUT_DIR = os.path.join(REPO, "risk_study", "agent_out")

RSI_PERIOD = 14
ATR_PERIOD = 14
CHOP_PERIOD = 14
EMA_PERIOD = 200
PIVOT_LEFT = 3
PIVOT_RIGHT = 3
MIN_PIVOT_DISTANCE = 3
MAX_PIVOT_AGE = 10          # triggering pivot must be <=10 bars old at scan bar
LOOKBACK_BARS = 50          # how far back to search for the pivot pair
SCAN_START = max(205, LOOKBACK_BARS + PIVOT_RIGHT + 1)
BOS_MAX_WAIT = 12           # bars after the signal bar to find a BOS close
CHOP_GATE = 52.0
TRAIL_TRIGGER_R = 3.0
TRAIL_ATR_MULT = 1.0        # trailing_stop.atr_mult (fixed, independent of entry atr_mult)

WINDOW_START = pd.Timestamp("2023-06-01")
WINDOW_END = pd.Timestamp("2026-07-25 23:59:59")

DIV_TYPES = {
    "REG_BULL": ("low", "long"),
    "HID_BULL": ("low", "long"),
    "REG_BEAR": ("high", "short"),
    "HID_BEAR": ("high", "short"),
}


def load_live_symbols():
    with open(os.path.join(REPO, "config.yaml")) as f:
        cfg = yaml.safe_load(f)
    syms = cfg["symbols"]
    live = {}
    for k, v in syms.items():
        if v.get("enabled") and v.get("configs"):
            live[k] = {c["divergence_type"]: (float(c["rr"]), float(c["atr_mult"])) for c in v["configs"]}
    return live


def load_klines(symbol):
    path = os.path.join(CACHE_DIR, f"{symbol}.parquet")
    if not os.path.exists(path):
        return None
    df = pd.read_parquet(path, columns=["start", "open", "high", "low", "close"])
    df = df.sort_values("start").drop_duplicates(subset="start").reset_index(drop=True)
    return df


def compute_indicators(df):
    close = df["close"]
    high = df["high"]
    low = df["low"]

    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.rolling(RSI_PERIOD).mean()
    avg_loss = loss.rolling(RSI_PERIOD).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - 100 / (1 + rs)

    hl = high - low
    hc = (high - close.shift()).abs()
    lc = (low - close.shift()).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    atr = tr.rolling(ATR_PERIOD).mean()

    ema200 = close.ewm(span=EMA_PERIOD, adjust=False).mean()

    atr_sum = tr.rolling(CHOP_PERIOD).sum()
    hh = high.rolling(CHOP_PERIOD).max()
    ll = low.rolling(CHOP_PERIOD).min()
    hl_range = (hh - ll).replace(0, np.nan)
    chop = 100 * np.log10(atr_sum / hl_range) / np.log10(CHOP_PERIOD)

    return (
        rsi.to_numpy(),
        atr.to_numpy(),
        ema200.to_numpy(),
        chop.to_numpy(),
    )


def find_pivots(vals, left=PIVOT_LEFT, right=PIVOT_RIGHT):
    """Vectorized strict-fractal pivot finder.
    pivot_high[i] = vals[i] iff vals[i] is strictly greater than every one of the
    `left` bars before it and every one of the `right` bars after it.
    Implemented via rolling max/min of the flanking windows rather than the
    brute-force per-index neighbour loop the live detector uses -- mathematically
    equivalent (close[i] > max(left window) AND close[i] > max(right window)
    <=> close[i] > max(left window, right window)), just computed differently.
    """
    s = pd.Series(vals)
    left_max = s.rolling(left).max().shift(1).to_numpy()
    rev = pd.Series(vals[::-1])
    right_max_rev = rev.rolling(right).max().shift(1).to_numpy()
    right_max = right_max_rev[::-1]

    left_min = s.rolling(left).min().shift(1).to_numpy()
    right_min_rev = rev.rolling(right).min().shift(1).to_numpy()
    right_min = right_min_rev[::-1]

    combined_max = np.fmax(left_max, right_max)
    combined_min = np.fmin(left_min, right_min)

    is_high = vals > combined_max
    is_low = vals < combined_min

    n = len(vals)
    is_high[:left] = False
    is_high[n - right:] = False
    is_low[:left] = False
    is_low[n - right:] = False

    ph = np.where(is_high, vals, np.nan)
    pl = np.where(is_low, vals, np.nan)
    return ph, pl


def detect_signals(df, close, high, low, rsi, ema200, allowed_types):
    """Returns list of dicts: divergence_code, side, sig_idx, pivot_idx, swing_level."""
    n = len(df)
    if n < 100:
        return []

    ph, pl = find_pivots(close)
    pivot_high_idx = np.flatnonzero(~np.isnan(ph))
    pivot_low_idx = np.flatnonzero(~np.isnan(pl))

    signals = []
    used = set()

    scan_end = n - PIVOT_RIGHT
    for i in range(SCAN_START, scan_end):
        if np.isnan(ema200[i]) or np.isnan(rsi[i]):
            continue

        lo_bound = max(i - LOOKBACK_BARS - PIVOT_RIGHT, 0) + 1
        hi_bound = i - PIVOT_RIGHT

        # ---- bullish (pivot lows) ----
        if hi_bound >= lo_bound:
            pos = bisect.bisect_right(pivot_low_idx, hi_bound)
            if pos > 0:
                curr_pli = pivot_low_idx[pos - 1]
                if curr_pli >= lo_bound and (i - curr_pli) <= MAX_PIVOT_AGE:
                    hi2 = curr_pli - MIN_PIVOT_DISTANCE - 1
                    if hi2 >= lo_bound:
                        pos2 = bisect.bisect_right(pivot_low_idx, hi2)
                        if pos2 > 0:
                            prev_pli = pivot_low_idx[pos2 - 1]
                            if prev_pli >= lo_bound:
                                curr_pl = close[curr_pli]
                                prev_pl = close[prev_pli]
                                key = (curr_pli, prev_pli, "BULL")
                                if key not in used:
                                    swing_high = high[curr_pli:i + 1].max()
                                    code = None
                                    if curr_pl < prev_pl and rsi[curr_pli] > rsi[prev_pli]:
                                        code = "REG_BULL"
                                    elif curr_pl > prev_pl and rsi[curr_pli] < rsi[prev_pli]:
                                        code = "HID_BULL"
                                    if code is not None and code in allowed_types:
                                        signals.append(dict(
                                            divergence_code=code, side="long", sig_idx=i,
                                            pivot_idx=curr_pli, swing_level=swing_high,
                                        ))
                                        used.add(key)

        # ---- bearish (pivot highs) ----
        if hi_bound >= lo_bound:
            pos = bisect.bisect_right(pivot_high_idx, hi_bound)
            if pos > 0:
                curr_phi = pivot_high_idx[pos - 1]
                if curr_phi >= lo_bound and (i - curr_phi) <= MAX_PIVOT_AGE:
                    hi2 = curr_phi - MIN_PIVOT_DISTANCE - 1
                    if hi2 >= lo_bound:
                        pos2 = bisect.bisect_right(pivot_high_idx, hi2)
                        if pos2 > 0:
                            prev_phi = pivot_high_idx[pos2 - 1]
                            if prev_phi >= lo_bound:
                                curr_ph = close[curr_phi]
                                prev_ph = close[prev_phi]
                                key = (curr_phi, prev_phi, "BEAR")
                                if key not in used:
                                    swing_low = low[curr_phi:i + 1].min()
                                    code = None
                                    if curr_ph > prev_ph and rsi[curr_phi] < rsi[prev_phi]:
                                        code = "REG_BEAR"
                                    elif curr_ph < prev_ph and rsi[curr_phi] > rsi[prev_phi]:
                                        code = "HID_BEAR"
                                    if code is not None and code in allowed_types:
                                        signals.append(dict(
                                            divergence_code=code, side="short", sig_idx=i,
                                            pivot_idx=curr_phi, swing_level=swing_low,
                                        ))
                                        used.add(key)

    return signals


def resolve_bos(sig, close, ema200, chop, n):
    i = sig["sig_idx"]
    side = sig["side"]
    swing = sig["swing_level"]
    lo = i + 1
    hi = min(i + BOS_MAX_WAIT, n - 1)
    bos_idx = None
    for k in range(lo, hi + 1):
        if side == "long" and close[k] > swing:
            bos_idx = k
            break
        if side == "short" and close[k] < swing:
            bos_idx = k
            break
    if bos_idx is None:
        return None
    if side == "long" and not (close[bos_idx] > ema200[bos_idx]):
        return None
    if side == "short" and not (close[bos_idx] < ema200[bos_idx]):
        return None
    c = chop[bos_idx]
    if np.isnan(c) or c >= CHOP_GATE:
        return None
    entry_idx = bos_idx + 1
    if entry_idx >= n:
        return None
    return bos_idx, entry_idx


def simulate_trailing_exit(side, entry_price, risk_dist, rr, entry_idx, open_, high, low, atr, n):
    if side == "long":
        stop = entry_price - risk_dist
        tp = entry_price + risk_dist * rr
    else:
        stop = entry_price + risk_dist
        tp = entry_price - risk_dist * rr

    cur_stop = stop
    for k in range(entry_idx, n):
        hb = high[k]
        lb = low[k]
        if side == "long":
            hit_stop = lb <= cur_stop
            hit_tp = hb >= tp
        else:
            hit_stop = hb >= cur_stop
            hit_tp = lb <= tp

        if hit_stop:
            r = (cur_stop - entry_price) / risk_dist if side == "long" else (entry_price - cur_stop) / risk_dist
            return k, cur_stop, r
        if hit_tp:
            return k, tp, rr

        atrk = atr[k]
        if not np.isnan(atrk):
            if side == "long":
                mfe = (hb - entry_price) / risk_dist
                if mfe >= TRAIL_TRIGGER_R:
                    cand = hb - TRAIL_ATR_MULT * atrk
                    if cand > cur_stop:
                        cur_stop = cand
            else:
                mfe = (entry_price - lb) / risk_dist
                if mfe >= TRAIL_TRIGGER_R:
                    cand = lb + TRAIL_ATR_MULT * atrk
                    if cand < cur_stop:
                        cur_stop = cand
    return None  # never resolved


def process_symbol(symbol, cfgmap):
    df = load_klines(symbol)
    if df is None or len(df) < 300:
        return [], []
    rsi, atr, ema200, chop = compute_indicators(df)
    close = df["close"].to_numpy()
    high = df["high"].to_numpy()
    low = df["low"].to_numpy()
    open_ = df["open"].to_numpy()
    start = df["start"].to_numpy()
    n = len(df)

    allowed_types = set(cfgmap.keys())
    signals = detect_signals(df, close, high, low, rsi, ema200, allowed_types)

    a0_trades = []
    a1_trades = []
    for sig in signals:
        res = resolve_bos(sig, close, ema200, chop, n)
        if res is None:
            continue
        bos_idx, entry_idx = res
        entry_time = pd.Timestamp(start[entry_idx])
        if entry_time < WINDOW_START or entry_time > WINDOW_END:
            continue
        atr_bos = atr[bos_idx]
        if np.isnan(atr_bos) or atr_bos <= 0:
            continue
        entry_price = open_[entry_idx]
        side = sig["side"]
        code = sig["divergence_code"]

        for arm_name, (rr, atr_mult) in (("A0", cfgmap[code]), ("A1", (10.0, 3.0))):
            risk_dist = atr_bos * atr_mult
            if risk_dist <= 0:
                continue
            out = simulate_trailing_exit(side, entry_price, risk_dist, rr, entry_idx, open_, high, low, atr, n)
            if out is None:
                continue
            exit_idx, exit_price, r_gross = out
            rec = dict(
                symbol=symbol, divergence_code=code, side=side, rr=rr, atr_mult=atr_mult,
                sig_idx=int(sig["sig_idx"]), bos_idx=int(bos_idx), entry_idx=int(entry_idx),
                entry_time=entry_time, exit_idx=int(exit_idx), exit_time=pd.Timestamp(start[exit_idx]),
                entry_price=float(entry_price), exit_price=float(exit_price),
                risk_dist=float(risk_dist), r_gross=float(r_gross),
            )
            if arm_name == "A0":
                a0_trades.append(rec)
            else:
                a1_trades.append(rec)

    return a0_trades, a1_trades


def main():
    live = load_live_symbols()
    symbols = sorted(live.keys())
    print(f"live symbols: {len(symbols)}  total configs: {sum(len(v) for v in live.values())}")

    t0 = time.time()
    all_a0 = []
    all_a1 = []
    for idx, sym in enumerate(symbols):
        a0, a1 = process_symbol(sym, live[sym])
        all_a0.extend(a0)
        all_a1.extend(a1)
        if (idx + 1) % 25 == 0 or idx == len(symbols) - 1:
            el = time.time() - t0
            print(f"[{idx+1}/{len(symbols)}] {sym}  cum A0={len(all_a0)} A1={len(all_a1)}  elapsed={el:.0f}s", flush=True)

    df0 = pd.DataFrame(all_a0)
    df1 = pd.DataFrame(all_a1)
    df0.to_parquet(os.path.join(OUT_DIR, "trades_A0.parquet"))
    df1.to_parquet(os.path.join(OUT_DIR, "trades_A1.parquet"))

    print("\n=== PART 1 SUMMARY ===")
    print(f"A0: {len(df0)} trades, mean gross R = {df0['r_gross'].mean():.4f}")
    print(f"A1: {len(df1)} trades, mean gross R = {df1['r_gross'].mean():.4f}")
    print("Reference to compare: A0 32,913 trades gross +0.2214; A1 32,102 trades gross +0.2631")


if __name__ == "__main__":
    main()
