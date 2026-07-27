"""Shared execution engine — identical fills, costs and timing for every candidate.

This is deliberately the ONLY place fills are modelled. If each strategy brought its own
execution assumptions (as the ~369 root scripts do) then a cross-strategy comparison would
mostly measure differences in optimism, not differences in edge.

Semantics are the live bot's, post the 2026-07-25 BOS timing fix:
  - a signal is confirmed on the last CLOSED bar (conf_idx)
  - entry at conf_idx+1's OPEN, with slippage against you
  - stop distance comes from the signal; target = sl_dist * rr
  - SL WINS TIES when a bar touches both (conservative, matches the shadow logger)
  - per-side fee + slippage, expressed in R via fee_r = round_trip * entry / sl_dist
  - trades unresolved at the end of data are DROPPED, never scored as flat

The cost model is the reason low-timeframe strategies fail in this repo: fee_r has
sl_distance in the denominator, so tight stops pay proportionally enormous fees. Keeping
it here means every candidate is penalised on the same footing.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .contract import Signal

FEE_PER_SIDE = 0.0006
SLIP_PER_SIDE = 0.0003
ROUND_TRIP = 2 * (FEE_PER_SIDE + SLIP_PER_SIDE)   # 0.0018


def replay(df: pd.DataFrame, signals: list[Signal], symbol: str,
           cost_mult: float = 1.0, max_hold_bars: int | None = None) -> list[dict]:
    """Execute signals against one symbol's bars. Returns trade dicts.

    cost_mult scales fees+slippage for the mandatory 1x/2x/3x sensitivity sweep.
    max_hold_bars optionally forces a time-based exit (used only where a strategy
    pre-registers one; None = run to stop or target).
    """
    o = df["open"].to_numpy(dtype=float)
    h = df["high"].to_numpy(dtype=float)
    l = df["low"].to_numpy(dtype=float)
    ts = df["start"].to_numpy() if "start" in df.columns else df.index.to_numpy()
    n = len(o)
    slip = SLIP_PER_SIDE * cost_mult
    rt = ROUND_TRIP * cost_mult

    out = []
    for s in signals:
        e = s.conf_idx + 1
        if e >= n or s.sl_dist <= 0 or not np.isfinite(s.sl_dist):
            continue
        raw = o[e]
        if not np.isfinite(raw) or raw <= 0:
            continue
        # slippage always against you
        entry = raw * (1 + slip) if s.side == "long" else raw * (1 - slip)
        if s.side == "long":
            sl, tp = entry - s.sl_dist, entry + s.sl_dist * s.rr
        else:
            sl, tp = entry + s.sl_dist, entry - s.sl_dist * s.rr
        fee_r = rt * entry / s.sl_dist

        stop_at = n if max_hold_bars is None else min(n, e + max_hold_bars + 1)
        r = None
        xi = None
        for k in range(e, stop_at):
            if s.side == "long":
                hit_sl, hit_tp = l[k] <= sl, h[k] >= tp
            else:
                hit_sl, hit_tp = h[k] >= sl, l[k] <= tp
            if hit_sl or hit_tp:
                r = -1.0 if hit_sl else s.rr        # SL wins ties
                xi = k
                break
        if r is None:
            if max_hold_bars is None or stop_at >= n:
                continue                            # unresolved -> dropped
            # pre-registered time exit: mark to market at the close of the last bar
            xi = stop_at - 1
            close = df["close"].to_numpy(dtype=float)[xi]
            move = (close - entry) if s.side == "long" else (entry - close)
            r = move / s.sl_dist

        out.append({
            "symbol": symbol, "side": s.side,
            "entry_time": ts[e], "exit_time": ts[xi],
            "entry_price": raw,                     # RAW — run_simulation adds its own costs
            "sl_price": (raw - s.sl_dist) if s.side == "long" else (raw + s.sl_dist),
            "r_result": r,                          # RAW (+rr / -1) for the dollar engine
            "r_net": r - fee_r,                     # net of costs, for R-space stats
            "rr": s.rr, "sl_dist": s.sl_dist,
            "conf_idx": s.conf_idx, "entry_idx": e, "exit_idx": xi,
            "bars_held": xi - e,
            **{f"m_{k}": v for k, v in (s.meta or {}).items()},
        })
    return out


def placebo_signals(df: pd.DataFrame, signals: list[Signal], seed: int,
                    atr_mult: float, multiple: int = 2,
                    atr_col: str = "atr") -> list[Signal]:
    """Random-entry control: same symbol, side mix and SIZING RULE — random timing.

    This is the bar every strategy must clear. A strategy that cannot beat its own placebo
    has no timing information, and its apparent performance is just the stop/target
    geometry interacting with drift. The incumbent clears it by +0.0377 R/trade (t=+4.66).

    The stop is sized from the ATR **at the random entry bar** (atr[pe] * atr_mult), not
    transplanted from the signal bar. That distinction matters and was caught by
    validate_harness.py: reusing the signal-bar distance at an unrelated bar breaks the
    stop's relationship to local volatility AND distorts fee_r (which has sl_dist in its
    denominator), making the placebo far too weak (-0.1585 vs the correct -0.0549) and so
    inflating every strategy's apparent edge over it.
    """
    if not signals:
        return []
    rng = np.random.default_rng(seed)
    n = len(df)
    lo, hi = 250, n - 2                             # leave indicator warmup + room to enter
    if hi <= lo or atr_col not in df.columns:
        return []
    atr = df[atr_col].to_numpy(dtype=float)
    sides = [s.side for s in signals]
    rrs = [s.rr for s in signals]

    out = []
    for _ in range(multiple):
        for j in range(len(signals)):
            i = int(rng.integers(lo, hi))
            a = atr[i]
            if not np.isfinite(a) or a <= 0:
                continue
            out.append(Signal(conf_idx=i, side=sides[j],
                              sl_dist=float(a * atr_mult), rr=float(rrs[j]),
                              meta={"placebo": True}))
    return out
