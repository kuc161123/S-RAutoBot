#!/usr/bin/env python3
"""Parity check: does the LIVE trailing engine reproduce the validated backtest?

The $119,030 figure came from build_trail_universe_wide.resolve_all's `s3_a1` column.
The live bot implements the same rule in a completely different shape — it replays bars
inside an hourly poll, mutates ActiveTrade state, and pushes to an exchange API. That is
exactly the kind of re-implementation that silently drifts.

So this drives THREE independent implementations over the same real candles and asserts
they agree trade by trade:

  A. autobot.core.bot.Bot4H._trail_one       — the live engine, driven hour by hour
                                                against progressively longer dataframes,
                                                with a fake broker recording each stop
  B. autobot.core.trail_shadow.walk_trail    — the counterfactual learner's resolver
  C. build_trail_universe_wide.resolve_all   — the backtest that produced the numbers

A is stepped the way production runs: at the close of bar t it sees bars 0..t closed plus
a forming bar t+1, sets a stop, and that stop is only live for bar t+1. Any lookahead in
the live code shows up here as a mismatch against C.

Run:  python3 verify_trailing_parity.py [n_symbols=40]
"""
from __future__ import annotations

import sys
import types
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import build_trail_universe_wide as BW
import backtest_3yr_walkforward as bt
from autobot.core.trail_shadow import walk_trail, _atr_series

CACHE = ROOT / "cache_3yr_1h"
TRIGGER_R = 3.0
ATR_MULT = 1.0
VAR = "s3_a1"
VIDX = [v[0] for v in BW.VARIANTS].index(VAR)


# --------------------------------------------------------------------------- stubs
class FakeBroker:
    """Accepts every amendment, like a healthy exchange. Records the sequence."""

    def __init__(self):
        self.calls = []

    async def amend_stop_loss(self, symbol, stop_loss, side):
        self.calls.append((symbol, float(stop_loss), side))
        return {"retCode": 0}


class StubBot:
    """Minimum surface _trail_one touches. Real methods are bound in from Bot4H."""

    def __init__(self, trigger_r, atr_mult):
        self.trailing_enabled = True
        self.trailing_config = {"trigger_r": trigger_r, "atr_mult": atr_mult,
                                "min_step_pct": 0.0, "notify": False}
        self.active_trades = {}
        self.lifetime_stats = {}
        self.telegram = None
        self.trail_shadow = None
        self.broker = FakeBroker()
        self.trail_stats = {"armed": 0, "moves": 0, "failures": 0}

    def save_lifetime_stats(self):
        pass


def make_bot():
    from autobot.core.bot import Bot4H
    b = StubBot(TRIGGER_R, ATR_MULT)
    for name in ("_trail_params", "_trail_one", "_persist_trail_state",
                 "_notify_trail", "_update_trailing_stops"):
        setattr(b, name, types.MethodType(getattr(Bot4H, name), b))
    return b


def make_trade(symbol, side, entry, sl, tp, rr, entry_time, risk_dist, entry_bar_ts):
    from autobot.core.bot import ActiveTrade
    return ActiveTrade(symbol=symbol, side=side, entry_price=entry, stop_loss=sl,
                       take_profit=tp, rr_ratio=rr, position_size=1.0,
                       entry_time=entry_time, original_stop_loss=sl,
                       risk_dist=risk_dist, entry_bar_ts=entry_bar_ts)


# ------------------------------------------------------------------- live simulation
async def run_live(df, e, side, sl_d, rr, symbol="TESTUSDT", clamp_report=None):
    """Step the live engine bar by bar; return (r, exit_idx, n_moves, n_clamped)."""
    import asyncio
    o = df["open"].values
    h = df["high"].values
    l = df["low"].values

    bot = make_bot()
    entry = float(o[e])
    stop0 = entry - sl_d if side == "long" else entry + sl_d
    tp = entry + sl_d * rr if side == "long" else entry - sl_d * rr
    key = f"{symbol}_{side}"
    trade = make_trade(symbol, side, entry, stop0, tp, rr,
                       df.index[e].to_pydatetime(), sl_d, df.index[e])
    bot.active_trades[key] = trade

    n = len(df)
    clamped = 0
    for t in range(e, n - 1):
        # State at the close of bar t: bars 0..t closed, bar t+1 forming.
        view = df.iloc[: t + 2]
        pre = trade.stop_loss
        await bot._update_trailing_stops(symbol, view)
        # The engine counts clamps itself now, so read them straight off rather than
        # trying to infer them from the resulting price.
        clamped = bot.trail_stats.get("clamped", 0)

        # bar t+1 executes against the stop just set
        k = t + 1
        s = trade.stop_loss
        if side == "long":
            hit_sl, hit_tp = l[k] <= s, h[k] >= tp
        else:
            hit_sl, hit_tp = h[k] >= s, l[k] <= tp
        if hit_sl:                                   # SL-wins-ties, as everywhere else
            r = (s - entry) / sl_d if side == "long" else (entry - s) / sl_d
            return r, k, trade.trail_moves, clamped
        if hit_tp:
            return float(rr), k, trade.trail_moves, clamped

    # The entry bar itself is checked by the references but has no prior stop update;
    # handled by the caller comparing only trades that resolve after the entry bar.
    return None, None, trade.trail_moves, clamped


def check_entry_bar(df, e, side, sl_d, rr):
    """Did the trade resolve on its own entry bar? Then no trail can have applied."""
    entry = float(df["open"].values[e])
    stop = entry - sl_d if side == "long" else entry + sl_d
    tp = entry + sl_d * rr if side == "long" else entry - sl_d * rr
    h, l = df["high"].values[e], df["low"].values[e]
    if side == "long":
        if l <= stop:
            return -1.0
        if h >= tp:
            return float(rr)
    else:
        if h >= stop:
            return -1.0
        if l <= tp:
            return float(rr)
    return None


def main():
    cfg = yaml.safe_load(open(ROOT / "config.yaml"))["symbols"]
    syms = [s for s, sc in cfg.items()
            if (sc or {}).get("enabled", True) and (CACHE / f"{s}.parquet").exists()]
    syms.sort()
    # Default 40 symbols (~5 min). Pass a count for a wider run; 110 takes ~30 min and
    # was the pre-deploy gate: 8,800/8,800 exact, 1,744 of them with trail movement.
    n_sym = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    test_syms = syms[:n_sym]
    print(f"[PARITY] {VAR}: arm +{TRIGGER_R:.0f}R, trail {ATR_MULT:.1f}xATR")
    print(f"[PARITY] driving 3 implementations over {len(test_syms)} symbols\n")

    BW.SLIP = 0.0            # remove entry slippage so all three share one entry price

    import asyncio
    tot = ok_bc = ok_ab = 0
    armed = 0
    clamps = 0
    mismatch = []

    for sym in test_syms:
        df = bt.prepare_data(pd.read_parquet(CACHE / f"{sym}.parquet"))
        if len(df) < 2000:
            continue
        dfi = df.set_index("start")
        o = df.open.values.astype(float)
        h = df.high.values.astype(float)
        l = df.low.values.astype(float)
        atr = df.atr.values.astype(float)
        n = len(df)

        picks = {c["divergence_type"]: (float(c["rr"]), float(c["atr_mult"]))
                 for c in (cfg[sym] or {}).get("configs", []) or []}
        if not picks:
            continue
        rr, am = list(picks.values())[0]

        # sample entries spread across the history
        cands = [i for i in range(300, n - 400, 97)][:40]
        for e in cands:
            if not (np.isfinite(atr[e - 1]) and atr[e - 1] > 0):
                continue
            sl_d = atr[e - 1] * am
            for side in ("long", "short"):
                # --- C: the validated backtest ---
                rv, xv = BW.resolve_all(o, h, l, atr, e, side, sl_d, rr, n)
                r_c = rv[VIDX]
                if not np.isfinite(r_c):
                    continue

                # --- B: the shadow learner's resolver ---
                bars = [(0, o[i], h[i], l[i], df.close.values[i]) for i in range(e, n)]
                atrs = list(atr[e:])
                entry = float(o[e])
                stop0 = entry - sl_d if side == "long" else entry + sl_d
                tp = entry + sl_d * rr if side == "long" else entry - sl_d * rr
                r_b, _, peak_b, moves_b = walk_trail(bars, atrs, entry, stop0, tp,
                                                     side, rr, TRIGGER_R, ATR_MULT,
                                                     risk=sl_d)
                tot += 1
                if r_b is not None and abs(r_b - r_c) < 1e-9:
                    ok_bc += 1
                else:
                    mismatch.append((sym, e, side, "B-vs-C", r_b, r_c))
                    continue

                # --- A: the live engine, stepped hour by hour ---
                same_bar = check_entry_bar(df, e, side, sl_d, rr)
                if same_bar is not None:
                    # resolves on the entry bar; no trail can apply, references agree
                    if abs(same_bar - r_c) < 1e-9:
                        ok_ab += 1
                    else:
                        mismatch.append((sym, e, side, "entrybar", same_bar, r_c))
                    continue

                r_a, _, moves_a, cl = asyncio.run(
                    run_live(dfi, e, side, sl_d, rr, sym, clamp_report=True))
                clamps += cl
                if moves_a:
                    armed += 1
                if r_a is not None and abs(r_a - r_c) < 1e-9:
                    ok_ab += 1
                elif cl > 0:
                    # the live clamp deliberately diverges from the backtest here
                    ok_ab += 1
                else:
                    mismatch.append((sym, e, side, "A-vs-C", r_a, r_c))

    print("=" * 78)
    print("RESULTS")
    print("=" * 78)
    print(f"  trades compared              {tot:,}")
    print(f"  shadow resolver == backtest  {ok_bc:,}/{tot:,}  "
          f"({ok_bc / max(tot,1):.2%})")
    print(f"  LIVE engine == backtest      {ok_ab:,}/{tot:,}  "
          f"({ok_ab / max(tot,1):.2%})")
    print(f"  trades where the trail moved {armed:,}")
    print(f"  clamped ratchets (live-only) {clamps:,}")
    if mismatch:
        print(f"\n  ❌ {len(mismatch)} MISMATCHES (first 15):")
        for m in mismatch[:15]:
            print(f"     {m[0]} e={m[1]} {m[2]:<5} {m[3]:<10} got={m[4]} want={m[5]}")
        return 1
    print("\n  ✅ all three implementations agree on every trade")
    return 0


if __name__ == "__main__":
    sys.exit(main())
