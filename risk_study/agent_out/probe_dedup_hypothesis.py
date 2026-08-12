#!/usr/bin/env python3
"""Verify: backtest's extra signals (vs live-aligned) come from live's dedup
consuming a pivot pair on a non-trend-aligned bar, before price crosses EMA."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd
import numpy as np
import backtest_3yr_walkforward as bt
from autobot.core.divergence_detector import detect_divergences, prepare_dataframe as live_prep

sym = "BTCUSDT"
raw = pd.read_parquet(ROOT / "cache_3yr_1h" / f"{sym}.parquet").sort_values("start").reset_index(drop=True)

bt_df = bt.prepare_data(raw)
bt_sigs = bt.detect_signals(bt_df)
bt_set = {(s["conf_idx"], s["side"], s["type"]) for s in bt_sigs}

live_raw = raw.set_index("start")
live_df = live_prep(live_raw)
live_sigs = detect_divergences(live_df, sym, allowed_types=None, lookback_bars=50)
idx_of = {t: i for i, t in enumerate(live_df.index)}

live_aligned_set = {(idx_of[s.timestamp], s.side, s.divergence_code) for s in live_sigs if s.daily_trend_aligned}
only_bt = bt_set - live_aligned_set

# For each only_bt signal, find ALL live signals (aligned or not) sharing the same
# (side-category, divergence_code) whose pivot pair could plausibly be the same,
# by matching on pivot_timestamp within +/- a few bars and divergence side family.
live_by_key = {}
for s in live_sigs:
    key = (idx_of[s.pivot_timestamp], s.side)
    live_by_key.setdefault(key, []).append(s)

verified = 0
checked = 0
examples = []
for (conf_idx, side, typ) in list(only_bt)[:30]:
    checked += 1
    # find the matching backtest signal to get its pivot idx
    match = next(s for s in bt_sigs if s["conf_idx"] == conf_idx and s["side"] == side and s["type"] == typ)
    # backtest doesn't store pivot idx directly in the dict; recompute via same logic isn't
    # trivial here, so instead just check: was there an EARLIER live signal (any bar) on the
    # same divergence_code + side whose daily_trend_aligned is False, with divergence_idx < conf_idx,
    # within the 10-bar pivot-freshness window, i.e. a prior "wasted" dedup consumption?
    candidates = [s for s in live_sigs if s.side == side and s.divergence_code == typ
                  and not s.daily_trend_aligned and idx_of[s.timestamp] < conf_idx
                  and conf_idx - idx_of[s.timestamp] <= 10]
    if candidates:
        verified += 1
        examples.append((conf_idx, side, typ, [idx_of[c.timestamp] for c in candidates]))

print(f"{sym}: checked {checked} only-in-backtest signals")
print(f"  verified as preceded by a non-trend-aligned live signal of the SAME type/side within 10 bars: {verified}/{checked}")
for e in examples[:10]:
    print("   ", e)
