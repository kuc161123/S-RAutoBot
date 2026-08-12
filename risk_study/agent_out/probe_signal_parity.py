#!/usr/bin/env python3
"""Compare backtest_3yr_walkforward.detect_signals() against the LIVE
autobot/core/divergence_detector.detect_divergences() on real cached data.

Goal: confirm the backtest reproduces exactly the signals the live bot would
detect (same conf_idx / side / type), or quantify any divergence.
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd
import numpy as np
import backtest_3yr_walkforward as bt
from autobot.core.divergence_detector import (
    detect_divergences, prepare_dataframe as live_prepare_dataframe,
)

SAMPLE = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "DOGEUSDT", "1000PEPEUSDT", "ARBUSDT", "SUIUSDT"]
CACHE = ROOT / "cache_3yr_1h"

for sym in SAMPLE:
    f = CACHE / f"{sym}.parquet"
    if not f.exists():
        continue
    raw = pd.read_parquet(f).sort_values("start").reset_index(drop=True)

    # --- backtest path ---
    bt_df = bt.prepare_data(raw)
    bt_sigs = bt.detect_signals(bt_df)
    bt_set = {(s["conf_idx"], s["side"], s["type"]) for s in bt_sigs}

    # --- live path ---
    live_raw = raw.set_index("start")
    live_df = live_prepare_dataframe(live_raw)
    live_sigs = detect_divergences(live_df, sym, allowed_types=None, lookback_bars=50)
    # live signal timestamps -> map back to positional idx via live_df.index
    idx_of = {t: i for i, t in enumerate(live_df.index)}
    live_all_set = {(idx_of[s.timestamp], s.side, s.divergence_code) for s in live_sigs}
    live_aligned_set = {(idx_of[s.timestamp], s.side, s.divergence_code)
                         for s in live_sigs if s.daily_trend_aligned}

    only_bt = bt_set - live_aligned_set
    only_live = live_aligned_set - bt_set

    print(f"\n=== {sym} ===  rows={len(raw)}")
    print(f"  backtest signals:               {len(bt_set)}")
    print(f"  live signals (all, incl. non-trend-aligned): {len(live_all_set)}")
    print(f"  live signals (trend-aligned only, i.e. what ingests): {len(live_aligned_set)}")
    print(f"  in backtest but NOT in live-aligned: {len(only_bt)}")
    print(f"  in live-aligned but NOT in backtest: {len(only_live)}")
    if only_bt:
        print(f"    sample only_bt: {list(only_bt)[:5]}")
    if only_live:
        print(f"    sample only_live: {list(only_live)[:5]}")
