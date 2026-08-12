#!/usr/bin/env python3
"""Quantify: how much of the backtest trade universe consists of signals the
live bot's ingestion pipeline would never generate (because live's dedup
consumes a pivot pair on a non-trend-aligned bar before EMA-gating it out,
while the backtest only searches a direction when already trend-aligned)?

For each symbol: compute backtest signal set, live-aligned signal set,
report overlap and the R-impact of the "only-in-backtest" (leaked) subset
at a representative (atr_mult, rr) using the SAME execute_trade_1h the
walk-forward and risk_study universes use.
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd
import numpy as np
import backtest_3yr_walkforward as bt
from autobot.core.divergence_detector import detect_divergences, prepare_dataframe as live_prep

SAMPLE = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "DOGEUSDT", "1000PEPEUSDT", "ARBUSDT", "SUIUSDT",
          "XRPUSDT", "AVAXUSDT", "LINKUSDT"]
CACHE = ROOT / "cache_3yr_1h"

tot_bt = tot_live_aligned = tot_only_bt = tot_only_live = 0
only_bt_rs = []
shared_rs = []

for sym in SAMPLE:
    f = CACHE / f"{sym}.parquet"
    if not f.exists():
        continue
    raw = pd.read_parquet(f).sort_values("start").reset_index(drop=True)
    if len(raw) < 2000:
        continue
    bt_df = bt.prepare_data(raw)
    bt_sigs = bt.detect_signals(bt_df)
    bt_map = {(s["conf_idx"], s["side"], s["type"]): s for s in bt_sigs}
    bt_set = set(bt_map.keys())

    live_raw = raw.set_index("start")
    live_df = live_prep(live_raw)
    live_sigs = detect_divergences(live_df, sym, allowed_types=None, lookback_bars=50)
    idx_of = {t: i for i, t in enumerate(live_df.index)}
    live_aligned_set = {(idx_of[s.timestamp], s.side, s.divergence_code)
                         for s in live_sigs if s.daily_trend_aligned}

    only_bt = bt_set - live_aligned_set
    only_live = live_aligned_set - bt_set
    shared = bt_set & live_aligned_set

    tot_bt += len(bt_set)
    tot_live_aligned += len(live_aligned_set)
    tot_only_bt += len(only_bt)
    tot_only_live += len(only_live)

    for key in only_bt:
        res = bt.execute_trade_1h(bt_map[key], bt_df, rr_ratio=3.0, atr_mult=1.5)
        if res:
            only_bt_rs.append(res["r"])
    for key in shared:
        res = bt.execute_trade_1h(bt_map[key], bt_df, rr_ratio=3.0, atr_mult=1.5)
        if res:
            shared_rs.append(res["r"])

    print(f"{sym:16} bt={len(bt_set):5} live_aligned={len(live_aligned_set):5} "
          f"only_bt={len(only_bt):4} ({len(only_bt)/max(1,len(bt_set)):5.1%})  only_live={len(only_live):3}")

print("\n=== TOTALS across sample ===")
print(f"  backtest signal-set size:        {tot_bt}")
print(f"  live trend-aligned signal-set:   {tot_live_aligned}")
print(f"  only-in-backtest (leaked):       {tot_only_bt}  ({tot_only_bt/tot_bt:.1%} of backtest universe)")
print(f"  only-in-live (missed by bt):     {tot_only_live}")

if only_bt_rs:
    print(f"\n  R @ (atr=1.5, rr=3.0), fee-adjusted, executed only:")
    print(f"    only-in-backtest (leaked) signals: n={len(only_bt_rs):5} avg_R={np.mean(only_bt_rs):+.4f}")
if shared_rs:
    print(f"    shared (bt & live agree) signals:  n={len(shared_rs):5} avg_R={np.mean(shared_rs):+.4f}")
