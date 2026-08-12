#!/usr/bin/env python3
"""Same as probe_dedup_impact.py but using each symbol's REAL live (rr, atr_mult)
picks from config.yaml (not a fixed 3.0/1.5), across a larger sample, to get a
realistic R-impact estimate of the backtest-vs-live signal-detection mismatch.
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd
import numpy as np
import yaml
import backtest_3yr_walkforward as bt
from autobot.core.divergence_detector import detect_divergences, prepare_dataframe as live_prep

CACHE = ROOT / "cache_3yr_1h"
cfg = yaml.safe_load(open(ROOT / "config.yaml"))["symbols"]

jobs = []
for s, sc in cfg.items():
    if not (sc or {}).get("enabled", True):
        continue
    picks = {c["divergence_type"]: (float(c["rr"]), float(c["atr_mult"]))
              for c in (sc or {}).get("configs", []) or []}
    if picks and (CACHE / f"{s}.parquet").exists():
        jobs.append((s, picks))
jobs.sort()

# Use a deterministic stride sample of ~60 symbols to keep runtime bounded
SAMPLE = jobs[::5][:60]
print(f"[dedup-impact] sampling {len(SAMPLE)} of {len(jobs)} live symbols with their real (rr, atr) picks")

tot_bt = tot_live_aligned = tot_only_bt = tot_only_live = 0
only_bt_rs = []
shared_rs = []

for sym, picks in SAMPLE:
    raw = pd.read_parquet(CACHE / f"{sym}.parquet").sort_values("start").reset_index(drop=True)
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
        conf_idx, side, typ = key
        if typ not in picks:
            continue
        rr, am = picks[typ]
        res = bt.execute_trade_1h(bt_map[key], bt_df, rr_ratio=rr, atr_mult=am)
        if res:
            only_bt_rs.append(res["r"])
    for key in shared:
        conf_idx, side, typ = key
        if typ not in picks:
            continue
        rr, am = picks[typ]
        res = bt.execute_trade_1h(bt_map[key], bt_df, rr_ratio=rr, atr_mult=am)
        if res:
            shared_rs.append(res["r"])

print(f"\n=== TOTALS across {len(SAMPLE)} symbols (real per-symbol RR/ATR configs) ===")
print(f"  backtest signal-set size:        {tot_bt}")
print(f"  live trend-aligned signal-set:   {tot_live_aligned}")
print(f"  only-in-backtest (leaked):       {tot_only_bt}  ({tot_only_bt/tot_bt:.1%} of backtest universe)")
print(f"  only-in-live (missed by bt):     {tot_only_live}")

if only_bt_rs:
    print(f"\n  R using each symbol's REAL live picks, fee-adjusted, executed-only:")
    print(f"    only-in-backtest (leaked) signals: n={len(only_bt_rs):5} avg_R={np.mean(only_bt_rs):+.4f}")
if shared_rs:
    print(f"    shared (bt & live agree) signals:  n={len(shared_rs):5} avg_R={np.mean(shared_rs):+.4f}")

if only_bt_rs and shared_rs:
    n1, n2 = len(only_bt_rs), len(shared_rs)
    combined_avg = (sum(only_bt_rs) + sum(shared_rs)) / (n1 + n2)
    print(f"\n  backtest-universe blended avg R (as currently built): {combined_avg:+.4f}")
    print(f"  what live would ACTUALLY see (shared only):           {np.mean(shared_rs):+.4f}")
    print(f"  bias introduced by the extra {n1/(n1+n2):.1%} of leaked signals: "
          f"{combined_avg - np.mean(shared_rs):+.4f} R/trade")
