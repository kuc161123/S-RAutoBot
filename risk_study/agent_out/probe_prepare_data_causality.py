#!/usr/bin/env python3
"""Structural causality check on prepare_data(): recompute rsi/atr/ema for a
symbol using ONLY data truncated at each of several points, and verify the
value at the last row of the truncated series exactly matches the value at
the same timestamp in the full-series computation. If prepare_data() used
any future information (centered windows, shift(-1), bfill, full-series
stats), these would NOT match.
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd
import numpy as np
import backtest_3yr_walkforward as bt

sym = "BTCUSDT"
raw = pd.read_parquet(ROOT / "cache_3yr_1h" / f"{sym}.parquet").sort_values("start").reset_index(drop=True)
full = bt.prepare_data(raw)

checkpoints = [500, 5000, 15000, 25000, len(raw) - 1]
print(f"{'cutoff_idx':>10}  {'rsi_match':>10}  {'atr_match':>10}  {'ema_match':>10}")
for cp in checkpoints:
    trunc_raw = raw.iloc[: cp + 1].copy()
    trunc = bt.prepare_data(trunc_raw)
    r_full = full.loc[cp, ["rsi", "atr", "ema"]]
    r_trunc = trunc.loc[cp, ["rsi", "atr", "ema"]]
    rsi_ok = np.isclose(r_full["rsi"], r_trunc["rsi"], equal_nan=True)
    atr_ok = np.isclose(r_full["atr"], r_trunc["atr"], equal_nan=True)
    ema_ok = np.isclose(r_full["ema"], r_trunc["ema"], equal_nan=True)
    print(f"{cp:>10}  {str(rsi_ok):>10}  {str(atr_ok):>10}  {str(ema_ok):>10}")

print("\nAll True => prepare_data() is strictly causal (no future info leaks into rsi/atr/ema).")
