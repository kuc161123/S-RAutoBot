#!/usr/bin/env python3
"""Probe cache_3yr_1h/ parquet files for gaps, dupes, timezone consistency."""
import pandas as pd
import numpy as np
from pathlib import Path

CACHE = Path(__file__).resolve().parent.parent.parent / "cache_3yr_1h"

SAMPLE = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "DOGEUSDT", "XRPUSDT",
          "1000PEPEUSDT", "ARBUSDT", "OPUSDT", "SUIUSDT", "AVAXUSDT",
          "CAMPUSDT", "RDNTUSDT", "TRUUSDT", "GODSUSDT", "SCUSDT"]

print(f"{'symbol':<16}{'rows':>8}{'dupe_ts':>9}{'gaps':>7}{'max_gap_h':>11}{'tz':>8}  start .. end")
for s in SAMPLE:
    f = CACHE / f"{s}.parquet"
    if not f.exists():
        print(f"{s:<16} FILE MISSING")
        continue
    df = pd.read_parquet(f)
    if df.empty:
        print(f"{s:<16} EMPTY")
        continue
    ts = df["start"]
    tz = str(ts.dtype)
    dupes = ts.duplicated().sum()
    ts_sorted = ts.sort_values()
    diffs = ts_sorted.diff().dropna()
    expected = pd.Timedelta(hours=1)
    gap_mask = diffs != expected
    n_gaps = gap_mask.sum()
    max_gap_h = (diffs.max().total_seconds() / 3600) if len(diffs) else 0
    is_monotonic = ts.is_monotonic_increasing
    print(f"{s:<16}{len(df):>8}{dupes:>9}{n_gaps:>7}{max_gap_h:>11.1f}{tz:>8}  "
          f"{ts.min()} .. {ts.max()}  monotonic_in_file={is_monotonic}")
