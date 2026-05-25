#!/usr/bin/env python3
"""
Incrementally extend the 1h and 5m candle caches to today.

For each cached parquet, reads the last timestamp and fetches only new
candles from Bybit, then appends and saves. Skips files that are already
current within one tick.

Usage:
  python3 update_caches.py --tf 60                       # 1h only
  python3 update_caches.py --tf 5                        # 5m only
  python3 update_caches.py --tf 60 5                     # both
"""
import argparse
import time
import sys
from pathlib import Path

import pandas as pd
import requests

BASE_URL = "https://api.bybit.com"
ROOT = Path(__file__).resolve().parent

CACHES = {
    '60': ROOT / 'cache_3yr_1h',
    '5':  ROOT / 'cache_data' / '5m',
}
INTERVAL_MS = {'60': 60 * 60 * 1000, '5': 5 * 60 * 1000}


def fetch_klines_range(symbol, interval, start_ts_ms, end_ts_ms):
    """Forward-paginated fetch from start_ts_ms to end_ts_ms (Bybit returns DESC)."""
    all_rows = []
    cur_end = end_ts_ms
    while cur_end > start_ts_ms:
        params = {
            'category': 'linear', 'symbol': symbol,
            'interval': interval, 'limit': 1000, 'end': cur_end,
        }
        try:
            r = requests.get(f"{BASE_URL}/v5/market/kline", params=params, timeout=10)
            data = r.json().get('result', {}).get('list', []) or []
        except Exception:
            time.sleep(0.5)
            continue
        if not data:
            break
        all_rows.extend(data)
        cur_end = int(data[-1][0]) - 1
        if len(data) < 1000:
            break
        # Also break if we've reached start
        if cur_end <= start_ts_ms:
            break
        time.sleep(0.03)
    if not all_rows:
        return pd.DataFrame()
    df = pd.DataFrame(all_rows, columns=['start', 'open', 'high', 'low', 'close', 'volume', 'turnover'])
    df['start'] = pd.to_datetime(df['start'].astype('int64'), unit='ms')
    for c in ('open', 'high', 'low', 'close', 'volume', 'turnover'):
        df[c] = df[c].astype(float)
    df = df.sort_values('start').drop_duplicates(subset='start').reset_index(drop=True)
    # Trim strictly to requested range
    cutoff_lo = pd.to_datetime(start_ts_ms, unit='ms')
    df = df[df['start'] >= cutoff_lo].reset_index(drop=True)
    return df


def update_timeframe(interval, target_end_ms):
    cache_dir = CACHES[interval]
    if not cache_dir.is_dir():
        print(f"[{interval}] cache dir missing: {cache_dir}")
        return
    files = sorted(cache_dir.glob('*.parquet'))
    print(f"\n[{interval}] cache_dir={cache_dir} files={len(files)}")
    if not files:
        return

    interval_ms = INTERVAL_MS[interval]
    target_end = pd.to_datetime(target_end_ms, unit='ms')

    updated = 0
    already_current = 0
    new_rows_total = 0
    failed = 0
    t0 = time.time()
    for i, f in enumerate(files):
        try:
            existing = pd.read_parquet(f)
        except Exception:
            failed += 1
            continue
        if existing.empty or 'start' not in existing.columns:
            failed += 1
            continue
        last_ts = existing['start'].max()
        next_start = int((last_ts.value // 1_000_000) + interval_ms)  # ms after the last bar
        if next_start >= target_end_ms:
            already_current += 1
            continue
        try:
            new_df = fetch_klines_range(f.stem, interval, next_start, target_end_ms)
        except Exception:
            failed += 1
            continue
        if new_df.empty:
            already_current += 1
            continue
        # Defensive: drop anything <= last_ts
        new_df = new_df[new_df['start'] > last_ts]
        if new_df.empty:
            already_current += 1
            continue
        combined = pd.concat([existing, new_df], ignore_index=True)
        combined = combined.drop_duplicates(subset='start').sort_values('start').reset_index(drop=True)
        combined.to_parquet(f, index=False)
        updated += 1
        new_rows_total += len(new_df)
        if (i + 1) % 50 == 0 or i == len(files) - 1:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            eta_s = (len(files) - (i + 1)) / rate if rate > 0 else 0
            print(f"  [{interval}] {i+1}/{len(files)}  updated={updated}  current={already_current}  fail={failed}  rows+={new_rows_total}  {elapsed:.0f}s  eta {eta_s:.0f}s")

    elapsed = time.time() - t0
    print(f"[{interval}] DONE in {elapsed:.0f}s | updated={updated} current={already_current} fail={failed} rows+={new_rows_total}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tf', nargs='+', default=['60', '5'], choices=['60', '5'])
    ap.add_argument('--end', default=None, help='Optional end (YYYY-MM-DD); default now')
    args = ap.parse_args()
    if args.end:
        end_ts_ms = int(pd.Timestamp(args.end).value // 1_000_000)
    else:
        end_ts_ms = int(time.time() * 1000)
    print(f"Target end: {pd.to_datetime(end_ts_ms, unit='ms')}")
    for tf in args.tf:
        update_timeframe(tf, end_ts_ms)


if __name__ == '__main__':
    main()
