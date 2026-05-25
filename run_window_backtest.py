#!/usr/bin/env python3
"""
Filter regime_backtest_all_trades.csv to a date window and run the same
production-correct simulation as backtest_production_correct.py.

Usage:
  python3 run_window_backtest.py --start 2025-11-01 --end 2026-05-25
"""
import argparse
import sys
from pathlib import Path
from collections import defaultdict

import pandas as pd
import numpy as np

# Re-use the simulation primitives from backtest_production_correct.
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import backtest_production_correct as bpc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--start', default='2025-11-01')
    ap.add_argument('--end', default='2026-05-25')
    ap.add_argument('--starting-balance', type=float, default=None,
                    help='Override starting balance (default 850 from bpc)')
    ap.add_argument('--csv', default=None,
                    help='Override trades CSV path (default: regime_backtest_all_trades.csv)')
    args = ap.parse_args()

    if args.starting_balance is not None:
        bpc.STARTING_BALANCE = args.starting_balance
    if args.csv:
        bpc.TRADES_CSV = Path(args.csv)

    start_ts = pd.Timestamp(args.start)
    end_ts = pd.Timestamp(args.end) + pd.Timedelta(days=1)  # inclusive end day

    print("=" * 70)
    print(f"  WINDOWED BACKTEST  {args.start} -> {args.end}")
    print(f"  Starting balance: ${bpc.STARTING_BALANCE:,.0f}")
    print("=" * 70)

    df = pd.read_csv(bpc.TRADES_CSV, parse_dates=['entry_time', 'exit_time'])
    full_min, full_max = df['entry_time'].min(), df['entry_time'].max()
    print(f"\n  CSV total: {len(df):,} trades  {full_min} -> {full_max}")

    df = df[(df['entry_time'] >= start_ts) & (df['entry_time'] < end_ts)]
    df = df.sort_values('entry_time').reset_index(drop=True)
    if df.empty:
        print("\n  No trades in requested window. Aborting.")
        return

    print(f"  Windowed: {len(df):,} trades  {df['entry_time'].min()} -> {df['entry_time'].max()}")
    print(f"  Symbols:  {df['symbol'].nunique()}")
    total_trades = len(df)

    chop_map = bpc.load_chop_data(df['symbol'].unique().tolist())

    print("\n  Running PRODUCTION scenario...")
    r_prod = bpc.run_simulation(df, chop_map, scenario='production')
    print("  Running NO REGIME scenario...")
    r_noreg = bpc.run_simulation(df, chop_map, scenario='no_regime')
    print("  Running NO FILTERS scenario...")
    r_nofilt = bpc.run_simulation(df, chop_map, scenario='no_filters')

    all_results = [r_prod, r_noreg, r_nofilt]
    bpc.print_head_to_head(all_results)
    for r in all_results:
        bpc.print_results(r, total_trades)
    for r in all_results:
        bpc.sanity_check(r, total_trades)

    print(f"\n{'=' * 70}")
    print("  REGIME DISTRIBUTION (PRODUCTION)")
    print(f"{'=' * 70}")
    regime_counts = defaultdict(int)
    for t in r_prod['entered_trades']:
        regime_counts[t['regime']] += 1
    for regime in ['favorable', 'cautious', 'adverse', 'critical']:
        cnt = regime_counts.get(regime, 0)
        pct = cnt / len(r_prod['entered_trades']) * 100 if r_prod['entered_trades'] else 0
        print(f"  {regime:<12} {cnt:>6} trades ({pct:>5.1f}%)")

    print(f"\n{'=' * 70}")
    print("  DONE")
    print(f"{'=' * 70}")


if __name__ == '__main__':
    main()
