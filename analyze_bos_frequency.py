#!/usr/bin/env python3
"""
BOS TRADE FREQUENCY ANALYSIS - 28 Validated Symbols
====================================================
Calculates expected BOS confirmations (executed trades) for the
28 symbols that are both validated and in current config.
"""

import pandas as pd
import yaml

print("="*80)
print("BOS TRADE FREQUENCY ANALYSIS - 28 VALIDATED SYMBOLS")
print("="*80)

# Load current config
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

current_symbols = set()
for symbol, data in config.get('symbols', {}).items():
    if data.get('enabled', False):
        current_symbols.add(symbol)

# Load walk-forward validated
wf_df = pd.read_csv('500_walkforward_validated.csv')
wf_symbols = set(wf_df['symbol'].values)

# Find overlap (Group 1)
group1_symbols = current_symbols & wf_symbols
group1_df = wf_df[wf_df['symbol'].isin(group1_symbols)].copy()

print(f"\n📊 Analyzing {len(group1_df)} validated symbols from your current config\n")

# Load initial backtest for detailed trade counts
initial_df = pd.read_csv('500_symbol_validated.csv')
initial_dict = dict(zip(initial_df['symbol'], initial_df['trades']))

# Add trade count from initial backtest (6 month period)
group1_df['trades_6mo'] = group1_df['symbol'].map(initial_dict)

# Calculate metrics
group1_df['avg_trades_per_month'] = group1_df['trades_6mo'] / 6

# Sort by profitability
group1_df = group1_df.sort_values('total_r_all_periods', ascending=False)

print("="*80)
print("EXPECTED BOS CONFIRMATIONS (TRADES) PER SYMBOL")
print("="*80)
print(f"\n{'Symbol':20} | {'Trades (6mo)':13} | {'Trades/Month':12} | {'Trades/Week':11} | {'Total R':10}")
print("-"*80)

total_trades_6mo = 0
for _, row in group1_df.iterrows():
    trades_6mo = row['trades_6mo'] if pd.notna(row['trades_6mo']) else 0
    trades_month = row['avg_trades_per_month'] if pd.notna(row['avg_trades_per_month']) else 0
    trades_week = trades_month / 4.33 if trades_month > 0 else 0
    
    total_trades_6mo += trades_6mo
    
    print(f"{row['symbol']:20} | {trades_6mo:13.0f} | {trades_month:12.1f} | {trades_week:11.1f} | {row['total_r_all_periods']:+9.1f}R")

print("-"*80)

# Calculate portfolio totals
total_trades_per_month = total_trades_6mo / 6
total_trades_per_week = total_trades_per_month / 4.33
total_trades_per_day = total_trades_per_month / 30

print(f"\n{'PORTFOLIO TOTAL':20} | {total_trades_6mo:13.0f} | {total_trades_per_month:12.1f} | {total_trades_per_week:11.1f} |")
print("-"*80)

print("\n" + "="*80)
print("EXPECTED TRADE FREQUENCY SUMMARY")
print("="*80)

print(f"\n📊 PORTFOLIO METRICS (28 Symbols):")
print(f"├─ Total Trades (6 months): {total_trades_6mo:.0f}")
print(f"├─ Trades per Month: {total_trades_per_month:.1f}")
print(f"├─ Trades per Week: {total_trades_per_week:.1f}")
print(f"└─ Trades per Day: {total_trades_per_day:.1f}")

print(f"\n📈 PER SYMBOL AVERAGES:")
avg_per_symbol_6mo = total_trades_6mo / len(group1_df)
avg_per_symbol_month = total_trades_per_month / len(group1_df)
print(f"├─ Avg Trades per Symbol (6mo): {avg_per_symbol_6mo:.1f}")
print(f"└─ Avg Trades per Symbol per Month: {avg_per_symbol_month:.1f}")

print(f"\n🎯 BOS CONFIRMATION EXPECTATIONS:")
print(f"├─ Total Divergences Detected: Higher (many won't reach BOS)")
print(f"├─ BOS Confirmations (Executed): {total_trades_6mo:.0f} in 6 months")
print(f"├─ BOS Confirmation Rate: ~{total_trades_per_day:.1f} trades/day")
print(f"└─ Max Wait per Signal: 6 candles (6 hours max)")

# Breakdown by consistency
print("\n" + "="*80)
print("BREAKDOWN BY CONSISTENCY")
print("="*80)

consistency_groups = group1_df.groupby('consistency').agg({
    'trades_6mo': 'sum',
    'symbol': 'count'
}).reset_index()
consistency_groups.columns = ['consistency', 'total_trades', 'symbol_count']
consistency_groups['trades_per_month'] = consistency_groups['total_trades'] / 6

print(f"\n{'Consistency':15} | {'Symbols':8} | {'Total Trades (6mo)':20} | {'Trades/Month':12}")
print("-"*80)
for _, row in consistency_groups.iterrows():
    print(f"{row['consistency']:15} | {row['symbol_count']:8.0f} | {row['total_trades']:20.0f} | {row['trades_per_month']:12.1f}")

# Most active symbols
print("\n" + "="*80)
print("TOP 10 MOST ACTIVE SYMBOLS (Most BOS Confirmations)")
print("="*80)

top_active = group1_df.nlargest(10, 'trades_6mo')
print(f"\n{'Symbol':20} | {'Trades (6mo)':13} | {'Trades/Month':12} | {'Consistency':11}")
print("-"*80)
for _, row in top_active.iterrows():
    trades_month = row['avg_trades_per_month'] if pd.notna(row['avg_trades_per_month']) else 0
    print(f"{row['symbol']:20} | {row['trades_6mo']:13.0f} | {trades_month:12.1f} | {row['consistency']:11}")

print("\n" + "="*80)
print("PRACTICAL EXPECTATIONS")
print("="*80)

print(f"""
With your 28 validated symbols:

📅 DAILY EXPECTATIONS:
   • ~{total_trades_per_day:.1f} BOS confirmations (executed trades) per day
   • Each symbol trades ~{avg_per_symbol_month / 30:.2f} times per day on average
   • Most active symbols: {top_active.iloc[0]['symbol']} (~{top_active.iloc[0]['avg_trades_per_month'] / 30:.1f}/day)

📅 WEEKLY EXPECTATIONS:
   • ~{total_trades_per_week:.0f} BOS confirmations per week
   • Spread across 28 symbols = plenty of opportunities

📅 MONTHLY EXPECTATIONS:
   • ~{total_trades_per_month:.0f} BOS confirmations per month
   • ~{avg_per_symbol_month:.1f} trades per symbol per month

⚡ DIVERGENCE vs BOS:
   • You'll detect MORE divergences than this
   • Only ~50-70% will reach BOS within 6 candles (6 hours)
   • Expect ~{total_trades_per_day * 1.5:.1f}-{total_trades_per_day * 2:.1f} divergence detections per day
   • But only ~{total_trades_per_day:.1f} will execute (BOS confirmed)

🎯 TRADING PACE:
   • With 1H timeframe, new candles every hour
   • Bot scans all 28 symbols every hour
   • Executes when BOS confirms
   • Average: {total_trades_per_day:.1f} positions opened per day
   • With max concurrent: 10 positions, turnover is healthy
""")

print("="*80)
print("ANALYSIS COMPLETE")
print("="*80)
