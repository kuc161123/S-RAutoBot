#!/usr/bin/env python3
"""
COMPARISON: Current 79 Symbols vs Walk-Forward Validated 63 Symbols
====================================================================
Analyzes the overlap and differences between:
1. Your current 79-symbol configuration
2. The 63 walk-forward validated symbols

Shows:
- Which current symbols passed walk-forward validation ✅
- Which current symbols failed/weren't tested ❌
- New discoveries not in your current config 🆕
- Detailed performance metrics for all groups
"""

import pandas as pd
import yaml

print("="*80)
print("PORTFOLIO COMPARISON ANALYSIS")
print("="*80)

# Load current config
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

current_symbols = {}
for symbol, data in config.get('symbols', {}).items():
    if data.get('enabled', False):
        current_symbols[symbol] = data.get('rr', 5.0)

print(f"\n📊 Current Configuration: {len(current_symbols)} symbols")

# Load walk-forward validated symbols
try:
    wf_df = pd.read_csv('500_walkforward_validated.csv')
    wf_symbols = dict(zip(wf_df['symbol'], wf_df['rr']))
    print(f"✅ Walk-Forward Validated: {len(wf_symbols)} symbols")
except FileNotFoundError:
    print("❌ 500_walkforward_validated.csv not found!")
    exit(1)

# Load initial backtest results for additional context
try:
    initial_df = pd.read_csv('500_symbol_validated.csv')
    initial_dict = dict(zip(initial_df['symbol'], initial_df['total_r']))
except:
    initial_df = None
    initial_dict = {}

print("\n" + "="*80)
print("OVERLAP ANALYSIS")
print("="*80)

# Find overlaps
validated_and_current = set(current_symbols.keys()) & set(wf_symbols.keys())
current_not_validated = set(current_symbols.keys()) - set(wf_symbols.keys())
new_discoveries = set(wf_symbols.keys()) - set(current_symbols.keys())

print(f"\n✅ Symbols in BOTH (Current + Validated): {len(validated_and_current)}")
print(f"❌ Current symbols NOT validated: {len(current_not_validated)}")
print(f"🆕 New discoveries (not in current): {len(new_discoveries)}")

# === GROUP 1: Validated AND Current (KEEP THESE!) ===
print("\n" + "="*80)
print(f"✅ GROUP 1: VALIDATED & IN CURRENT CONFIG ({len(validated_and_current)} SYMBOLS)")
print("="*80)
print("These symbols passed walk-forward validation AND are already in your config.")
print("RECOMMENDATION: KEEP ALL OF THESE\n")

if validated_and_current:
    group1_data = []
    for symbol in sorted(validated_and_current):
        wf_data = wf_df[wf_df['symbol'] == symbol].iloc[0]
        group1_data.append({
            'symbol': symbol,
            'current_rr': current_symbols[symbol],
            'validated_rr': wf_symbols[symbol],
            'consistency': wf_data['consistency'],
            'total_r_18mo': wf_data['total_r_all_periods'],
            'avg_r_per_period': wf_data['avg_r_per_period']
        })
    
    group1_df = pd.DataFrame(group1_data).sort_values('total_r_18mo', ascending=False)
    
    print(f"{'Symbol':20} | {'Current R:R':11} | {'Valid R:R':9} | {'Consistency':11} | {'Total R (18mo)':15} | {'Avg R/Period':12}")
    print("-"*80)
    for _, row in group1_df.iterrows():
        rr_match = "✓" if row['current_rr'] == row['validated_rr'] else f"→{row['validated_rr']:.0f}:1"
        print(f"{row['symbol']:20} | {row['current_rr']:5.1f}:1      | {row['validated_rr']:3.0f}:1    | {row['consistency']:11} | {row['total_r_18mo']:+14.1f}R | {row['avg_r_per_period']:+11.1f}R")
    
    total_r_group1 = group1_df['total_r_18mo'].sum()
    avg_r_group1 = group1_df['total_r_18mo'].mean()
    print("-"*80)
    print(f"GROUP 1 TOTAL: {len(group1_df)} symbols | Total R: {total_r_group1:+.1f}R | Avg: {avg_r_group1:+.1f}R per symbol")

# === GROUP 2: Current but NOT Validated (REVIEW THESE!) ===
print("\n" + "="*80)
print(f"❌ GROUP 2: IN CURRENT CONFIG BUT NOT VALIDATED ({len(current_not_validated)} SYMBOLS)")
print("="*80)
print("These symbols are in your current config but did NOT pass walk-forward validation.")
print("RECOMMENDATION: REVIEW CAREFULLY - Consider removing or monitoring closely\n")

if current_not_validated:
    group2_data = []
    for symbol in sorted(current_not_validated):
        # Check if they were in initial backtest
        initial_r = initial_dict.get(symbol, 'N/A')
        group2_data.append({
            'symbol': symbol,
            'current_rr': current_symbols[symbol],
            'initial_6mo_r': initial_r,
            'status': 'Not tested' if symbol not in initial_dict else 'Failed validation'
        })
    
    group2_df = pd.DataFrame(group2_data)
    
    print(f"{'Symbol':20} | {'Current R:R':11} | {'Initial 6mo R':13} | {'Status':20}")
    print("-"*80)
    for _, row in group2_df.iterrows():
        r_str = f"{row['initial_6mo_r']:+.1f}R" if isinstance(row['initial_6mo_r'], (int, float)) else 'N/A'
        print(f"{row['symbol']:20} | {row['current_rr']:5.1f}:1      | {r_str:13} | {row['status']:20}")
    
    print("-"*80)
    print(f"GROUP 2 TOTAL: {len(group2_df)} symbols")
    print("\n⚠️  These symbols may be:")
    print("   - Recently added and not in backtest database")
    print("   - Profitable in recent 6mo but not consistent over 18mo")
    print("   - Worth monitoring but higher risk")

# === GROUP 3: New Discoveries (ADD THESE!) ===
print("\n" + "="*80)
print(f"🆕 GROUP 3: NEW VALIDATED DISCOVERIES ({len(new_discoveries)} SYMBOLS)")
print("="*80)
print("These symbols passed walk-forward validation but are NOT in your current config.")
print("RECOMMENDATION: STRONG CANDIDATES TO ADD\n")

if new_discoveries:
    group3_data = []
    for symbol in new_discoveries:
        wf_data = wf_df[wf_df['symbol'] == symbol].iloc[0]
        group3_data.append({
            'symbol': symbol,
            'validated_rr': wf_symbols[symbol],
            'consistency': wf_data['consistency'],
            'total_r_18mo': wf_data['total_r_all_periods'],
            'avg_r_per_period': wf_data['avg_r_per_period']
        })
    
    group3_df = pd.DataFrame(group3_data).sort_values('total_r_18mo', ascending=False)
    
    print("TOP 20 NEW DISCOVERIES:")
    print(f"{'Symbol':20} | {'R:R':5} | {'Consistency':11} | {'Total R (18mo)':15} | {'Avg R/Period':12}")
    print("-"*80)
    for _, row in group3_df.head(20).iterrows():
        print(f"{row['symbol']:20} | {row['validated_rr']:2.0f}:1 | {row['consistency']:11} | {row['total_r_18mo']:+14.1f}R | {row['avg_r_per_period']:+11.1f}R")
    
    total_r_group3 = group3_df['total_r_18mo'].sum()
    avg_r_group3 = group3_df['total_r_18mo'].mean()
    print("-"*80)
    print(f"GROUP 3 TOTAL: {len(group3_df)} symbols | Total R: {total_r_group3:+.1f}R | Avg: {avg_r_group3:+.1f}R per symbol")

# === PORTFOLIO RECOMMENDATIONS ===
print("\n" + "="*80)
print("PORTFOLIO RECOMMENDATIONS")
print("="*80)

print(f"\n📊 CURRENT PORTFOLIO (79 symbols):")
print(f"   ✅ Validated: {len(validated_and_current)} symbols")
print(f"   ❌ Not Validated: {len(current_not_validated)} symbols")
print(f"   Coverage: {len(validated_and_current)/len(current_symbols)*100:.1f}% validated")

print(f"\n📊 POTENTIAL NEW PORTFOLIO OPTIONS:")

# Option 1: Keep only validated current symbols
print(f"\n   OPTION 1: Conservative (Keep only validated current)")
print(f"   • Symbols: {len(validated_and_current)}")
print(f"   • Expected Performance: Based on GROUP 1 metrics")
print(f"   • Risk: LOW (all symbols validated)")

# Option 2: Validated current + top new discoveries
top_new = min(15, len(new_discoveries))
print(f"\n   OPTION 2: Balanced (Validated current + top {top_new} new)")
print(f"   • Symbols: {len(validated_and_current) + top_new}")
print(f"   • Expected Performance: Higher than current")
print(f"   • Risk: LOW-MEDIUM (all validated, diversified)")

# Option 3: All 63 validated
print(f"\n   OPTION 3: Aggressive (All 63 validated)")
print(f"   • Symbols: 63")
print(f"   • Expected Performance: Optimized for validation")
print(f"   • Risk: LOW (all walk-forward validated)")

# Option 4: Current + all new validated
print(f"\n   OPTION 4: Maximum Diversification (Current 79 + All new validated)")
print(f"   • Symbols: {len(current_symbols) + len(new_discoveries)}")
print(f"   • Expected Performance: Maximum coverage")
print(f"   • Risk: MEDIUM (includes unvalidated current symbols)")

# === SYMBOLS TO REVIEW ===
if current_not_validated:
    print("\n" + "="*80)
    print("⚠️  SYMBOLS REQUIRING REVIEW")
    print("="*80)
    print("Current symbols that did NOT pass walk-forward validation:\n")
    for symbol in sorted(current_not_validated):
        print(f"   • {symbol} (R:R {current_symbols[symbol]}:1)")
    print("\nConsider:")
    print("   1. Monitoring these closely in live trading")
    print("   2. Reducing position size for these symbols")
    print("   3. Removing if they continue to underperform")

# === R:R COMPARISON ===
if validated_and_current:
    print("\n" + "="*80)
    print("R:R OPTIMIZATION OPPORTUNITIES")
    print("="*80)
    
    rr_changes = group1_df[group1_df['current_rr'] != group1_df['validated_rr']]
    if len(rr_changes) > 0:
        print(f"\n{len(rr_changes)} symbols have different R:R in validation vs current config:\n")
        for _, row in rr_changes.iterrows():
            print(f"   • {row['symbol']:20} Current: {row['current_rr']:.1f}:1 → Validated: {row['validated_rr']:.0f}:1")
        print("\nConsider updating to validated R:R ratios for optimal performance.")
    else:
        print("\n✅ All validated symbols already have optimal R:R ratios!")

print("\n" + "="*80)
print("📁 ANALYSIS COMPLETE")
print("="*80)
print("\nNext Steps:")
print("1. Review Group 2 symbols (current but not validated)")
print("2. Consider adding top Group 3 symbols (new discoveries)")
print("3. Update R:R ratios where validation differs from current")
print("4. Choose a portfolio option based on your risk tolerance")
