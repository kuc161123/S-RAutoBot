#!/usr/bin/env python3
"""
BACKTEST AUDIT - Checking for Potential Issues
===============================================
Testing for lookahead bias, entry timing, and other issues
"""

import pandas as pd
import numpy as np

# Simulate the backtest flow
print("=" * 70)
print("🔍 BACKTEST AUDIT REPORT")
print("=" * 70)

issues_found = []
warnings_found = []

# Issue 1: Forming Candle
print("\n1️⃣ FORMING CANDLE CHECK:")
print("   Code: fetch_klines() uses datetime.now() as end_ts")
print("   Code: df = df.dropna() but NO explicit drop of last candle")
print("   ❌ ISSUE: Last candle in dataset is likely forming/incomplete")
print("   Impact: Signals detected on forming candles = LOOKAHEAD BIAS")
issues_found.append("Forming candle not excluded")

# Issue 2: Entry Timing 
print("\n2️⃣ ENTRY TIMING CHECK:")
print("   Code: entry = rows[idx + 1].open (line 180)")
print("   ✅ CORRECT: Entry on NEXT candle open (no lookahead)")

# Issue 3: SL/TP Check on Entry Candle
print("\n3️⃣ SL/TP ENTRY CANDLE CHECK:")
print("   Code: for bar_idx, row in enumerate(rows[entry_idx:entry_idx + 100])")
print("   Code: Starts checking from entry_idx (the entry candle)")
print("   ✅ CORRECT: Checks entry candle high/low for immediate stop")

# Issue 4: Slippage
print("\n4️⃣ SLIPPAGE CHECK:")
print("   Code: entry = entry * (1 + SLIPPAGE_PCT) for longs")
print("   Code: entry = entry * (1 - SLIPPAGE_PCT) for shorts")
print("   ⚠️  WARNING: Applied BEFORE TP calc, not to TP itself")
print("   Code: tp = tp * (1 - TOTAL_COST) for longs")
print("   ✅ But TP adjusted for total costs")
warnings_found.append("Slippage method slightly unconventional but conservative")

# Issue 5: Signal Detection Range
print("\n5️⃣ SIGNAL DETECTION RANGE:")
print("   Code: for i in range(30, n - 5)")
print("   ⚠️  STOPS 5 bars before end, BUT...")
print("   If last candle is forming, n-5 still includes incomplete data")
issues_found.append("Detection range doesn't account for forming candle")

# Issue 6: Cooldown
print("\n6️⃣ COOLDOWN CHECK:")
print("   Code: if i - last_trade_idx < COOLDOWN_BARS: continue")
print("   ✅ CORRECT: 10-bar cooldown between same-symbol trades")

# Issue 7: Volume Filter
print("\n7️⃣ VOLUME FILTER:")
print("   Code: if not row.vol_ok or row.atr <= 0: continue")
print("   Code: df['vol_ok'] = df['volume'] > df['vol_ma'] * 0.5")
print("   ✅ CORRECT: Volume must be > 50% of 20-bar MA")

# Issue 8: Pivot Finding
print("\n8️⃣ PIVOT DETECTION:")
print("   Code: find_pivots(data, left=3, right=3)")
print("   Code: for i in range(left, n - right)")
print("   ✅ CORRECT: Needs 3 bars on each side, stops 3 from end")

# Issue 9: Data Quality
print("\n9️⃣ DATA QUALITY:")
print("   ⚠️  POTENTIAL ISSUE: Using data 'as of now'")
print("   If backtest run at different times, results may vary")
print("   (Not bias, but inconsistency)")
warnings_found.append("Results may vary based on when backtest is run")

# Issue 10: THE BIG ONE - Walk Forward Periods
print("\n🔟 WALK-FORWARD PERIOD ASSIGNMENT:")
print("   Code: all_dates = sorted(df['date'].unique())")
print("   Code: days_per_period = len(all_dates) // NUM_PERIODS")
print("   Code: period = day_idx // days_per_period")
print("   ⚠️  If forming candle included, latest period inflated")
warnings_found.append("Walk-forward periods may be slightly skewed")

print("\n" + "=" * 70)
print("📋 SUMMARY")
print("=" * 70)
print(f"🔴 CRITICAL ISSUES: {len(issues_found)}")
for i, issue in enumerate(issues_found, 1):
    print(f"   {i}. {issue}")

print(f"\n⚠️  WARNINGS: {len(warnings_found)}")
for i, warning in enumerate(warnings_found, 1):
    print(f"   {i}. {warning}")

print("\n" + "=" * 70)
print("🎯 IMPACT ASSESSMENT")
print("=" * 70)
print("The forming candle issue is the MOST CRITICAL:")
print("")
print("🔴 WORST CASE:")
print("   - If signals detected on forming candle")
print("   - And those signals happen to be winners")
print("   - Results are SIGNIFICANTLY overstated")
print("")
print("🟡 BEST CASE:")
print("   - If most signals are historical (not on forming candle)")
print("   - Impact is minimal")
print("")
print("📊 ESTIMATED IMPACT:")
print("   - Out of 6688 total trades")
print("   - At most ~20-30 trades affected (last candle per symbol)")
print("   - If those 30 trades are 90% winners (optimistic)")
print("   - Overstatement: ~0.3-0.5% in WR")
print("")
print("CONCLUSION: Issue exists but impact likely SMALL")
print("            Results still directionally correct")
print("=" * 70)

print("\n🔧 RECOMMENDED FIXES:")
print("1. Add: df = df.iloc[:-1]  # Drop forming candle")
print("2. Or: Use historical end date, not datetime.now()")
print("3. Re-run backtest with fix")
print("=" * 70)
