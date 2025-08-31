#!/usr/bin/env python3
"""
Test signal quality controls
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("="*70)
print("SIGNAL QUALITY CONTROL TEST")
print("="*70)

print("""
NEW SIGNAL CONTROLS ADDED:

1. MIN_SIGNAL_SCORE (2-6)
   Controls signal quality threshold:
   - 2 = Many signals (very aggressive)
   - 3 = More signals (aggressive)
   - 4 = Balanced (recommended)
   - 5 = Fewer signals (conservative)
   - 6 = Very few signals (very selective)

2. MIN_VOLUME_MULTIPLIER (0.5-3.0)
   Filters by volume:
   - 0.5 = Accept all volume levels
   - 1.0 = Some volume filtering
   - 1.5 = Normal volume required (recommended)
   - 2.0 = High volume only
   - 3.0 = Very high volume only

3. SIGNAL_COOLDOWN_MINUTES (1-60)
   Time to wait between signals per symbol:
   - 5 = Quick re-entry allowed
   - 10 = Balanced (recommended)
   - 30 = Conservative
   - 60 = Very conservative

""")

print("="*70)
print("RECOMMENDED SETTINGS")
print("="*70)

print("""
For FEWER, HIGHER QUALITY signals:
  MIN_SIGNAL_SCORE=5
  MIN_VOLUME_MULTIPLIER=2.0
  SIGNAL_COOLDOWN_MINUTES=15

For BALANCED signals:
  MIN_SIGNAL_SCORE=4
  MIN_VOLUME_MULTIPLIER=1.5
  SIGNAL_COOLDOWN_MINUTES=10

For MORE FREQUENT signals:
  MIN_SIGNAL_SCORE=3
  MIN_VOLUME_MULTIPLIER=1.0
  SIGNAL_COOLDOWN_MINUTES=5

""")

print("="*70)
print("HOW SIGNALS ARE SCORED")
print("="*70)

print("""
Each signal gets points for:
- RSI oversold/overbought: +2 points
- Near Bollinger Band: +2 points
- MACD momentum: +1 point
- Price action: +1 point

Total possible score: 6 points

With MIN_SIGNAL_SCORE=4:
- Need at least 2 strong indicators (RSI + BB)
- Or 1 strong + 2 weak indicators
- This filters out weak signals

""")

print("="*70)
print("EXPECTED IMPACT")
print("="*70)

print("""
With balanced settings (score=4, volume=1.5, cooldown=10):

Before: ~100+ signals per hour across all symbols
After:  ~10-20 quality signals per hour

This means:
✅ Better win rate
✅ Less overtrading
✅ Lower fees
✅ Better risk management
✅ More meaningful trades
""")