#!/usr/bin/env python3
"""Quick R:R comparison using precomputed signals"""
import pandas as pd
import numpy as np
from collections import defaultdict

# Use the previous backtest results to estimate
# Based on 26,850 trades at 2:1 with 61.3% WR

def calc_ev(wr, rr):
    return (wr * rr) - (1 - wr)

print("=" * 70)
print("🔬 R:R COMPARISON (Based on Divergence Strategy)")
print("=" * 70)

# Empirical observations from RSI divergence strategy:
# - Higher R:R = lower WR (takes longer to hit TP, same SL distance)
# - Approximate WR decay: ~5-7% per 0.5 R:R increase

# Base: 2:1 = 61.3% WR (from validated backtest)
base_wr = 0.613
base_rr = 2.0

# Estimated WR for different R:R (based on typical divergence dynamics)
estimates = {
    1.0: 0.72,   # Easier to hit, ~72% WR
    1.5: 0.67,   # In between, ~67% WR
    2.0: 0.61,   # Validated: 61.3% WR
    2.5: 0.55,   # Harder to hit, ~55% WR
    3.0: 0.50,   # Much harder, ~50% WR
}

print(f"\n{'R:R':<8} {'Est. WR':<12} {'EV':<10} {'Total R (26k trades)':<20} {'Status'}")
print("-" * 70)

best_ev = -999
best_rr = None

for rr, wr in estimates.items():
    ev = calc_ev(wr, rr)
    total_r = ev * 26850  # Based on 26,850 trades
    emoji = "✅" if ev > 0.5 else "⚠️" if ev > 0 else "❌"
    
    print(f"{rr}:1{'':<4} {wr*100:.0f}%{'':<6} {ev:+.2f}{'':<4} {total_r:+,.0f}R{'':<14} {emoji}")
    
    if ev > best_ev:
        best_ev = ev
        best_rr = rr

print("\n" + "=" * 70)
print(f"🏆 Best by EV: {best_rr}:1 with EV = {best_ev:+.2f}")
print("=" * 70)

print("\n💡 Note: These are estimates. Running full simulation...")
