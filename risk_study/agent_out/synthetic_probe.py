#!/usr/bin/env python3
"""Minimal synthetic reproduction of the close-batch ordering bug in
run_simulation's STEP A (backtest_production_correct.py:378-439), isolated
from CHOP/regime cascade effects (scenario='no_filters', flat risk).
"""
import sys
sys.path.insert(0, '/Users/lualakol/AutoTrading Bot/risk_study/agent_out')
import pandas as pd
from engine_probe import run_simulation

rows = [
    # Trade A opens first (insertion order #1), closes LAST chronologically (T5) - a big loser.
    dict(entry_time='2024-01-01 00:00', exit_time='2024-01-06 00:00',
         entry_price=100.0, sl_price=50.0, r_result=-1.0, side='long', symbol='SOLUSDT'),
    # Trade B opens second (insertion order #2), closes FIRST chronologically (T2) - a winner.
    dict(entry_time='2024-01-01 01:00', exit_time='2024-01-02 00:00',
         entry_price=100.0, sl_price=50.0, r_result=+1.0, side='long', symbol='XRPUSDT'),
    # Trade C's entry is what finally triggers Step A to close BOTH A and B in one batch,
    # since neither A's nor B's exit_time had a prior row's entry_time land on/after it.
    dict(entry_time='2024-01-07 00:00', exit_time='2024-01-07 05:00',
         entry_price=10.0, sl_price=9.0, r_result=0.0, side='long', symbol='ADAUSDT'),
]
df = pd.DataFrame(rows)
df['entry_time'] = pd.to_datetime(df['entry_time'])
df['exit_time'] = pd.to_datetime(df['exit_time'])

kwargs = dict(scenario='no_filters', base_risk=0.5, starting_balance=1000.0)

r_buggy = run_simulation(df, {}, sort_close_batch=False, **kwargs)
r_fixed = run_simulation(df, {}, sort_close_batch=True, **kwargs)

print("Trade close order (as processed by Step A, insertion/AS-IS):")
for t in r_buggy['entered_trades']:
    print(f"  {t['symbol']:<10} entry={t['entry_time']}  exit={t['exit_time']}  "
          f"r={t['r_result']:+.1f}  pnl=${t['pnl']:+.2f}  balance_after=${t['balance_after']:.2f}")

print(f"\nAS-IS   (Step A batch processed in dict-insertion / entry order):")
print(f"  max_dd_pct = {r_buggy['max_dd_pct']:.2f}%   final balance = ${r_buggy['wallet_balance']:.2f}")

print(f"\nCORRECTED (Step A batch processed in true exit_time order):")
print(f"  max_dd_pct = {r_fixed['max_dd_pct']:.2f}%   final balance = ${r_fixed['wallet_balance']:.2f}")

print(f"\nSame final balance both ways (order-independent): "
      f"{'YES' if abs(r_buggy['wallet_balance']-r_fixed['wallet_balance'])<0.01 else 'NO'}")
print(f"max_dd_pct distortion from wrong ordering: "
      f"{r_buggy['max_dd_pct'] - r_fixed['max_dd_pct']:+.2f} percentage points "
      f"({(r_buggy['max_dd_pct']/r_fixed['max_dd_pct']-1)*100:+.0f}% relative)")
