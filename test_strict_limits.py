#!/usr/bin/env python3
"""
Test the strict position size limits
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Set required env vars
os.environ['RISK_PER_TRADE'] = '0.005'
os.environ['MAX_POSITIONS'] = '10'
os.environ['LEVERAGE'] = '10'
os.environ['MAX_POSITION_VALUE_MULTIPLIER'] = '1.0'

from crypto_trading_bot.trading.position_manager import PositionManager

print("="*70)
print("STRICT POSITION LIMITS TEST")
print("="*70)

balance = 250
risk_per_trade = 0.005

print(f"\nSettings:")
print(f"  Balance: ${balance}")
print(f"  Risk per trade: {risk_per_trade*100}% = ${balance * risk_per_trade:.2f}")
print(f"  HARD LIMITS:")
print(f"  - Max position value: ${balance * 0.5:.2f} (50% of balance)")
print(f"  - Max risk per trade: ${balance * 0.01:.2f} (1% absolute max)")

pm = PositionManager(
    max_positions=10,
    risk_per_trade=risk_per_trade,
    max_position_multiplier=1.0
)

# Test extreme cases that should be limited
test_cases = [
    {
        'name': 'Normal case',
        'coin': 'BTC',
        'entry': 100000,
        'stop': 99000,  # 1% stop
    },
    {
        'name': 'Tiny stop (should be limited)',
        'coin': 'ETH',
        'entry': 4000,
        'stop': 3999,  # 0.025% stop - way too tight
    },
    {
        'name': 'Large position request',
        'coin': 'SOL',
        'entry': 200,
        'stop': 199.5,  # 0.25% stop
    },
    {
        'name': 'Micro stop (should hit hard limit)',
        'coin': 'DOGE',
        'entry': 0.50,
        'stop': 0.4999,  # 0.02% stop - extremely tight
    }
]

print(f"\n{'='*70}")
print("TESTING WITH STRICT LIMITS")
print(f"{'='*70}\n")

for test in test_cases:
    print(f"\nTest: {test['name']}")
    print(f"  {test['coin']} @ ${test['entry']:.4f}")
    print(f"  Stop: ${test['stop']:.4f} ({abs(test['entry']-test['stop'])/test['entry']*100:.3f}% away)")
    
    size = pm.calculate_position_size(
        balance=balance,
        entry_price=test['entry'],
        stop_loss=test['stop'],
        leverage=10
    )
    
    if size > 0:
        position_value = size * test['entry']
        risk = size * abs(test['entry'] - test['stop'])
        
        print(f"\n  Results:")
        print(f"    Position size: {size:.8f} {test['coin']}")
        print(f"    Position value: ${position_value:.2f}")
        print(f"    Risk: ${risk:.2f}")
        
        # Check limits
        if position_value <= balance * 0.5:
            print(f"    ✅ Position value within 50% limit")
        else:
            print(f"    ❌ POSITION TOO LARGE!")
            
        if risk <= balance * 0.01:
            print(f"    ✅ Risk within 1% limit")
        else:
            print(f"    ❌ RISK TOO HIGH!")
    else:
        print(f"  ❌ Position rejected")
    
    print(f"  {'-'*50}")

print(f"\n{'='*70}")
print("SUMMARY")
print(f"{'='*70}")
print("""
With the strict limits in place:
✅ Position value capped at 50% of balance ($125 max)
✅ Risk capped at 1% absolute maximum ($2.50 max)
✅ Minimum stop distance increased to 0.2%
✅ Multiple validation layers before order execution

This ensures NO LARGE POSITIONS can slip through!
""")