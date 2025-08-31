#!/usr/bin/env python3
"""
Test safer position sizing with all safety checks
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Set required env vars
os.environ['RISK_PER_TRADE'] = '0.005'  # 0.5%
os.environ['MAX_POSITIONS'] = '10'
os.environ['LEVERAGE'] = '10'

from crypto_trading_bot.trading.position_manager import PositionManager

print("="*70)
print("SAFER POSITION SIZING TEST")
print("="*70)

# Test parameters
balance = 250  # $250 balance
risk_per_trade = 0.005  # 0.5% risk
leverage = 10  # 10x leverage

print(f"\n📊 Test Parameters:")
print(f"   Balance: ${balance}")
print(f"   Risk per trade: {risk_per_trade*100}% = ${balance * risk_per_trade:.2f}")
print(f"   Leverage: {leverage}x")
print(f"   Max margin: ${balance * 0.95:.2f} (95% of balance)")
print(f"   Max position value: ${balance * 2:.2f} (2x balance)")

# Initialize position manager
pm = PositionManager(
    max_positions=10,
    risk_per_trade=risk_per_trade
)

# Test cases including edge cases
test_cases = [
    {
        'name': 'Normal: BTC with 1% stop',
        'entry_price': 100000,
        'stop_loss': 99000,  # 1% below
        'expected_risk': 1.25
    },
    {
        'name': 'Tiny stop: BTC with 0.05% stop',
        'entry_price': 100000,
        'stop_loss': 99950,  # 0.05% below - should trigger minimum
        'expected_risk': 1.25
    },
    {
        'name': 'Large stop: ETH with 5% stop',
        'entry_price': 4000,
        'stop_loss': 3800,  # 5% below
        'expected_risk': 1.25
    },
    {
        'name': 'Micro price: SHIB with 2% stop',
        'entry_price': 0.00001,
        'stop_loss': 0.0000098,  # 2% below
        'expected_risk': 1.25
    },
    {
        'name': 'Extreme: SOL with 0.01% stop',
        'entry_price': 200,
        'stop_loss': 199.98,  # 0.01% below - should trigger minimum
        'expected_risk': 1.25
    }
]

print(f"\n🧪 Testing Position Sizes with Safety Checks:\n")

for test in test_cases:
    print(f"{'='*60}")
    print(f"Test: {test['name']}")
    print(f"   Entry: ${test['entry_price']}")
    print(f"   Stop Loss: ${test['stop_loss']}")
    
    # Calculate stop distance
    stop_distance = abs(test['entry_price'] - test['stop_loss']) / test['entry_price']
    print(f"   Stop Distance: {stop_distance*100:.3f}%")
    
    # Calculate position size
    position_size = pm.calculate_position_size(
        balance=balance,
        entry_price=test['entry_price'],
        stop_loss=test['stop_loss'],
        leverage=leverage
    )
    
    if position_size > 0:
        # Calculate values
        position_value = position_size * test['entry_price']
        margin_required = position_value / leverage
        potential_loss = position_size * abs(test['entry_price'] - test['stop_loss'])
        
        print(f"\n   ✅ Results:")
        print(f"   Position Size: {position_size:.8f} coins")
        print(f"   Position Value: ${position_value:.2f}")
        print(f"   Margin Required: ${margin_required:.2f}")
        print(f"   Potential Loss: ${potential_loss:.2f}")
        print(f"   Expected Risk: ${test['expected_risk']:.2f}")
        
        # Check safety
        safety_checks = []
        
        if margin_required <= balance * 0.95:
            safety_checks.append("✅ Margin within limits")
        else:
            safety_checks.append("❌ Margin too high")
            
        if position_value <= balance * 2:
            safety_checks.append("✅ Position value within cap")
        else:
            safety_checks.append("❌ Position value too high")
            
        if potential_loss <= test['expected_risk'] * 1.5:
            safety_checks.append("✅ Risk within tolerance")
        else:
            safety_checks.append("❌ Risk too high")
            
        print(f"\n   Safety Checks:")
        for check in safety_checks:
            print(f"   {check}")
    else:
        print(f"   ❌ Position rejected by safety checks")
    
    print()

print("="*70)
print("SAFETY FEATURES")
print("="*70)
print("""
The improved position sizing now includes:

1. ✅ Minimum stop distance (0.1%) to prevent huge positions
2. ✅ Maximum margin usage (95% of balance)
3. ✅ Maximum position value (2x balance)
4. ✅ Risk validation (never > 150% of target risk)
5. ✅ Final risk check before returning

This ensures:
- Positions are never too large
- Risk is always controlled
- Margin usage is safe
- No accidental account blow-ups
""")