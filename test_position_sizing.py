#!/usr/bin/env python3
"""
Test position sizing calculations with the fix
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
print("POSITION SIZING TEST")
print("="*70)

# Test parameters
balance = 250  # $250 balance
risk_per_trade = 0.005  # 0.5% risk
leverage = 10  # 10x leverage

print(f"\n📊 Test Parameters:")
print(f"   Balance: ${balance}")
print(f"   Risk per trade: {risk_per_trade*100}%")
print(f"   Leverage: {leverage}x")

# Initialize position manager with correct parameters
pm = PositionManager(
    max_positions=10,
    risk_per_trade=risk_per_trade
)

# Test cases
test_cases = [
    {
        'name': 'BTC with 1% stop loss',
        'entry_price': 100000,
        'stop_loss': 99000,  # 1% below
    },
    {
        'name': 'ETH with 2% stop loss',
        'entry_price': 4000,
        'stop_loss': 3920,  # 2% below
    },
    {
        'name': 'SOL with 0.5% stop loss',
        'entry_price': 200,
        'stop_loss': 199,  # 0.5% below
    },
    {
        'name': 'DOGE with 3% stop loss',
        'entry_price': 0.50,
        'stop_loss': 0.485,  # 3% below
    }
]

print(f"\n🧪 Testing Position Sizes:\n")

for test in test_cases:
    print(f"Test: {test['name']}")
    print(f"   Entry: ${test['entry_price']}")
    print(f"   Stop Loss: ${test['stop_loss']}")
    
    # Calculate stop distance
    stop_distance = abs(test['entry_price'] - test['stop_loss']) / test['entry_price']
    print(f"   Stop Distance: {stop_distance*100:.2f}%")
    
    # Calculate position size
    position_size = pm.calculate_position_size(
        balance=balance,
        entry_price=test['entry_price'],
        stop_loss=test['stop_loss'],
        leverage=leverage
    )
    
    # Calculate values
    risk_amount = balance * risk_per_trade
    position_value = position_size * test['entry_price']
    margin_required = position_value / leverage
    potential_loss = position_size * abs(test['entry_price'] - test['stop_loss'])
    
    print(f"   ✅ Position Size: {position_size:.6f} coins")
    print(f"   Position Value: ${position_value:.2f}")
    print(f"   Margin Required: ${margin_required:.2f}")
    print(f"   Risk Amount: ${risk_amount:.2f}")
    print(f"   Potential Loss: ${potential_loss:.2f}")
    
    # Verify risk is correct
    if abs(potential_loss - risk_amount) < 0.01:
        print(f"   ✅ Risk calculation CORRECT!")
    else:
        print(f"   ❌ Risk calculation ERROR! Expected ${risk_amount:.2f}, got ${potential_loss:.2f}")
    
    print()

print("="*70)
print("EXPECTED BEHAVIOR")
print("="*70)
print("""
With RISK_PER_TRADE=0.005 (0.5%) and $250 balance:
- Risk per trade should be $1.25
- Regardless of leverage, you should only lose $1.25 if stop loss hits
- Higher leverage allows larger positions with same risk
- Position value = Risk / Stop Distance
- Margin required = Position Value / Leverage

Example:
- If BTC is $100,000 with 1% stop loss
- Risk $1.25 on 1% move = $125 position value
- With 10x leverage, margin required = $12.50
- Position size = $125 / $100,000 = 0.00125 BTC
""")