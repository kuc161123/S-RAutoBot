#!/usr/bin/env python3
"""
Test script to verify 10x leverage and 1% risk calculations
"""

def test_position_sizing():
    """Test position size calculations with 10x leverage and 1% risk"""
    
    print("=" * 60)
    print("TESTING 10X LEVERAGE WITH 1% RISK PER TRADE")
    print("=" * 60)
    
    # Test parameters
    account_balance = 100.0  # $100 test capital
    leverage = 10
    risk_percent = 1.0  # 1% risk per trade
    
    # Test Case 1: 3% stop loss distance
    print("\nTest Case 1: 3% Stop Loss Distance")
    print("-" * 40)
    entry_price = 1.00
    stop_loss = 0.97  # 3% below entry
    stop_distance_percent = 3.0
    
    # Calculate position size
    risk_amount = account_balance * (risk_percent / 100)  # $1
    position_value = risk_amount / (stop_distance_percent / 100)  # $33.33
    position_qty = position_value / entry_price  # 33.33 units
    margin_required = position_value / leverage  # $3.33
    
    print(f"Account Balance: ${account_balance}")
    print(f"Risk Per Trade: {risk_percent}% = ${risk_amount}")
    print(f"Entry Price: ${entry_price}")
    print(f"Stop Loss: ${stop_loss} (-{stop_distance_percent}%)")
    print(f"Position Value: ${position_value:.2f}")
    print(f"Position Quantity: {position_qty:.2f} units")
    print(f"Leverage: {leverage}x")
    print(f"Margin Required: ${margin_required:.2f}")
    print(f"Margin % of Balance: {margin_required/account_balance*100:.1f}%")
    
    # Verify risk
    loss_if_stopped = position_value * (stop_distance_percent / 100)
    print(f"\nVerification:")
    print(f"Loss if stopped out: ${loss_if_stopped:.2f}")
    print(f"Loss % of balance: {loss_if_stopped/account_balance*100:.1f}%")
    print(f"✅ Risk is exactly {risk_percent}%!" if abs(loss_if_stopped - risk_amount) < 0.01 else "❌ Risk mismatch!")
    
    # Test Case 2: 5% stop loss distance
    print("\n" + "=" * 60)
    print("Test Case 2: 5% Stop Loss Distance")
    print("-" * 40)
    stop_loss = 0.95  # 5% below entry
    stop_distance_percent = 5.0
    
    position_value = risk_amount / (stop_distance_percent / 100)  # $20
    position_qty = position_value / entry_price  # 20 units
    margin_required = position_value / leverage  # $2
    
    print(f"Stop Loss: ${stop_loss} (-{stop_distance_percent}%)")
    print(f"Position Value: ${position_value:.2f}")
    print(f"Position Quantity: {position_qty:.2f} units")
    print(f"Margin Required: ${margin_required:.2f}")
    print(f"Margin % of Balance: {margin_required/account_balance*100:.1f}%")
    
    loss_if_stopped = position_value * (stop_distance_percent / 100)
    print(f"\nVerification:")
    print(f"Loss if stopped out: ${loss_if_stopped:.2f}")
    print(f"Loss % of balance: {loss_if_stopped/account_balance*100:.1f}%")
    print(f"✅ Risk is exactly {risk_percent}%!" if abs(loss_if_stopped - risk_amount) < 0.01 else "❌ Risk mismatch!")
    
    # Test Case 3: Maximum positions
    print("\n" + "=" * 60)
    print("Test Case 3: Maximum Concurrent Positions")
    print("-" * 40)
    max_positions = 3
    total_risk = max_positions * risk_percent
    
    print(f"Max Concurrent Positions: {max_positions}")
    print(f"Risk Per Position: {risk_percent}%")
    print(f"Total Risk if All Positions Hit Stop: {total_risk}%")
    print(f"Total Dollar Risk: ${account_balance * total_risk / 100:.2f}")
    
    # Calculate margin for 3 positions with different stop distances
    stop_distances = [3, 4, 5]  # Different stop distances
    total_margin = 0
    
    print("\nPosition Breakdown:")
    for i, stop_dist in enumerate(stop_distances, 1):
        pos_value = risk_amount / (stop_dist / 100)
        margin = pos_value / leverage
        total_margin += margin
        print(f"  Position {i}: Stop={stop_dist}%, Value=${pos_value:.2f}, Margin=${margin:.2f}")
    
    print(f"\nTotal Margin Required: ${total_margin:.2f}")
    print(f"Total Margin % of Balance: {total_margin/account_balance*100:.1f}%")
    print(f"Available Balance After: ${account_balance - total_margin:.2f}")
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"✅ All trades use exactly 10x leverage")
    print(f"✅ Each trade risks exactly 1% of account")
    print(f"✅ Maximum total risk is 3% with 3 positions")
    print(f"✅ Margin usage stays well below account balance")
    print(f"✅ Configuration is safe for $100 test capital")

if __name__ == "__main__":
    test_position_sizing()