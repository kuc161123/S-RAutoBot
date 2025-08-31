#!/usr/bin/env python3
"""
Diagnose intermittent API failures - why it works sometimes but not others
"""
import os
import sys
import time
import json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Set required env vars
os.environ.setdefault('RISK_PER_TRADE', '0.005')
os.environ.setdefault('MAX_POSITIONS', '10')
os.environ.setdefault('LEVERAGE', '10')

from pybit.unified_trading import HTTP
from crypto_trading_bot.config import settings
from datetime import datetime

print("="*70)
print("INTERMITTENT API FAILURE DIAGNOSTIC")
print("="*70)
print(f"Time: {datetime.now()}")

api_key = settings.bybit_api_key
api_secret = settings.bybit_api_secret

# The error suggests it might be environment-specific
print(f"\n1. Environment Check:")
print(f"   BYBIT_TESTNET env var: {os.environ.get('BYBIT_TESTNET', 'not set')}")
print(f"   Config testnet value: {settings.bybit_testnet}")
print(f"   API Key: {api_key[:10]}...{api_key[-4:]}")

# Test multiple times to see if it's intermittent
print(f"\n2. Testing API 5 times with different configurations...")

configs = [
    ("Mainnet", False),
    ("Testnet", True),
]

for config_name, use_testnet in configs:
    print(f"\n   Testing {config_name} (testnet={use_testnet}):")
    
    success_count = 0
    fail_count = 0
    errors = []
    
    for i in range(3):
        try:
            client = HTTP(
                testnet=use_testnet,
                api_key=api_key.strip(),  # Remove any whitespace
                api_secret=api_secret.strip()
            )
            
            # Try a simple authenticated call
            response = client.get_wallet_balance(accountType="UNIFIED")
            
            if response['retCode'] == 0:
                success_count += 1
                print(f"   ✅ Attempt {i+1}: SUCCESS")
                
                # If successful, show some account info
                if i == 0 and 'result' in response:
                    result = response['result']
                    if 'list' in result and len(result['list']) > 0:
                        account = result['list'][0]
                        if 'totalEquity' in account:
                            print(f"      Balance: ${float(account['totalEquity']):.2f}")
            else:
                fail_count += 1
                error_msg = f"{response['retCode']}: {response['retMsg']}"
                errors.append(error_msg)
                print(f"   ❌ Attempt {i+1}: {error_msg}")
                
        except Exception as e:
            fail_count += 1
            error_msg = str(e)[:100]
            errors.append(error_msg)
            print(f"   ❌ Attempt {i+1}: {error_msg}")
        
        # Small delay between attempts
        if i < 2:
            time.sleep(1)
    
    print(f"   Results: {success_count} success, {fail_count} failed")
    if errors:
        print(f"   Unique errors: {list(set(errors))}")

# Check if it's a network-specific issue
print(f"\n3. Checking if it's network-specific...")

# Force mainnet with explicit clearing of testnet
client_mainnet = HTTP(
    testnet=False,  # Explicitly mainnet
    api_key=api_key.strip(),
    api_secret=api_secret.strip()
)

# Try to get positions (this would show if trades were made)
try:
    response = client_mainnet.get_positions(category="linear", settleCoin="USDT")
    if response['retCode'] == 0:
        print(f"   ✅ Can get positions on mainnet")
        positions = response['result']['list']
        if positions:
            print(f"   Found {len(positions)} open positions:")
            for pos in positions[:3]:  # Show first 3
                if float(pos.get('size', 0)) > 0:
                    print(f"   - {pos['symbol']}: {pos['side']} {pos['size']}")
    else:
        print(f"   ❌ Cannot get positions: {response['retMsg']}")
except Exception as e:
    print(f"   ❌ Error getting positions: {str(e)[:100]}")

# Check the actual testnet setting being used
print(f"\n4. Configuration Issues:")

# Check if Railway might be setting BYBIT_TESTNET differently
testnet_values = ['true', 'True', 'TRUE', '1', 'yes', 'false', 'False', 'FALSE', '0', 'no']
for val in testnet_values:
    os.environ['BYBIT_TESTNET'] = val
    # Reload settings
    from crypto_trading_bot.config import Settings
    test_settings = Settings()
    if val in ['true', 'True', 'TRUE', '1', 'yes']:
        expected = True
    else:
        expected = False
    
    if test_settings.bybit_testnet != expected:
        print(f"   ⚠️ BYBIT_TESTNET='{val}' results in testnet={test_settings.bybit_testnet}")

print(f"\n5. Possible Issues:")

issues = []

# Issue 1: Wrong network
if success_count == 0 and fail_count > 0:
    if "33004" in str(errors):
        issues.append("API key might be for different network (testnet vs mainnet)")
        issues.append("Railway might have BYBIT_TESTNET=true set")

# Issue 2: IP restrictions
if "permission" in str(errors).lower() or "ip" in str(errors).lower():
    issues.append("IP whitelist might be blocking Railway's servers")

# Issue 3: Rate limiting
if success_count > 0 and fail_count > 0:
    issues.append("Intermittent failures suggest rate limiting or network issues")

for issue in issues:
    print(f"   - {issue}")

print("\n" + "="*70)
print("SOLUTION")
print("="*70)

print("""
Since trades were working this morning, the issue is likely:

1. **BYBIT_TESTNET Configuration**
   - Railway might have BYBIT_TESTNET=true
   - Change it to BYBIT_TESTNET=false
   - Or remove BYBIT_TESTNET entirely (defaults to false)

2. **IP Restrictions**
   - Your local IP might work but Railway's doesn't
   - Check API settings on Bybit for IP whitelist
   - Consider removing IP restrictions temporarily

3. **Mixed Network Usage**
   - Bot might be trying testnet when you have mainnet funds
   - Ensure all references use mainnet

4. **Rate Limiting**
   - Too many failed attempts might trigger temporary blocks
   - Wait a few minutes and try again

IMMEDIATE FIX:
In Railway, set: BYBIT_TESTNET=false (or remove it entirely)
""")