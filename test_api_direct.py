#!/usr/bin/env python3
"""
Test API key directly to diagnose the issue
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Set required env vars
os.environ.setdefault('RISK_PER_TRADE', '0.005')
os.environ.setdefault('MAX_POSITIONS', '10')
os.environ.setdefault('LEVERAGE', '10')

from pybit.unified_trading import HTTP
from crypto_trading_bot.config import settings
import time
import hmac
import hashlib

print("="*70)
print("API KEY DIAGNOSTIC")
print("="*70)

# Show API key info (masked)
api_key = settings.bybit_api_key
api_secret = settings.bybit_api_secret

print(f"\n1. API Key Configuration:")
print(f"   API Key: {api_key[:10]}...{api_key[-4:]}")
print(f"   Secret: {'*' * 20}")
print(f"   Testnet: {settings.bybit_testnet}")

# Test 1: Try testnet
print(f"\n2. Testing TESTNET connection...")
try:
    testnet_client = HTTP(
        testnet=True,
        api_key=api_key,
        api_secret=api_secret
    )
    
    response = testnet_client.get_wallet_balance(accountType="UNIFIED")
    if response['retCode'] == 0:
        print(f"   ✅ TESTNET API works!")
    else:
        print(f"   ❌ TESTNET failed: {response['retMsg']}")
except Exception as e:
    print(f"   ❌ TESTNET error: {e}")

# Test 2: Try mainnet
print(f"\n3. Testing MAINNET connection...")
try:
    mainnet_client = HTTP(
        testnet=False,
        api_key=api_key,
        api_secret=api_secret
    )
    
    response = mainnet_client.get_wallet_balance(accountType="UNIFIED")
    if response['retCode'] == 0:
        print(f"   ✅ MAINNET API works!")
        # Try to get balance
        if 'result' in response and 'list' in response['result']:
            for account in response['result']['list']:
                if 'totalEquity' in account:
                    print(f"   Balance: ${float(account['totalEquity']):.2f}")
    else:
        print(f"   ❌ MAINNET failed: {response['retMsg']}")
        print(f"   Error Code: {response['retCode']}")
except Exception as e:
    print(f"   ❌ MAINNET error: {e}")

# Test 3: Check server time sync
print(f"\n4. Checking time synchronization...")
try:
    # Get server time
    server_response = mainnet_client.get_server_time()
    if server_response['retCode'] == 0:
        server_time = int(server_response['result']['timeSecond'])
        local_time = int(time.time())
        diff = abs(server_time - local_time)
        
        print(f"   Server time: {server_time}")
        print(f"   Local time:  {local_time}")
        print(f"   Difference:  {diff} seconds")
        
        if diff > 5:
            print(f"   ⚠️ TIME SYNC ISSUE! Difference > 5 seconds")
            print(f"   This can cause 'expired' errors")
        else:
            print(f"   ✅ Time sync OK")
except Exception as e:
    print(f"   Error checking time: {e}")

# Test 4: Try a simple public endpoint
print(f"\n5. Testing public endpoint (no auth)...")
try:
    response = mainnet_client.get_tickers(category="linear", symbol="BTCUSDT")
    if response['retCode'] == 0:
        price = float(response['result']['list'][0]['lastPrice'])
        print(f"   ✅ Public API works - BTC: ${price:.2f}")
    else:
        print(f"   ❌ Public API failed: {response['retMsg']}")
except Exception as e:
    print(f"   Error: {e}")

# Test 5: Check API key permissions
print(f"\n6. Checking API key info...")
try:
    response = mainnet_client.get_api_key_information()
    if response['retCode'] == 0:
        print(f"   ✅ API Key Info Retrieved:")
        result = response['result']
        print(f"   - Type: {result.get('type', 'N/A')}")
        print(f"   - Permissions: {result.get('permissions', {})}")
        print(f"   - IPs: {result.get('ips', [])}")
        print(f"   - Status: {result.get('status', 'N/A')}")
        print(f"   - Expired: {result.get('expiredAt', 'N/A')}")
    else:
        print(f"   ❌ Failed to get API info: {response['retMsg']}")
except Exception as e:
    print(f"   Error: {e}")

print("\n" + "="*70)
print("DIAGNOSIS SUMMARY")
print("="*70)

print("""
Common issues and solutions:

1. If TESTNET works but MAINNET doesn't:
   → Your API key is for testnet only
   → Create a new MAINNET API key

2. If you get 'expired' errors:
   → Check time sync (should be < 5 seconds difference)
   → API key might have an expiry date set
   → IP whitelist might be blocking Railway's IPs

3. If you get 'permission denied':
   → Enable "Derivatives" trading permission
   → Enable "Orders" permission
   → Enable "Positions" permission

4. If nothing works:
   → Double-check API key and secret are copied correctly
   → No extra spaces or newlines in the credentials
   → Make sure using API v5 (Unified Trading Account)
""")