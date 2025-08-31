#!/usr/bin/env python3
"""
Determine if API key is for testnet or mainnet
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

print("="*70)
print("API KEY NETWORK DETECTION")
print("="*70)

api_key = settings.bybit_api_key
api_secret = settings.bybit_api_secret

print(f"\nAPI Key (partial): {api_key[:10]}...{api_key[-4:]}")

# Remove any whitespace that might have been copied
api_key = api_key.strip()
api_secret = api_secret.strip()

results = []

# Test 1: Try as TESTNET key
print(f"\n1. Testing as TESTNET API key...")
try:
    client = HTTP(
        testnet=True,  # Force testnet
        api_key=api_key,
        api_secret=api_secret
    )
    
    # Try to get account info
    response = client.get_wallet_balance(accountType="UNIFIED")
    
    if response['retCode'] == 0:
        print(f"   ✅ SUCCESS! This is a TESTNET API key")
        results.append("TESTNET")
        
        # Show balance
        if 'result' in response and 'list' in response['result']:
            for account in response['result']['list']:
                if 'totalEquity' in account:
                    equity = float(account.get('totalEquity', 0))
                    if equity > 0:
                        print(f"   Testnet Balance: ${equity:.2f}")
    elif response['retCode'] == 10002:
        print(f"   ❌ Invalid API key for testnet")
    elif response['retCode'] == 10003:
        print(f"   ❌ Invalid signature - check API secret")
    elif response['retCode'] == 33004:
        print(f"   ❌ API key expired (according to testnet)")
    else:
        print(f"   ❌ Failed: {response['retMsg']} (Code: {response['retCode']})")
        
except Exception as e:
    error_str = str(e)
    if "401" in error_str:
        print(f"   ❌ Authentication failed - not a testnet key")
    else:
        print(f"   ❌ Error: {error_str[:100]}")

# Test 2: Try as MAINNET key
print(f"\n2. Testing as MAINNET API key...")
try:
    client = HTTP(
        testnet=False,  # Force mainnet
        api_key=api_key,
        api_secret=api_secret
    )
    
    # Try to get account info
    response = client.get_wallet_balance(accountType="UNIFIED")
    
    if response['retCode'] == 0:
        print(f"   ✅ SUCCESS! This is a MAINNET API key")
        results.append("MAINNET")
        
        # Show balance
        if 'result' in response and 'list' in response['result']:
            for account in response['result']['list']:
                if 'totalEquity' in account:
                    equity = float(account.get('totalEquity', 0))
                    if equity > 0:
                        print(f"   Mainnet Balance: ${equity:.2f}")
    elif response['retCode'] == 10002:
        print(f"   ❌ Invalid API key for mainnet")
    elif response['retCode'] == 10003:
        print(f"   ❌ Invalid signature - check API secret")
    elif response['retCode'] == 33004:
        print(f"   ❌ API key shows as 'expired' on mainnet")
        print(f"      (This often means it's a testnet-only key)")
    else:
        print(f"   ❌ Failed: {response['retMsg']} (Code: {response['retCode']})")
        
except Exception as e:
    error_str = str(e)
    if "33004" in error_str:
        print(f"   ❌ API key 'expired' error - likely a testnet key")
    else:
        print(f"   ❌ Error: {error_str[:100]}")

print("\n" + "="*70)
print("CONCLUSION")
print("="*70)

if "TESTNET" in results and "MAINNET" not in results:
    print("""
🔴 Your API key is for TESTNET only!

To fix this:
1. Log into https://www.bybit.com (not testnet.bybit.com)
2. Go to API Management
3. Create a NEW API key for MAINNET
4. Make sure to enable:
   - Derivatives trading
   - Orders (Read/Write)
   - Positions (Read/Write)
5. Update Railway environment with the new MAINNET credentials
6. Set BYBIT_TESTNET=false in Railway
""")
elif "MAINNET" in results and "TESTNET" not in results:
    print("""
✅ Your API key is for MAINNET!

The bot should work. Make sure:
- BYBIT_TESTNET=false in Railway environment
- API has correct permissions enabled
""")
elif "TESTNET" in results and "MAINNET" in results:
    print("""
⚠️ Unusual: API key works on both networks.
Use BYBIT_TESTNET=false for mainnet trading.
""")
else:
    print("""
❌ API key doesn't work on either network!

Possible issues:
1. Invalid API key or secret
2. Extra spaces/newlines in credentials
3. Wrong API version (need v5 Unified Trading)
4. API key actually expired
5. IP restrictions blocking access

Please double-check your API credentials.
""")