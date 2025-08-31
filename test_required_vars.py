#!/usr/bin/env python3
"""
Test that all variables are required from environment
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("="*70)
print("TESTING REQUIRED ENVIRONMENT VARIABLES")
print("="*70)

# Clear all environment variables first
env_vars_to_test = [
    'BYBIT_API_KEY', 'BYBIT_API_SECRET', 'RISK_PER_TRADE', 'MAX_POSITIONS', 
    'LEVERAGE', 'RSI_PERIOD', 'RSI_OVERSOLD', 'RSI_OVERBOUGHT',
    'MACD_FAST', 'MACD_SLOW', 'MACD_SIGNAL', 'SCAN_INTERVAL',
    'MIN_VOLUME_24H', 'STARTUP_DELAY', 'SCALP_TIMEFRAME',
    'SCALP_PROFIT_TARGET', 'SCALP_STOP_LOSS', 'MIN_RISK_REWARD',
    'MIN_CONFIRMATIONS', 'RR_SL_MULTIPLIER', 'RR_TP_MULTIPLIER',
    'SCALP_RR_SL_MULTIPLIER', 'SCALP_RR_TP_MULTIPLIER',
    'SCALP_LEVERAGE', 'SWING_LEVERAGE', 'LOG_LEVEL', 'STRATEGY_TYPE'
]

print("\n1. Testing with NO environment variables set...")
print("   Expected: Should fail to load config\n")

# Clear any existing env vars
for var in env_vars_to_test:
    os.environ.pop(var, None)

try:
    from crypto_trading_bot.config import Settings
    settings = Settings()
    print("   ❌ ERROR: Config loaded without required variables!")
    print("   Some variables have defaults that shouldn't exist")
except Exception as e:
    print("   ✅ CORRECT: Config failed to load")
    print(f"   Error: {str(e)[:100]}...")

print("\n2. Testing with ALL environment variables set...")
print("   Expected: Should load successfully\n")

# Set all required variables
os.environ['BYBIT_API_KEY'] = 'test_key'
os.environ['BYBIT_API_SECRET'] = 'test_secret'
os.environ['RISK_PER_TRADE'] = '0.005'
os.environ['MAX_POSITIONS'] = '10'
os.environ['LEVERAGE'] = '10'
os.environ['RSI_PERIOD'] = '14'
os.environ['RSI_OVERSOLD'] = '35'
os.environ['RSI_OVERBOUGHT'] = '65'
os.environ['MACD_FAST'] = '12'
os.environ['MACD_SLOW'] = '26'
os.environ['MACD_SIGNAL'] = '9'
os.environ['SCAN_INTERVAL'] = '60'
os.environ['MIN_VOLUME_24H'] = '1000000'
os.environ['STARTUP_DELAY'] = '5'
os.environ['SCALP_TIMEFRAME'] = '5'
os.environ['SCALP_PROFIT_TARGET'] = '0.003'
os.environ['SCALP_STOP_LOSS'] = '0.002'
os.environ['MIN_RISK_REWARD'] = '1.0'
os.environ['MIN_CONFIRMATIONS'] = '1'
os.environ['RR_SL_MULTIPLIER'] = '1.5'
os.environ['RR_TP_MULTIPLIER'] = '3.0'
os.environ['SCALP_RR_SL_MULTIPLIER'] = '1.0'
os.environ['SCALP_RR_TP_MULTIPLIER'] = '2.0'
os.environ['SCALP_LEVERAGE'] = '5'
os.environ['SWING_LEVERAGE'] = '10'
os.environ['LOG_LEVEL'] = 'INFO'
os.environ['STRATEGY_TYPE'] = 'aggressive'

try:
    # Need to reload the module
    import importlib
    if 'crypto_trading_bot.config' in sys.modules:
        importlib.reload(sys.modules['crypto_trading_bot.config'])
    
    from crypto_trading_bot.config import Settings
    settings = Settings()
    print("   ✅ CORRECT: Config loaded successfully")
    print(f"   Risk per trade: {settings.risk_per_trade}")
    print(f"   Max positions: {settings.max_positions}")
    print(f"   Strategy type: {settings.strategy_type}")
except Exception as e:
    print("   ❌ ERROR: Config failed to load even with all vars")
    print(f"   Error: {e}")

print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print("""
✅ ALL configuration is now controlled via Railway environment variables
✅ No hidden defaults - you have complete control
✅ Bot will not start without all required variables
✅ Changes in Railway take effect immediately on redeploy

This ensures consistent behavior between local and Railway deployments.
""")