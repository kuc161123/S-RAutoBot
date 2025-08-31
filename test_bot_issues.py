#!/usr/bin/env python3
"""
Comprehensive test to identify why trades aren't happening
"""
import asyncio
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from crypto_trading_bot.config import settings
from crypto_trading_bot.exchange.bybit_client import BybitClient
from crypto_trading_bot.strategy.scalping_strategy import ScalpingStrategy
from crypto_trading_bot.utils.indicators import add_all_indicators
import pandas as pd
import structlog

logger = structlog.get_logger()

async def run_tests():
    """Run comprehensive tests"""
    print("=" * 60)
    print("CRYPTO BOT DIAGNOSTIC TESTS")
    print("=" * 60)
    
    results = {
        'passed': 0,
        'failed': 0,
        'issues': []
    }
    
    # Test 1: Configuration
    print("\n[TEST 1] Checking configuration...")
    try:
        assert settings.bybit_api_key, "API key missing"
        assert settings.bybit_api_secret, "API secret missing"
        print("✅ Configuration loaded")
        results['passed'] += 1
    except Exception as e:
        print(f"❌ Configuration error: {e}")
        results['failed'] += 1
        results['issues'].append(f"Config: {e}")
    
    # Test 2: Exchange connection
    print("\n[TEST 2] Connecting to Bybit...")
    exchange = None
    try:
        exchange = BybitClient(
            api_key=settings.bybit_api_key,
            api_secret=settings.bybit_api_secret,
            testnet=settings.bybit_testnet,
            config=settings
        )
        print("✅ Connected to Bybit")
        results['passed'] += 1
    except Exception as e:
        print(f"❌ Connection error: {e}")
        results['failed'] += 1
        results['issues'].append(f"Connection: {e}")
        return results
    
    # Test 3: Balance check
    print("\n[TEST 3] Checking balance...")
    balance = exchange.get_account_balance()
    if balance and balance > 0:
        print(f"✅ Balance: ${balance:.2f}")
        results['passed'] += 1
    else:
        print(f"❌ Balance issue: ${balance if balance else 0:.2f}")
        results['failed'] += 1
        results['issues'].append("Balance showing as 0 or None")
        
        # Try alternative balance fetch
        print("   Trying alternative balance fetch...")
        try:
            response = exchange.client.get_wallet_balance(accountType="UNIFIED")
            print(f"   Raw response: {response.get('result', {}).get('list', [{}])[0].get('totalEquity', 'N/A')}")
        except Exception as e:
            print(f"   Alternative fetch failed: {e}")
    
    # Test 4: Fetch market data
    print("\n[TEST 4] Fetching market data for BTCUSDT...")
    klines = None
    try:
        klines = exchange.get_klines("BTCUSDT", interval="5", limit=200)
        if klines is not None and len(klines) > 0:
            print(f"✅ Fetched {len(klines)} candles")
            results['passed'] += 1
        else:
            print("❌ No klines data")
            results['failed'] += 1
            results['issues'].append("Cannot fetch klines")
    except Exception as e:
        print(f"❌ Klines error: {e}")
        results['failed'] += 1
        results['issues'].append(f"Klines: {e}")
    
    # Test 5: Strategy initialization
    print("\n[TEST 5] Initializing strategy...")
    strategy = None
    try:
        strategy = ScalpingStrategy(vars(settings), ml_enabled=False)  # Disable ML for testing
        print("✅ Strategy initialized")
        results['passed'] += 1
    except Exception as e:
        print(f"❌ Strategy error: {e}")
        results['failed'] += 1
        results['issues'].append(f"Strategy: {e}")
        return results
    
    # Test 6: Add indicators
    print("\n[TEST 6] Adding technical indicators...")
    df_with_indicators = None
    if klines is not None:
        try:
            df_with_indicators = add_all_indicators(klines, vars(settings))
            required_indicators = ['rsi', 'macd', 'macd_signal', 'bb_upper', 'bb_lower', 'atr']
            missing = [ind for ind in required_indicators if ind not in df_with_indicators.columns]
            
            if not missing:
                print(f"✅ All indicators added")
                results['passed'] += 1
            else:
                print(f"❌ Missing indicators: {missing}")
                results['failed'] += 1
                results['issues'].append(f"Missing indicators: {missing}")
        except Exception as e:
            print(f"❌ Indicators error: {e}")
            results['failed'] += 1
            results['issues'].append(f"Indicators: {e}")
    
    # Test 7: Signal generation
    print("\n[TEST 7] Testing signal generation...")
    if strategy and df_with_indicators is not None:
        try:
            signal = strategy.analyze("BTCUSDT", df_with_indicators)
            if signal:
                print(f"✅ Signal generated: {signal.action} at ${signal.price:.2f}")
                print(f"   Confidence: {signal.confidence:.2%}")
                print(f"   Reason: {signal.reason}")
                results['passed'] += 1
            else:
                print("⚠️  No signal generated (market conditions may not be favorable)")
                print("   This is normal - strategy is selective")
                results['passed'] += 1
        except Exception as e:
            print(f"❌ Signal generation error: {e}")
            results['failed'] += 1
            results['issues'].append(f"Signal: {e}")
    
    # Test 8: Check strategy parameters
    print("\n[TEST 8] Checking strategy parameters...")
    print(f"   Min confirmations: {strategy.min_confirmations}")
    print(f"   RSI oversold: {strategy.rsi_oversold}")
    print(f"   RSI overbought: {strategy.rsi_overbought}")
    
    if strategy.min_confirmations > 4:
        print("⚠️  Min confirmations might be too high")
        results['issues'].append("Min confirmations > 4 (too strict)")
    else:
        print("✅ Strategy parameters reasonable")
        results['passed'] += 1
    
    # Test 9: Test multiple symbols
    print("\n[TEST 9] Testing top 5 symbols for signals...")
    test_symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "DOGEUSDT", "BNBUSDT"]
    signals_found = 0
    
    for symbol in test_symbols:
        try:
            klines = exchange.get_klines(symbol, interval="5", limit=200)
            if klines is not None:
                df = add_all_indicators(klines, vars(settings))
                signal = strategy.analyze(symbol, df)
                if signal:
                    signals_found += 1
                    print(f"   ✅ {symbol}: {signal.action} signal")
                else:
                    print(f"   - {symbol}: No signal")
        except Exception as e:
            print(f"   ❌ {symbol}: Error - {e}")
    
    if signals_found > 0:
        print(f"✅ Found {signals_found} signals across {len(test_symbols)} symbols")
        results['passed'] += 1
    else:
        print(f"⚠️  No signals found across {len(test_symbols)} symbols")
        results['issues'].append("No signals on any tested symbols")
        results['passed'] += 1  # This might be normal
    
    # Test 10: Risk calculations
    print("\n[TEST 10] Testing risk management...")
    try:
        risk_per_trade = settings.risk_per_trade
        max_positions = settings.max_positions
        leverage = settings.leverage
        
        if balance and balance > 0:
            position_size = (balance * risk_per_trade) * leverage
            print(f"   Balance: ${balance:.2f}")
            print(f"   Risk per trade: {risk_per_trade*100:.1f}%")
            print(f"   Leverage: {leverage}x")
            print(f"   Position size: ${position_size:.2f}")
            
            if position_size < 10:
                print("⚠️  Position size might be too small")
                results['issues'].append(f"Position size ${position_size:.2f} < $10")
            else:
                print("✅ Risk management OK")
            results['passed'] += 1
        else:
            print("❌ Cannot calculate - balance is 0")
            results['failed'] += 1
    except Exception as e:
        print(f"❌ Risk management error: {e}")
        results['failed'] += 1
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"Passed: {results['passed']}/10")
    print(f"Failed: {results['failed']}/10")
    
    if results['issues']:
        print("\n🔧 ISSUES TO FIX:")
        for i, issue in enumerate(results['issues'], 1):
            print(f"   {i}. {issue}")
    
    # Recommendations
    print("\n💡 RECOMMENDATIONS:")
    if balance == 0 or balance is None:
        print("   1. CRITICAL: Fix balance detection - bot cannot trade without balance")
        print("      - Check if using correct account type (UNIFIED)")
        print("      - Verify API has trading permissions")
        print("      - Try CONTRACT account if UNIFIED doesn't work")
    
    if signals_found == 0:
        print("   2. Consider reducing strategy strictness:")
        print("      - Lower min_confirmations to 2")
        print("      - Adjust RSI thresholds (25/75 instead of 30/70)")
        print("      - Reduce min_score requirements")
    
    print("\n" + "=" * 60)
    
    # Cleanup
    if exchange:
        await exchange.cleanup()
    
    return results

if __name__ == "__main__":
    print("Starting diagnostic tests...")
    results = asyncio.run(run_tests())
    
    # Exit with error code if critical issues
    if results['failed'] > 3:
        sys.exit(1)