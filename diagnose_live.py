#!/usr/bin/env python3
"""
Live diagnostic to see exactly why no trades are happening
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

async def diagnose():
    print("=" * 70)
    print("LIVE TRADING BOT DIAGNOSTIC")
    print("=" * 70)
    
    # Force mainnet
    settings.bybit_testnet = False
    
    print(f"\n1. Configuration:")
    print(f"   Mode: {'TESTNET' if settings.bybit_testnet else 'MAINNET'}")
    print(f"   Min Confirmations: {settings.min_confirmations}")
    print(f"   Risk per trade: {settings.risk_per_trade * 100:.1f}%")
    print(f"   Leverage: {settings.leverage}x")
    
    # Initialize exchange
    print(f"\n2. Connecting to Bybit...")
    exchange = BybitClient(
        api_key=settings.bybit_api_key,
        api_secret=settings.bybit_api_secret,
        testnet=False,  # Force mainnet
        config=settings
    )
    
    # Check balance with detailed info
    print(f"\n3. Checking Balance...")
    try:
        # Try multiple methods to get balance
        methods = [
            ("UNIFIED", None),
            ("UNIFIED", "USDT"),
            ("CONTRACT", None),
            ("SPOT", None)
        ]
        
        for account_type, coin in methods:
            try:
                params = {"accountType": account_type}
                if coin:
                    params["coin"] = coin
                    
                response = exchange.client.get_wallet_balance(**params)
                
                if response['retCode'] == 0:
                    result = response.get('result', {})
                    
                    # Try to find USDT balance
                    if 'list' in result and len(result['list']) > 0:
                        account = result['list'][0]
                        
                        # Check totalEquity
                        if 'totalEquity' in account:
                            equity = float(account['totalEquity'])
                            if equity > 0:
                                print(f"   ✅ Found balance via {account_type}: ${equity:.2f}")
                                break
                        
                        # Check coins
                        if 'coin' in account:
                            for coin_data in account['coin']:
                                if coin_data.get('coin') == 'USDT':
                                    for field in ['availableToWithdraw', 'walletBalance', 'equity', 'availableBalanceWithoutConvert']:
                                        val = coin_data.get(field, '0')
                                        if val and val != '' and float(val) > 0:
                                            print(f"   ✅ Found USDT via {account_type}.{field}: ${float(val):.2f}")
                                            exchange.balance = float(val)  # Store it
                                            break
            except Exception as e:
                print(f"   - {account_type} failed: {str(e)[:50]}")
    except Exception as e:
        print(f"   ❌ Balance check failed: {e}")
    
    # Initialize strategy with looser settings
    print(f"\n4. Initializing Strategy (with looser settings)...")
    config = vars(settings)
    config['min_confirmations'] = 1  # Even looser for testing
    config['min_risk_reward'] = 1.0  # Lower R:R requirement
    
    strategy = ScalpingStrategy(config, ml_enabled=False)
    strategy.min_confirmations = 1  # Override
    strategy.min_volume_multiplier = 1.2  # Even lower
    
    print(f"   Strategy settings:")
    print(f"   - Min confirmations: {strategy.min_confirmations}")
    print(f"   - RSI oversold/overbought: {strategy.rsi_oversold}/{strategy.rsi_overbought}")
    print(f"   - Volume multiplier: {strategy.min_volume_multiplier}x")
    
    # Test top symbols
    print(f"\n5. Testing Signal Generation on Top Symbols...")
    test_symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "DOGEUSDT", "1000PEPEUSDT", 
                   "WIFUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT", "AVAXUSDT"]
    
    signals_found = []
    
    for symbol in test_symbols:
        try:
            print(f"\n   Testing {symbol}...")
            
            # Fetch klines
            response = exchange.client.get_kline(
                category="linear",
                symbol=symbol,
                interval="5",
                limit=200
            )
            
            if response['retCode'] != 0:
                print(f"   ❌ Failed to fetch data: {response['retMsg']}")
                continue
            
            # Convert to DataFrame
            klines = response['result']['list']
            df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'])
            df = df.astype({
                'open': float, 'high': float, 'low': float, 
                'close': float, 'volume': float
            })
            df['timestamp'] = pd.to_datetime(df['timestamp'].astype(float), unit='ms')
            df = df.sort_values('timestamp')
            
            # Add indicators
            df = add_all_indicators(df, config)
            
            # Check latest values
            current = df.iloc[-1]
            print(f"      Price: ${current['close']:.4f}")
            print(f"      RSI: {current.get('rsi', 0):.1f}")
            print(f"      Volume: {current['volume']:.0f}")
            
            # Try to generate signal
            signal = strategy.analyze(symbol, df)
            
            if signal:
                print(f"   🎯 SIGNAL FOUND: {signal.action}")
                print(f"      Confidence: {signal.confidence:.1%}")
                print(f"      Reason: {signal.reason}")
                print(f"      Entry: ${signal.price:.4f}")
                print(f"      SL: ${signal.stop_loss:.4f}")
                print(f"      TP: ${signal.take_profit:.4f}")
                signals_found.append(signal)
            else:
                # Check why no signal
                print(f"   No signal - checking why...")
                
                # Manually check conditions
                if current.get('rsi', 50) < 30:
                    print(f"      ✓ RSI oversold ({current['rsi']:.1f})")
                elif current.get('rsi', 50) > 70:
                    print(f"      ✓ RSI overbought ({current['rsi']:.1f})")
                else:
                    print(f"      - RSI neutral ({current.get('rsi', 50):.1f})")
                
                # Check MACD
                if 'macd' in current and 'macd_signal' in current:
                    if current['macd'] > current['macd_signal']:
                        print(f"      ✓ MACD bullish")
                    else:
                        print(f"      - MACD bearish")
                
                # Check volume
                if 'volume' in current:
                    avg_vol = df['volume'].rolling(20).mean().iloc[-1]
                    vol_ratio = current['volume'] / avg_vol if avg_vol > 0 else 0
                    if vol_ratio > 1.2:
                        print(f"      ✓ Volume high ({vol_ratio:.1f}x)")
                    else:
                        print(f"      - Volume low ({vol_ratio:.1f}x)")
                        
        except Exception as e:
            print(f"   ❌ Error testing {symbol}: {e}")
    
    # Summary
    print(f"\n" + "=" * 70)
    print(f"DIAGNOSTIC SUMMARY")
    print(f"=" * "70")
    
    if len(signals_found) > 0:
        print(f"✅ Found {len(signals_found)} potential signals")
        print(f"The strategy IS working but might be too selective")
    else:
        print(f"⚠️ No signals found on {len(test_symbols)} symbols")
        print(f"Possible issues:")
        print(f"1. Market conditions not favorable (sideways/quiet)")
        print(f"2. Strategy still too strict")
        print(f"3. Indicators not aligning")
    
    print(f"\nRECOMMENDATIONS:")
    print(f"1. Set MIN_CONFIRMATIONS=1 in Railway")
    print(f"2. Consider using simpler entry conditions")
    print(f"3. Monitor logs for any errors")
    
    await exchange.cleanup()

if __name__ == "__main__":
    asyncio.run(diagnose())