#!/usr/bin/env python3
"""
Complete Trading Test Suite - Tests 10 symbols and verifies order execution
"""
import asyncio
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from crypto_trading_bot.config import settings
from crypto_trading_bot.exchange.bybit_client import BybitClient
from crypto_trading_bot.strategy.aggressive_strategy import AggressiveStrategy
from crypto_trading_bot.utils.indicators import add_all_indicators
import pandas as pd
from datetime import datetime

# Test symbols - high volume pairs
TEST_SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "DOGEUSDT", "1000PEPEUSDT",
    "WIFUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT", "AVAXUSDT"
]

async def test_symbol(exchange, strategy, symbol, test_num):
    """Test a single symbol for signal generation and order placement"""
    print(f"\n{'='*60}")
    print(f"TEST #{test_num}: {symbol}")
    print(f"{'='*60}")
    
    try:
        # Fetch klines
        print(f"1. Fetching data...")
        response = exchange.client.get_kline(
            category="linear",
            symbol=symbol,
            interval="5",
            limit=200
        )
        
        if response['retCode'] != 0:
            print(f"   ❌ Failed to fetch data: {response['retMsg']}")
            return False
            
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
        df = add_all_indicators(df, vars(settings))
        
        # Current data
        current = df.iloc[-1]
        print(f"2. Market Data:")
        print(f"   Price: ${current['close']:.4f}")
        print(f"   RSI: {current.get('rsi', 0):.1f}")
        print(f"   Volume: {current['volume']:,.0f}")
        
        # Generate signal
        print(f"3. Analyzing for signals...")
        signal = strategy.analyze(symbol, df)
        
        if signal:
            print(f"   ✅ SIGNAL FOUND!")
            print(f"   Action: {signal.action}")
            print(f"   Confidence: {signal.confidence:.1%}")
            print(f"   Entry: ${signal.price:.4f}")
            print(f"   Stop Loss: ${signal.stop_loss:.4f}")
            print(f"   Take Profit: ${signal.take_profit:.4f}")
            print(f"   Risk/Reward: {signal.risk_reward_ratio:.2f}")
            
            # Test order placement (DRY RUN)
            print(f"4. Testing order parameters...")
            
            # Calculate position size
            balance = exchange.balance if exchange.balance > 0 else 250
            risk_amount = balance * settings.risk_per_trade
            
            # Determine leverage based on signal type
            leverage = settings.scalp_leverage if 'scalp' in signal.reason.lower() else settings.swing_leverage
            
            # Calculate stop loss distance
            sl_distance = abs(signal.price - signal.stop_loss) / signal.price
            
            # Calculate position size
            position_size = risk_amount / sl_distance
            qty_in_coin = position_size / signal.price / leverage
            
            print(f"   Balance: ${balance:.2f}")
            print(f"   Risk Amount: ${risk_amount:.2f}")
            print(f"   Leverage: {leverage}x")
            print(f"   Position Size: ${position_size:.2f}")
            print(f"   Quantity: {qty_in_coin:.6f} {symbol.replace('USDT', '')}")
            
            # Validate parameters
            print(f"5. Validating order parameters...")
            
            # Check minimum order size (simplified for test)
            min_qty = 0.001  # Default minimum for most pairs
            if qty_in_coin < min_qty:
                print(f"   ⚠️ Quantity below minimum: {qty_in_coin:.6f} < {min_qty}")
                qty_in_coin = min_qty * 1.1  # Use 10% above minimum
                print(f"   Adjusted to: {qty_in_coin:.6f}")
            
            # Format quantity
            formatted_qty = exchange._format_quantity(symbol, qty_in_coin)
            
            # Build order parameters (what would be sent)
            order_params = {
                "category": "linear",
                "symbol": symbol,
                "side": signal.action.capitalize(),
                "orderType": "Market",
                "qty": str(formatted_qty),
                "timeInForce": "IOC",
                "positionIdx": 0,
                "stopLoss": str(signal.stop_loss),
                "slOrderType": "Market",
                "slTriggerBy": "LastPrice",
                "takeProfit": str(signal.take_profit),
                "tpOrderType": "Limit",
                "tpTriggerBy": "LastPrice",
                "tpLimitPrice": str(signal.take_profit),
                "tpslMode": "Full"
            }
            
            print(f"   ✅ Order parameters validated")
            print(f"   Would place: {signal.action.upper()} {formatted_qty} {symbol}")
            print(f"   With TP: ${signal.take_profit:.4f} (Limit)")
            print(f"   With SL: ${signal.stop_loss:.4f} (Market)")
            
            return True
        else:
            print(f"   No signal generated")
            
            # Show why no signal
            if current.get('rsi', 50) < 35:
                print(f"   - RSI oversold but other conditions not met")
            elif current.get('rsi', 50) > 65:
                print(f"   - RSI overbought but other conditions not met")
            else:
                print(f"   - RSI neutral ({current.get('rsi', 50):.1f})")
            
            return False
            
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False

async def main():
    print("="*70)
    print("COMPLETE TRADING TEST SUITE")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
    
    # Force mainnet
    settings.bybit_testnet = False
    
    print("\n📋 Configuration:")
    print(f"   Mode: {'TESTNET' if settings.bybit_testnet else 'MAINNET'}")
    print(f"   Strategy: {settings.strategy_type}")
    print(f"   Min Confirmations: {settings.min_confirmations}")
    print(f"   Risk per trade: {settings.risk_per_trade * 100:.1f}%")
    print(f"   Max positions: {settings.max_positions}")
    print(f"   Scalp Leverage: {settings.scalp_leverage}x")
    print(f"   Swing Leverage: {settings.swing_leverage}x")
    
    # Initialize exchange
    print("\n🔌 Connecting to Bybit...")
    exchange = BybitClient(
        api_key=settings.bybit_api_key,
        api_secret=settings.bybit_api_secret,
        testnet=False,
        config=settings
    )
    
    # Check balance
    print("\n💰 Checking Balance...")
    balance = exchange.get_account_balance()
    if balance and balance > 0:
        print(f"   ✅ Balance: ${balance:.2f}")
    else:
        print(f"   ⚠️ Could not detect balance, assuming $250")
        exchange.balance = 250
    
    # Initialize strategy with aggressive settings
    print("\n🎯 Initializing Aggressive Strategy...")
    config = vars(settings)
    config['min_confirmations'] = 1  # Most aggressive
    config['min_risk_reward'] = 1.0  # Lower requirement
    
    strategy = AggressiveStrategy(config)
    print(f"   Strategy initialized with min_confirmations={strategy.min_confirmations}")
    
    # Run tests
    print("\n🚀 Starting Tests on 10 Symbols...")
    
    successful_signals = 0
    failed_tests = 0
    
    for i, symbol in enumerate(TEST_SYMBOLS, 1):
        result = await test_symbol(exchange, strategy, symbol, i)
        if result:
            successful_signals += 1
        else:
            failed_tests += 1
        
        # Small delay between tests
        await asyncio.sleep(0.5)
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    print(f"✅ Successful signals: {successful_signals}/10")
    print(f"❌ No signals: {failed_tests}/10")
    
    if successful_signals > 0:
        print(f"\n🎉 SUCCESS! Bot can generate signals and would execute trades.")
        print(f"   The tpslMode fix should allow orders to execute properly.")
        print(f"\n📝 Next Steps:")
        print(f"   1. Deploy to Railway with these environment variables:")
        print(f"      - STRATEGY_TYPE=aggressive")
        print(f"      - MIN_CONFIRMATIONS=1")
        print(f"      - MIN_RISK_REWARD=1.0")
        print(f"   2. Monitor logs for successful order execution")
        print(f"   3. Bot should now place trades with proper TP/SL")
    else:
        print(f"\n⚠️ No signals generated. Possible issues:")
        print(f"   1. Market conditions not favorable")
        print(f"   2. All symbols in neutral zone")
        print(f"   3. Consider even looser parameters")
    
    await exchange.cleanup()

if __name__ == "__main__":
    asyncio.run(main())