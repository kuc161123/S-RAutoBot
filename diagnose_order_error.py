#!/usr/bin/env python3
"""
Diagnose the exact error preventing order execution
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Set required env vars BEFORE importing settings
os.environ.setdefault('RISK_PER_TRADE', '0.005')
os.environ.setdefault('MAX_POSITIONS', '10')
os.environ.setdefault('LEVERAGE', '10')

from crypto_trading_bot.config import settings
from crypto_trading_bot.exchange.bybit_client import BybitClient
import logging

# Set up detailed logging
logging.basicConfig(level=logging.DEBUG, format='%(message)s')
logger = logging.getLogger()

def test_order():
    print("="*70)
    print("ORDER EXECUTION DIAGNOSTIC")
    print("="*70)
    
    # Force mainnet
    settings.bybit_testnet = False
    
    print(f"\n1. Initializing Bybit Client...")
    exchange = BybitClient(
        api_key=settings.bybit_api_key,
        api_secret=settings.bybit_api_secret,
        testnet=False,
        config=settings
    )
    
    print(f"\n2. Testing with BTCUSDT...")
    
    # Get current price
    try:
        response = exchange.client.get_tickers(
            category="linear",
            symbol="BTCUSDT"
        )
        
        if response['retCode'] == 0:
            current_price = float(response['result']['list'][0]['lastPrice'])
            print(f"   Current price: ${current_price:.2f}")
            
            # Calculate test order parameters
            # Very small order for testing
            qty = 0.001  # Small BTC amount
            stop_loss = current_price * 0.98  # 2% below
            take_profit = current_price * 1.02  # 2% above
            
            print(f"\n3. Test Order Parameters:")
            print(f"   Symbol: BTCUSDT")
            print(f"   Side: Buy")
            print(f"   Quantity: {qty} BTC")
            print(f"   Stop Loss: ${stop_loss:.2f}")
            print(f"   Take Profit: ${take_profit:.2f}")
            
            # Try to place order (will fail but show exact error)
            print(f"\n4. Attempting to place order...")
            
            order_params = {
                "category": "linear",
                "symbol": "BTCUSDT",
                "side": "Buy",
                "orderType": "Market",
                "qty": str(qty),
                "timeInForce": "IOC",
                "positionIdx": 0,
                "stopLoss": str(stop_loss),
                "slOrderType": "Market",
                "slTriggerBy": "LastPrice",
                "takeProfit": str(take_profit),
                "tpOrderType": "Limit",
                "tpTriggerBy": "LastPrice",
                "tpLimitPrice": str(take_profit),
                "tpslMode": "Full"
            }
            
            print(f"\n   Order parameters being sent:")
            for key, value in order_params.items():
                print(f"   - {key}: {value}")
            
            print(f"\n5. Sending order to Bybit...")
            response = exchange.client.place_order(**order_params)
            
            print(f"\n6. Response from Bybit:")
            print(f"   retCode: {response.get('retCode')}")
            print(f"   retMsg: {response.get('retMsg')}")
            
            if response['retCode'] != 0:
                print(f"\n❌ ORDER FAILED!")
                print(f"   Error Code: {response.get('retCode')}")
                print(f"   Error Message: {response.get('retMsg')}")
                print(f"   Full Response: {response}")
                
                # Common error explanations
                if response['retCode'] == 10001:
                    print(f"\n   💡 This means: Invalid parameters")
                elif response['retCode'] == 10002:
                    print(f"\n   💡 This means: Invalid API key")
                elif response['retCode'] == 10003:
                    print(f"\n   💡 This means: Invalid signature")
                elif response['retCode'] == 10004:
                    print(f"\n   💡 This means: Invalid timestamp")
                elif response['retCode'] == 10005:
                    print(f"\n   💡 This means: Permission denied")
                elif response['retCode'] == 33004:
                    print(f"\n   💡 This means: API key expired")
                elif response['retCode'] == 110001:
                    print(f"\n   💡 This means: Order value too small")
                elif response['retCode'] == 110003:
                    print(f"\n   💡 This means: Order price out of range")
                elif response['retCode'] == 110004:
                    print(f"\n   💡 This means: Insufficient balance")
                elif response['retCode'] == 110007:
                    print(f"\n   💡 This means: Insufficient available balance")
                elif response['retCode'] == 110013:
                    print(f"\n   💡 This means: Invalid qty")
                elif response['retCode'] == 110043:
                    print(f"\n   💡 This means: Leverage not modified")
                elif response['retCode'] == 170131:
                    print(f"\n   💡 This means: Balance insufficient")
                elif response['retCode'] == 170213:
                    print(f"\n   💡 This means: Position mode not modified")
                
            else:
                print(f"\n✅ ORDER PLACED SUCCESSFULLY!")
                print(f"   Order ID: {response['result']['orderId']}")
                
        else:
            print(f"❌ Failed to get price: {response['retMsg']}")
            
    except Exception as e:
        print(f"\n❌ Exception occurred:")
        print(f"   {type(e).__name__}: {str(e)}")
        import traceback
        print(f"\nFull traceback:")
        traceback.print_exc()

if __name__ == "__main__":
    test_order()