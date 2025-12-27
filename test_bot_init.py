import asyncio
import os
import sys

# Add project root to path
sys.path.append(os.getcwd())

from autobot.core.bot import Bot4H

async def test_bot_init():
    print("Testing Bot4H initialization...")
    try:
        # We need to mock config or environment variables if Bot4H reads them on init
        # Bot4H loads config.yaml which exists.
        # It initializes Bybit broker which requires API keys from Env.
        # We can set dummy env vars for the test if not present.
        if "BYBIT_API_KEY" not in os.environ:
            os.environ["BYBIT_API_KEY"] = "dummy_key"
        if "BYBIT_API_SECRET" not in os.environ:
            os.environ["BYBIT_API_SECRET"] = "dummy_secret"
            
        bot = Bot4H()
        print("✅ Bot4H instantiated successfully.")
        
        # Test broker instantiation check
        if bot.broker:
            print("✅ Broker initialized.")
            
        # We can't easily test connection without real keys, 
        # but we can check if the methods exist and are coroutines.
        import inspect
        
        methods_to_check = [
            'get_balance',
            'place_market',
            'place_reduce_only_limit',
            'place_conditional_stop',
            'get_klines'
        ]
        
        print("\nChecking Broker methods:")
        for m in methods_to_check:
            method = getattr(bot.broker, m, None)
            if method:
                is_async = inspect.iscoroutinefunction(method)
                print(f"  - {m}: {'✅ Async' if is_async else '❌ NOT Async'}")
            else:
                print(f"  - {m}: ❌ MISSING")
                
    except Exception as e:
        print(f"❌ Initialization failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_bot_init())
