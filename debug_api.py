
import asyncio
import aiohttp
import json
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("DebugAPI")

async def test_request(method, endpoint, params=None):
    url = f"https://api.bybit.com{endpoint}"
    logger.info(f"REQUEST: {method} {url} Params: {params}")
    
    async with aiohttp.ClientSession() as session:
        start = time.time()
        try:
            async with session.request(method, url, params=params, timeout=10) as resp:
                logger.info(f"STATUS: {resp.status}")
                logger.info(f"HEADERS: {resp.headers}")
                
                # Read raw bytes first
                raw_data = await resp.read()
                logger.info(f"RAW BYTES LENGTH: {len(raw_data)}")
                
                # Try decoding
                text = raw_data.decode('utf-8')
                logger.info(f"DECODED TEXT LENGTH: {len(text)}")
                
                # Preview
                preview = text[:200]
                logger.info(f"PREVIEW: {preview}...")
                
                # JSON Load
                data = json.loads(text)
                logger.info("✅ JSON PARSE SUCCESS")
                return data
                
        except Exception as e:
            logger.error(f"❌ ERROR: {e}")
            return None

async def main():
    logger.info("--- TEST 1: Single Symbol (No Limit) ---")
    await test_request("GET", "/v5/market/instruments-info", {"category": "linear", "symbol": "BTCUSDT"})
    
    logger.info("\n--- TEST 2: Single Symbol (Limit=1) ---")
    await test_request("GET", "/v5/market/instruments-info", {"category": "linear", "symbol": "BTCUSDT", "limit": 1})
    
    logger.info("\n--- TEST 3: Bybit Class Implementation ---")
    try:
        from autobot.brokers.bybit import Bybit, BybitConfig
        # Dummy keys (public endpoint doesn't check signature for instruments-info usually, but we sign anyway)
        cfg = BybitConfig(api_key="test", api_secret="test", base_url="https://api.bybit.com")
        broker = Bybit(cfg)
        
        # This calls get_instruments_info -> _request
        logger.info("Calling broker.get_instruments_info(symbol='BTCUSDT')...")
        res = await broker.get_instruments_info(symbol='BTCUSDT')
        logger.info(f"Class Result Count: {len(res)}")
        if len(res) > 0:
            logger.info("✅ Class Test Passed")
        else:
            logger.error("❌ Class Test Failed (Empty)")
            
        await broker.close()
    except Exception as e:
        logger.error(f"❌ Class Test Exception: {e}")

if __name__ == "__main__":
    asyncio.run(main())
