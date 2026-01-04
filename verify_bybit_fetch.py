
import asyncio
import logging
import os
import yaml
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from autobot.brokers.bybit import Bybit, BybitConfig

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

async def verify_fetch():
    # Load config for keys (optional, public endpoint might work without)
    try:
        with open('/Users/lualakol/AutoTrading Bot/config.yaml', 'r') as f:
            config = yaml.safe_load(f)
            api_key = os.path.expandvars(config['bybit']['api_key'])
            api_secret = os.path.expandvars(config['bybit']['api_secret'])
    except:
        api_key = ""
        api_secret = ""

    cfg = BybitConfig(api_key=api_key, api_secret=api_secret, base_url='https://api.bybit.com')
    broker = Bybit(cfg)
    
    logger.info("Testing batch fetch with limit=200...")
    items = await broker.get_instruments_info(category="linear")
    
    logger.info(f"Fetched {len(items)} total instruments")
    
    if len(items) == 0:
        logger.error("❌ Fetch failed! 0 items returned.")
    else:
        logger.info("✅ Fetch successful!")
        
        # Check a few random symbols
        check_syms = ['BTCUSDT', 'ETHUSDT', '1000BONKUSDT']
        found = 0
        for item in items:
            sym = item.get('symbol')
            if sym in check_syms:
                lev = item.get('leverageFilter', {}).get('maxLeverage')
                logger.info(f"Found {sym}: Max Lev {lev}x")
                found += 1
                
        if found > 0:
             logger.info(f"✅ Found {found}/{len(check_syms)} target symbols in batch data")
        
    await broker.close()

if __name__ == "__main__":
    asyncio.run(verify_fetch())
