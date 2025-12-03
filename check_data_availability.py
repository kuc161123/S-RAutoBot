import asyncio
import aiohttp
import yaml
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

async def check_symbol(session, symbol):
    url = "https://api.bybit.com/v5/market/kline"
    params = {"category": "linear", "symbol": symbol, "interval": "3", "limit": 1}
    try:
        async with session.get(url, params=params) as resp:
            data = await resp.json()
            if data['retCode'] == 0 and data['result']['list']:
                return True
    except Exception:
        pass
    return False

async def run():
    with open('symbols_400.yaml', 'r') as f:
        sym_data = yaml.safe_load(f)
        symbols = sym_data['symbols']
        
    print(f"Checking {len(symbols)} symbols...")
    
    valid = 0
    invalid = 0
    
    async with aiohttp.ClientSession() as session:
        tasks = [check_symbol(session, s) for s in symbols]
        results = await asyncio.gather(*tasks)
        
    for res in results:
        if res: valid += 1
        else: invalid += 1
        
    print(f"✅ Valid: {valid}")
    print(f"❌ Invalid/Error: {invalid}")

if __name__ == "__main__":
    asyncio.run(run())
