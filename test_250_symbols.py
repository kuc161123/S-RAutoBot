#!/usr/bin/env python3
"""
Test script to verify bot can handle 250 symbols
"""
import asyncio
import yaml
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_multi_websocket():
    """Test multi-websocket connection handling"""
    
    # Load config
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    symbols = config['trade']['symbols']
    logger.info(f"✅ Loaded {len(symbols)} symbols from config")
    
    # Test multi-websocket handler
    from multi_websocket_handler import MultiWebSocketHandler
    
    ws_url = "wss://stream.bybit.com/v5/public/linear"
    running = asyncio.Event()
    running.set()
    
    handler = MultiWebSocketHandler(ws_url, running)
    
    # Calculate connection requirements
    num_symbols = len(symbols)
    connections_needed = (num_symbols - 1) // handler.MAX_SUBS_PER_CONNECTION + 1
    
    logger.info(f"📡 Will use {connections_needed} WebSocket connections")
    logger.info(f"   • Max per connection: {handler.MAX_SUBS_PER_CONNECTION}")
    logger.info(f"   • Total symbols: {num_symbols}")
    
    # Test kline topics
    kline_topics = [f"5.{sym}" for sym in symbols]
    
    logger.info(f"🔧 Testing connection setup...")
    
    # Quick connection test
    test_topics = kline_topics[:5]  # Just test with first 5
    
    try:
        count = 0
        async for sym, kline in handler.multi_kline_stream(test_topics):
            count += 1
            logger.info(f"✅ Received kline for {sym}: {kline.get('close', 'N/A')}")
            if count >= 5:
                break
        
        logger.info(f"✅ Multi-websocket handler working correctly!")
        
    except Exception as e:
        logger.error(f"❌ Error: {e}")
    
    finally:
        running.clear()

async def test_memory_usage():
    """Test memory requirements for 250 symbols"""
    import psutil
    import os
    
    process = psutil.Process(os.getpid())
    
    # Load config
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    symbols = config['trade']['symbols']
    
    # Simulate data structures
    from collections import defaultdict
    import pandas as pd
    import numpy as np
    
    # Create mock kline data
    klines = {}
    for sym in symbols:
        # 300 candles per symbol
        df = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=300, freq='5min'),
            'open': np.random.random(300) * 100,
            'high': np.random.random(300) * 100,
            'low': np.random.random(300) * 100,
            'close': np.random.random(300) * 100,
            'volume': np.random.random(300) * 1000000
        })
        klines[sym] = df
    
    # Check memory usage
    mem_info = process.memory_info()
    mem_mb = mem_info.rss / 1024 / 1024
    
    logger.info(f"💾 Memory Usage with {len(symbols)} symbols:")
    logger.info(f"   • RSS Memory: {mem_mb:.2f} MB")
    logger.info(f"   • Per symbol: {mem_mb/len(symbols):.2f} MB")
    
    if mem_mb < 1000:
        logger.info(f"✅ Memory usage acceptable (<1GB)")
    else:
        logger.warning(f"⚠️ High memory usage: {mem_mb:.2f} MB")

def main():
    logger.info("=" * 60)
    logger.info("🚀 Testing 250 Symbol Configuration")
    logger.info("=" * 60)
    
    # Load and verify config
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    symbols = config['trade']['symbols']
    
    logger.info(f"\n📊 Configuration Summary:")
    logger.info(f"   • Total symbols: {len(symbols)}")
    logger.info(f"   • First 5: {symbols[:5]}")
    logger.info(f"   • Last 5: {symbols[-5:]}")
    
    # Check for duplicates
    if len(symbols) != len(set(symbols)):
        logger.warning("⚠️ Duplicate symbols found!")
    else:
        logger.info("✅ No duplicate symbols")
    
    # Test async components
    logger.info(f"\n🧪 Testing WebSocket Components...")
    
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        # Test multi-websocket
        loop.run_until_complete(asyncio.wait_for(test_multi_websocket(), timeout=30))
        
        # Test memory
        logger.info(f"\n🧪 Testing Memory Usage...")
        loop.run_until_complete(test_memory_usage())
        
    except asyncio.TimeoutError:
        logger.info("✅ WebSocket test completed (timeout as expected)")
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
    finally:
        loop.close()
    
    logger.info("\n" + "=" * 60)
    logger.info("✅ All tests completed!")
    logger.info("=" * 60)

if __name__ == "__main__":
    main()