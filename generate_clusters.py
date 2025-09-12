#!/usr/bin/env python
"""
Generate initial symbol clusters for the trading bot
Run this once to create symbol_clusters.json
"""

import asyncio
import json
import logging
from datetime import datetime
from candle_storage_postgres import CandleStorage
from symbol_clustering import SymbolClusterer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def main():
    """Generate symbol clusters from historical data"""
    
    # Initialize storage
    storage = CandleStorage()
    
    # Load recent data for all symbols
    logger.info("Loading historical data...")
    frames = {}
    
    # Get symbols from config
    with open('config.yaml', 'r') as f:
        import yaml
        config = yaml.safe_load(f)
        symbols = config['symbols']
    
    # Load data for each symbol
    for symbol in symbols:
        try:
            df = storage.load_candles(symbol, limit=1000)
            if df is not None and len(df) >= 300:
                frames[symbol] = df
                logger.info(f"Loaded {len(df)} candles for {symbol}")
        except Exception as e:
            logger.warning(f"Failed to load {symbol}: {e}")
    
    logger.info(f"Loaded data for {len(frames)} symbols")
    
    # Perform clustering
    clusterer = SymbolClusterer(frames)
    
    # Calculate metrics
    metrics = clusterer.calculate_metrics(min_candles=300)
    logger.info(f"Calculated metrics for {len(metrics)} symbols")
    
    # Save clusters
    clusters = clusterer.save_clusters()
    
    # Print summary
    descriptions = clusterer.get_cluster_descriptions()
    print("\nCluster Summary:")
    print("="*50)
    for cluster_id, desc in descriptions.items():
        count = sum(1 for c in clusters.values() if c == cluster_id)
        print(f"Cluster {cluster_id}: {desc}")
        print(f"  Count: {count} symbols")
        
        # Show example symbols
        examples = [s for s, c in clusters.items() if c == cluster_id][:5]
        if examples:
            print(f"  Examples: {', '.join(examples)}")
        print()

if __name__ == "__main__":
    asyncio.run(main())