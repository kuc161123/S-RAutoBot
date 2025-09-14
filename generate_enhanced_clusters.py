#!/usr/bin/env python
"""
Generate enhanced symbol clusters with confidence scores
Run this to create the initial enhanced clusters file
"""

import asyncio
import json
import logging
import yaml
from datetime import datetime
from candle_storage_postgres import CandleStorage
from symbol_clustering_enhanced import EnhancedSymbolClusterer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def main():
    """Generate enhanced symbol clusters from historical data"""
    
    # Initialize storage
    storage = CandleStorage()
    
    # Load recent data for all symbols
    logger.info("Loading historical data...")
    frames = {}
    
    # Get symbols from config
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
        symbols = config['trade']['symbols']
    
    logger.info(f"Processing {len(symbols)} symbols...")
    
    # Load data for each symbol
    for symbol in symbols:
        try:
            df = storage.load_candles(symbol, limit=1000)
            if df is not None and len(df) >= 500:
                frames[symbol] = df
                logger.info(f"Loaded {len(df)} candles for {symbol}")
            else:
                logger.warning(f"Insufficient data for {symbol}: {len(df) if df is not None else 0} candles")
        except Exception as e:
            logger.warning(f"Failed to load {symbol}: {e}")
    
    logger.info(f"Successfully loaded data for {len(frames)} symbols")
    
    if not frames:
        logger.error("No data loaded, cannot generate clusters")
        return
    
    # Perform enhanced clustering
    clusterer = EnhancedSymbolClusterer(frames)
    
    # Calculate enhanced metrics
    metrics = clusterer.calculate_enhanced_metrics(min_candles=500)
    logger.info(f"Calculated enhanced metrics for {len(metrics)} symbols")
    
    # Get confidence-based clusters
    confidence_clusters = clusterer.cluster_with_confidence()
    
    # Save enhanced clusters
    clusterer.save_enhanced_clusters()
    
    # Print summary
    print("\n" + "="*60)
    print("ENHANCED CLUSTER SUMMARY")
    print("="*60)
    
    cluster_names = {
        1: "Blue Chip",
        2: "Stable/Low Vol",
        3: "Meme/High Vol",
        4: "Mid-Cap Alts",
        5: "Small Caps"
    }
    
    # Count symbols per cluster and borderline cases
    cluster_counts = {i: 0 for i in range(1, 6)}
    borderline_counts = {i: 0 for i in range(1, 6)}
    confidence_sums = {i: 0.0 for i in range(1, 6)}
    
    for symbol, assignment in confidence_clusters.items():
        cluster_counts[assignment.primary_cluster] += 1
        confidence_sums[assignment.primary_cluster] += assignment.primary_confidence
        if assignment.is_borderline:
            borderline_counts[assignment.primary_cluster] += 1
    
    for cluster_id in range(1, 6):
        count = cluster_counts[cluster_id]
        if count > 0:
            avg_conf = confidence_sums[cluster_id] / count
            borderline = borderline_counts[cluster_id]
            
            print(f"\nCluster {cluster_id}: {cluster_names[cluster_id]}")
            print(f"  Total symbols: {count}")
            print(f"  Average confidence: {avg_conf:.1%}")
            print(f"  Borderline symbols: {borderline} ({borderline/count*100:.1f}%)")
            
            # Show example symbols
            examples = [s for s, a in confidence_clusters.items() 
                       if a.primary_cluster == cluster_id][:5]
            if examples:
                print(f"  Examples: {', '.join(examples)}")
    
    # Show most borderline symbols
    print("\n" + "="*60)
    print("MOST BORDERLINE SYMBOLS (need special attention):")
    print("="*60)
    
    borderline_symbols = [(s, a) for s, a in confidence_clusters.items() 
                         if a.is_borderline]
    borderline_symbols.sort(key=lambda x: x[1].primary_confidence)
    
    for symbol, assignment in borderline_symbols[:10]:
        primary_name = cluster_names[assignment.primary_cluster]
        secondary_name = cluster_names.get(assignment.secondary_cluster, "None")
        
        print(f"{symbol:15} Primary: {primary_name} ({assignment.primary_confidence:.0%}) | "
              f"Secondary: {secondary_name} ({assignment.secondary_confidence:.0%})")
    
    print("\n✅ Enhanced clusters generated successfully!")
    print(f"📊 Output saved to: symbol_clusters_enhanced.json")


if __name__ == "__main__":
    asyncio.run(main())