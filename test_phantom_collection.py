#!/usr/bin/env python3
"""
Test script to verify phantom trade data collection
"""
import os
import logging
from datetime import datetime, timedelta
from symbol_data_collector import get_symbol_collector
from phantom_trade_tracker import get_phantom_tracker

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_phantom_collection():
    """Test that phantom trades are properly collected"""
    
    # Get instances
    collector = get_symbol_collector()
    phantom_tracker = get_phantom_tracker()
    
    # Test recording a phantom trade
    test_symbol = "BTCUSDT"
    test_ml_score = 58.5  # Below threshold
    test_features = {
        'trend_strength': 65,
        'volume_ratio': 1.8,
        'risk_reward_ratio': 2.5,
        'support_resistance_strength': 3
    }
    
    # Simulate recording a phantom trade
    signal_time = datetime.now() - timedelta(hours=1)
    exit_time = datetime.now()
    
    try:
        # Record phantom trade
        collector.record_phantom_trade(
            symbol=test_symbol,
            df=None,
            btc_price=45000,
            ml_score=test_ml_score,
            features=test_features,
            outcome='win',
            pnl_percent=2.5,
            signal_time=signal_time,
            exit_time=exit_time
        )
        logger.info("✅ Successfully recorded phantom trade")
        
        # Get phantom stats
        stats = collector.get_phantom_stats(test_symbol)
        logger.info(f"Phantom stats for {test_symbol}: {stats}")
        
        # Get overall phantom stats
        overall_stats = collector.get_phantom_stats()
        logger.info(f"Overall phantom stats: {overall_stats}")
        
        print("\n✅ Phantom trade collection is working!")
        print(f"Total phantoms: {overall_stats.get('total_phantoms', 0)}")
        print(f"Would have won: {overall_stats.get('would_have_won', 0)}")
        print(f"Would have lost: {overall_stats.get('would_have_lost', 0)}")
        print(f"Missed profits: {overall_stats.get('missed_profits', 0):.2f}%")
        
    except Exception as e:
        logger.error(f"❌ Failed to record phantom trade: {e}")
        return False
    
    return True

if __name__ == "__main__":
    print("Testing phantom trade data collection...")
    success = test_phantom_collection()
    
    if success:
        print("\n✅ All tests passed! Phantom data collection is ready.")
    else:
        print("\n❌ Tests failed. Check the logs above.")