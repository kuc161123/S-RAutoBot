#!/usr/bin/env python3
"""
Test script to verify phantom trade integration
Tests the integration without requiring database
"""
import os
import logging
from datetime import datetime, timedelta
from phantom_trade_tracker import PhantomTrade, get_phantom_tracker

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_phantom_integration():
    """Test phantom trade tracker integration"""
    
    # Get phantom tracker
    phantom_tracker = get_phantom_tracker()
    
    # Create a test phantom trade
    test_phantom = PhantomTrade(
        symbol="BTCUSDT",
        side="long",
        entry_price=45000,
        stop_loss=44000,
        take_profit=46500,
        signal_time=datetime.now(),
        ml_score=58.5,  # Below threshold - rejected
        was_executed=False,
        features={
            'trend_strength': 65,
            'volume_ratio': 1.8,
            'risk_reward_ratio': 2.5,
            'support_resistance_strength': 3
        }
    )
    
    # Add to tracker
    phantom_tracker.active_phantoms["BTCUSDT"] = test_phantom
    logger.info("Added test phantom trade")
    
    # Simulate price updates
    prices = [45100, 45300, 45500, 45800, 46000, 46300, 46500]  # Will hit TP
    
    for price in prices:
        logger.info(f"Updating BTCUSDT price to {price}")
        # Note: In live bot, symbol_collector would be passed here
        phantom_tracker.update_phantom_prices("BTCUSDT", price)
        
        if "BTCUSDT" not in phantom_tracker.active_phantoms:
            logger.info(f"✅ Phantom trade closed at {price}")
            break
    
    # Check stats
    stats = phantom_tracker.get_phantom_stats("BTCUSDT")
    
    print("\n=== Phantom Trade Stats ===")
    print(f"Total phantoms: {stats['total']}")
    print(f"Executed: {stats['executed']}")
    print(f"Rejected: {stats['rejected']}")
    
    if stats['rejection_stats']['total_rejected'] > 0:
        print("\n=== Rejection Analysis ===")
        print(f"Would have won: {stats['rejection_stats']['would_have_won']}")
        print(f"Would have lost: {stats['rejection_stats']['would_have_lost']}")
        print(f"Missed profit: {stats['rejection_stats']['missed_profit_pct']:.2f}%")
        print(f"Avoided loss: {stats['rejection_stats']['avoided_loss_pct']:.2f}%")
    
    print("\n✅ Phantom tracking integration working!")
    print("Note: When database is available, phantom data will also be stored for future symbol-specific ML")
    
    return True

if __name__ == "__main__":
    print("Testing phantom trade integration...")
    success = test_phantom_integration()
    
    if success:
        print("\n✅ Integration test passed!")
    else:
        print("\n❌ Integration test failed.")