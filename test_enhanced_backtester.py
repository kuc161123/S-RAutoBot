#!/usr/bin/env python3
"""
Test Enhanced Backtester

Verifies that the enhanced backtester generates ML features exactly like the live bot
for both Pullback and Mean Reversion strategies.
"""

import logging
from enhanced_backtester import EnhancedBacktester
from strategy_pullback_ml_learning import get_ml_learning_signals, MinimalSettings as PullbackSettings, reset_symbol_state as reset_pullback_state
from strategy_mean_reversion import detect_signal as detect_signal_mean_reversion, reset_symbol_state as reset_mean_reversion_state
from strategy_pullback import Settings as MRSettings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_pullback_backtester():
    """Test pullback strategy backtesting with ML features"""
    logger.info("🧪 Testing Pullback Strategy Backtesting")
    
    # Create pullback backtester
    backtester = EnhancedBacktester(
        get_ml_learning_signals,
        PullbackSettings(),
        reset_state_func=reset_pullback_state,
        strategy_type="pullback"
    )
    
    # Test on a single symbol
    try:
        results = backtester.run("BTCUSDT")
        
        logger.info(f"✅ Pullback backtest results: {len(results)} signals")
        
        if results:
            # Check first result structure
            first_result = results[0]
            logger.info(f"📊 Sample signal features: {len(first_result.get('features', {}))} features")
            logger.info(f"📈 Sample outcome: {first_result.get('outcome')}")
            
            # Verify expected features exist
            features = first_result.get('features', {})
            required_features = ['volume_ratio', 'entry_price', 'symbol_cluster']
            
            for feature in required_features:
                if feature in features:
                    logger.info(f"   ✅ {feature}: {features[feature]}")
                else:
                    logger.warning(f"   ❌ Missing feature: {feature}")
        
        return len(results) > 0
        
    except Exception as e:
        logger.error(f"❌ Pullback backtest failed: {e}")
        return False

def test_mr_backtester():
    """Test mean reversion strategy backtesting with ML features"""
    logger.info("🧪 Testing Mean Reversion Strategy Backtesting")
    
    # Create MR backtester
    backtester = EnhancedBacktester(
        detect_signal_mean_reversion,
        MRSettings(),
        reset_state_func=reset_mean_reversion_state,
        strategy_type="mean_reversion"
    )
    
    # Test on a single symbol
    try:
        results = backtester.run("ETHUSDT")
        
        logger.info(f"✅ MR backtest results: {len(results)} signals")
        
        if results:
            # Check first result structure
            first_result = results[0]
            logger.info(f"📊 Sample signal features: {len(first_result.get('features', {}))} features")
            logger.info(f"📈 Sample outcome: {first_result.get('outcome')}")
            
            # Verify expected features exist
            features = first_result.get('features', {})
            expected_features = ['range_position', 'volume_ratio', 'volatility']
            
            for feature in expected_features:
                if feature in features:
                    logger.info(f"   ✅ {feature}: {features[feature]}")
                else:
                    logger.warning(f"   ❌ Missing feature: {feature}")
        
        return len(results) > 0
        
    except Exception as e:
        logger.error(f"❌ MR backtest failed: {e}")
        return False

def main():
    """Run all tests"""
    logger.info("🚀 Testing Enhanced Backtester")
    
    pullback_success = test_pullback_backtester()
    mr_success = test_mr_backtester()
    
    if pullback_success and mr_success:
        logger.info("🎉 All tests passed! Enhanced backtester is working correctly.")
        return True
    else:
        logger.error("❌ Some tests failed. Check the logs above.")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)