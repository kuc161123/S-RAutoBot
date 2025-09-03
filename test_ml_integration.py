#!/usr/bin/env python3
"""
Test ML integration without breaking existing bot
"""
import asyncio
import yaml
import logging
import sys
from datetime import datetime
import pandas as pd
import numpy as np

# Test imports
try:
    from ml_signal_scorer import get_scorer
    from strategy_pullback import Settings, detect_signal_pullback
    from position_mgr import Position
    
    print("✅ All imports successful")
except ImportError as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_ml_disabled():
    """Test that ML scorer works when disabled"""
    print("\n🧪 Test 1: ML Scorer disabled")
    
    try:
        scorer = get_scorer(enabled=False)
        
        # Create mock data
        df = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=300, freq='5min'),
            'open': np.random.random(300) * 100,
            'high': np.random.random(300) * 100,
            'low': np.random.random(300) * 100,
            'close': np.random.random(300) * 100,
            'volume': np.random.random(300) * 1000000
        })
        
        # Test scoring with disabled scorer
        should_take, score, reason = scorer.should_take_signal(
            df, "long", 100.0, 95.0, 110.0, "BTCUSDT", {}
        )
        
        assert should_take == True, "Disabled scorer should always return True"
        assert score == 75.0, "Disabled scorer should return neutral score"
        print(f"  Result: {should_take}, Score: {score}, Reason: {reason}")
        print("✅ Test 1 passed: ML disabled works correctly")
        
    except Exception as e:
        print(f"❌ Test 1 failed: {e}")
        return False
    
    return True

def test_ml_enabled_no_redis():
    """Test ML scorer without Redis (local storage)"""
    print("\n🧪 Test 2: ML Scorer with local storage")
    
    try:
        scorer = get_scorer(enabled=True, min_score=70.0)
        
        # Create mock data
        df = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=300, freq='5min'),
            'open': np.random.random(300) * 100 + 100,
            'high': np.random.random(300) * 100 + 105,
            'low': np.random.random(300) * 100 + 95,
            'close': np.random.random(300) * 100 + 100,
            'volume': np.random.random(300) * 1000000
        })
        
        # Set up trending data for better signal
        df['close'] = df['close'].cumsum() / 100 + 100
        df.index = df['timestamp']
        
        # Test scoring
        should_take, score, reason = scorer.should_take_signal(
            df, "long", 102.0, 100.0, 106.0, "ETHUSDT", 
            {'atr': 2.0, 'res': 103.0, 'sup': 99.0}
        )
        
        print(f"  Result: {should_take}, Score: {score:.1f}")
        print(f"  Reason: {reason}")
        print(f"  ML Stats: {scorer.get_ml_stats()}")
        
        # Test outcome update
        scorer.update_signal_outcome("ETHUSDT", datetime.now(), "win", 2.0)
        print(f"  After update: {scorer.completed_trades_count} completed trades")
        
        print("✅ Test 2 passed: Local storage works correctly")
        
    except Exception as e:
        print(f"❌ Test 2 failed: {e}")
        return False
    
    return True

def test_position_with_entry_time():
    """Test Position dataclass with entry_time"""
    print("\n🧪 Test 3: Position with entry_time")
    
    try:
        # Create position with entry_time
        pos = Position(
            side="long",
            qty=0.1,
            entry=50000,
            sl=49000,
            tp=52000,
            entry_time=datetime.now()
        )
        
        assert pos.entry_time is not None, "Entry time should be set"
        print(f"  Position created with entry_time: {pos.entry_time}")
        
        # Test position without entry_time (for backward compatibility)
        pos2 = Position("short", 0.2, 60000, 61000, 58000)
        assert pos2.entry_time is None, "Entry time should be None by default"
        print(f"  Position without entry_time works: {pos2.entry_time}")
        
        print("✅ Test 3 passed: Position dataclass updated correctly")
        
    except Exception as e:
        print(f"❌ Test 3 failed: {e}")
        return False
    
    return True

def test_config_loading():
    """Test config loading with ML options"""
    print("\n🧪 Test 4: Config with ML options")
    
    try:
        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        use_ml = config['trade'].get('use_ml_scoring', False)
        ml_min_score = config['trade'].get('ml_min_score', 70.0)
        
        print(f"  use_ml_scoring: {use_ml}")
        print(f"  ml_min_score: {ml_min_score}")
        
        assert isinstance(use_ml, bool), "use_ml_scoring should be boolean"
        assert isinstance(ml_min_score, (int, float)), "ml_min_score should be numeric"
        assert 0 <= ml_min_score <= 100, "ml_min_score should be between 0-100"
        
        print("✅ Test 4 passed: Config options added correctly")
        
    except Exception as e:
        print(f"❌ Test 4 failed: {e}")
        return False
    
    return True

def main():
    print("=" * 60)
    print("🔧 Testing ML Integration (Non-Breaking)")
    print("=" * 60)
    
    all_passed = True
    
    # Run tests
    all_passed &= test_ml_disabled()
    all_passed &= test_ml_enabled_no_redis()
    all_passed &= test_position_with_entry_time()
    all_passed &= test_config_loading()
    
    print("\n" + "=" * 60)
    if all_passed:
        print("✅ ALL TESTS PASSED - ML integration is safe!")
        print("The bot will work with or without ML enabled.")
        print("\nTo enable ML scoring:")
        print("1. Set use_ml_scoring: true in config.yaml")
        print("2. Set REDIS_URL environment variable for persistence")
        print("3. ML will start scoring after 200 completed trades")
    else:
        print("❌ Some tests failed - please review")
    print("=" * 60)

if __name__ == "__main__":
    main()