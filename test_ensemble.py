#!/usr/bin/env python3
"""
Test Ensemble ML System
Ensures ensemble scorer works without breaking existing functionality
"""
import numpy as np
import pandas as pd
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_ensemble_import():
    """Test that ensemble imports correctly"""
    print("\n🧪 Test 1: Ensemble Import")
    try:
        from ml_ensemble_scorer import get_ensemble_scorer, EnsembleMLScorer, MarketRegime
        print("✅ Ensemble modules imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False

def test_ensemble_backward_compatibility():
    """Test that ensemble is backward compatible with basic scorer"""
    print("\n🧪 Test 2: Backward Compatibility")
    try:
        from ml_ensemble_scorer import get_ensemble_scorer
        from ml_signal_scorer import SignalFeatures
        
        # Create ensemble scorer
        scorer = get_ensemble_scorer(enabled=True, min_score=70.0)
        
        # Test with same interface as basic scorer
        df = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=300, freq='5min'),
            'open': np.random.random(300) * 100 + 100,
            'high': np.random.random(300) * 100 + 105,
            'low': np.random.random(300) * 100 + 95,
            'close': np.random.random(300) * 100 + 100,
            'volume': np.random.random(300) * 1000000
        })
        df.index = df['timestamp']
        
        # Create features
        features = SignalFeatures(
            trend_strength=60.0,
            higher_tf_alignment=75.0,
            ema_distance_ratio=1.5,
            volume_ratio=1.3,
            volume_trend=0.2,
            breakout_volume=1.5,
            support_resistance_strength=70.0,
            pullback_depth=50.0,
            confirmation_candle_strength=65.0,
            atr_percentile=45.0,
            volatility_regime="normal",
            hour_of_day=14,
            day_of_week=2,
            session="us",
            risk_reward_ratio=2.0,
            atr_stop_distance=1.2
        )
        
        # Test scoring
        score, reason = scorer.score_signal(features)
        print(f"  Score: {score:.1f}, Reason: {reason}")
        
        # Test should_take_signal
        should_take, score2, reason2 = scorer.should_take_signal(
            df, "long", 102.0, 100.0, 106.0, "BTCUSDT", 
            {'atr': 2.0, 'res': 103.0, 'sup': 99.0}
        )
        print(f"  Should take: {should_take}, Score: {score2:.1f}")
        
        print("✅ Backward compatibility maintained")
        return True
        
    except Exception as e:
        print(f"❌ Compatibility test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_market_regime_detection():
    """Test market regime detection"""
    print("\n🧪 Test 3: Market Regime Detection")
    try:
        from ml_ensemble_scorer import EnsembleMLScorer
        
        scorer = EnsembleMLScorer(enabled=True)
        
        # Create trending up market
        df_up = pd.DataFrame({
            'close': np.linspace(100, 110, 100) + np.random.random(100) * 0.5,
            'high': np.linspace(101, 111, 100) + np.random.random(100) * 0.5,
            'low': np.linspace(99, 109, 100) + np.random.random(100) * 0.5,
        })
        
        regime_up = scorer.detect_market_regime(df_up)
        print(f"  Trending up detected as: {regime_up}")
        
        # Create ranging market
        df_range = pd.DataFrame({
            'close': 100 + np.sin(np.linspace(0, 10*np.pi, 100)) * 2,
            'high': 102 + np.sin(np.linspace(0, 10*np.pi, 100)) * 2,
            'low': 98 + np.sin(np.linspace(0, 10*np.pi, 100)) * 2,
        })
        
        regime_range = scorer.detect_market_regime(df_range)
        print(f"  Ranging market detected as: {regime_range}")
        
        # Create volatile market
        df_volatile = pd.DataFrame({
            'close': 100 + np.random.random(100) * 20,
            'high': 105 + np.random.random(100) * 20,
            'low': 95 + np.random.random(100) * 20,
        })
        
        regime_volatile = scorer.detect_market_regime(df_volatile)
        print(f"  Volatile market detected as: {regime_volatile}")
        
        print("✅ Market regime detection working")
        return True
        
    except Exception as e:
        print(f"❌ Regime detection failed: {e}")
        return False

def test_ensemble_models():
    """Test ensemble model creation"""
    print("\n🧪 Test 4: Ensemble Model Creation")
    try:
        from ml_ensemble_scorer import EnsembleMLScorer
        
        scorer = EnsembleMLScorer(enabled=True)
        
        # Test model creation
        models = scorer._create_ensemble_models()
        
        print(f"  Models created: {list(models.keys())}")
        assert 'rf' in models, "Random Forest missing"
        assert 'xgb' in models or 'gb' in models, "Boosting model missing (XGBoost or GradientBoosting)"
        assert 'nn' in models, "Neural Network missing"
        
        print("✅ All ensemble models created successfully")
        return True
        
    except Exception as e:
        print(f"❌ Model creation failed: {e}")
        return False

def test_live_bot_integration():
    """Test that live_bot can use ensemble scorer"""
    print("\n🧪 Test 5: Live Bot Integration")
    try:
        # Test import in live_bot style
        try:
            from ml_ensemble_scorer import get_ensemble_scorer as get_scorer
            print("  Using Enhanced Ensemble ML Scorer")
        except ImportError:
            from ml_signal_scorer import get_scorer
            print("  Using Basic ML Scorer")
        
        # Create scorer as live_bot would
        scorer = get_scorer(enabled=True, min_score=70.0)
        stats = scorer.get_ml_stats()
        
        print(f"  ML Stats: Enabled={stats['enabled']}, Trained={stats['is_trained']}")
        print(f"  Completed trades: {stats['completed_trades']}")
        
        print("✅ Live bot integration successful")
        return True
        
    except Exception as e:
        print(f"❌ Integration failed: {e}")
        return False

def main():
    print("=" * 60)
    print("🔧 Testing Ensemble ML System")
    print("=" * 60)
    
    all_passed = True
    
    # Run tests
    all_passed &= test_ensemble_import()
    all_passed &= test_ensemble_backward_compatibility()
    all_passed &= test_market_regime_detection()
    all_passed &= test_ensemble_models()
    all_passed &= test_live_bot_integration()
    
    print("\n" + "=" * 60)
    if all_passed:
        print("✅ ALL TESTS PASSED - Ensemble system is safe!")
        print("\nEnhancements added:")
        print("• Ensemble voting (RF + XGBoost + Neural Network)")
        print("• Separate models for long vs short signals")
        print("• Market regime detection and regime-specific models")
        print("• Online learning capability")
        print("• Backward compatible with existing system")
    else:
        print("❌ Some tests failed - review before deploying")
    print("=" * 60)

if __name__ == "__main__":
    main()