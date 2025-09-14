"""
Test script for ML improvements
Demonstrates how to use the new features without breaking existing functionality
"""
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Import the improvements
from dynamic_rr_calculator import get_dynamic_rr_calculator

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_dynamic_rr():
    """Test dynamic R:R calculator with various market conditions"""
    logger.info("=== Testing Dynamic R:R Calculator ===")
    
    # Initialize calculator (disabled by default)
    rr_calc = get_dynamic_rr_calculator(base_rr=2.5, enabled=False)
    
    # Test 1: Disabled mode (should return base RR)
    features = {
        'volatility_regime': 'high',
        'atr_percentile': 80,
        'trend_strength': 30,
        'higher_tf_alignment': 40,
        'session': 'us',
        'hour_of_day': 14,
        'day_of_week': 2
    }
    
    disabled_rr = rr_calc.calculate_rr(features, "BTCUSDT")
    logger.info(f"Test 1 - Disabled mode: {disabled_rr} (expected: 2.5)")
    assert disabled_rr == 2.5, "Should return base RR when disabled"
    
    # Enable calculator
    rr_calc.enabled = True
    
    # Test 2: High volatility (should reduce RR)
    high_vol_rr = rr_calc.calculate_rr(features, "BTCUSDT")
    logger.info(f"Test 2 - High volatility: {high_vol_rr:.2f} (expected: < 2.5)")
    assert high_vol_rr < 2.5, "High volatility should reduce RR"
    
    # Test 3: Low volatility + strong trend (should increase RR)
    features.update({
        'volatility_regime': 'low',
        'atr_percentile': 20,
        'trend_strength': 80,
        'higher_tf_alignment': 85
    })
    
    low_vol_rr = rr_calc.calculate_rr(features, "ETHUSDT")
    logger.info(f"Test 3 - Low vol + strong trend: {low_vol_rr:.2f} (expected: > 2.5)")
    assert low_vol_rr > 2.5, "Low volatility with strong trend should increase RR"
    
    # Test 4: Off hours (should slightly reduce RR)
    features.update({
        'volatility_regime': 'normal',
        'atr_percentile': 50,
        'trend_strength': 50,
        'session': 'off_hours'
    })
    
    off_hours_rr = rr_calc.calculate_rr(features, "ADAUSDT")
    logger.info(f"Test 4 - Off hours: {off_hours_rr:.2f} (expected: < 2.5)")
    assert off_hours_rr < 2.5, "Off hours should slightly reduce RR"
    
    # Test 5: Safety bounds
    features.update({
        'volatility_regime': 'extreme',
        'atr_percentile': 95,
        'trend_strength': 10,
        'higher_tf_alignment': 20
    })
    
    extreme_rr = rr_calc.calculate_rr(features, "DOGEUSDT")
    logger.info(f"Test 5 - Extreme conditions: {extreme_rr:.2f} (expected: >= 1.5)")
    assert extreme_rr >= rr_calc.min_rr, "Should not go below minimum RR"
    
    # Test 6: Update performance tracking
    logger.info("\n=== Testing Performance Tracking ===")
    rr_calc.update_performance(2.2, 'win', 2.2)
    rr_calc.update_performance(2.2, 'loss', -1.0)
    rr_calc.update_performance(2.8, 'win', 2.8)
    rr_calc.update_performance(2.8, 'win', 2.8)
    
    stats = rr_calc.get_stats()
    logger.info(f"Calculator stats: {stats}")
    
    logger.info("✅ All dynamic R:R tests passed!\n")

def test_correlation_concept():
    """Demonstrate how correlation filtering would work"""
    logger.info("=== Correlation Filter Concept ===")
    
    # Simulated correlation matrix
    symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'ADAUSDT', 'DOGEUSDT']
    
    # Create correlation matrix (example values)
    correlations = {
        'BTCUSDT': {'BTCUSDT': 1.0, 'ETHUSDT': 0.85, 'BNBUSDT': 0.75, 'ADAUSDT': 0.6, 'DOGEUSDT': 0.4},
        'ETHUSDT': {'BTCUSDT': 0.85, 'ETHUSDT': 1.0, 'BNBUSDT': 0.8, 'ADAUSDT': 0.65, 'DOGEUSDT': 0.45},
        'BNBUSDT': {'BTCUSDT': 0.75, 'ETHUSDT': 0.8, 'BNBUSDT': 1.0, 'ADAUSDT': 0.55, 'DOGEUSDT': 0.35},
        'ADAUSDT': {'BTCUSDT': 0.6, 'ETHUSDT': 0.65, 'ADAUSDT': 0.55, 'ADAUSDT': 1.0, 'DOGEUSDT': 0.3},
        'DOGEUSDT': {'BTCUSDT': 0.4, 'ETHUSDT': 0.45, 'BNBUSDT': 0.35, 'ADAUSDT': 0.3, 'DOGEUSDT': 1.0}
    }
    
    # Simulate open positions
    open_positions = [('BTCUSDT', 'long'), ('ETHUSDT', 'long')]
    
    # Check if we should take a new signal
    def should_filter_signal(symbol, signal_type, threshold=0.7):
        high_corr_count = 0
        reasons = []
        
        for pos_symbol, pos_side in open_positions:
            corr = correlations[symbol].get(pos_symbol, 0)
            
            # Same direction with high correlation = risky
            if abs(corr) > threshold:
                if (corr > 0 and signal_type == pos_side):
                    high_corr_count += 1
                    reasons.append(f"{pos_symbol} (corr: {corr:.2f})")
                elif (corr < 0 and signal_type != pos_side):
                    high_corr_count += 1
                    reasons.append(f"{pos_symbol} (inverse corr: {corr:.2f})")
        
        should_filter = high_corr_count >= 2
        return should_filter, reasons
    
    # Test scenarios
    test_cases = [
        ('BNBUSDT', 'long'),   # High correlation with BTC and ETH
        ('DOGEUSDT', 'long'),  # Low correlation
        ('ADAUSDT', 'short'),  # Opposite direction
    ]
    
    for symbol, direction in test_cases:
        should_filter, reasons = should_filter_signal(symbol, direction)
        if should_filter:
            logger.info(f"❌ Filter {symbol} {direction} - High correlation with: {', '.join(reasons)}")
        else:
            logger.info(f"✅ Allow {symbol} {direction} - Correlation within limits")
    
    logger.info("")

def test_adaptive_threshold_concept():
    """Demonstrate adaptive ML threshold concept"""
    logger.info("=== Adaptive ML Threshold Concept ===")
    
    # Simulate threshold evolution
    class SimpleAdaptiveThreshold:
        def __init__(self, base=70.0):
            self.base = base
            self.current = base
            self.recent_performance = []
        
        def update(self, ml_score, actual_outcome):
            self.recent_performance.append({
                'score': ml_score,
                'outcome': actual_outcome,
                'correct': (ml_score >= self.current and actual_outcome == 'win') or 
                          (ml_score < self.current and actual_outcome == 'loss')
            })
            
            # Keep last 20
            if len(self.recent_performance) > 20:
                self.recent_performance.pop(0)
            
            # Adjust threshold
            if len(self.recent_performance) >= 10:
                accuracy = sum(1 for p in self.recent_performance if p['correct']) / len(self.recent_performance)
                
                if accuracy < 0.5:  # Poor performance
                    # Threshold might be too low (letting bad trades through)
                    avg_losing_score = np.mean([p['score'] for p in self.recent_performance 
                                               if p['outcome'] == 'loss' and p['score'] >= self.current])
                    if avg_losing_score > 0:
                        self.current = min(85.0, self.current + 2.0)
                elif accuracy > 0.7:  # Good performance
                    # Threshold might be too high (missing good trades)
                    avg_winning_score = np.mean([p['score'] for p in self.recent_performance 
                                                if p['outcome'] == 'win' and p['score'] < self.current])
                    if avg_winning_score > 0:
                        self.current = max(60.0, self.current - 2.0)
        
        def get_threshold(self):
            return self.current
    
    # Simulate trading
    threshold = SimpleAdaptiveThreshold(70.0)
    
    # Simulate some trades
    simulated_trades = [
        (75, 'win'), (68, 'loss'), (82, 'win'), (65, 'loss'),
        (71, 'loss'), (78, 'win'), (69, 'loss'), (85, 'win'),
        (72, 'loss'), (76, 'loss'), (80, 'win'), (67, 'win'),
    ]
    
    logger.info(f"Starting threshold: {threshold.get_threshold()}")
    
    for i, (score, outcome) in enumerate(simulated_trades):
        old_threshold = threshold.get_threshold()
        threshold.update(score, outcome)
        new_threshold = threshold.get_threshold()
        
        if old_threshold != new_threshold:
            logger.info(f"Trade {i+1}: Score {score}, Outcome {outcome} → "
                       f"Threshold adjusted from {old_threshold} to {new_threshold}")
    
    logger.info(f"Final threshold: {threshold.get_threshold()}\n")

def test_volatility_sizing_concept():
    """Demonstrate volatility-based position sizing"""
    logger.info("=== Volatility-Based Position Sizing Concept ===")
    
    base_risk_usd = 100  # Base risk per trade
    
    scenarios = [
        ("Low volatility", "low", 20, base_risk_usd),
        ("Normal volatility", "normal", 50, base_risk_usd),
        ("High volatility", "high", 80, base_risk_usd),
        ("Extreme volatility", "extreme", 95, base_risk_usd),
    ]
    
    volatility_multipliers = {
        'low': 1.2,
        'normal': 1.0,
        'high': 0.7,
        'extreme': 0.5
    }
    
    for name, regime, percentile, base_risk in scenarios:
        # Base adjustment from regime
        regime_mult = volatility_multipliers[regime]
        
        # Fine-tune with percentile
        if percentile > 80:
            percentile_mult = 0.9
        elif percentile < 20:
            percentile_mult = 1.1
        else:
            percentile_mult = 1.0
        
        # Final risk
        adjusted_risk = base_risk * regime_mult * percentile_mult
        
        # Cap at 150% of base
        adjusted_risk = min(adjusted_risk, base_risk * 1.5)
        
        logger.info(f"{name} (percentile: {percentile}): "
                   f"${base_risk} → ${adjusted_risk:.0f} "
                   f"({(adjusted_risk/base_risk-1)*100:+.0f}%)")
    
    logger.info("")

def main():
    """Run all tests"""
    logger.info("Starting ML Improvements Test Suite\n")
    
    test_dynamic_rr()
    test_correlation_concept()
    test_adaptive_threshold_concept()
    test_volatility_sizing_concept()
    
    logger.info("✅ All tests completed successfully!")
    logger.info("\nThese improvements can be gradually enabled via config without breaking existing functionality.")

if __name__ == "__main__":
    main()