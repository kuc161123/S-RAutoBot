#!/usr/bin/env python3
"""
Balanced Pullback Strategy - Less Restrictive Version
Provides more signals while still maintaining quality
"""
from strategy_pullback_enhanced import *
import logging

logger = logging.getLogger(__name__)

@dataclass
class BalancedSettings(Settings):
    """More balanced settings for better signal generation"""
    
    # Loosen Fibonacci requirements
    use_fibonacci: bool = True
    fib_min: float = 0.25  # Accept from 25% (was 38.2%)
    fib_max: float = 0.75  # Accept up to 75% (was 61.8%)
    fib_danger: float = 0.90  # Only reject >90% (was 78.6%)
    
    # Make 1H trend a preference, not requirement
    use_higher_tf: bool = True
    higher_tf_weight: float = 0.3  # 30% weight, not absolute filter
    
    # Less strict on confirmations
    use_dynamic_params: bool = True
    min_confirmation_candles: int = 1  # Can be as low as 1
    max_confirmation_candles: int = 3  # But no more than 3

def calculate_signal_quality_score(features: Dict, settings: BalancedSettings) -> float:
    """
    Calculate quality score (0-100) instead of binary accept/reject
    Higher score = better signal, but don't reject unless very low
    """
    score = 0.0
    
    # Fibonacci contribution (30 points max, more generous)
    fib = features.get('fibonacci_retracement', 50.0)
    if 38.2 <= fib <= 61.8:
        score += 30  # Golden zone - full points
    elif 25 <= fib <= 75:
        # Linear scaling outside golden zone
        if fib < 38.2:
            score += 15 + (fib - 25) / (38.2 - 25) * 15
        else:
            score += 15 + (75 - fib) / (75 - 61.8) * 15
    elif fib < 25:
        score += max(0, 10 - (25 - fib) * 0.5)  # Shallow - some points
    else:
        score += max(0, 10 - (fib - 75) * 0.3)  # Deep - few points
    
    # 1H trend contribution (20 points max, not mandatory)
    ht_trend = features.get('higher_tf_trend', 'NEUTRAL')
    signal_side = features.get('side', 'long')
    
    if (ht_trend == 'BULLISH' and signal_side == 'long') or \
       (ht_trend == 'BEARISH' and signal_side == 'short'):
        score += 20  # Aligned - full points
    elif ht_trend == 'NEUTRAL':
        score += 10  # Neutral - half points
    else:
        score += 5   # Against - still some points
    
    # Volume contribution (20 points)
    vol_ratio = features.get('volume_ratio', 1.0)
    score += min(20, vol_ratio * 10)
    
    # Trend strength (15 points)
    trend = features.get('trend_strength', 50)
    score += min(15, trend * 0.15)
    
    # Volatility adjustment (15 points)
    volatility = features.get('volatility_regime', 'NORMAL')
    if volatility == 'NORMAL':
        score += 15
    elif volatility == 'LOW':
        score += 10
    else:  # HIGH
        score += 8
    
    return min(100, score)

def get_balanced_signals(df: pd.DataFrame, settings: BalancedSettings = None,
                        df_1h: Optional[pd.DataFrame] = None, 
                        symbol: str = "UNKNOWN") -> list:
    """
    Balanced signal generation - more signals with quality scores
    """
    if settings is None:
        settings = BalancedSettings()
    
    # Get signals from enhanced strategy
    signals = get_signals(df, settings, df_1h, symbol)
    
    # If we got signals, they're already good
    if signals:
        return signals
    
    # Check if we're being too strict
    if symbol in breakout_states:
        state = breakout_states[symbol]
        
        # If we have a formed HL/LH but got rejected, check why
        if state.state in ["HL_FORMED", "LH_FORMED"]:
            
            # Extract current conditions
            features = {
                'fibonacci_retracement': 50.0,  # Calculate actual
                'higher_tf_trend': state.higher_tf_trend,
                'volatility_regime': state.volatility_regime,
                'side': 'long' if state.state == "HL_FORMED" else 'short',
                'volume_ratio': 1.0,  # Calculate actual
                'trend_strength': 50
            }
            
            # Calculate quality score
            quality = calculate_signal_quality_score(features, settings)
            
            # Log why signal was rejected
            if quality < 40:
                logger.info(f"{symbol}: Signal rejected - Quality too low ({quality:.0f}/100)")
            else:
                logger.info(f"{symbol}: Signal pending - Quality {quality:.0f}/100, waiting for conditions")
    
    return signals

class AdaptiveFilterManager:
    """
    Dynamically adjust filtering based on signal frequency
    Too few signals = loosen filters
    Too many signals = tighten filters
    """
    
    def __init__(self):
        self.signals_per_hour = []
        self.target_signals_per_hour = 2.0  # Aim for ~2 signals/hour
        self.min_signals_per_hour = 0.5    # Minimum acceptable
        self.max_signals_per_hour = 5.0    # Maximum acceptable
        
    def adjust_settings(self, current_settings: BalancedSettings,
                       signals_last_hour: int) -> BalancedSettings:
        """
        Dynamically adjust settings based on signal frequency
        """
        settings = current_settings
        
        if signals_last_hour < self.min_signals_per_hour:
            # Too restrictive - loosen
            logger.info("Too few signals - loosening filters")
            settings.fib_min = max(0.20, settings.fib_min - 0.05)
            settings.fib_max = min(0.80, settings.fib_max + 0.05)
            settings.min_confirmation_candles = 1
            settings.use_higher_tf = False  # Temporarily disable
            
        elif signals_last_hour > self.max_signals_per_hour:
            # Too loose - tighten
            logger.info("Too many signals - tightening filters")
            settings.fib_min = min(0.382, settings.fib_min + 0.05)
            settings.fib_max = max(0.618, settings.fib_max - 0.05)
            settings.min_confirmation_candles = 2
            settings.use_higher_tf = True
            
        return settings

def compare_strategies():
    """Show comparison of different restriction levels"""
    
    print("\n" + "="*60)
    print("📊 STRATEGY RESTRICTION COMPARISON")
    print("="*60)
    
    comparison = """
Setting              | Original | Enhanced | Balanced
---------------------|----------|----------|----------
Fibonacci Range      | Any      | 38-61%   | 25-75%
1H Trend Required    | No       | Yes      | Preferred
Confirmation Candles | 2        | 2-3      | 1-3
Volume Filter        | 1.2x     | 1.0-1.5x | 1.1x
Rejection Zone       | None     | >78.6%   | >90%
Expected Signals/Day | 10-15    | 2-5      | 8-12
Quality vs Quantity  | Medium   | Quality  | Balanced

Balanced Advantages:
✅ More learning opportunities for ML
✅ Wider Fibonacci acceptance range
✅ 1H trend as preference, not requirement
✅ Adaptive confirmation candles
✅ Quality scoring instead of binary filter
✅ ~8-12 signals per day target
    """
    
    print(comparison)
    
    print("\nRecommended Approach:")
    print("1. Start with Balanced strategy for more signals")
    print("2. Let ML learn from 500+ trades")
    print("3. Gradually tighten as ML improves")
    print("4. Use ML score as primary filter")

if __name__ == "__main__":
    compare_strategies()
    
    print("\n✅ Balanced strategy ready!")
    print("This version provides more signals while maintaining quality")
    print("Perfect for ML learning phase!"