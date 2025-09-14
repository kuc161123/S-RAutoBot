"""
Dynamic Risk/Reward Calculator
Adjusts R:R ratio based on market conditions while maintaining safety
Designed to be non-breaking with fallback to base R:R
"""
import logging
import numpy as np
from typing import Dict, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

class DynamicRRCalculator:
    """
    Calculates dynamic R:R based on:
    - Volatility regime and ATR percentile
    - Trend strength and alignment
    - Trading session and time factors
    - Recent performance metrics
    """
    
    def __init__(self, base_rr: float = 2.5, enabled: bool = False):
        self.base_rr = base_rr
        self.enabled = enabled
        
        # Safety bounds
        self.min_rr = 1.5  # Never go below 1.5:1
        self.max_rr = 4.0  # Never go above 4:1
        
        # Adjustment weights
        self.volatility_weight = 0.4
        self.trend_weight = 0.4
        self.session_weight = 0.2
        
        # Performance tracking
        self.recent_adjustments = []
        self.performance_by_rr = {}
        
        logger.info(f"Dynamic RR Calculator initialized: {'ENABLED' if enabled else 'DISABLED'}")
    
    def calculate_rr(self, features: Dict[str, any], symbol: str = "") -> float:
        """
        Calculate dynamic R:R based on market conditions
        Always returns base_rr if disabled or on error
        """
        # Safety: return base if disabled
        if not self.enabled:
            return self.base_rr
        
        try:
            # Get adjustment factors
            vol_factor = self._volatility_adjustment(features)
            trend_factor = self._trend_strength_adjustment(features)
            session_factor = self._session_adjustment(features)
            
            # Log individual factors for analysis
            logger.debug(f"[{symbol}] RR Factors - Vol: {vol_factor:.2f}, "
                        f"Trend: {trend_factor:.2f}, Session: {session_factor:.2f}")
            
            # Weighted combination
            weighted_adjustment = (
                vol_factor * self.volatility_weight +
                trend_factor * self.trend_weight +
                session_factor * self.session_weight
            )
            
            # Apply to base R:R
            dynamic_rr = self.base_rr * weighted_adjustment
            
            # Safety: clamp to reasonable bounds
            final_rr = max(self.min_rr, min(self.max_rr, dynamic_rr))
            
            # Log significant adjustments
            if abs(final_rr - self.base_rr) > 0.2:
                logger.info(f"[{symbol}] Dynamic RR: {final_rr:.2f} "
                           f"(base: {self.base_rr}, adjustment: {weighted_adjustment:.2f})")
            
            # Track for analysis
            self._track_adjustment(symbol, final_rr, features)
            
            return final_rr
            
        except Exception as e:
            logger.error(f"Dynamic RR calculation failed: {e}")
            return self.base_rr  # Safety fallback
    
    def _volatility_adjustment(self, features: Dict) -> float:
        """
        Adjust based on volatility
        High volatility = lower R:R (more conservative)
        Low volatility = higher R:R (capitalize on stability)
        """
        volatility_regime = features.get('volatility_regime', 'normal')
        atr_percentile = features.get('atr_percentile', 50)
        
        # Base adjustment from regime
        regime_factors = {
            'low': 1.2,      # +20% in low volatility
            'normal': 1.0,   # No change
            'high': 0.85,    # -15% in high volatility
            'extreme': 0.7   # -30% in extreme volatility
        }
        
        base_factor = regime_factors.get(volatility_regime, 1.0)
        
        # Fine-tune with ATR percentile
        if atr_percentile > 80:
            # Very high volatility
            percentile_factor = 0.9
        elif atr_percentile > 60:
            # Elevated volatility
            percentile_factor = 0.95
        elif atr_percentile < 20:
            # Very low volatility
            percentile_factor = 1.1
        elif atr_percentile < 40:
            # Low volatility
            percentile_factor = 1.05
        else:
            # Normal
            percentile_factor = 1.0
        
        return base_factor * percentile_factor
    
    def _trend_strength_adjustment(self, features: Dict) -> float:
        """
        Adjust based on trend strength
        Strong trend = higher R:R (ride the trend)
        Weak trend = lower R:R (be conservative)
        """
        trend_strength = features.get('trend_strength', 50)
        higher_tf_alignment = features.get('higher_tf_alignment', 50)
        
        # Normalize to 0-1
        trend_normalized = trend_strength / 100.0
        alignment_normalized = higher_tf_alignment / 100.0
        
        # Combined trend score
        trend_score = (trend_normalized * 0.6 + alignment_normalized * 0.4)
        
        # Convert to adjustment factor
        # Strong trend (>70) = up to +20%
        # Weak trend (<30) = up to -20%
        if trend_score > 0.7:
            factor = 1.0 + (trend_score - 0.7) * 0.67  # Max 1.2
        elif trend_score < 0.3:
            factor = 0.8 + trend_score * 0.67  # Min 0.8
        else:
            # Linear interpolation in middle range
            factor = 0.8 + (trend_score - 0.3) * 0.5
        
        return factor
    
    def _session_adjustment(self, features: Dict) -> float:
        """
        Adjust based on trading session
        Prime sessions = normal to higher R:R
        Off hours = lower R:R
        """
        session = features.get('session', 'us')
        hour_of_day = features.get('hour_of_day', 12)
        day_of_week = features.get('day_of_week', 3)
        
        # Session factors
        session_factors = {
            'asian': 0.95,     # Slightly lower
            'european': 1.0,   # Normal
            'us': 1.05,        # Slightly higher (most liquid)
            'off_hours': 0.9   # Lower in off hours
        }
        
        base_factor = session_factors.get(session, 1.0)
        
        # Weekend adjustment
        if day_of_week >= 5:  # Saturday/Sunday
            base_factor *= 0.95
        
        # Extreme hours adjustment (very early/late)
        if hour_of_day < 2 or hour_of_day > 22:
            base_factor *= 0.95
        
        return base_factor
    
    def _track_adjustment(self, symbol: str, rr: float, features: Dict):
        """Track adjustments for performance analysis"""
        self.recent_adjustments.append({
            'timestamp': datetime.now(),
            'symbol': symbol,
            'rr': rr,
            'base_rr': self.base_rr,
            'adjustment': rr / self.base_rr,
            'volatility_regime': features.get('volatility_regime'),
            'trend_strength': features.get('trend_strength'),
            'session': features.get('session')
        })
        
        # Keep only last 100 adjustments
        if len(self.recent_adjustments) > 100:
            self.recent_adjustments.pop(0)
    
    def update_performance(self, rr_used: float, outcome: str, pnl_r: float):
        """
        Track performance by R:R level for future optimization
        Called after trade completion
        """
        # Round R:R to nearest 0.1 for bucketing
        rr_bucket = round(rr_used, 1)
        
        if rr_bucket not in self.performance_by_rr:
            self.performance_by_rr[rr_bucket] = {
                'trades': 0,
                'wins': 0,
                'total_r': 0.0
            }
        
        stats = self.performance_by_rr[rr_bucket]
        stats['trades'] += 1
        if outcome == 'win':
            stats['wins'] += 1
        stats['total_r'] += pnl_r
        
        # Log performance update
        win_rate = stats['wins'] / stats['trades']
        avg_r = stats['total_r'] / stats['trades']
        logger.info(f"RR {rr_bucket} performance: {stats['trades']} trades, "
                   f"{win_rate:.1%} win rate, {avg_r:.2f}R average")
    
    def get_stats(self) -> Dict:
        """Get dynamic RR statistics"""
        if not self.recent_adjustments:
            return {
                'enabled': self.enabled,
                'base_rr': self.base_rr,
                'average_rr': self.base_rr,
                'adjustment_range': [1.0, 1.0],
                'total_adjustments': 0
            }
        
        recent_rrs = [adj['rr'] for adj in self.recent_adjustments[-20:]]
        
        return {
            'enabled': self.enabled,
            'base_rr': self.base_rr,
            'average_rr': np.mean(recent_rrs),
            'adjustment_range': [min(recent_rrs), max(recent_rrs)],
            'total_adjustments': len(self.recent_adjustments),
            'performance_by_rr': self.performance_by_rr
        }

# Global instance
_rr_calculator = None

def get_dynamic_rr_calculator(base_rr: float = 2.5, enabled: bool = False) -> DynamicRRCalculator:
    """Get or create the global dynamic RR calculator"""
    global _rr_calculator
    if _rr_calculator is None:
        _rr_calculator = DynamicRRCalculator(base_rr=base_rr, enabled=enabled)
    return _rr_calculator