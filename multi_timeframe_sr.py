"""
Multi-Timeframe Support/Resistance Detection
Finds major S/R levels from higher timeframes for better pullback entries
"""
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class MultiTimeframeSR:
    """Detects support/resistance levels from multiple timeframes"""
    
    def __init__(self, timeframes: List[int] = [240, 60]):
        """
        Initialize with higher timeframes to analyze
        Default: 4H (240) and 1H (60) for 15min base timeframe
        """
        self.timeframes = timeframes
        self.sr_levels = {}  # symbol -> list of SR levels
        
    def aggregate_candles(self, df: pd.DataFrame, from_tf: int, to_tf: int) -> pd.DataFrame:
        """
        Aggregate lower timeframe candles to higher timeframe
        e.g., 15min -> 1H (4 candles) or 15min -> 4H (16 candles)
        """
        if to_tf % from_tf != 0:
            raise ValueError(f"Target timeframe {to_tf} must be multiple of source {from_tf}")
            
        factor = to_tf // from_tf
        
        # Resample to higher timeframe
        agg_df = df.resample(f'{to_tf}T').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()
        
        return agg_df
    
    def find_pivot_levels(self, df: pd.DataFrame, left: int = 5, right: int = 5, 
                         min_touches: int = 2, tolerance_pct: float = 0.002) -> List[Tuple[float, int, str]]:
        """
        Find significant pivot levels that have been tested multiple times
        Returns: [(level, strength, type), ...] where type is 'resistance' or 'support'
        """
        levels = []
        
        # Find pivot highs (resistance)
        for i in range(left, len(df) - right):
            pivot_high = df['high'].iloc[i]
            is_pivot = True
            
            # Check left side
            for j in range(i - left, i):
                if df['high'].iloc[j] >= pivot_high:
                    is_pivot = False
                    break
            
            # Check right side
            if is_pivot:
                for j in range(i + 1, i + right + 1):
                    if df['high'].iloc[j] > pivot_high:
                        is_pivot = False
                        break
            
            if is_pivot:
                # Count how many times this level was tested
                touches = self.count_level_touches(df, pivot_high, tolerance_pct, 'resistance')
                if touches >= min_touches:
                    levels.append((pivot_high, touches, 'resistance'))
        
        # Find pivot lows (support)
        for i in range(left, len(df) - right):
            pivot_low = df['low'].iloc[i]
            is_pivot = True
            
            # Check left side
            for j in range(i - left, i):
                if df['low'].iloc[j] <= pivot_low:
                    is_pivot = False
                    break
            
            # Check right side
            if is_pivot:
                for j in range(i + 1, i + right + 1):
                    if df['low'].iloc[j] < pivot_low:
                        is_pivot = False
                        break
            
            if is_pivot:
                # Count how many times this level was tested
                touches = self.count_level_touches(df, pivot_low, tolerance_pct, 'support')
                if touches >= min_touches:
                    levels.append((pivot_low, touches, 'support'))
        
        # Cluster nearby levels
        clustered_levels = self.cluster_levels(levels, tolerance_pct)
        
        # Sort by strength (number of touches)
        clustered_levels.sort(key=lambda x: x[1], reverse=True)
        
        return clustered_levels
    
    def count_level_touches(self, df: pd.DataFrame, level: float, 
                           tolerance_pct: float, level_type: str) -> int:
        """Count how many times a level has been tested"""
        tolerance = level * tolerance_pct
        touches = 0
        
        if level_type == 'resistance':
            # Count highs that came close to level
            mask = (df['high'] >= level - tolerance) & (df['high'] <= level + tolerance)
            touches = mask.sum()
        else:  # support
            # Count lows that came close to level
            mask = (df['low'] >= level - tolerance) & (df['low'] <= level + tolerance)
            touches = mask.sum()
        
        return touches
    
    def cluster_levels(self, levels: List[Tuple[float, int, str]], 
                      tolerance_pct: float) -> List[Tuple[float, int, str]]:
        """Cluster nearby levels together - keeping support and resistance separate"""
        if not levels:
            return []
        
        clustered = []
        used = set()
        
        for i, (level1, touches1, type1) in enumerate(levels):
            if i in used:
                continue
            
            cluster_levels = [(level1, touches1)]
            cluster_type = type1
            
            # Find all levels within tolerance OF THE SAME TYPE
            for j, (level2, touches2, type2) in enumerate(levels):
                if j != i and j not in used:
                    # Only cluster if same type (support with support, resistance with resistance)
                    if type1 == type2 and abs(level2 - level1) / level1 <= tolerance_pct:
                        cluster_levels.append((level2, touches2))
                        used.add(j)
            
            # Use weighted average for cluster level
            total_touches = sum(t for _, t in cluster_levels)
            avg_level = sum(l * t for l, t in cluster_levels) / total_touches
            
            clustered.append((avg_level, total_touches, cluster_type))
        
        return clustered
    
    def get_nearest_levels(self, symbol: str, current_price: float, 
                          above_count: int = 3, below_count: int = 3) -> Dict[str, List[float]]:
        """
        Get nearest resistance levels above and support levels below current price
        """
        if symbol not in self.sr_levels:
            return {'resistance': [], 'support': []}
        
        all_levels = self.sr_levels[symbol]
        
        # Separate and filter levels
        resistance_levels = sorted([l for l, s, t in all_levels 
                                  if t == 'resistance' and l > current_price])
        support_levels = sorted([l for l, s, t in all_levels 
                               if t == 'support' and l < current_price], reverse=True)
        
        return {
            'resistance': resistance_levels[:above_count],
            'support': support_levels[:below_count]
        }
    
    def update_sr_levels(self, symbol: str, df_15min: pd.DataFrame):
        """
        Update support/resistance levels for a symbol using multiple timeframes
        """
        all_levels = []
        
        # Get 15min pivots (minor levels)
        minor_levels = self.find_pivot_levels(df_15min, left=10, right=10, 
                                            min_touches=3, tolerance_pct=0.001)
        # Add with lower weight
        all_levels.extend([(l, s * 0.5, t) for l, s, t in minor_levels])
        
        # Get higher timeframe levels (major levels)
        for tf in self.timeframes:
            try:
                # Aggregate to higher timeframe
                df_htf = self.aggregate_candles(df_15min, 15, tf)
                
                if len(df_htf) < 50:  # Need enough data
                    continue
                
                # Find pivots on higher timeframe
                if tf == 60:  # 1H
                    htf_levels = self.find_pivot_levels(df_htf, left=5, right=5,
                                                      min_touches=2, tolerance_pct=0.002)
                    # Medium weight for 1H levels
                    all_levels.extend([(l, s * 1.5, t) for l, s, t in htf_levels])
                    
                elif tf == 240:  # 4H
                    htf_levels = self.find_pivot_levels(df_htf, left=3, right=3,
                                                      min_touches=2, tolerance_pct=0.003)
                    # High weight for 4H levels
                    all_levels.extend([(l, s * 3.0, t) for l, s, t in htf_levels])
                    
            except Exception as e:
                logger.warning(f"Failed to analyze {tf}min timeframe for {symbol}: {e}")
        
        # Cluster all levels across timeframes
        self.sr_levels[symbol] = self.cluster_levels(all_levels, tolerance_pct=0.002)
        
        # Keep only top 20 levels by strength
        self.sr_levels[symbol] = sorted(self.sr_levels[symbol], 
                                      key=lambda x: x[1], reverse=True)[:20]
        
        logger.info(f"Updated S/R for {symbol}: {len(self.sr_levels[symbol])} levels found")
    
    def is_near_major_level(self, symbol: str, price: float, 
                           tolerance_pct: float = 0.002) -> Tuple[bool, float, str]:
        """
        Check if price is near a major S/R level
        Returns: (is_near, level, level_type)
        """
        if symbol not in self.sr_levels:
            return False, 0.0, ""
        
        tolerance = price * tolerance_pct
        
        for level, strength, level_type in self.sr_levels[symbol]:
            if abs(price - level) <= tolerance:
                # Higher strength = more major level
                if strength >= 3.0:  # Threshold for "major"
                    return True, level, level_type
        
        return False, 0.0, ""
    
    def get_level_strength(self, symbol: str, level: float, 
                          tolerance_pct: float = 0.002) -> float:
        """Get the strength score of a specific level"""
        if symbol not in self.sr_levels:
            return 0.0
        
        tolerance = level * tolerance_pct
        
        for l, strength, _ in self.sr_levels[symbol]:
            if abs(l - level) <= tolerance:
                return strength
        
        return 0.0


# Global instance for the strategy to use
mtf_sr = MultiTimeframeSR(timeframes=[240, 60])  # 4H and 1H


def should_use_mtf_level(symbol: str, breakout_level: float, 
                        current_price: float, df: pd.DataFrame) -> Tuple[bool, float, str]:
    """
    Determine if we should use MTF level instead of recent pivot
    Returns: (use_mtf, level, reason)
    """
    # Update MTF levels periodically (every 100 candles)
    if len(df) % 100 == 0:
        mtf_sr.update_sr_levels(symbol, df)
    
    # Check if breakout level is near a major level
    is_major, major_level, level_type = mtf_sr.is_near_major_level(symbol, breakout_level)
    
    if is_major:
        strength = mtf_sr.get_level_strength(symbol, major_level)
        return True, major_level, f"Major {level_type} (strength: {strength:.1f})"
    
    # Get nearest major levels
    nearest = mtf_sr.get_nearest_levels(symbol, current_price, above_count=1, below_count=1)
    
    # Check if we're very close to a major resistance (must be above current price)
    if nearest['resistance'] and len(nearest['resistance']) > 0:
        resistance_level = nearest['resistance'][0]
        # Validate resistance is actually above price
        if resistance_level > current_price and abs(current_price - resistance_level) / current_price < 0.005:
            strength = mtf_sr.get_level_strength(symbol, resistance_level)
            return True, resistance_level, f"Near major resistance (strength: {strength:.1f})"
    
    # Check if we're very close to a major support (must be below current price)
    if nearest['support'] and len(nearest['support']) > 0:
        support_level = nearest['support'][0]
        # Validate support is actually below price
        if support_level < current_price and abs(current_price - support_level) / current_price < 0.005:
            strength = mtf_sr.get_level_strength(symbol, support_level)
            return True, support_level, f"Near major support (strength: {strength:.1f})"
    
    return False, 0.0, ""