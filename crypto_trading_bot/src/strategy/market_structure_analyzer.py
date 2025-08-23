"""
Market Structure Analyzer for Multi-Timeframe Trading
Identifies HH, HL, LH, LL patterns and structure breaks
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
from datetime import datetime
import structlog

logger = structlog.get_logger(__name__)

class StructureType(Enum):
    """Market structure patterns"""
    HIGHER_HIGH = "HH"
    HIGHER_LOW = "HL"
    LOWER_HIGH = "LH"
    LOWER_LOW = "LL"
    EQUAL_HIGH = "EH"
    EQUAL_LOW = "EL"

class TrendDirection(Enum):
    """Overall trend direction"""
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"
    TRANSITIONING = "transitioning"

@dataclass
class SwingPoint:
    """Represents a swing high or low"""
    price: float
    time: datetime
    type: str  # 'high' or 'low'
    strength: float  # 0-100 based on volume and candle pattern
    volume: float

@dataclass
class MarketStructure:
    """Current market structure state"""
    trend: TrendDirection
    last_high: SwingPoint
    last_low: SwingPoint
    structure_pattern: List[StructureType]  # Recent pattern sequence
    structure_break_level: Optional[float]  # Price level for structure break
    confidence: float  # 0-100 confidence in structure

@dataclass
class StructureSignal:
    """Signal generated from market structure"""
    type: str  # 'continuation', 'reversal', 'break'
    direction: str  # 'long' or 'short'
    entry_zone: Tuple[float, float]  # (lower, upper) price range
    stop_loss: float
    confidence: float
    pattern: List[StructureType]
    timeframe: str

class MarketStructureAnalyzer:
    """
    Analyzes market structure to identify trading opportunities
    based on HH/HL/LH/LL patterns and their breaks
    """
    
    def __init__(self):
        self.min_swing_strength = 2  # Reduced from 3 for more sensitive detection
        self.structure_lookback = 30  # Reduced from 50 for faster structure detection
        self.volume_threshold = 1.2  # Volume must be 1.2x average for strong swing
        
    def analyze_structure(
        self,
        df: pd.DataFrame,
        timeframe: str = '15m'
    ) -> MarketStructure:
        """
        Analyze market structure for a given timeframe
        """
        if df is None or len(df) < self.structure_lookback:
            return self._get_neutral_structure()
            
        # Find swing points
        swings = self._identify_swing_points(df)
        
        if len(swings['highs']) < 2 or len(swings['lows']) < 2:
            return self._get_neutral_structure()
        
        # Analyze structure patterns
        patterns = self._analyze_patterns(swings)
        
        # Determine trend
        trend = self._determine_trend(patterns)
        
        # Find structure break levels
        break_level = self._find_structure_break_level(swings, trend)
        
        # Calculate confidence
        confidence = self._calculate_structure_confidence(patterns, swings, df)
        
        return MarketStructure(
            trend=trend,
            last_high=swings['highs'][-1] if swings['highs'] else None,
            last_low=swings['lows'][-1] if swings['lows'] else None,
            structure_pattern=patterns[-5:] if patterns else [],  # Last 5 patterns
            structure_break_level=break_level,
            confidence=confidence
        )
    
    def _identify_swing_points(self, df: pd.DataFrame) -> Dict[str, List[SwingPoint]]:
        """
        Identify swing highs and lows using pivot points
        """
        highs = []
        lows = []
        
        # Calculate average volume for strength scoring
        avg_volume = df['volume'].rolling(20).mean().iloc[-1]
        
        for i in range(self.min_swing_strength, len(df) - self.min_swing_strength):
            # Check for swing high
            if self._is_swing_high(df, i):
                strength = self._calculate_swing_strength(df, i, 'high', avg_volume)
                highs.append(SwingPoint(
                    price=df.iloc[i]['high'],
                    time=df.index[i] if hasattr(df.index, '__iter__') else datetime.now(),
                    type='high',
                    strength=strength,
                    volume=df.iloc[i]['volume']
                ))
            
            # Check for swing low
            if self._is_swing_low(df, i):
                strength = self._calculate_swing_strength(df, i, 'low', avg_volume)
                lows.append(SwingPoint(
                    price=df.iloc[i]['low'],
                    time=df.index[i] if hasattr(df.index, '__iter__') else datetime.now(),
                    type='low',
                    strength=strength,
                    volume=df.iloc[i]['volume']
                ))
        
        return {'highs': highs, 'lows': lows}
    
    def _is_swing_high(self, df: pd.DataFrame, index: int) -> bool:
        """Check if a point is a swing high"""
        high = df.iloc[index]['high']
        
        # Check left side
        for i in range(1, self.min_swing_strength + 1):
            if df.iloc[index - i]['high'] >= high:
                return False
        
        # Check right side
        for i in range(1, min(self.min_swing_strength + 1, len(df) - index)):
            if df.iloc[index + i]['high'] >= high:
                return False
        
        return True
    
    def _is_swing_low(self, df: pd.DataFrame, index: int) -> bool:
        """Check if a point is a swing low"""
        low = df.iloc[index]['low']
        
        # Check left side
        for i in range(1, self.min_swing_strength + 1):
            if df.iloc[index - i]['low'] <= low:
                return False
        
        # Check right side
        for i in range(1, min(self.min_swing_strength + 1, len(df) - index)):
            if df.iloc[index + i]['low'] <= low:
                return False
        
        return True
    
    def _calculate_swing_strength(
        self,
        df: pd.DataFrame,
        index: int,
        swing_type: str,
        avg_volume: float
    ) -> float:
        """Calculate strength of a swing point (0-100)"""
        strength = 50.0  # Base strength
        
        # Volume factor
        if df.iloc[index]['volume'] > avg_volume * self.volume_threshold:
            strength += 20
        
        # Candle body size factor
        body_size = abs(df.iloc[index]['close'] - df.iloc[index]['open'])
        wick_size = df.iloc[index]['high'] - df.iloc[index]['low']
        if wick_size > 0:
            body_ratio = body_size / wick_size
            if swing_type == 'high' and body_ratio < 0.3:  # Long upper wick
                strength += 15
            elif swing_type == 'low' and body_ratio < 0.3:  # Long lower wick
                strength += 15
        
        # Rejection speed factor (how quickly price moved away)
        if index < len(df) - 2:
            next_move = abs(df.iloc[index + 1]['close'] - df.iloc[index]['close'])
            avg_move = df['close'].diff().abs().rolling(20).mean().iloc[index]
            if avg_move > 0 and next_move > avg_move * 2:
                strength += 15
        
        return min(100, strength)
    
    def _analyze_patterns(self, swings: Dict[str, List[SwingPoint]]) -> List[StructureType]:
        """Analyze swing points to identify HH/HL/LH/LL patterns"""
        patterns = []
        highs = swings['highs']
        lows = swings['lows']
        
        # Analyze high patterns
        for i in range(1, len(highs)):
            if highs[i].price > highs[i-1].price * 1.001:  # 0.1% threshold
                patterns.append(StructureType.HIGHER_HIGH)
            elif highs[i].price < highs[i-1].price * 0.999:
                patterns.append(StructureType.LOWER_HIGH)
            else:
                patterns.append(StructureType.EQUAL_HIGH)
        
        # Analyze low patterns
        for i in range(1, len(lows)):
            if lows[i].price > lows[i-1].price * 1.001:
                patterns.append(StructureType.HIGHER_LOW)
            elif lows[i].price < lows[i-1].price * 0.999:
                patterns.append(StructureType.LOWER_LOW)
            else:
                patterns.append(StructureType.EQUAL_LOW)
        
        return patterns
    
    def _determine_trend(self, patterns: List[StructureType]) -> TrendDirection:
        """Determine overall trend from pattern sequence"""
        if not patterns:
            return TrendDirection.NEUTRAL
        
        recent_patterns = patterns[-10:] if len(patterns) >= 10 else patterns
        
        bullish_count = sum(1 for p in recent_patterns if p in [
            StructureType.HIGHER_HIGH, StructureType.HIGHER_LOW
        ])
        bearish_count = sum(1 for p in recent_patterns if p in [
            StructureType.LOWER_HIGH, StructureType.LOWER_LOW
        ])
        
        total = len(recent_patterns)
        
        if bullish_count > total * 0.6:
            return TrendDirection.BULLISH
        elif bearish_count > total * 0.6:
            return TrendDirection.BEARISH
        elif bullish_count > 0 and bearish_count > 0:
            return TrendDirection.TRANSITIONING
        else:
            return TrendDirection.NEUTRAL
    
    def _find_structure_break_level(
        self,
        swings: Dict[str, List[SwingPoint]],
        trend: TrendDirection
    ) -> Optional[float]:
        """Find the price level that would confirm a structure break"""
        if trend == TrendDirection.BULLISH and swings['lows']:
            # In uptrend, break below recent higher low
            return swings['lows'][-1].price * 0.999  # Small buffer
        elif trend == TrendDirection.BEARISH and swings['highs']:
            # In downtrend, break above recent lower high
            return swings['highs'][-1].price * 1.001  # Small buffer
        return None
    
    def _calculate_structure_confidence(
        self,
        patterns: List[StructureType],
        swings: Dict[str, List[SwingPoint]],
        df: pd.DataFrame
    ) -> float:
        """Calculate confidence in current structure (0-100)"""
        confidence = 50.0
        
        # Pattern consistency
        if patterns:
            recent = patterns[-5:] if len(patterns) >= 5 else patterns
            if all(p in [StructureType.HIGHER_HIGH, StructureType.HIGHER_LOW] for p in recent):
                confidence += 25  # Strong bullish structure
            elif all(p in [StructureType.LOWER_HIGH, StructureType.LOWER_LOW] for p in recent):
                confidence += 25  # Strong bearish structure
        
        # Swing strength
        all_swings = swings['highs'] + swings['lows']
        if all_swings:
            avg_strength = sum(s.strength for s in all_swings[-5:]) / min(5, len(all_swings))
            confidence += (avg_strength - 50) * 0.5  # Add up to 25 points
        
        return min(100, max(0, confidence))
    
    def _get_neutral_structure(self) -> MarketStructure:
        """Return neutral market structure when analysis isn't possible"""
        return MarketStructure(
            trend=TrendDirection.NEUTRAL,
            last_high=None,
            last_low=None,
            structure_pattern=[],
            structure_break_level=None,
            confidence=0
        )
    
    def detect_entry_signal(
        self,
        structure: MarketStructure,
        current_price: float,
        supply_zone: Optional[Tuple[float, float]] = None,
        demand_zone: Optional[Tuple[float, float]] = None,
        timeframe: str = '15m'
    ) -> Optional[StructureSignal]:
        """
        Detect entry signals based on structure and zones
        """
        if structure.confidence < 40:  # Lowered from 60 for testing
            return None
        
        signal = None
        
        # Check for structure break signals
        if structure.structure_break_level:
            if structure.trend == TrendDirection.BULLISH:
                # Look for bullish continuation at demand zone
                if demand_zone and demand_zone[0] <= current_price <= demand_zone[1]:
                    if StructureType.HIGHER_LOW in structure.structure_pattern[-2:]:
                        signal = StructureSignal(
                            type='continuation',
                            direction='long',
                            entry_zone=(demand_zone[0], demand_zone[1]),
                            stop_loss=demand_zone[0] * 0.995,
                            confidence=structure.confidence,
                            pattern=structure.structure_pattern,
                            timeframe=timeframe
                        )
            
            elif structure.trend == TrendDirection.BEARISH:
                # Look for bearish continuation at supply zone
                if supply_zone and supply_zone[0] <= current_price <= supply_zone[1]:
                    if StructureType.LOWER_HIGH in structure.structure_pattern[-2:]:
                        signal = StructureSignal(
                            type='continuation',
                            direction='short',
                            entry_zone=(supply_zone[0], supply_zone[1]),
                            stop_loss=supply_zone[1] * 1.005,
                            confidence=structure.confidence,
                            pattern=structure.structure_pattern,
                            timeframe=timeframe
                        )
        
        # Check for reversal signals (structure breaks)
        if structure.trend == TrendDirection.TRANSITIONING:
            # Bullish reversal at demand zone
            if demand_zone and current_price <= demand_zone[1]:
                if (StructureType.HIGHER_LOW in structure.structure_pattern[-2:] and
                    StructureType.LOWER_LOW in structure.structure_pattern[-4:-2]):
                    signal = StructureSignal(
                        type='reversal',
                        direction='long',
                        entry_zone=(demand_zone[0], demand_zone[1]),
                        stop_loss=structure.last_low.price * 0.995 if structure.last_low else demand_zone[0] * 0.99,
                        confidence=structure.confidence * 0.9,  # Slightly higher confidence for testing
                        pattern=structure.structure_pattern,
                        timeframe=timeframe
                    )
            
            # Bearish reversal at supply zone
            elif supply_zone and current_price >= supply_zone[0]:
                if (StructureType.LOWER_HIGH in structure.structure_pattern[-2:] and
                    StructureType.HIGHER_HIGH in structure.structure_pattern[-4:-2]):
                    signal = StructureSignal(
                        type='reversal',
                        direction='short',
                        entry_zone=(supply_zone[0], supply_zone[1]),
                        stop_loss=structure.last_high.price * 1.005 if structure.last_high else supply_zone[1] * 1.01,
                        confidence=structure.confidence * 0.9,  # Slightly higher for testing
                        pattern=structure.structure_pattern,
                        timeframe=timeframe
                    )
        
        return signal