"""
Wyckoff Method Pattern Detection
Identifies accumulation/distribution phases and spring patterns
Used by institutional traders to identify smart money movements
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from datetime import datetime, timedelta
import structlog

logger = structlog.get_logger(__name__)

class WyckoffPhase(Enum):
    """Wyckoff market phases"""
    # Accumulation phases
    PHASE_A_ACC = "phase_a_accumulation"  # Stopping of downtrend
    PHASE_B_ACC = "phase_b_accumulation"  # Building a cause
    PHASE_C_ACC = "phase_c_accumulation"  # Spring test
    PHASE_D_ACC = "phase_d_accumulation"  # Sign of strength
    PHASE_E_ACC = "phase_e_accumulation"  # Markup begins
    
    # Distribution phases
    PHASE_A_DIST = "phase_a_distribution"  # Stopping of uptrend
    PHASE_B_DIST = "phase_b_distribution"  # Building a cause
    PHASE_C_DIST = "phase_c_distribution"  # UTAD test
    PHASE_D_DIST = "phase_d_distribution"  # Sign of weakness
    PHASE_E_DIST = "phase_e_distribution"  # Markdown begins
    
    # Neutral
    RANGING = "ranging"
    TRENDING = "trending"
    UNKNOWN = "unknown"

class WyckoffEvent(Enum):
    """Key Wyckoff events"""
    # Accumulation events
    PS = "preliminary_support"      # First support after decline
    SC = "selling_climax"           # Heavy selling, high volume
    AR = "automatic_rally"          # Bounce after SC
    ST = "secondary_test"           # Test of SC low
    SPRING = "spring"               # False breakdown below support
    TEST = "test"                   # Test after spring
    SOS = "sign_of_strength"        # Strong move up
    LPS = "last_point_support"      # Final pullback before markup
    BU = "backup"                   # Pullback to support
    
    # Distribution events
    PSY = "preliminary_supply"       # First resistance after advance
    BC = "buying_climax"            # Heavy buying, high volume
    AR_DIST = "automatic_reaction"  # Drop after BC
    ST_DIST = "secondary_test_dist" # Test of BC high
    UTAD = "upthrust_after_distribution"  # False breakout above resistance
    SOW = "sign_of_weakness"        # Strong move down
    LPSY = "last_point_supply"      # Final rally before markdown

@dataclass
class WyckoffPattern:
    """Detected Wyckoff pattern"""
    phase: WyckoffPhase
    events: List[WyckoffEvent]
    trading_range: Tuple[float, float]  # Support and resistance
    volume_profile: Dict
    strength_score: float  # 0-100
    spring_detected: bool
    spring_location: Optional[float]
    accumulation_time: int  # Bars in range
    institutional_activity: float  # 0-100
    breakout_target: Optional[float]
    confidence: float  # 0-100

class WyckoffDetector:
    """
    Detects Wyckoff accumulation/distribution patterns
    Identifies institutional trading activity
    """
    
    def __init__(self):
        self.min_range_bars = 20  # Minimum bars for valid range
        self.volume_threshold = 1.5  # Volume spike threshold
        self.spring_penetration = 0.03  # 3% below support for spring
        self.institutional_volume_ratio = 2.0  # Institutional activity threshold
        
    def analyze(self, df: pd.DataFrame) -> Dict:
        """
        Main analysis method for Wyckoff patterns
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            Dictionary with detected patterns and signals
        """
        results = {
            'phase': WyckoffPhase.UNKNOWN,
            'patterns': [],
            'events': [],
            'trading_signals': [],
            'institutional_zones': []
        }
        
        if len(df) < self.min_range_bars * 2:
            return results
        
        try:
            # Add technical indicators
            df = self._add_indicators(df)
            
            # Identify trading ranges
            ranges = self._identify_trading_ranges(df)
            
            for range_data in ranges:
                # Analyze each range for Wyckoff patterns
                pattern = self._analyze_range(df, range_data)
                
                if pattern:
                    results['patterns'].append(pattern)
                    
                    # Generate trading signals
                    signals = self._generate_signals(df, pattern)
                    results['trading_signals'].extend(signals)
                    
                    # Identify institutional zones
                    zones = self._identify_institutional_zones(df, pattern)
                    results['institutional_zones'].extend(zones)
            
            # Determine current phase
            if results['patterns']:
                results['phase'] = results['patterns'][-1].phase
                results['events'] = results['patterns'][-1].events
            
        except Exception as e:
            logger.error(f"Wyckoff analysis error: {e}")
        
        return results
    
    def _add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators for Wyckoff analysis"""
        
        # Volume analysis
        df['volume_ma'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma']
        
        # Price analysis
        df['atr'] = self._calculate_atr(df, 14)
        df['price_ma'] = df['close'].rolling(window=20).mean()
        
        # Volatility
        df['volatility'] = df['close'].rolling(window=20).std()
        
        # Cumulative volume delta (simplified)
        df['volume_delta'] = df.apply(
            lambda x: x['volume'] if x['close'] > x['open'] else -x['volume'],
            axis=1
        )
        df['cum_volume_delta'] = df['volume_delta'].cumsum()
        
        return df
    
    def _calculate_atr(self, df: pd.DataFrame, period: int) -> pd.Series:
        """Calculate Average True Range"""
        high_low = df['high'] - df['low']
        high_close = abs(df['high'] - df['close'].shift())
        low_close = abs(df['low'] - df['close'].shift())
        
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        
        return true_range.rolling(window=period).mean()
    
    def _identify_trading_ranges(self, df: pd.DataFrame) -> List[Dict]:
        """Identify potential accumulation/distribution ranges"""
        
        ranges = []
        window = self.min_range_bars
        
        for i in range(window, len(df) - window, window // 2):
            segment = df.iloc[i-window:i+window]
            
            # Calculate range metrics
            high = segment['high'].max()
            low = segment['low'].min()
            range_size = (high - low) / low
            
            # Check if price is ranging (not trending strongly)
            price_std = segment['close'].std()
            price_mean = segment['close'].mean()
            cv = price_std / price_mean  # Coefficient of variation
            
            if cv < 0.1 and range_size < 0.3:  # Ranging market
                # Check for volume characteristics
                avg_volume = segment['volume'].mean()
                volume_spikes = len(segment[segment['volume_ratio'] > self.volume_threshold])
                
                range_data = {
                    'start_idx': i - window,
                    'end_idx': i + window,
                    'high': high,
                    'low': low,
                    'range_size': range_size,
                    'avg_volume': avg_volume,
                    'volume_spikes': volume_spikes,
                    'duration': window * 2
                }
                
                ranges.append(range_data)
        
        return ranges
    
    def _analyze_range(self, df: pd.DataFrame, range_data: Dict) -> Optional[WyckoffPattern]:
        """Analyze a trading range for Wyckoff patterns"""
        
        start_idx = range_data['start_idx']
        end_idx = min(range_data['end_idx'], len(df) - 1)
        segment = df.iloc[start_idx:end_idx]
        
        if len(segment) < self.min_range_bars:
            return None
        
        # Identify key levels
        support = range_data['low']
        resistance = range_data['high']
        mid_point = (support + resistance) / 2
        
        # Detect events
        events = []
        
        # Look for selling climax (high volume at lows)
        low_points = segment[segment['low'] <= support * 1.01]
        if not low_points.empty:
            max_vol_at_low = low_points['volume'].max()
            if max_vol_at_low > segment['volume'].mean() * self.volume_threshold:
                events.append(WyckoffEvent.SC)
        
        # Look for automatic rally
        if WyckoffEvent.SC in events:
            rally_after_sc = segment[segment.index > low_points.index[0]]
            if not rally_after_sc.empty:
                if rally_after_sc['high'].max() > support * 1.05:
                    events.append(WyckoffEvent.AR)
        
        # Detect spring pattern
        spring_detected, spring_location = self._detect_spring(segment, support)
        if spring_detected:
            events.append(WyckoffEvent.SPRING)
        
        # Detect accumulation vs distribution
        phase = self._determine_phase(segment, events, support, resistance)
        
        # Calculate institutional activity
        institutional_activity = self._calculate_institutional_activity(segment)
        
        # Calculate pattern strength
        strength_score = self._calculate_pattern_strength(
            events, 
            spring_detected,
            institutional_activity,
            range_data['volume_spikes']
        )
        
        # Calculate breakout target
        breakout_target = None
        if phase in [WyckoffPhase.PHASE_D_ACC, WyckoffPhase.PHASE_E_ACC]:
            breakout_target = resistance + (resistance - support)
        elif phase in [WyckoffPhase.PHASE_D_DIST, WyckoffPhase.PHASE_E_DIST]:
            breakout_target = support - (resistance - support)
        
        # Create pattern object
        pattern = WyckoffPattern(
            phase=phase,
            events=events,
            trading_range=(support, resistance),
            volume_profile=self._create_volume_profile(segment),
            strength_score=strength_score,
            spring_detected=spring_detected,
            spring_location=spring_location,
            accumulation_time=len(segment),
            institutional_activity=institutional_activity,
            breakout_target=breakout_target,
            confidence=min(100, strength_score * 1.2)
        )
        
        return pattern
    
    def _detect_spring(self, segment: pd.DataFrame, support: float) -> Tuple[bool, Optional[float]]:
        """
        Detect spring pattern (false breakdown below support)
        
        The spring is a critical Wyckoff pattern where price briefly breaks
        below support to trigger stop losses before reversing sharply
        """
        
        # Look for penetration below support
        below_support = segment[segment['low'] < support * (1 - self.spring_penetration)]
        
        if below_support.empty:
            return False, None
        
        for idx in below_support.index:
            # Check for quick reversal
            after_break = segment[segment.index > idx].head(5)
            
            if not after_break.empty:
                # Check if price quickly recovered above support
                recovery = after_break[after_break['close'] > support]
                
                if not recovery.empty:
                    # Check for volume confirmation
                    break_volume = segment.loc[idx, 'volume']
                    recovery_volume = recovery['volume'].mean()
                    
                    if recovery_volume > segment['volume'].mean():
                        # Spring detected
                        spring_location = segment.loc[idx, 'low']
                        return True, spring_location
        
        return False, None
    
    def _determine_phase(
        self, 
        segment: pd.DataFrame, 
        events: List[WyckoffEvent],
        support: float,
        resistance: float
    ) -> WyckoffPhase:
        """Determine current Wyckoff phase"""
        
        # Check price position relative to range
        current_price = segment['close'].iloc[-1]
        range_position = (current_price - support) / (resistance - support)
        
        # Check volume trend
        first_half_vol = segment.iloc[:len(segment)//2]['volume'].mean()
        second_half_vol = segment.iloc[len(segment)//2:]['volume'].mean()
        volume_increasing = second_half_vol > first_half_vol
        
        # Check cumulative volume delta trend
        cvd_trend = segment['cum_volume_delta'].iloc[-1] > segment['cum_volume_delta'].iloc[0]
        
        # Accumulation phases
        if WyckoffEvent.SC in events:
            if WyckoffEvent.SPRING in events:
                if range_position > 0.6:
                    return WyckoffPhase.PHASE_D_ACC
                else:
                    return WyckoffPhase.PHASE_C_ACC
            elif WyckoffEvent.AR in events:
                return WyckoffPhase.PHASE_B_ACC
            else:
                return WyckoffPhase.PHASE_A_ACC
        
        # Distribution phases (simplified)
        if range_position > 0.8 and not cvd_trend:
            if volume_increasing:
                return WyckoffPhase.PHASE_A_DIST
            else:
                return WyckoffPhase.PHASE_B_DIST
        
        # Default to ranging
        return WyckoffPhase.RANGING
    
    def _calculate_institutional_activity(self, segment: pd.DataFrame) -> float:
        """
        Calculate institutional activity score (0-100)
        
        Institutional traders leave specific footprints:
        - Large volume at key levels
        - Absorption of selling/buying
        - Tests on low volume
        """
        
        score = 0.0
        
        # Check for high volume at extremes
        high_volume_bars = segment[segment['volume_ratio'] > self.institutional_volume_ratio]
        if len(high_volume_bars) > 0:
            score += 20
        
        # Check for absorption (high volume, small price movement)
        absorption_bars = segment[
            (segment['volume_ratio'] > 1.5) & 
            (abs(segment['close'] - segment['open']) / segment['open'] < 0.005)
        ]
        if len(absorption_bars) > 2:
            score += 30
        
        # Check for tests on low volume
        low_volume_tests = segment[
            (segment['volume_ratio'] < 0.5) &
            ((segment['low'] == segment['low'].min()) | 
             (segment['high'] == segment['high'].max()))
        ]
        if len(low_volume_tests) > 0:
            score += 25
        
        # Check for consistent accumulation/distribution
        cvd_slope = np.polyfit(range(len(segment)), segment['cum_volume_delta'], 1)[0]
        if abs(cvd_slope) > segment['cum_volume_delta'].std() * 0.1:
            score += 25
        
        return min(100, score)
    
    def _calculate_pattern_strength(
        self,
        events: List[WyckoffEvent],
        spring_detected: bool,
        institutional_activity: float,
        volume_spikes: int
    ) -> float:
        """Calculate overall pattern strength (0-100)"""
        
        score = 0.0
        
        # Event-based scoring
        if WyckoffEvent.SC in events:
            score += 15
        if WyckoffEvent.AR in events:
            score += 10
        if WyckoffEvent.SPRING in events:
            score += 25
        if spring_detected:
            score += 20
        
        # Institutional activity
        score += institutional_activity * 0.2
        
        # Volume characteristics
        if volume_spikes >= 3:
            score += 10
        
        return min(100, score)
    
    def _create_volume_profile(self, segment: pd.DataFrame) -> Dict:
        """Create volume profile for the range"""
        
        price_bins = 20
        min_price = segment['low'].min()
        max_price = segment['high'].max()
        
        bins = np.linspace(min_price, max_price, price_bins)
        volume_profile = {}
        
        for i in range(len(bins) - 1):
            mask = (segment['close'] >= bins[i]) & (segment['close'] < bins[i + 1])
            volume_profile[f"{bins[i]:.2f}-{bins[i+1]:.2f}"] = segment[mask]['volume'].sum()
        
        return volume_profile
    
    def _generate_signals(self, df: pd.DataFrame, pattern: WyckoffPattern) -> List[Dict]:
        """Generate trading signals based on Wyckoff pattern"""
        
        signals = []
        
        # Long signal for accumulation with spring
        if pattern.spring_detected and pattern.phase in [
            WyckoffPhase.PHASE_C_ACC,
            WyckoffPhase.PHASE_D_ACC
        ]:
            signal = {
                'type': 'BUY',
                'pattern': 'wyckoff_spring',
                'entry': pattern.trading_range[0] * 1.01,  # Above support
                'stop_loss': pattern.spring_location * 0.99 if pattern.spring_location else pattern.trading_range[0] * 0.97,
                'target': pattern.breakout_target or pattern.trading_range[1],
                'confidence': pattern.confidence,
                'reason': f"Wyckoff Spring in {pattern.phase.value}"
            }
            signals.append(signal)
        
        # Long signal for sign of strength
        if WyckoffEvent.SOS in pattern.events:
            signal = {
                'type': 'BUY',
                'pattern': 'wyckoff_sos',
                'entry': df['close'].iloc[-1],
                'stop_loss': pattern.trading_range[0],
                'target': pattern.breakout_target or pattern.trading_range[1] * 1.1,
                'confidence': pattern.confidence * 0.9,
                'reason': "Wyckoff Sign of Strength"
            }
            signals.append(signal)
        
        # Short signal for distribution
        if pattern.phase in [WyckoffPhase.PHASE_D_DIST, WyckoffPhase.PHASE_E_DIST]:
            signal = {
                'type': 'SELL',
                'pattern': 'wyckoff_distribution',
                'entry': pattern.trading_range[1] * 0.99,  # Below resistance
                'stop_loss': pattern.trading_range[1] * 1.02,
                'target': pattern.breakout_target or pattern.trading_range[0],
                'confidence': pattern.confidence * 0.85,
                'reason': f"Wyckoff Distribution {pattern.phase.value}"
            }
            signals.append(signal)
        
        return signals
    
    def _identify_institutional_zones(self, df: pd.DataFrame, pattern: WyckoffPattern) -> List[Dict]:
        """Identify zones of institutional interest"""
        
        zones = []
        
        if pattern.institutional_activity > 70:
            # Create supply/demand zone based on pattern
            if pattern.phase in [WyckoffPhase.PHASE_C_ACC, WyckoffPhase.PHASE_D_ACC]:
                zone = {
                    'type': 'demand',
                    'upper': pattern.trading_range[0] * 1.02,
                    'lower': pattern.spring_location if pattern.spring_location else pattern.trading_range[0] * 0.98,
                    'strength': pattern.institutional_activity,
                    'pattern': 'wyckoff_accumulation'
                }
                zones.append(zone)
            
            elif pattern.phase in [WyckoffPhase.PHASE_C_DIST, WyckoffPhase.PHASE_D_DIST]:
                zone = {
                    'type': 'supply',
                    'upper': pattern.trading_range[1] * 1.02,
                    'lower': pattern.trading_range[1] * 0.98,
                    'strength': pattern.institutional_activity,
                    'pattern': 'wyckoff_distribution'
                }
                zones.append(zone)
        
        return zones