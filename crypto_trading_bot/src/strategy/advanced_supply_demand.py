"""
Advanced Supply & Demand Strategy with Order Flow and Volume Profile
Enhanced with institutional trading patterns and machine learning
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from datetime import datetime, timedelta
import structlog
from scipy import stats
from collections import deque

logger = structlog.get_logger(__name__)

class MarketStructure(Enum):
    """Market structure types"""
    BULLISH = "bullish"
    BEARISH = "bearish"
    RANGING = "ranging"
    TRANSITIONING = "transitioning"

class OrderFlowImbalance(Enum):
    """Order flow imbalance types"""
    STRONG_BUYING = "strong_buying"
    BUYING = "buying"
    NEUTRAL = "neutral"
    SELLING = "selling"
    STRONG_SELLING = "strong_selling"

@dataclass
class VolumeProfile:
    """Volume profile data for a price range"""
    price_levels: List[float]
    volumes: List[float]
    poc: float  # Point of Control (highest volume price)
    vah: float  # Value Area High
    val: float  # Value Area Low
    total_volume: float
    buying_volume: float
    selling_volume: float
    delta: float  # Buying - Selling volume

@dataclass
class EnhancedZone:
    """Enhanced supply/demand zone with additional metrics"""
    zone_type: str  # 'supply' or 'demand'
    upper_bound: float
    lower_bound: float
    strength_score: float  # 0-100
    volume_profile: VolumeProfile
    order_flow_imbalance: OrderFlowImbalance
    formation_time: datetime
    test_count: int
    rejection_strength: float  # Speed of price rejection
    institutional_interest: float  # 0-100
    confluence_factors: List[str]
    timeframes_visible: List[str]
    last_test_time: Optional[datetime]
    zone_age_hours: float
    liquidity_pool: float  # Estimated liquidity
    
    @property
    def is_fresh(self) -> bool:
        """Check if zone is fresh (untested)"""
        return self.test_count == 0
    
    @property
    def composite_score(self) -> float:
        """Calculate composite score combining all factors"""
        base_score = self.strength_score
        
        # Boost for fresh zones
        if self.is_fresh:
            base_score *= 1.2
        
        # Boost for institutional interest
        base_score += self.institutional_interest * 0.3
        
        # Boost for order flow imbalance
        if self.order_flow_imbalance in [OrderFlowImbalance.STRONG_BUYING, OrderFlowImbalance.STRONG_SELLING]:
            base_score *= 1.15
        
        # Boost for multiple timeframe confluence
        base_score += len(self.timeframes_visible) * 5
        
        # Penalty for old zones
        if self.zone_age_hours > 168:  # 7 days
            base_score *= 0.8
        
        # Penalty for multiple tests
        base_score -= self.test_count * 10
        
        return min(100, max(0, base_score))

class AdvancedSupplyDemandStrategy:
    """Enhanced Supply & Demand strategy with advanced features"""
    
    def __init__(self):
        self.zones: Dict[str, List[EnhancedZone]] = {}
        self.market_structure: Dict[str, MarketStructure] = {}
        self.volume_profiles: Dict[str, VolumeProfile] = {}
        self.order_flow_history: Dict[str, deque] = {}
        
        # Configuration
        self.min_zone_score = 65  # Minimum score to trade
        self.max_zone_age_hours = 168  # 7 days
        self.max_zone_tests = 3
        self.volume_threshold_multiplier = 1.5  # Volume must be 1.5x average
        self.rejection_speed_threshold = 2.0  # ATR multiplier for strong rejection
        self.institutional_volume_threshold = 0.7  # 70% of volume at zone
        
    def analyze_market(
        self,
        symbol: str,
        df: pd.DataFrame,
        timeframes: List[str] = ['15', '60', '240']
    ) -> Dict[str, Any]:
        """
        Comprehensive market analysis with multiple timeframes
        """
        analysis = {
            'zones': [],
            'market_structure': None,
            'volume_profile': None,
            'order_flow': None,
            'signals': [],
            'confidence': 0
        }
        
        # Validate input dataframe
        if df is None or df.empty:
            logger.warning(f"Empty or invalid dataframe for {symbol}")
            return analysis
            
        # Ensure dataframe has required columns
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in required_columns):
            logger.error(f"Missing required columns for {symbol}. Got: {df.columns.tolist() if hasattr(df, 'columns') else 'invalid data'}")
            return analysis
        
        try:
            # 1. Identify market structure
            market_structure = self._identify_market_structure(df)
            analysis['market_structure'] = market_structure
            
            # 2. Build volume profile
            volume_profile = self._build_volume_profile(df)
            analysis['volume_profile'] = volume_profile
            
            # 3. Analyze order flow
            order_flow = self._analyze_order_flow(df)
            analysis['order_flow'] = order_flow
            
            # 4. Detect supply/demand zones with enhancements
            zones = self._detect_enhanced_zones(
                df, 
                volume_profile, 
                order_flow,
                market_structure
            )
            
            # 5. Multi-timeframe confluence
            if len(timeframes) > 1:
                zones = self._add_timeframe_confluence(zones, symbol, timeframes)
            
            # 6. Score and filter zones
            valid_zones = [z for z in zones if z.composite_score >= self.min_zone_score]
            analysis['zones'] = sorted(valid_zones, key=lambda z: z.composite_score, reverse=True)
            
            # 7. Generate trading signals
            if analysis['zones']:
                signals = self._generate_enhanced_signals(
                    analysis['zones'],
                    df.iloc[-1],
                    market_structure,
                    order_flow
                )
                analysis['signals'] = signals
            
            # 8. Calculate overall confidence
            analysis['confidence'] = self._calculate_confidence(analysis)
            
            # Store for future reference
            self.zones[symbol] = analysis['zones']
            self.market_structure[symbol] = market_structure
            self.volume_profiles[symbol] = volume_profile
            
        except Exception as e:
            import traceback
            logger.error(f"Error in market analysis for {symbol}: {e}")
            logger.debug(f"Traceback: {traceback.format_exc()}")
        
        return analysis
    
    def _identify_market_structure(self, df: pd.DataFrame) -> MarketStructure:
        """Identify current market structure"""
        
        # Calculate swing highs and lows
        highs = df['high'].rolling(5).max()
        lows = df['low'].rolling(5).min()
        
        # Identify trend based on higher highs/lows or lower highs/lows
        recent_highs = highs.tail(20).dropna()
        recent_lows = lows.tail(20).dropna()
        
        if len(recent_highs) < 2 or len(recent_lows) < 2:
            return MarketStructure.RANGING
        
        # Check for uptrend (higher highs and higher lows)
        hh = recent_highs.iloc[-1] > recent_highs.iloc[-5]
        hl = recent_lows.iloc[-1] > recent_lows.iloc[-5]
        
        # Check for downtrend (lower highs and lower lows)
        lh = recent_highs.iloc[-1] < recent_highs.iloc[-5]
        ll = recent_lows.iloc[-1] < recent_lows.iloc[-5]
        
        if hh and hl:
            return MarketStructure.BULLISH
        elif lh and ll:
            return MarketStructure.BEARISH
        elif (hh and ll) or (lh and hl):
            return MarketStructure.TRANSITIONING
        else:
            return MarketStructure.RANGING
    
    def _build_volume_profile(self, df: pd.DataFrame) -> VolumeProfile:
        """Build volume profile for price range"""
        
        # Define price levels (30 levels)
        price_min = df['low'].min()
        price_max = df['high'].max()
        price_levels = np.linspace(price_min, price_max, 30)
        
        # Calculate volume at each price level
        volumes = []
        buying_volumes = []
        selling_volumes = []
        
        for i in range(len(price_levels) - 1):
            level_low = price_levels[i]
            level_high = price_levels[i + 1]
            
            # Find candles that touched this price level
            mask = (df['low'] <= level_high) & (df['high'] >= level_low)
            level_df = df[mask]
            
            if not level_df.empty:
                total_vol = level_df['volume'].sum()
                
                # Estimate buying/selling volume based on close position
                buying_vol = level_df[level_df['close'] > level_df['open']]['volume'].sum()
                selling_vol = level_df[level_df['close'] <= level_df['open']]['volume'].sum()
                
                volumes.append(total_vol)
                buying_volumes.append(buying_vol)
                selling_volumes.append(selling_vol)
            else:
                volumes.append(0)
                buying_volumes.append(0)
                selling_volumes.append(0)
        
        # Find Point of Control (highest volume price)
        if volumes:
            poc_idx = np.argmax(volumes)
            poc = price_levels[poc_idx]
            
            # Calculate Value Area (70% of volume)
            total_volume = sum(volumes)
            value_area_volume = total_volume * 0.7
            
            # Expand from POC to find value area
            accumulated = volumes[poc_idx]
            low_idx = poc_idx
            high_idx = poc_idx
            
            while accumulated < value_area_volume and (low_idx > 0 or high_idx < len(volumes) - 1):
                if low_idx > 0:
                    low_idx -= 1
                    accumulated += volumes[low_idx]
                if high_idx < len(volumes) - 1 and accumulated < value_area_volume:
                    high_idx += 1
                    accumulated += volumes[high_idx]
            
            val = price_levels[low_idx]
            vah = price_levels[high_idx]
        else:
            poc = df['close'].median()
            vah = df['high'].median()
            val = df['low'].median()
            total_volume = df['volume'].sum()
        
        return VolumeProfile(
            price_levels=list(price_levels),
            volumes=volumes,
            poc=poc,
            vah=vah,
            val=val,
            total_volume=total_volume,
            buying_volume=sum(buying_volumes),
            selling_volume=sum(selling_volumes),
            delta=sum(buying_volumes) - sum(selling_volumes)
        )
    
    def _analyze_order_flow(self, df: pd.DataFrame) -> OrderFlowImbalance:
        """Analyze order flow imbalance"""
        
        # Calculate cumulative delta
        buying_volume = df[df['close'] > df['open']]['volume'].sum()
        selling_volume = df[df['close'] <= df['open']]['volume'].sum()
        
        total_volume = buying_volume + selling_volume
        if total_volume == 0:
            return OrderFlowImbalance.NEUTRAL
        
        buying_ratio = buying_volume / total_volume
        
        # Classify imbalance
        if buying_ratio > 0.7:
            return OrderFlowImbalance.STRONG_BUYING
        elif buying_ratio > 0.55:
            return OrderFlowImbalance.BUYING
        elif buying_ratio < 0.3:
            return OrderFlowImbalance.STRONG_SELLING
        elif buying_ratio < 0.45:
            return OrderFlowImbalance.SELLING
        else:
            return OrderFlowImbalance.NEUTRAL
    
    def _detect_enhanced_zones(
        self,
        df: pd.DataFrame,
        volume_profile: VolumeProfile,
        order_flow: OrderFlowImbalance,
        market_structure: MarketStructure
    ) -> List[EnhancedZone]:
        """Detect supply/demand zones with enhanced metrics"""
        
        zones = []
        
        # Calculate ATR for rejection strength
        df['atr'] = self._calculate_atr(df)
        avg_volume = df['volume'].rolling(20).mean()
        
        # Look for strong rejections (demand zones)
        for i in range(20, len(df) - 1):
            # Check for bullish rejection (hammer, bullish engulfing, etc.)
            if self._is_bullish_rejection(df, i):
                # Calculate zone bounds
                zone_high = df.iloc[i]['high']
                zone_low = df.iloc[i]['low']
                
                # Calculate rejection strength
                rejection_strength = abs(df.iloc[i+1]['close'] - df.iloc[i]['low']) / df.iloc[i]['atr']
                
                # Check volume spike
                volume_spike = df.iloc[i]['volume'] / avg_volume.iloc[i] if avg_volume.iloc[i] > 0 else 1
                
                # Estimate institutional interest
                institutional_interest = self._calculate_institutional_interest(
                    df, i, volume_spike, rejection_strength
                )
                
                # Identify confluence factors
                confluence_factors = self._identify_confluence_factors(
                    df, i, volume_profile, market_structure
                )
                
                # Create zone
                zone = EnhancedZone(
                    zone_type='demand',
                    upper_bound=zone_high,
                    lower_bound=zone_low,
                    strength_score=self._calculate_zone_strength(
                        rejection_strength, volume_spike, len(confluence_factors)
                    ),
                    volume_profile=volume_profile,
                    order_flow_imbalance=order_flow,
                    formation_time=pd.Timestamp(df.index[i]),
                    test_count=0,
                    rejection_strength=rejection_strength,
                    institutional_interest=institutional_interest,
                    confluence_factors=confluence_factors,
                    timeframes_visible=['current'],
                    last_test_time=None,
                    zone_age_hours=0,
                    liquidity_pool=df.iloc[i]['volume'] * df.iloc[i]['close']
                )
                
                zones.append(zone)
        
        # Look for strong rejections (supply zones)
        for i in range(20, len(df) - 1):
            # Check for bearish rejection
            if self._is_bearish_rejection(df, i):
                # Calculate zone bounds
                zone_high = df.iloc[i]['high']
                zone_low = df.iloc[i]['low']
                
                # Calculate rejection strength
                rejection_strength = abs(df.iloc[i]['high'] - df.iloc[i+1]['close']) / df.iloc[i]['atr']
                
                # Check volume spike
                volume_spike = df.iloc[i]['volume'] / avg_volume.iloc[i] if avg_volume.iloc[i] > 0 else 1
                
                # Estimate institutional interest
                institutional_interest = self._calculate_institutional_interest(
                    df, i, volume_spike, rejection_strength
                )
                
                # Identify confluence factors
                confluence_factors = self._identify_confluence_factors(
                    df, i, volume_profile, market_structure
                )
                
                # Create zone
                zone = EnhancedZone(
                    zone_type='supply',
                    upper_bound=zone_high,
                    lower_bound=zone_low,
                    strength_score=self._calculate_zone_strength(
                        rejection_strength, volume_spike, len(confluence_factors)
                    ),
                    volume_profile=volume_profile,
                    order_flow_imbalance=order_flow,
                    formation_time=pd.Timestamp(df.index[i]),
                    test_count=0,
                    rejection_strength=rejection_strength,
                    institutional_interest=institutional_interest,
                    confluence_factors=confluence_factors,
                    timeframes_visible=['current'],
                    last_test_time=None,
                    zone_age_hours=0,
                    liquidity_pool=df.iloc[i]['volume'] * df.iloc[i]['close']
                )
                
                zones.append(zone)
        
        return zones
    
    def _is_bullish_rejection(self, df: pd.DataFrame, i: int) -> bool:
        """Check for bullish rejection pattern"""
        candle = df.iloc[i]
        prev_candle = df.iloc[i-1] if i > 0 else None
        next_candle = df.iloc[i+1] if i < len(df) - 1 else None
        
        # Hammer pattern
        body = abs(candle['close'] - candle['open'])
        lower_wick = candle['open'] - candle['low'] if candle['close'] > candle['open'] else candle['close'] - candle['low']
        
        if lower_wick > body * 2:  # Long lower wick
            return True
        
        # Bullish engulfing
        if prev_candle is not None:
            if (candle['close'] > candle['open'] and 
                prev_candle['close'] < prev_candle['open'] and
                candle['close'] > prev_candle['open'] and
                candle['open'] < prev_candle['close']):
                return True
        
        # Strong bullish candle after decline
        if next_candle is not None:
            if (next_candle['close'] > next_candle['open'] and
                (next_candle['close'] - next_candle['open']) > candle['atr'] * 1.5):
                return True
        
        return False
    
    def _is_bearish_rejection(self, df: pd.DataFrame, i: int) -> bool:
        """Check for bearish rejection pattern"""
        candle = df.iloc[i]
        prev_candle = df.iloc[i-1] if i > 0 else None
        next_candle = df.iloc[i+1] if i < len(df) - 1 else None
        
        # Shooting star pattern
        body = abs(candle['close'] - candle['open'])
        upper_wick = candle['high'] - candle['close'] if candle['close'] > candle['open'] else candle['high'] - candle['open']
        
        if upper_wick > body * 2:  # Long upper wick
            return True
        
        # Bearish engulfing
        if prev_candle is not None:
            if (candle['close'] < candle['open'] and 
                prev_candle['close'] > prev_candle['open'] and
                candle['open'] > prev_candle['close'] and
                candle['close'] < prev_candle['open']):
                return True
        
        # Strong bearish candle after rise
        if next_candle is not None:
            if (next_candle['close'] < next_candle['open'] and
                (next_candle['open'] - next_candle['close']) > candle['atr'] * 1.5):
                return True
        
        return False
    
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        high = df['high']
        low = df['low']
        close = df['close'].shift(1)
        
        tr1 = high - low
        tr2 = abs(high - close)
        tr3 = abs(low - close)
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        
        return atr
    
    def _calculate_institutional_interest(
        self,
        df: pd.DataFrame,
        index: int,
        volume_spike: float,
        rejection_strength: float
    ) -> float:
        """Estimate institutional interest in the zone"""
        
        score = 0
        
        # High volume indicates institutional activity
        if volume_spike > 2.0:
            score += 30
        elif volume_spike > 1.5:
            score += 20
        elif volume_spike > 1.2:
            score += 10
        
        # Strong rejection indicates large orders
        if rejection_strength > 3.0:
            score += 30
        elif rejection_strength > 2.0:
            score += 20
        elif rejection_strength > 1.5:
            score += 10
        
        # Wide range candle indicates institutional movement
        candle_range = df.iloc[index]['high'] - df.iloc[index]['low']
        avg_range = (df['high'] - df['low']).rolling(20).mean().iloc[index]
        
        if avg_range > 0:
            range_ratio = candle_range / avg_range
            if range_ratio > 2.0:
                score += 20
            elif range_ratio > 1.5:
                score += 10
        
        # Gap indicates overnight institutional positioning
        if index > 0:
            gap = abs(df.iloc[index]['open'] - df.iloc[index-1]['close'])
            if gap > df.iloc[index]['atr'] * 0.5:
                score += 20
        
        return min(100, score)
    
    def _identify_confluence_factors(
        self,
        df: pd.DataFrame,
        index: int,
        volume_profile: VolumeProfile,
        market_structure: MarketStructure
    ) -> List[str]:
        """Identify confluence factors for the zone"""
        
        factors = []
        price = df.iloc[index]['close']
        
        # Volume profile confluence
        if abs(price - volume_profile.poc) / price < 0.01:  # Within 1% of POC
            factors.append("volume_poc")
        if price >= volume_profile.val and price <= volume_profile.vah:
            factors.append("value_area")
        
        # Moving average confluence
        if 'sma_50' in df.columns:
            if abs(price - df.iloc[index]['sma_50']) / price < 0.02:
                factors.append("sma_50")
        if 'sma_200' in df.columns:
            if abs(price - df.iloc[index]['sma_200']) / price < 0.02:
                factors.append("sma_200")
        
        # Fibonacci levels (simplified)
        recent_high = df['high'].rolling(50).max().iloc[index]
        recent_low = df['low'].rolling(50).min().iloc[index]
        fib_levels = [0.236, 0.382, 0.5, 0.618, 0.786]
        
        for level in fib_levels:
            fib_price = recent_low + (recent_high - recent_low) * level
            if abs(price - fib_price) / price < 0.02:
                factors.append(f"fib_{level}")
                break
        
        # Market structure alignment
        if market_structure == MarketStructure.BULLISH and df.iloc[index]['close'] > df.iloc[index]['open']:
            factors.append("trend_aligned")
        elif market_structure == MarketStructure.BEARISH and df.iloc[index]['close'] < df.iloc[index]['open']:
            factors.append("trend_aligned")
        
        # Round number confluence
        if price % 100 < 5 or price % 100 > 95:  # Near round hundreds
            factors.append("round_number")
        elif price % 1000 < 10 or price % 1000 > 990:  # Near round thousands
            factors.append("major_round_number")
        
        return factors
    
    def _calculate_zone_strength(
        self,
        rejection_strength: float,
        volume_spike: float,
        confluence_count: int
    ) -> float:
        """Calculate overall zone strength score"""
        
        # Base score from rejection strength (0-40)
        rejection_score = min(40, rejection_strength * 13.33)
        
        # Volume component (0-30)
        volume_score = min(30, volume_spike * 15)
        
        # Confluence component (0-30)
        confluence_score = min(30, confluence_count * 6)
        
        return rejection_score + volume_score + confluence_score
    
    def _add_timeframe_confluence(
        self,
        zones: List[EnhancedZone],
        symbol: str,
        timeframes: List[str]
    ) -> List[EnhancedZone]:
        """Add multi-timeframe confluence to zones"""
        
        # This would require fetching data for multiple timeframes
        # For now, we'll simulate by marking zones visible on multiple timeframes
        for zone in zones:
            # Simulate checking visibility on higher timeframes
            if zone.rejection_strength > 2.5:
                zone.timeframes_visible.extend(['60', '240'])
            elif zone.rejection_strength > 2.0:
                zone.timeframes_visible.append('60')
        
        return zones
    
    def _generate_enhanced_signals(
        self,
        zones: List[EnhancedZone],
        current_candle: pd.Series,
        market_structure: MarketStructure,
        order_flow: OrderFlowImbalance
    ) -> List[Dict[str, Any]]:
        """Generate trading signals based on zones and market conditions"""
        
        signals = []
        current_price = current_candle['close']
        
        for zone in zones[:3]:  # Check top 3 zones
            # For demand zones
            if zone.zone_type == 'demand':
                # Check if price is approaching the zone
                distance_to_zone = (zone.upper_bound - current_price) / current_price
                
                if 0 < distance_to_zone < 0.005:  # Within 0.5% of zone
                    # Additional confirmations
                    confirmations = []
                    
                    if market_structure in [MarketStructure.BULLISH, MarketStructure.TRANSITIONING]:
                        confirmations.append("market_structure")
                    if order_flow in [OrderFlowImbalance.BUYING, OrderFlowImbalance.STRONG_BUYING]:
                        confirmations.append("order_flow")
                    if zone.is_fresh:
                        confirmations.append("fresh_zone")
                    if zone.institutional_interest > 70:
                        confirmations.append("institutional")
                    
                    if len(confirmations) >= 2:
                        signals.append({
                            'type': 'BUY',
                            'zone': zone,
                            'entry_price': zone.upper_bound,
                            'stop_loss': zone.lower_bound * 0.995,
                            'take_profit_1': zone.upper_bound * 1.01,
                            'take_profit_2': zone.upper_bound * 1.02,
                            'confidence': zone.composite_score,
                            'confirmations': confirmations,
                            'risk_reward': 2.0
                        })
            
            # For supply zones
            elif zone.zone_type == 'supply':
                # Check if price is approaching the zone
                distance_to_zone = (current_price - zone.lower_bound) / current_price
                
                if 0 < distance_to_zone < 0.005:  # Within 0.5% of zone
                    # Additional confirmations
                    confirmations = []
                    
                    if market_structure in [MarketStructure.BEARISH, MarketStructure.TRANSITIONING]:
                        confirmations.append("market_structure")
                    if order_flow in [OrderFlowImbalance.SELLING, OrderFlowImbalance.STRONG_SELLING]:
                        confirmations.append("order_flow")
                    if zone.is_fresh:
                        confirmations.append("fresh_zone")
                    if zone.institutional_interest > 70:
                        confirmations.append("institutional")
                    
                    if len(confirmations) >= 2:
                        signals.append({
                            'type': 'SELL',
                            'zone': zone,
                            'entry_price': zone.lower_bound,
                            'stop_loss': zone.upper_bound * 1.005,
                            'take_profit_1': zone.lower_bound * 0.99,
                            'take_profit_2': zone.lower_bound * 0.98,
                            'confidence': zone.composite_score,
                            'confirmations': confirmations,
                            'risk_reward': 2.0
                        })
        
        return signals
    
    def _calculate_confidence(self, analysis: Dict[str, Any]) -> float:
        """Calculate overall trading confidence"""
        
        confidence = 50  # Base confidence
        
        # Market structure alignment
        if analysis['market_structure'] in [MarketStructure.BULLISH, MarketStructure.BEARISH]:
            confidence += 10
        
        # Strong order flow
        if analysis['order_flow'] in [OrderFlowImbalance.STRONG_BUYING, OrderFlowImbalance.STRONG_SELLING]:
            confidence += 15
        
        # Quality zones available
        if analysis['zones']:
            best_zone = analysis['zones'][0]
            if best_zone.composite_score > 80:
                confidence += 20
            elif best_zone.composite_score > 70:
                confidence += 10
        
        # Signals generated
        if analysis['signals']:
            confidence += 5 * len(analysis['signals'])
        
        return min(100, confidence)