"""
Advanced Order Flow Analysis
Analyzes buy/sell pressure, volume delta, and liquidity imbalances
Based on institutional trading techniques
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from datetime import datetime, timedelta
import structlog
from scipy import stats
from collections import deque

logger = structlog.get_logger(__name__)

class OrderFlowImbalance(Enum):
    """Order flow imbalance levels"""
    EXTREME_BUYING = "extreme_buying"
    STRONG_BUYING = "strong_buying"
    BUYING = "buying"
    NEUTRAL = "neutral"
    SELLING = "selling"
    STRONG_SELLING = "strong_selling"
    EXTREME_SELLING = "extreme_selling"

class LiquidityType(Enum):
    """Liquidity zone types"""
    BUY_SIDE_LIQUIDITY = "buy_side_liquidity"
    SELL_SIDE_LIQUIDITY = "sell_side_liquidity"
    BALANCED_LIQUIDITY = "balanced_liquidity"
    THIN_LIQUIDITY = "thin_liquidity"

@dataclass
class DeltaDivergence:
    """Delta divergence pattern"""
    divergence_type: str  # 'bullish' or 'bearish'
    price_trend: str  # 'up' or 'down'
    delta_trend: str  # 'up' or 'down'
    strength: float  # 0-100
    bars_count: int
    start_price: float
    end_price: float
    total_delta: float

@dataclass
class ImbalanceZone:
    """Order flow imbalance zone"""
    zone_type: str  # 'buying' or 'selling'
    start_price: float
    end_price: float
    volume_imbalance: float
    delta: float
    timestamp: datetime
    filled: bool = False
    test_count: int = 0

@dataclass
class OrderFlowMetrics:
    """Comprehensive order flow metrics"""
    cumulative_delta: float
    session_delta: float
    delta_momentum: float
    buy_volume: float
    sell_volume: float
    volume_ratio: float
    imbalance: OrderFlowImbalance
    poc: float  # Point of Control
    vwap: float  # Volume Weighted Average Price
    liquidity_zones: List[Dict]
    delta_divergences: List[DeltaDivergence]
    imbalance_zones: List[ImbalanceZone]
    absorption_detected: bool
    exhaustion_detected: bool

class OrderFlowAnalyzer:
    """
    Advanced order flow analysis for institutional trading
    Identifies smart money movements through volume analysis
    """
    
    def __init__(self):
        # Configuration
        self.delta_lookback = 20
        self.imbalance_threshold = 0.3  # 30% imbalance
        self.absorption_ratio = 2.0  # Volume to range ratio
        self.exhaustion_volume = 3.0  # Volume spike for exhaustion
        self.divergence_periods = 10
        
        # State tracking
        self.historical_deltas = deque(maxlen=100)
        self.imbalance_zones = []
        self.liquidity_map = {}
        
    def analyze(self, df: pd.DataFrame, orderbook_data: Optional[Dict] = None) -> OrderFlowMetrics:
        """
        Comprehensive order flow analysis
        
        Args:
            df: OHLCV DataFrame
            orderbook_data: Optional orderbook snapshot
            
        Returns:
            OrderFlowMetrics with complete analysis
        """
        
        # Calculate volume delta
        df = self._calculate_volume_delta(df)
        
        # Calculate cumulative metrics
        cumulative_delta = df['volume_delta'].sum()
        session_delta = df['volume_delta'].iloc[-20:].sum() if len(df) > 20 else cumulative_delta
        
        # Calculate buy/sell volumes
        buy_volume = df[df['volume_delta'] > 0]['volume'].sum()
        sell_volume = df[df['volume_delta'] < 0]['volume'].sum()
        volume_ratio = buy_volume / max(sell_volume, 1)
        
        # Determine order flow imbalance
        imbalance = self._determine_imbalance(df)
        
        # Calculate POC and VWAP
        poc = self._calculate_poc(df)
        vwap = self._calculate_vwap(df)
        
        # Detect delta divergences
        delta_divergences = self._detect_delta_divergences(df)
        
        # Identify imbalance zones
        imbalance_zones = self._identify_imbalance_zones(df)
        
        # Identify liquidity zones
        liquidity_zones = self._identify_liquidity_zones(df, orderbook_data)
        
        # Detect absorption
        absorption_detected = self._detect_absorption(df)
        
        # Detect exhaustion
        exhaustion_detected = self._detect_exhaustion(df)
        
        # Calculate delta momentum
        delta_momentum = self._calculate_delta_momentum(df)
        
        return OrderFlowMetrics(
            cumulative_delta=cumulative_delta,
            session_delta=session_delta,
            delta_momentum=delta_momentum,
            buy_volume=buy_volume,
            sell_volume=sell_volume,
            volume_ratio=volume_ratio,
            imbalance=imbalance,
            poc=poc,
            vwap=vwap,
            liquidity_zones=liquidity_zones,
            delta_divergences=delta_divergences,
            imbalance_zones=imbalance_zones,
            absorption_detected=absorption_detected,
            exhaustion_detected=exhaustion_detected
        )
    
    def _calculate_volume_delta(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate volume delta (buy volume - sell volume)
        Uses price action to estimate buy/sell pressure
        """
        
        # Method 1: Close vs Open
        df['volume_delta_co'] = df.apply(
            lambda x: x['volume'] if x['close'] > x['open'] else -x['volume'],
            axis=1
        )
        
        # Method 2: Close position in range
        df['close_position'] = (df['close'] - df['low']) / (df['high'] - df['low'] + 0.0001)
        df['volume_delta_range'] = df['volume'] * (2 * df['close_position'] - 1)
        
        # Method 3: Tick-based estimation (simplified)
        df['price_change'] = df['close'] - df['close'].shift(1)
        df['volume_delta_tick'] = df.apply(
            lambda x: x['volume'] if x['price_change'] > 0 else -x['volume'] if x['price_change'] < 0 else 0,
            axis=1
        )
        
        # Weighted average of methods
        df['volume_delta'] = (
            df['volume_delta_co'] * 0.4 +
            df['volume_delta_range'] * 0.4 +
            df['volume_delta_tick'] * 0.2
        )
        
        # Cumulative delta
        df['cum_delta'] = df['volume_delta'].cumsum()
        
        # Delta momentum
        df['delta_ma'] = df['volume_delta'].rolling(window=5).mean()
        
        return df
    
    def _determine_imbalance(self, df: pd.DataFrame) -> OrderFlowImbalance:
        """Determine current order flow imbalance"""
        
        recent_delta = df['volume_delta'].iloc[-10:].mean() if len(df) > 10 else 0
        recent_volume = df['volume'].iloc[-10:].mean() if len(df) > 10 else 1
        
        imbalance_ratio = recent_delta / max(recent_volume, 1)
        
        if imbalance_ratio > 0.5:
            return OrderFlowImbalance.EXTREME_BUYING
        elif imbalance_ratio > 0.3:
            return OrderFlowImbalance.STRONG_BUYING
        elif imbalance_ratio > 0.1:
            return OrderFlowImbalance.BUYING
        elif imbalance_ratio < -0.5:
            return OrderFlowImbalance.EXTREME_SELLING
        elif imbalance_ratio < -0.3:
            return OrderFlowImbalance.STRONG_SELLING
        elif imbalance_ratio < -0.1:
            return OrderFlowImbalance.SELLING
        else:
            return OrderFlowImbalance.NEUTRAL
    
    def _calculate_poc(self, df: pd.DataFrame) -> float:
        """
        Calculate Point of Control (price with highest volume)
        """
        
        # Create price bins
        price_bins = 50
        min_price = df['low'].min()
        max_price = df['high'].max()
        
        bins = np.linspace(min_price, max_price, price_bins)
        volume_profile = {}
        
        for i in range(len(bins) - 1):
            mask = (df['close'] >= bins[i]) & (df['close'] < bins[i + 1])
            volume_profile[bins[i]] = df[mask]['volume'].sum()
        
        # Find POC
        if volume_profile:
            poc = max(volume_profile, key=volume_profile.get)
            return poc
        
        return df['close'].median()
    
    def _calculate_vwap(self, df: pd.DataFrame) -> float:
        """Calculate Volume Weighted Average Price"""
        
        df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
        df['pv'] = df['typical_price'] * df['volume']
        
        cumulative_pv = df['pv'].sum()
        cumulative_volume = df['volume'].sum()
        
        if cumulative_volume > 0:
            return cumulative_pv / cumulative_volume
        
        return df['close'].mean()
    
    def _detect_delta_divergences(self, df: pd.DataFrame) -> List[DeltaDivergence]:
        """
        Detect divergences between price and delta
        Bullish: Price down, delta up (accumulation)
        Bearish: Price up, delta down (distribution)
        """
        
        divergences = []
        
        if len(df) < self.divergence_periods * 2:
            return divergences
        
        for i in range(self.divergence_periods, len(df) - self.divergence_periods):
            segment = df.iloc[i:i+self.divergence_periods]
            
            # Calculate price trend
            price_slope = np.polyfit(range(len(segment)), segment['close'], 1)[0]
            price_trend = 'up' if price_slope > 0 else 'down'
            
            # Calculate delta trend
            delta_slope = np.polyfit(range(len(segment)), segment['cum_delta'], 1)[0]
            delta_trend = 'up' if delta_slope > 0 else 'down'
            
            # Check for divergence
            if price_trend == 'down' and delta_trend == 'up':
                # Bullish divergence
                divergence = DeltaDivergence(
                    divergence_type='bullish',
                    price_trend=price_trend,
                    delta_trend=delta_trend,
                    strength=abs(delta_slope) / segment['volume'].mean() * 100,
                    bars_count=len(segment),
                    start_price=segment['close'].iloc[0],
                    end_price=segment['close'].iloc[-1],
                    total_delta=segment['volume_delta'].sum()
                )
                divergences.append(divergence)
                
            elif price_trend == 'up' and delta_trend == 'down':
                # Bearish divergence
                divergence = DeltaDivergence(
                    divergence_type='bearish',
                    price_trend=price_trend,
                    delta_trend=delta_trend,
                    strength=abs(delta_slope) / segment['volume'].mean() * 100,
                    bars_count=len(segment),
                    start_price=segment['close'].iloc[0],
                    end_price=segment['close'].iloc[-1],
                    total_delta=segment['volume_delta'].sum()
                )
                divergences.append(divergence)
        
        return divergences
    
    def _identify_imbalance_zones(self, df: pd.DataFrame) -> List[ImbalanceZone]:
        """
        Identify order flow imbalance zones (unfilled liquidity gaps)
        These are areas where buying/selling was so aggressive that
        price moved quickly without much opposition
        """
        
        zones = []
        
        for i in range(1, len(df)):
            current = df.iloc[i]
            previous = df.iloc[i-1]
            
            # Calculate imbalance
            price_move = abs(current['close'] - previous['close'])
            avg_move = df['close'].diff().abs().rolling(window=20).mean().iloc[i]
            
            if avg_move > 0 and price_move > avg_move * 2:
                # Significant move detected
                
                # Check volume imbalance
                volume_ratio = current['volume'] / df['volume'].rolling(window=20).mean().iloc[i]
                
                if volume_ratio > 1.5:
                    # Create imbalance zone
                    if current['close'] > previous['close']:
                        # Buying imbalance
                        zone = ImbalanceZone(
                            zone_type='buying',
                            start_price=previous['high'],
                            end_price=current['low'],
                            volume_imbalance=volume_ratio,
                            delta=current['volume_delta'],
                            timestamp=current.name if hasattr(current, 'name') else datetime.now()
                        )
                    else:
                        # Selling imbalance
                        zone = ImbalanceZone(
                            zone_type='selling',
                            start_price=current['high'],
                            end_price=previous['low'],
                            volume_imbalance=volume_ratio,
                            delta=current['volume_delta'],
                            timestamp=current.name if hasattr(current, 'name') else datetime.now()
                        )
                    
                    zones.append(zone)
        
        # Update zone status (check if filled)
        for zone in zones:
            for i in range(len(df)):
                if zone.zone_type == 'buying':
                    if df.iloc[i]['low'] <= zone.start_price:
                        zone.filled = True
                        zone.test_count += 1
                else:
                    if df.iloc[i]['high'] >= zone.end_price:
                        zone.filled = True
                        zone.test_count += 1
        
        return zones
    
    def _identify_liquidity_zones(self, df: pd.DataFrame, orderbook_data: Optional[Dict]) -> List[Dict]:
        """
        Identify liquidity zones from price action and orderbook
        """
        
        zones = []
        
        # Price-based liquidity zones (swing highs/lows)
        window = 10
        
        for i in range(window, len(df) - window):
            # Check for swing high
            if df['high'].iloc[i] == df['high'].iloc[i-window:i+window+1].max():
                zone = {
                    'type': 'sell_side_liquidity',
                    'price': df['high'].iloc[i],
                    'strength': df['volume'].iloc[i-window:i+window+1].sum(),
                    'timestamp': df.index[i] if hasattr(df, 'index') else i
                }
                zones.append(zone)
            
            # Check for swing low
            if df['low'].iloc[i] == df['low'].iloc[i-window:i+window+1].min():
                zone = {
                    'type': 'buy_side_liquidity',
                    'price': df['low'].iloc[i],
                    'strength': df['volume'].iloc[i-window:i+window+1].sum(),
                    'timestamp': df.index[i] if hasattr(df, 'index') else i
                }
                zones.append(zone)
        
        # Orderbook-based liquidity (if available)
        if orderbook_data:
            # Identify large bid/ask walls
            if 'bids' in orderbook_data:
                for bid in orderbook_data['bids'][:5]:  # Top 5 bid levels
                    if float(bid[1]) > df['volume'].mean() * 2:  # Large size
                        zone = {
                            'type': 'buy_side_liquidity',
                            'price': float(bid[0]),
                            'strength': float(bid[1]),
                            'source': 'orderbook'
                        }
                        zones.append(zone)
            
            if 'asks' in orderbook_data:
                for ask in orderbook_data['asks'][:5]:  # Top 5 ask levels
                    if float(ask[1]) > df['volume'].mean() * 2:  # Large size
                        zone = {
                            'type': 'sell_side_liquidity',
                            'price': float(ask[0]),
                            'strength': float(ask[1]),
                            'source': 'orderbook'
                        }
                        zones.append(zone)
        
        return zones
    
    def _detect_absorption(self, df: pd.DataFrame) -> bool:
        """
        Detect absorption pattern (high volume, small price movement)
        Indicates institutional accumulation/distribution
        """
        
        if len(df) < 5:
            return False
        
        recent = df.iloc[-5:]
        
        # Calculate average volume and price range
        avg_volume = df['volume'].rolling(window=20).mean().iloc[-1] if len(df) > 20 else df['volume'].mean()
        
        for i in range(len(recent)):
            bar = recent.iloc[i]
            
            # High volume
            if bar['volume'] > avg_volume * self.absorption_ratio:
                # Small price movement
                bar_range = bar['high'] - bar['low']
                avg_range = df['high'].sub(df['low']).rolling(window=20).mean().iloc[-1]
                
                if bar_range < avg_range * 0.5:
                    # Absorption detected
                    return True
        
        return False
    
    def _detect_exhaustion(self, df: pd.DataFrame) -> bool:
        """
        Detect exhaustion pattern (climactic volume at extremes)
        Indicates potential reversal
        """
        
        if len(df) < 20:
            return False
        
        recent = df.iloc[-3:]
        avg_volume = df['volume'].rolling(window=20).mean().iloc[-1]
        
        for i in range(len(recent)):
            bar = recent.iloc[i]
            
            # Extreme volume
            if bar['volume'] > avg_volume * self.exhaustion_volume:
                # At price extreme
                if bar['high'] == df['high'].iloc[-20:].max() or bar['low'] == df['low'].iloc[-20:].min():
                    # Followed by reversal
                    if i < len(recent) - 1:
                        next_bar = recent.iloc[i + 1]
                        if bar['close'] > bar['open'] and next_bar['close'] < next_bar['open']:
                            return True  # Buying exhaustion
                        elif bar['close'] < bar['open'] and next_bar['close'] > next_bar['open']:
                            return True  # Selling exhaustion
        
        return False
    
    def _calculate_delta_momentum(self, df: pd.DataFrame) -> float:
        """
        Calculate delta momentum (rate of change in cumulative delta)
        """
        
        if len(df) < 10:
            return 0.0
        
        recent_delta = df['cum_delta'].iloc[-10:]
        
        # Calculate momentum
        momentum = (recent_delta.iloc[-1] - recent_delta.iloc[0]) / len(recent_delta)
        
        # Normalize by average volume
        avg_volume = df['volume'].iloc[-10:].mean()
        if avg_volume > 0:
            normalized_momentum = momentum / avg_volume
        else:
            normalized_momentum = 0
        
        return normalized_momentum
    
    def get_trading_bias(self, metrics: OrderFlowMetrics) -> str:
        """
        Determine trading bias based on order flow analysis
        
        Returns:
            'long', 'short', or 'neutral'
        """
        
        bias_score = 0
        
        # Delta analysis
        if metrics.cumulative_delta > 0:
            bias_score += 1
        else:
            bias_score -= 1
        
        if metrics.delta_momentum > 0:
            bias_score += 1
        else:
            bias_score -= 1
        
        # Volume analysis
        if metrics.volume_ratio > 1.2:
            bias_score += 1
        elif metrics.volume_ratio < 0.8:
            bias_score -= 1
        
        # Imbalance analysis
        if metrics.imbalance in [OrderFlowImbalance.STRONG_BUYING, OrderFlowImbalance.EXTREME_BUYING]:
            bias_score += 2
        elif metrics.imbalance in [OrderFlowImbalance.STRONG_SELLING, OrderFlowImbalance.EXTREME_SELLING]:
            bias_score -= 2
        
        # Divergence analysis
        for divergence in metrics.delta_divergences:
            if divergence.divergence_type == 'bullish':
                bias_score += 1
            elif divergence.divergence_type == 'bearish':
                bias_score -= 1
        
        # Special patterns
        if metrics.absorption_detected:
            # Absorption often precedes moves
            if metrics.cumulative_delta > 0:
                bias_score += 2
            else:
                bias_score -= 2
        
        if metrics.exhaustion_detected:
            # Exhaustion suggests reversal
            if metrics.cumulative_delta > 0:
                bias_score -= 2  # Buying exhaustion = bearish
            else:
                bias_score += 2  # Selling exhaustion = bullish
        
        # Determine bias
        if bias_score >= 3:
            return 'long'
        elif bias_score <= -3:
            return 'short'
        else:
            return 'neutral'