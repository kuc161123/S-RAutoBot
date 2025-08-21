import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Optional, Tuple, Dict, Any
from enum import Enum
import ta

from ..config import settings
import structlog

logger = structlog.get_logger(__name__)

class ZoneType(Enum):
    DEMAND = "demand"
    SUPPLY = "supply"

class ZoneStatus(Enum):
    FRESH = "fresh"
    TESTED = "tested"
    INVALIDATED = "invalidated"

@dataclass
class Zone:
    """Supply or Demand zone"""
    zone_type: ZoneType
    upper_bound: float
    lower_bound: float
    created_at: datetime
    timeframe: str
    symbol: str
    score: float = 0.0
    touches: int = 0
    status: ZoneStatus = ZoneStatus.FRESH
    base_candles: int = 0
    departure_strength: float = 0.0
    volume_ratio: float = 1.0
    
    @property
    def midpoint(self) -> float:
        return (self.upper_bound + self.lower_bound) / 2
    
    @property
    def height(self) -> float:
        return self.upper_bound - self.lower_bound
    
    @property
    def age_hours(self) -> float:
        return (datetime.now() - self.created_at).total_seconds() / 3600
    
    def is_valid(self) -> bool:
        """Check if zone is still valid for trading"""
        return (
            self.status != ZoneStatus.INVALIDATED and
            self.touches < settings.sd_max_zone_touches and
            self.age_hours < settings.sd_zone_max_age_hours and
            self.score >= settings.sd_min_zone_score
        )
    
    def contains_price(self, price: float) -> bool:
        """Check if price is within the zone"""
        return self.lower_bound <= price <= self.upper_bound

@dataclass
class TradingSignal:
    """Trading signal generated from zone"""
    zone: Zone
    entry_price: float
    stop_loss: float
    take_profit_1: float
    take_profit_2: float
    position_size: float
    side: str  # "Buy" or "Sell"
    confidence: float
    reason: str
    timestamp: datetime = field(default_factory=datetime.now)

class SupplyDemandStrategy:
    """Supply and Demand zone detection and trading strategy"""
    
    def __init__(self):
        self.zones: Dict[str, List[Zone]] = {}  # symbol -> zones
        self.active_signals: Dict[str, TradingSignal] = {}  # symbol -> signal
        
    def detect_zones(self, df: pd.DataFrame, symbol: str, timeframe: str) -> List[Zone]:
        """
        Detect supply and demand zones from price data
        
        Args:
            df: DataFrame with OHLCV data
            symbol: Trading symbol
            timeframe: Timeframe of the data
        
        Returns:
            List of detected zones
        """
        if len(df) < 50:
            return []
        
        zones = []
        
        # Calculate indicators
        df = self._add_indicators(df)
        
        # Find potential bases
        bases = self._find_bases(df)
        
        for base_start, base_end in bases:
            # Check for departure after base
            zone = self._analyze_departure(df, base_start, base_end, symbol, timeframe)
            if zone:
                zones.append(zone)
        
        # Score and filter zones
        zones = self._score_zones(zones, df)
        
        # Remove overlapping zones (keep higher scored ones)
        zones = self._remove_overlapping_zones(zones)
        
        logger.info(f"Detected {len(zones)} zones for {symbol} on {timeframe}")
        
        return zones
    
    def _add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators to the dataframe"""
        df = df.copy()
        
        # ATR for volatility measurement
        df['atr'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=14)
        
        # Volume indicators
        df['volume_sma'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        
        # Price action
        df['body'] = abs(df['close'] - df['open'])
        df['range'] = df['high'] - df['low']
        df['body_ratio'] = df['body'] / df['range'].replace(0, 1)
        
        # Trend indicators
        df['ema_20'] = ta.trend.ema_indicator(df['close'], window=20)
        df['ema_50'] = ta.trend.ema_indicator(df['close'], window=50)
        
        return df
    
    def _find_bases(self, df: pd.DataFrame) -> List[Tuple[int, int]]:
        """Find consolidation bases in the price data"""
        bases = []
        min_candles = settings.sd_min_base_candles
        max_candles = settings.sd_max_base_candles
        
        i = 0
        while i < len(df) - max_candles - 5:  # Need room for departure
            # Check if this could be a base
            if self._is_consolidation(df.iloc[i:i+max_candles]):
                # Find the actual base length
                for length in range(min_candles, max_candles + 1):
                    if i + length >= len(df):
                        break
                    
                    base_slice = df.iloc[i:i+length]
                    
                    # Check if still consolidating
                    if not self._is_consolidation(base_slice):
                        if length > min_candles:
                            bases.append((i, i + length - 1))
                        break
                    
                    # Check if next candle breaks out
                    if i + length < len(df) - 1:
                        next_candle = df.iloc[i + length]
                        base_high = base_slice['high'].max()
                        base_low = base_slice['low'].min()
                        
                        if (next_candle['close'] > base_high * 1.002 or 
                            next_candle['close'] < base_low * 0.998):
                            bases.append((i, i + length - 1))
                            i += length
                            break
                
                i += 1
            else:
                i += 1
        
        return bases
    
    def _is_consolidation(self, df_slice: pd.DataFrame) -> bool:
        """Check if a slice of data represents consolidation"""
        if len(df_slice) < settings.sd_min_base_candles:
            return False
        
        # Check if ATR is available
        if 'atr' not in df_slice.columns or df_slice['atr'].isna().all():
            return False
        
        # Get the average ATR for this period
        avg_atr = df_slice['atr'].mean()
        if avg_atr == 0:
            return False
        
        # Check range relative to ATR
        total_range = df_slice['high'].max() - df_slice['low'].min()
        if total_range > avg_atr * 1.5:  # Range should be compact
            return False
        
        # Check body sizes
        avg_body_ratio = df_slice['body_ratio'].mean()
        if avg_body_ratio > 0.7:  # Bodies should be small
            return False
        
        # Check for overlap between candles
        overlaps = 0
        for i in range(1, len(df_slice)):
            prev = df_slice.iloc[i-1]
            curr = df_slice.iloc[i]
            
            # Check if candles overlap
            if not (curr['low'] > prev['high'] or curr['high'] < prev['low']):
                overlaps += 1
        
        overlap_ratio = overlaps / (len(df_slice) - 1)
        if overlap_ratio < 0.6:  # Should have significant overlap
            return False
        
        return True
    
    def _analyze_departure(
        self, 
        df: pd.DataFrame, 
        base_start: int, 
        base_end: int,
        symbol: str,
        timeframe: str
    ) -> Optional[Zone]:
        """Analyze the departure from a base to determine zone type"""
        if base_end >= len(df) - 5:
            return None
        
        base_slice = df.iloc[base_start:base_end+1]
        departure_slice = df.iloc[base_end+1:min(base_end+6, len(df))]
        
        if departure_slice.empty:
            return None
        
        # Calculate base boundaries
        base_high = base_slice['high'].max()
        base_low = base_slice['low'].min()
        base_mid = (base_high + base_low) / 2
        
        # Get ATR for departure measurement
        avg_atr = base_slice['atr'].mean() if 'atr' in base_slice.columns else 0
        if avg_atr == 0:
            return None
        
        # Measure departure strength
        cumulative_move = 0
        direction = None
        
        for i, row in departure_slice.iterrows():
            if row['close'] > base_high:
                move = row['close'] - base_high
                if direction is None:
                    direction = 'up'
                elif direction == 'down':
                    break
                cumulative_move += move
            elif row['close'] < base_low:
                move = base_low - row['close']
                if direction is None:
                    direction = 'down'
                elif direction == 'up':
                    break
                cumulative_move += move
        
        # Check if departure is strong enough
        departure_strength = cumulative_move / avg_atr if avg_atr > 0 else 0
        
        if departure_strength < settings.sd_departure_atr_multiplier:
            return None
        
        # Create zone based on departure direction
        if direction == 'up':
            # Demand zone (price rallied from here)
            zone_type = ZoneType.DEMAND
        elif direction == 'down':
            # Supply zone (price dropped from here)
            zone_type = ZoneType.SUPPLY
        else:
            return None
        
        # Create the zone
        zone = Zone(
            zone_type=zone_type,
            upper_bound=base_high,
            lower_bound=base_low,
            created_at=pd.Timestamp.now(),
            timeframe=timeframe,
            symbol=symbol,
            base_candles=base_end - base_start + 1,
            departure_strength=departure_strength,
            volume_ratio=base_slice['volume_ratio'].mean() if 'volume_ratio' in base_slice.columns else 1.0
        )
        
        return zone
    
    def _score_zones(self, zones: List[Zone], df: pd.DataFrame) -> List[Zone]:
        """Score zones based on multiple factors"""
        for zone in zones:
            score = 0.0
            
            # 1. Departure strength (0-30 points)
            departure_score = min(30, zone.departure_strength * 10)
            score += departure_score
            
            # 2. Base quality (0-20 points)
            base_score = 20 * (1 - (zone.base_candles - settings.sd_min_base_candles) / 
                              (settings.sd_max_base_candles - settings.sd_min_base_candles))
            score += max(0, base_score)
            
            # 3. Zone compactness (0-15 points)
            if 'atr' in df.columns:
                avg_atr = df['atr'].mean()
                if avg_atr > 0:
                    compactness = 1 - (zone.height / avg_atr) / 3
                    score += max(0, compactness * 15)
            
            # 4. Freshness (0-20 points)
            if zone.touches == 0:
                score += 20
            elif zone.touches == 1:
                score += 10
            elif zone.touches == 2:
                score += 5
            
            # 5. Volume at base (0-15 points)
            if zone.volume_ratio > 1.5:
                score += 15
            elif zone.volume_ratio > 1.2:
                score += 10
            elif zone.volume_ratio > 1.0:
                score += 5
            
            zone.score = min(100, max(0, score))
        
        return zones
    
    def _remove_overlapping_zones(self, zones: List[Zone]) -> List[Zone]:
        """Remove overlapping zones, keeping the higher scored ones"""
        if not zones:
            return zones
        
        # Sort by score (descending)
        zones.sort(key=lambda z: z.score, reverse=True)
        
        filtered_zones = []
        for zone in zones:
            # Check if this zone overlaps with any already selected zone
            overlaps = False
            for selected in filtered_zones:
                if (zone.lower_bound <= selected.upper_bound and 
                    zone.upper_bound >= selected.lower_bound):
                    overlaps = True
                    break
            
            if not overlaps:
                filtered_zones.append(zone)
        
        return filtered_zones
    
    def update_zones(self, symbol: str, price: float) -> None:
        """Update zone touches and status based on current price"""
        if symbol not in self.zones:
            return
        
        for zone in self.zones[symbol]:
            if zone.contains_price(price):
                zone.touches += 1
                
                if zone.touches == 1:
                    zone.status = ZoneStatus.TESTED
                elif zone.touches >= settings.sd_max_zone_touches:
                    zone.status = ZoneStatus.INVALIDATED
                
                logger.info(f"Zone touched: {zone.zone_type.value} zone at "
                          f"{zone.midpoint:.2f} (touches: {zone.touches})")
    
    def get_active_zones(self, symbol: str) -> List[Zone]:
        """Get all valid zones for a symbol"""
        if symbol not in self.zones:
            return []
        
        # Filter valid zones and sort by score
        active = [z for z in self.zones[symbol] if z.is_valid()]
        active.sort(key=lambda z: z.score, reverse=True)
        
        return active
    
    def check_entry_signal(
        self, 
        symbol: str, 
        current_price: float,
        account_balance: float,
        risk_percent: float,
        instrument_info: Dict[str, Any]
    ) -> Optional[TradingSignal]:
        """Check if current price triggers an entry signal"""
        zones = self.get_active_zones(symbol)
        
        for zone in zones:
            # Check if price is entering the zone
            if zone.contains_price(current_price):
                # Check if this is first touch (fresh zone)
                if zone.touches == 0:
                    # Generate trading signal
                    signal = self._create_signal(
                        zone, 
                        current_price, 
                        account_balance,
                        risk_percent,
                        instrument_info
                    )
                    
                    if signal:
                        # Mark zone as tested
                        zone.touches += 1
                        zone.status = ZoneStatus.TESTED
                        
                        # Store active signal
                        self.active_signals[symbol] = signal
                        
                        logger.info(f"Entry signal generated for {symbol}: "
                                  f"{signal.side} at {signal.entry_price:.4f}")
                        
                        return signal
        
        return None
    
    def _create_signal(
        self,
        zone: Zone,
        current_price: float,
        account_balance: float,
        risk_percent: float,
        instrument_info: Dict[str, Any]
    ) -> Optional[TradingSignal]:
        """Create a trading signal from a zone"""
        from ..utils.rounding import (
            calculate_position_size,
            calculate_stop_loss,
            calculate_take_profits
        )
        
        # Determine side based on zone type
        if zone.zone_type == ZoneType.DEMAND:
            side = "Buy"
            is_long = True
            entry_price = zone.upper_bound  # Enter at top of demand zone
            zone_edge = zone.lower_bound
        else:  # SUPPLY
            side = "Sell"
            is_long = False
            entry_price = zone.lower_bound  # Enter at bottom of supply zone
            zone_edge = zone.upper_bound
        
        # Calculate stop loss
        stop_loss = calculate_stop_loss(
            entry_price,
            zone_edge,
            settings.sd_zone_buffer_percent,
            float(instrument_info['tick_size']),
            is_long
        )
        
        # Calculate take profits
        tp1, tp2 = calculate_take_profits(
            entry_price,
            stop_loss,
            settings.tp1_risk_ratio,
            settings.tp2_risk_ratio,
            float(instrument_info['tick_size']),
            is_long
        )
        
        # Calculate position size
        stop_distance_percent = abs(entry_price - stop_loss) / entry_price * 100
        
        position_size = calculate_position_size(
            account_balance,
            risk_percent,
            stop_distance_percent,
            entry_price,
            float(instrument_info['qty_step']),
            float(instrument_info.get('min_notional', 0))
        )
        
        if position_size <= 0:
            logger.warning(f"Position size too small for {zone.symbol}")
            return None
        
        # Create signal
        signal = TradingSignal(
            zone=zone,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit_1=tp1,
            take_profit_2=tp2,
            position_size=position_size,
            side=side,
            confidence=zone.score / 100,
            reason=f"{zone.zone_type.value} zone entry (score: {zone.score:.1f})"
        )
        
        return signal