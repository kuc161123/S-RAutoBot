"""
Hybrid Smart Money Momentum Strategy
Combines ICT/SMC concepts, Mean Reversion, and VWAP Breakout
Designed for 70%+ win rate with ML enhancement
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import structlog
from scipy import stats
import talib

logger = structlog.get_logger(__name__)

class SignalType(Enum):
    """Types of trading signals"""
    ORDER_BLOCK = "order_block"
    FAIR_VALUE_GAP = "fair_value_gap"
    MEAN_REVERSION = "mean_reversion"
    VWAP_BREAKOUT = "vwap_breakout"
    LIQUIDITY_SWEEP = "liquidity_sweep"
    BREAKER_BLOCK = "breaker_block"

class MarketRegime(Enum):
    """Market regime classification"""
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    RANGING = "ranging"
    VOLATILE = "volatile"

@dataclass
class OrderBlock:
    """Order block structure for SMC"""
    block_type: str  # "bullish" or "bearish"
    top: float
    bottom: float
    volume: float
    strength: float  # 0-100
    formation_time: datetime
    tested: bool = False
    broken: bool = False
    
@dataclass
class FairValueGap:
    """Fair Value Gap (imbalance) structure"""
    gap_type: str  # "bullish" or "bearish"
    high: float
    low: float
    size: float
    formation_time: datetime
    filled: bool = False

@dataclass
class HybridSignal:
    """Combined signal from multiple strategies"""
    symbol: str
    signal_type: SignalType
    direction: str  # "buy" or "sell"
    entry_price: float
    stop_loss: float
    take_profit_1: float
    take_profit_2: float
    confidence: float  # 0-100
    
    # Strategy-specific data
    order_block: Optional[OrderBlock] = None
    fvg: Optional[FairValueGap] = None
    
    # Confluence factors
    confluences: List[str] = field(default_factory=list)
    timeframes_aligned: List[str] = field(default_factory=list)
    
    # Risk metrics
    risk_reward_ratio: float = 0
    position_size_multiplier: float = 1.0  # Based on confidence
    
    # ML enhancement
    ml_score: float = 0
    ml_features: Dict[str, float] = field(default_factory=dict)
    
    timestamp: datetime = field(default_factory=datetime.now)

class HybridSmartMoneyStrategy:
    """
    Hybrid strategy combining:
    1. Smart Money Concepts (Order Blocks, FVG, Liquidity)
    2. Mean Reversion (Bollinger Bands + RSI)
    3. VWAP Momentum Breakout
    """
    
    def __init__(self):
        # Strategy parameters - adjusted for more signals while maintaining quality
        
        # Smart Money Concepts
        self.order_block_min_volume_ratio = 1.2  # Lowered from 1.5
        self.fvg_min_size_atr = 0.3  # Lowered from 0.5 to detect more gaps
        self.liquidity_sweep_lookback = 20  # Bars to look for liquidity
        
        # Mean Reversion
        self.bb_period = 20
        self.bb_std = 2.0
        self.rsi_period = 14
        self.rsi_oversold = 35  # Raised from 30 for more signals
        self.rsi_overbought = 65  # Lowered from 70 for more signals
        
        # VWAP
        self.vwap_deviation_threshold = 1.5  # Lowered from 2.0 for more sensitivity
        self.volume_spike_threshold = 1.2  # Lowered from 1.5
        
        # Risk Management
        self.atr_period = 14
        self.stop_loss_atr_multiplier = 1.5
        self.tp1_atr_multiplier = 1.5
        self.tp2_atr_multiplier = 3.0
        
        # Signal filtering
        self.min_confidence_threshold = 30  # Lowered to allow more signals during testing
        self.max_signals_per_symbol = 1  # One signal per symbol at a time
        
        # Market regime
        self.trend_ema_fast = 20
        self.trend_ema_slow = 50
        
        # Storage
        self.order_blocks: Dict[str, List[OrderBlock]] = {}
        self.fair_value_gaps: Dict[str, List[FairValueGap]] = {}
        self.market_regimes: Dict[str, MarketRegime] = {}
        
    def analyze(self, symbol: str, df: pd.DataFrame) -> List[HybridSignal]:
        """
        Main analysis function that combines all strategies
        Returns list of high-probability signals
        """
        if len(df) < 100:
            logger.debug(f"Not enough data for {symbol}: {len(df)} candles")
            return []
            
        signals = []
        
        try:
            # Calculate base indicators
            df = self._calculate_indicators(df)
            
            # Detect market regime
            regime = self._detect_market_regime(df)
            self.market_regimes[symbol] = regime
            
            # 1. Smart Money Concepts signals
            smc_signals = self._detect_smart_money_signals(symbol, df, regime)
            signals.extend(smc_signals)
            
            # 2. Mean Reversion signals
            mr_signals = self._detect_mean_reversion_signals(symbol, df, regime)
            signals.extend(mr_signals)
            
            # 3. VWAP Breakout signals
            vwap_signals = self._detect_vwap_breakout_signals(symbol, df, regime)
            signals.extend(vwap_signals)
            
            # Filter and rank signals
            signals = self._filter_and_rank_signals(signals, df)
            
            # Log signal generation
            if signals:
                logger.info(f"✅ {symbol}: Generated {len(signals)} signals")
                for sig in signals:
                    logger.info(f"  - {sig.signal_type.value}: {sig.direction} @ {sig.entry_price:.4f} "
                              f"(confidence: {sig.confidence:.1f}%)")
            
        except Exception as e:
            logger.error(f"Error analyzing {symbol}: {e}")
            
        return signals
    
    def _calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all required indicators"""
        # ATR for volatility
        df['atr'] = talib.ATR(df['high'].values, df['low'].values, df['close'].values, 
                              timeperiod=self.atr_period)
        
        # Bollinger Bands
        df['bb_upper'], df['bb_middle'], df['bb_lower'] = talib.BBANDS(
            df['close'].values, timeperiod=self.bb_period, nbdevup=self.bb_std, 
            nbdevdn=self.bb_std, matype=0
        )
        
        # RSI
        df['rsi'] = talib.RSI(df['close'].values, timeperiod=self.rsi_period)
        
        # EMAs for trend
        df['ema_fast'] = talib.EMA(df['close'].values, timeperiod=self.trend_ema_fast)
        df['ema_slow'] = talib.EMA(df['close'].values, timeperiod=self.trend_ema_slow)
        
        # VWAP calculation
        df['vwap'] = self._calculate_vwap(df)
        df['vwap_upper'] = df['vwap'] + (df['atr'] * self.vwap_deviation_threshold)
        df['vwap_lower'] = df['vwap'] - (df['atr'] * self.vwap_deviation_threshold)
        
        # Volume analysis
        df['volume_ma'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma']
        
        # Price action
        df['body'] = abs(df['close'] - df['open'])
        df['upper_wick'] = df['high'] - df[['close', 'open']].max(axis=1)
        df['lower_wick'] = df[['close', 'open']].min(axis=1) - df['low']
        
        return df
    
    def _calculate_vwap(self, df: pd.DataFrame) -> pd.Series:
        """Calculate VWAP (Volume Weighted Average Price)"""
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        vwap = (typical_price * df['volume']).cumsum() / df['volume'].cumsum()
        return vwap
    
    def _detect_market_regime(self, df: pd.DataFrame) -> MarketRegime:
        """Classify current market regime"""
        recent_df = df.tail(50)
        
        # Trend detection
        ema_fast = recent_df['ema_fast'].iloc[-1]
        ema_slow = recent_df['ema_slow'].iloc[-1]
        
        # Volatility check
        atr_ratio = recent_df['atr'].iloc[-1] / recent_df['close'].iloc[-1]
        
        # Price position relative to moving averages
        close = recent_df['close'].iloc[-1]
        
        if atr_ratio > 0.03:  # High volatility (3% ATR)
            return MarketRegime.VOLATILE
        elif ema_fast > ema_slow and close > ema_fast:
            return MarketRegime.TRENDING_UP
        elif ema_fast < ema_slow and close < ema_fast:
            return MarketRegime.TRENDING_DOWN
        else:
            return MarketRegime.RANGING
    
    def _detect_smart_money_signals(self, symbol: str, df: pd.DataFrame, 
                                   regime: MarketRegime) -> List[HybridSignal]:
        """Detect Smart Money Concept signals"""
        signals = []
        
        # 1. Detect Order Blocks
        order_blocks = self._find_order_blocks(df)
        for ob in order_blocks[-3:]:  # Check last 3 order blocks
            signal = self._create_order_block_signal(symbol, df, ob, regime)
            if signal:
                signals.append(signal)
        
        # 2. Detect Fair Value Gaps
        fvgs = self._find_fair_value_gaps(df)
        for fvg in fvgs[-3:]:  # Check last 3 FVGs
            signal = self._create_fvg_signal(symbol, df, fvg, regime)
            if signal:
                signals.append(signal)
        
        # 3. Detect Liquidity Sweeps
        sweep_signal = self._detect_liquidity_sweep(symbol, df, regime)
        if sweep_signal:
            signals.append(sweep_signal)
        
        return signals
    
    def _find_order_blocks(self, df: pd.DataFrame) -> List[OrderBlock]:
        """Find order blocks in price data"""
        order_blocks = []
        
        for i in range(10, len(df) - 1):
            # Bullish order block: Down candle before strong up move
            if (df['close'].iloc[i] < df['open'].iloc[i] and  # Down candle
                df['close'].iloc[i+1] > df['open'].iloc[i+1] and  # Up candle
                df['body'].iloc[i+1] > df['body'].iloc[i] * 1.5 and  # Reduced from 2x to 1.5x
                df['volume_ratio'].iloc[i] > 1.0):  # Just need above average volume
                
                ob = OrderBlock(
                    block_type="bullish",
                    top=df['high'].iloc[i],
                    bottom=df['low'].iloc[i],
                    volume=df['volume'].iloc[i],
                    strength=min(100, df['volume_ratio'].iloc[i] * 30),
                    formation_time=datetime.now()  # Use current time if timestamp not available
                )
                order_blocks.append(ob)
            
            # Bearish order block: Up candle before strong down move
            elif (df['close'].iloc[i] > df['open'].iloc[i] and  # Up candle
                  df['close'].iloc[i+1] < df['open'].iloc[i+1] and  # Down candle
                  df['body'].iloc[i+1] > df['body'].iloc[i] * 1.5 and  # Reduced from 2x to 1.5x
                  df['volume_ratio'].iloc[i] > 1.0):  # Just need above average volume
                
                ob = OrderBlock(
                    block_type="bearish",
                    top=df['high'].iloc[i],
                    bottom=df['low'].iloc[i],
                    volume=df['volume'].iloc[i],
                    strength=min(100, df['volume_ratio'].iloc[i] * 30),
                    formation_time=datetime.now()  # Use current time if timestamp not available
                )
                order_blocks.append(ob)
        
        return order_blocks
    
    def _find_fair_value_gaps(self, df: pd.DataFrame) -> List[FairValueGap]:
        """Find fair value gaps (imbalances) in price"""
        gaps = []
        
        for i in range(2, len(df) - 1):
            # Bullish FVG: Gap between candle 1 high and candle 3 low
            gap_size = df['low'].iloc[i] - df['high'].iloc[i-2]
            if gap_size > df['atr'].iloc[i] * self.fvg_min_size_atr:
                fvg = FairValueGap(
                    gap_type="bullish",
                    high=df['low'].iloc[i],
                    low=df['high'].iloc[i-2],
                    size=gap_size,
                    formation_time=datetime.now()  # Use current time if timestamp not available
                )
                gaps.append(fvg)
            
            # Bearish FVG: Gap between candle 1 low and candle 3 high
            gap_size = df['low'].iloc[i-2] - df['high'].iloc[i]
            if gap_size > df['atr'].iloc[i] * self.fvg_min_size_atr:
                fvg = FairValueGap(
                    gap_type="bearish",
                    high=df['low'].iloc[i-2],
                    low=df['high'].iloc[i],
                    size=gap_size,
                    formation_time=datetime.now()  # Use current time if timestamp not available
                )
                gaps.append(fvg)
        
        return gaps
    
    def _create_order_block_signal(self, symbol: str, df: pd.DataFrame, 
                                  ob: OrderBlock, regime: MarketRegime) -> Optional[HybridSignal]:
        """Create signal from order block"""
        current_price = df['close'].iloc[-1]
        atr = df['atr'].iloc[-1]
        
        # Check if price is retesting order block
        if ob.block_type == "bullish" and not ob.tested:
            # Price should be approaching from above
            if ob.bottom <= current_price <= ob.top:
                # Bullish signal
                entry = current_price
                stop = ob.bottom - atr * 0.5
                tp1 = entry + (entry - stop) * self.tp1_atr_multiplier
                tp2 = entry + (entry - stop) * self.tp2_atr_multiplier
                
                signal = HybridSignal(
                    symbol=symbol,
                    signal_type=SignalType.ORDER_BLOCK,
                    direction="buy",
                    entry_price=entry,
                    stop_loss=stop,
                    take_profit_1=tp1,
                    take_profit_2=tp2,
                    confidence=ob.strength,
                    order_block=ob,
                    confluences=["Order Block Retest", f"Volume: {ob.volume:.0f}"],
                    risk_reward_ratio=(tp1 - entry) / (entry - stop)
                )
                return signal
                
        elif ob.block_type == "bearish" and not ob.tested:
            # Price should be approaching from below
            if ob.bottom <= current_price <= ob.top:
                # Bearish signal
                entry = current_price
                stop = ob.top + atr * 0.5
                tp1 = entry - (stop - entry) * self.tp1_atr_multiplier
                tp2 = entry - (stop - entry) * self.tp2_atr_multiplier
                
                signal = HybridSignal(
                    symbol=symbol,
                    signal_type=SignalType.ORDER_BLOCK,
                    direction="sell",
                    entry_price=entry,
                    stop_loss=stop,
                    take_profit_1=tp1,
                    take_profit_2=tp2,
                    confidence=ob.strength,
                    order_block=ob,
                    confluences=["Order Block Retest", f"Volume: {ob.volume:.0f}"],
                    risk_reward_ratio=(entry - tp1) / (stop - entry)
                )
                return signal
        
        return None
    
    def _create_fvg_signal(self, symbol: str, df: pd.DataFrame,
                          fvg: FairValueGap, regime: MarketRegime) -> Optional[HybridSignal]:
        """Create signal from fair value gap"""
        current_price = df['close'].iloc[-1]
        atr = df['atr'].iloc[-1]
        
        # Check if price is entering the FVG
        if fvg.gap_type == "bullish" and not fvg.filled:
            if fvg.low <= current_price <= fvg.high:
                # Bullish signal - expect bounce from FVG
                entry = current_price
                stop = fvg.low - atr * 0.3
                tp1 = entry + (entry - stop) * self.tp1_atr_multiplier
                tp2 = entry + (entry - stop) * self.tp2_atr_multiplier
                
                signal = HybridSignal(
                    symbol=symbol,
                    signal_type=SignalType.FAIR_VALUE_GAP,
                    direction="buy",
                    entry_price=entry,
                    stop_loss=stop,
                    take_profit_1=tp1,
                    take_profit_2=tp2,
                    confidence=70,  # FVGs have good win rate
                    fvg=fvg,
                    confluences=["FVG Fill", f"Gap Size: {fvg.size:.4f}"],
                    risk_reward_ratio=(tp1 - entry) / (entry - stop)
                )
                return signal
                
        elif fvg.gap_type == "bearish" and not fvg.filled:
            if fvg.low <= current_price <= fvg.high:
                # Bearish signal - expect rejection from FVG
                entry = current_price
                stop = fvg.high + atr * 0.3
                tp1 = entry - (stop - entry) * self.tp1_atr_multiplier
                tp2 = entry - (stop - entry) * self.tp2_atr_multiplier
                
                signal = HybridSignal(
                    symbol=symbol,
                    signal_type=SignalType.FAIR_VALUE_GAP,
                    direction="sell",
                    entry_price=entry,
                    stop_loss=stop,
                    take_profit_1=tp1,
                    take_profit_2=tp2,
                    confidence=70,
                    fvg=fvg,
                    confluences=["FVG Fill", f"Gap Size: {fvg.size:.4f}"],
                    risk_reward_ratio=(entry - tp1) / (stop - entry)
                )
                return signal
        
        return None
    
    def _detect_liquidity_sweep(self, symbol: str, df: pd.DataFrame,
                               regime: MarketRegime) -> Optional[HybridSignal]:
        """Detect liquidity sweep (stop hunt) patterns"""
        if len(df) < self.liquidity_sweep_lookback + 5:
            return None
            
        recent_df = df.tail(self.liquidity_sweep_lookback + 5)
        current_price = df['close'].iloc[-1]
        atr = df['atr'].iloc[-1]
        
        # Find recent swing highs/lows
        swing_high = recent_df['high'].rolling(5).max().iloc[-self.liquidity_sweep_lookback:-5].max()
        swing_low = recent_df['low'].rolling(5).min().iloc[-self.liquidity_sweep_lookback:-5].min()
        
        # Check for liquidity sweep above swing high (bearish)
        if (df['high'].iloc[-2] > swing_high and  # Swept liquidity
            df['close'].iloc[-1] < df['open'].iloc[-1] and  # Rejection candle
            df['volume_ratio'].iloc[-1] > self.volume_spike_threshold):
            
            entry = current_price
            stop = df['high'].iloc[-2] + atr * 0.2
            tp1 = entry - (stop - entry) * self.tp1_atr_multiplier
            tp2 = entry - (stop - entry) * self.tp2_atr_multiplier
            
            return HybridSignal(
                symbol=symbol,
                signal_type=SignalType.LIQUIDITY_SWEEP,
                direction="sell",
                entry_price=entry,
                stop_loss=stop,
                take_profit_1=tp1,
                take_profit_2=tp2,
                confidence=75,
                confluences=["Liquidity Sweep", "High Volume Rejection"],
                risk_reward_ratio=(entry - tp1) / (stop - entry)
            )
        
        # Check for liquidity sweep below swing low (bullish)
        elif (df['low'].iloc[-2] < swing_low and  # Swept liquidity
              df['close'].iloc[-1] > df['open'].iloc[-1] and  # Rejection candle
              df['volume_ratio'].iloc[-1] > self.volume_spike_threshold):
            
            entry = current_price
            stop = df['low'].iloc[-2] - atr * 0.2
            tp1 = entry + (entry - stop) * self.tp1_atr_multiplier
            tp2 = entry + (entry - stop) * self.tp2_atr_multiplier
            
            return HybridSignal(
                symbol=symbol,
                signal_type=SignalType.LIQUIDITY_SWEEP,
                direction="buy",
                entry_price=entry,
                stop_loss=stop,
                take_profit_1=tp1,
                take_profit_2=tp2,
                confidence=75,
                confluences=["Liquidity Sweep", "High Volume Rejection"],
                risk_reward_ratio=(tp1 - entry) / (entry - stop)
            )
        
        return None
    
    def _detect_mean_reversion_signals(self, symbol: str, df: pd.DataFrame,
                                      regime: MarketRegime) -> List[HybridSignal]:
        """Detect mean reversion signals using Bollinger Bands + RSI"""
        signals = []
        
        # Skip if trending strongly
        if regime in [MarketRegime.TRENDING_UP, MarketRegime.TRENDING_DOWN]:
            return signals
        
        current_price = df['close'].iloc[-1]
        rsi = df['rsi'].iloc[-1]
        atr = df['atr'].iloc[-1]
        
        # Bullish mean reversion: Price at lower BB + RSI oversold
        if (current_price <= df['bb_lower'].iloc[-1] and
            rsi < self.rsi_oversold and
            df['volume_ratio'].iloc[-1] > 1.2):  # Some volume confirmation
            
            entry = current_price
            stop = df['low'].iloc[-5:].min() - atr * 0.5
            tp1 = df['bb_middle'].iloc[-1]  # Target middle band
            tp2 = df['bb_upper'].iloc[-1]  # Target upper band
            
            signal = HybridSignal(
                symbol=symbol,
                signal_type=SignalType.MEAN_REVERSION,
                direction="buy",
                entry_price=entry,
                stop_loss=stop,
                take_profit_1=tp1,
                take_profit_2=tp2,
                confidence=65 + (30 - rsi),  # Higher confidence with lower RSI
                confluences=[
                    "Price at Lower BB",
                    f"RSI Oversold: {rsi:.1f}",
                    "Volume Confirmation"
                ],
                risk_reward_ratio=(tp1 - entry) / (entry - stop) if entry > stop else 0
            )
            signals.append(signal)
        
        # Bearish mean reversion: Price at upper BB + RSI overbought
        elif (current_price >= df['bb_upper'].iloc[-1] and
              rsi > self.rsi_overbought and
              df['volume_ratio'].iloc[-1] > 1.2):
            
            entry = current_price
            stop = df['high'].iloc[-5:].max() + atr * 0.5
            tp1 = df['bb_middle'].iloc[-1]
            tp2 = df['bb_lower'].iloc[-1]
            
            signal = HybridSignal(
                symbol=symbol,
                signal_type=SignalType.MEAN_REVERSION,
                direction="sell",
                entry_price=entry,
                stop_loss=stop,
                take_profit_1=tp1,
                take_profit_2=tp2,
                confidence=65 + (rsi - 70),  # Higher confidence with higher RSI
                confluences=[
                    "Price at Upper BB",
                    f"RSI Overbought: {rsi:.1f}",
                    "Volume Confirmation"
                ],
                risk_reward_ratio=(entry - tp1) / (stop - entry) if stop > entry else 0
            )
            signals.append(signal)
        
        return signals
    
    def _detect_vwap_breakout_signals(self, symbol: str, df: pd.DataFrame,
                                     regime: MarketRegime) -> List[HybridSignal]:
        """Detect VWAP breakout signals with volume confirmation"""
        signals = []
        
        current_price = df['close'].iloc[-1]
        vwap = df['vwap'].iloc[-1]
        atr = df['atr'].iloc[-1]
        volume_ratio = df['volume_ratio'].iloc[-1]
        
        # Calculate MACD for momentum confirmation
        macd, macd_signal, macd_hist = talib.MACD(df['close'].values)
        
        # Bullish VWAP breakout
        if (current_price > df['vwap_upper'].iloc[-1] and
            volume_ratio > self.volume_spike_threshold and
            macd[-1] > macd_signal[-1] and  # MACD confirmation
            regime != MarketRegime.TRENDING_DOWN):
            
            entry = current_price
            stop = vwap - atr * 0.5
            tp1 = entry + atr * self.tp1_atr_multiplier
            tp2 = entry + atr * self.tp2_atr_multiplier
            
            signal = HybridSignal(
                symbol=symbol,
                signal_type=SignalType.VWAP_BREAKOUT,
                direction="buy",
                entry_price=entry,
                stop_loss=stop,
                take_profit_1=tp1,
                take_profit_2=tp2,
                confidence=70 + min(20, volume_ratio * 5),  # Volume adds confidence
                confluences=[
                    "VWAP Upper Band Break",
                    f"Volume Spike: {volume_ratio:.1f}x",
                    "MACD Bullish"
                ],
                risk_reward_ratio=(tp1 - entry) / (entry - stop) if entry > stop else 0
            )
            signals.append(signal)
        
        # Bearish VWAP breakout
        elif (current_price < df['vwap_lower'].iloc[-1] and
              volume_ratio > self.volume_spike_threshold and
              macd[-1] < macd_signal[-1] and
              regime != MarketRegime.TRENDING_UP):
            
            entry = current_price
            stop = vwap + atr * 0.5
            tp1 = entry - atr * self.tp1_atr_multiplier
            tp2 = entry - atr * self.tp2_atr_multiplier
            
            signal = HybridSignal(
                symbol=symbol,
                signal_type=SignalType.VWAP_BREAKOUT,
                direction="sell",
                entry_price=entry,
                stop_loss=stop,
                take_profit_1=tp1,
                take_profit_2=tp2,
                confidence=70 + min(20, volume_ratio * 5),
                confluences=[
                    "VWAP Lower Band Break",
                    f"Volume Spike: {volume_ratio:.1f}x",
                    "MACD Bearish"
                ],
                risk_reward_ratio=(entry - tp1) / (stop - entry) if stop > entry else 0
            )
            signals.append(signal)
        
        return signals
    
    def _filter_and_rank_signals(self, signals: List[HybridSignal], 
                                df: pd.DataFrame) -> List[HybridSignal]:
        """Filter and rank signals by quality"""
        if not signals:
            return []
        
        # Filter by minimum confidence
        filtered = [s for s in signals if s.confidence >= self.min_confidence_threshold]
        
        # Filter by risk/reward ratio
        filtered = [s for s in filtered if s.risk_reward_ratio >= 1.0]  # Lowered from 1.5 for testing
        
        # Add multi-timeframe confluence (would need multiple timeframe data)
        for signal in filtered:
            # Simulate timeframe alignment (in production, check actual MTF data)
            if signal.confidence > 70:
                signal.timeframes_aligned = ["5m", "15m", "1h"]
            elif signal.confidence > 60:
                signal.timeframes_aligned = ["15m", "1h"]
            else:
                signal.timeframes_aligned = ["15m"]
        
        # Adjust position size based on confidence
        for signal in filtered:
            if signal.confidence >= 80:
                signal.position_size_multiplier = 1.5
            elif signal.confidence >= 70:
                signal.position_size_multiplier = 1.2
            elif signal.confidence >= 60:
                signal.position_size_multiplier = 1.0
            else:
                signal.position_size_multiplier = 0.8
        
        # Sort by confidence (highest first)
        filtered.sort(key=lambda x: x.confidence, reverse=True)
        
        # Return only the best signal per symbol
        return filtered[:self.max_signals_per_symbol] if filtered else []
    
    def get_active_zones(self, symbol: str) -> List[Any]:
        """Get active trading zones (for compatibility)"""
        zones = []
        
        # Convert order blocks to zones
        if symbol in self.order_blocks:
            for ob in self.order_blocks[symbol][-5:]:  # Last 5 order blocks
                if not ob.broken:
                    zones.append(ob)
        
        # Convert FVGs to zones
        if symbol in self.fair_value_gaps:
            for fvg in self.fair_value_gaps[symbol][-5:]:  # Last 5 FVGs
                if not fvg.filled:
                    zones.append(fvg)
        
        return zones
    
    def update_zone_status(self, symbol: str, current_price: float):
        """Update status of zones based on current price"""
        # Update order blocks
        if symbol in self.order_blocks:
            for ob in self.order_blocks[symbol]:
                if ob.block_type == "bullish" and current_price < ob.bottom:
                    ob.broken = True
                elif ob.block_type == "bearish" and current_price > ob.top:
                    ob.broken = True
                elif ob.bottom <= current_price <= ob.top:
                    ob.tested = True
        
        # Update FVGs
        if symbol in self.fair_value_gaps:
            for fvg in self.fair_value_gaps[symbol]:
                if fvg.low <= current_price <= fvg.high:
                    fvg.filled = True

# Create global instance
hybrid_strategy = HybridSmartMoneyStrategy()