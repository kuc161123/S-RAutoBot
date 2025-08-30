"""
Simple RSI + MACD trading strategy
"""
import pandas as pd
from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime
import structlog

logger = structlog.get_logger(__name__)

@dataclass
class TradingSignal:
    """Trading signal data"""
    symbol: str
    action: str  # BUY, SELL, or HOLD
    price: float
    confidence: float  # 0-1 confidence score
    reason: str
    stop_loss: float
    take_profit: float
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

class SimpleStrategy:
    """Simple RSI + MACD strategy"""
    
    def __init__(self, config: dict):
        self.config = config
        self.rsi_oversold = config.get('rsi_oversold', 30)
        self.rsi_overbought = config.get('rsi_overbought', 70)
        self.min_volume_multiplier = 1.5  # Volume should be 1.5x average
        
        logger.info(f"Strategy initialized - RSI: {self.rsi_oversold}/{self.rsi_overbought}")
    
    def analyze(self, symbol: str, df: pd.DataFrame) -> Optional[TradingSignal]:
        """Analyze symbol for trading signals"""
        try:
            # Need at least 200 candles for proper analysis
            if len(df) < 200:
                return None
            
            # Get latest values
            current = df.iloc[-1]
            prev = df.iloc[-2]
            
            # Check if indicators are valid
            if pd.isna(current['rsi']) or pd.isna(current['macd']):
                return None
            
            # Get current price
            price = current['close']
            
            # Calculate ATR for stop loss
            atr = current.get('atr', price * 0.02)  # Default 2% if ATR not available
            
            # Volume check - must be above average
            volume_ok = current['volume'] > (current.get('volume_ma', current['volume']) * self.min_volume_multiplier)
            
            # Generate signal
            signal = self._generate_signal(current, prev, price, atr, volume_ok)
            
            if signal:
                signal.symbol = symbol
                logger.info(f"Signal generated for {symbol}: {signal.action} at {price:.4f} ({signal.reason})")
                
            return signal
            
        except Exception as e:
            logger.error(f"Error analyzing {symbol}: {e}")
            return None
    
    def _generate_signal(self, current: pd.Series, prev: pd.Series, 
                        price: float, atr: float, volume_ok: bool) -> Optional[TradingSignal]:
        """Generate trading signal based on indicators"""
        
        # BUY Signal Conditions
        buy_conditions = []
        
        # RSI oversold
        if current['rsi'] < self.rsi_oversold:
            buy_conditions.append("RSI oversold")
        
        # MACD bullish crossover
        if (prev['macd'] < prev['macd_signal'] and 
            current['macd'] > current['macd_signal']):
            buy_conditions.append("MACD bullish cross")
        
        # Price above Bollinger lower band (potential bounce)
        if (current['close'] < current['bb_lower'] * 1.01 and 
            current['close'] > current['bb_lower'] * 0.99):
            buy_conditions.append("BB lower band bounce")
        
        # Trend confirmation - price above SMA 200
        if current['close'] > current.get('sma_200', current['close']):
            buy_conditions.append("Above SMA 200")
        
        # SELL Signal Conditions
        sell_conditions = []
        
        # RSI overbought
        if current['rsi'] > self.rsi_overbought:
            sell_conditions.append("RSI overbought")
        
        # MACD bearish crossover
        if (prev['macd'] > prev['macd_signal'] and 
            current['macd'] < current['macd_signal']):
            sell_conditions.append("MACD bearish cross")
        
        # Price at Bollinger upper band (potential resistance)
        if (current['close'] > current['bb_upper'] * 0.99 and 
            current['close'] < current['bb_upper'] * 1.01):
            sell_conditions.append("BB upper band resistance")
        
        # Generate signal if conditions are met
        if len(buy_conditions) >= 2 and volume_ok:
            # Calculate stop loss and take profit
            stop_loss = price - (atr * 2)  # 2 ATR stop loss
            take_profit = price + (atr * 4)  # 4 ATR take profit (2:1 RR)
            
            confidence = min(len(buy_conditions) / 4, 1.0)  # More conditions = higher confidence
            
            return TradingSignal(
                symbol="",  # Will be set by caller
                action="BUY",
                price=price,
                confidence=confidence,
                reason=", ".join(buy_conditions),
                stop_loss=stop_loss,
                take_profit=take_profit
            )
        
        elif len(sell_conditions) >= 2 and volume_ok:
            # For short positions
            stop_loss = price + (atr * 2)  # 2 ATR stop loss
            take_profit = price - (atr * 4)  # 4 ATR take profit (2:1 RR)
            
            confidence = min(len(sell_conditions) / 4, 1.0)
            
            return TradingSignal(
                symbol="",
                action="SELL",
                price=price,
                confidence=confidence,
                reason=", ".join(sell_conditions),
                stop_loss=stop_loss,
                take_profit=take_profit
            )
        
        return None
    
    def scan_symbols(self, market_data: Dict[str, pd.DataFrame]) -> List[TradingSignal]:
        """Scan all symbols for signals"""
        signals = []
        
        for symbol, df in market_data.items():
            signal = self.analyze(symbol, df)
            if signal:
                signals.append(signal)
        
        # Sort by confidence
        signals.sort(key=lambda x: x.confidence, reverse=True)
        
        logger.info(f"Scan complete: {len(signals)} signals found from {len(market_data)} symbols")
        
        return signals