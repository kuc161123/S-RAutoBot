"""
Aggressive Strategy - More signals for active trading
Simplified conditions to ensure trades happen
"""
import pandas as pd
from typing import Optional
from dataclasses import dataclass
from datetime import datetime
import structlog

logger = structlog.get_logger(__name__)

@dataclass
class TradingSignal:
    """Trading signal data"""
    symbol: str
    action: str
    price: float
    confidence: float
    reason: str
    stop_loss: float
    take_profit: float
    signal_type: str = "AGGRESSIVE"
    risk_reward: float = 1.5
    timestamp: datetime = None
    trade_id: Optional[str] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

class AggressiveStrategy:
    """Aggressive strategy for more frequent trading"""
    
    def __init__(self, config: dict):
        self.config = config
        
        # Much looser settings for more signals
        self.rsi_oversold = 35  # Was 30
        self.rsi_overbought = 65  # Was 70
        self.min_volume_multiplier = 0.8  # Was 1.5
        
        # Risk settings from config
        self.rr_sl_multiplier = config.get('rr_sl_multiplier', 1.5)
        self.rr_tp_multiplier = config.get('rr_tp_multiplier', 2.5)
        
        logger.info(f"Aggressive strategy initialized - RSI: {self.rsi_oversold}/{self.rsi_overbought}")
    
    def analyze(self, symbol: str, df: pd.DataFrame) -> Optional[TradingSignal]:
        """Analyze with simplified conditions for more signals"""
        try:
            if len(df) < 50:  # Need less data
                return None
            
            current = df.iloc[-1]
            prev = df.iloc[-2]
            
            # Get current price
            price = current['close']
            
            # Simple ATR for stops
            atr = current.get('atr', price * 0.015)
            
            # Volume check - much looser
            volume_ok = True  # Always pass volume check
            if 'volume' in current and 'volume_ma' in current:
                volume_ok = current['volume'] > (current.get('volume_ma', current['volume']) * self.min_volume_multiplier)
            
            # SIMPLIFIED BUY CONDITIONS - Just need 2
            buy_score = 0
            buy_reasons = []
            
            # 1. RSI oversold
            if 'rsi' in current and current['rsi'] < self.rsi_oversold:
                buy_score += 2
                buy_reasons.append(f"RSI oversold ({current['rsi']:.1f})")
            
            # 2. Price bounce from support (Bollinger)
            if 'bb_lower' in current and current['close'] < current['bb_lower'] * 1.02:
                buy_score += 2
                buy_reasons.append("Near BB lower")
            
            # 3. MACD turning bullish
            if 'macd' in current and 'macd_signal' in current:
                if current['macd'] > prev['macd']:  # Just improving
                    buy_score += 1
                    buy_reasons.append("MACD improving")
            
            # 4. Price above recent low
            recent_low = df['low'].tail(10).min()
            if price > recent_low * 1.001:
                buy_score += 1
                buy_reasons.append("Above recent low")
            
            # SIMPLIFIED SELL CONDITIONS
            sell_score = 0
            sell_reasons = []
            
            # 1. RSI overbought
            if 'rsi' in current and current['rsi'] > self.rsi_overbought:
                sell_score += 2
                sell_reasons.append(f"RSI overbought ({current['rsi']:.1f})")
            
            # 2. Price at resistance (Bollinger)
            if 'bb_upper' in current and current['close'] > current['bb_upper'] * 0.98:
                sell_score += 2
                sell_reasons.append("Near BB upper")
            
            # 3. MACD turning bearish
            if 'macd' in current and 'macd_signal' in current:
                if current['macd'] < prev['macd']:  # Just declining
                    sell_score += 1
                    sell_reasons.append("MACD declining")
            
            # 4. Price below recent high
            recent_high = df['high'].tail(10).max()
            if price < recent_high * 0.999:
                sell_score += 1
                sell_reasons.append("Below recent high")
            
            # Generate signal with VERY LOW requirements
            if buy_score >= 2 and buy_score > sell_score:  # Just need score of 2!
                stop_loss = price - (atr * self.rr_sl_multiplier)
                take_profit = price + (atr * self.rr_tp_multiplier)
                
                confidence = min(buy_score / 4, 1.0)
                
                logger.info(f"BUY signal for {symbol}: Score {buy_score}, Reasons: {buy_reasons}")
                
                return TradingSignal(
                    symbol=symbol,
                    action="BUY",
                    price=price,
                    confidence=confidence,
                    reason=", ".join(buy_reasons[:2]),
                    stop_loss=stop_loss,
                    take_profit=take_profit
                )
            
            elif sell_score >= 2 and sell_score > buy_score:
                stop_loss = price + (atr * self.rr_sl_multiplier)
                take_profit = price - (atr * self.rr_tp_multiplier)
                
                confidence = min(sell_score / 4, 1.0)
                
                logger.info(f"SELL signal for {symbol}: Score {sell_score}, Reasons: {sell_reasons}")
                
                return TradingSignal(
                    symbol=symbol,
                    action="SELL",
                    price=price,
                    confidence=confidence,
                    reason=", ".join(sell_reasons[:2]),
                    stop_loss=stop_loss,
                    take_profit=take_profit
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Error analyzing {symbol}: {e}")
            return None
    
    def scan_symbols(self, market_data: dict) -> list:
        """Scan all symbols for signals"""
        signals = []
        
        for symbol, df in market_data.items():
            signal = self.analyze(symbol, df)
            if signal:
                signals.append(signal)
        
        # Sort by confidence
        signals.sort(key=lambda x: x.confidence, reverse=True)
        
        logger.info(f"Aggressive scan: {len(signals)} signals from {len(market_data)} symbols")
        
        return signals[:10]  # Return up to 10 signals