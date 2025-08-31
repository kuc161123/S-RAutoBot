"""
Enhanced Aggressive Strategy - Combines best of scalping and aggressive approaches
Includes trend analysis, market structure, and dynamic R:R
"""
import pandas as pd
import numpy as np
from typing import Optional, List, Dict
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
    """Enhanced aggressive strategy with trend and structure analysis"""
    
    def __init__(self, config: dict):
        self.config = config
        
        # Configurable RSI settings from environment
        self.rsi_oversold = config.get('rsi_oversold', 35)
        self.rsi_overbought = config.get('rsi_overbought', 65)
        
        # Volume filter - higher = fewer but better signals
        self.min_volume_multiplier = config.get('min_volume_multiplier', 1.2)
        
        # Minimum score for signal quality (3-5 recommended)
        self.min_score = config.get('min_signal_score', 3)
        
        # Risk settings from config
        self.rr_sl_multiplier = config.get('rr_sl_multiplier', 1.5)
        self.rr_tp_multiplier = config.get('rr_tp_multiplier', 2.5)
        
        # Scalping features for better accuracy
        self.scalp_rr_sl_multiplier = config.get('scalp_rr_sl_multiplier', 1.0)
        self.scalp_rr_tp_multiplier = config.get('scalp_rr_tp_multiplier', 1.5)
        
        # Trend analysis (from scalping)
        self.trend_ema_fast = 9
        self.trend_ema_slow = 21
        self.use_trend_filter = True
        
        logger.info(f"Enhanced aggressive strategy initialized - RSI: {self.rsi_oversold}/{self.rsi_overbought}, Min score: {self.min_score}")
    
    def analyze(self, symbol: str, df: pd.DataFrame) -> Optional[TradingSignal]:
        """Enhanced analysis with trend and market structure"""
        try:
            if len(df) < 50:  # Need less data
                return None
            
            # Calculate EMAs for trend if not present
            if 'ema_9' not in df.columns:
                df['ema_9'] = df['close'].ewm(span=self.trend_ema_fast, adjust=False).mean()
            if 'ema_21' not in df.columns:
                df['ema_21'] = df['close'].ewm(span=self.trend_ema_slow, adjust=False).mean()
            
            current = df.iloc[-1]
            prev = df.iloc[-2]
            
            # Get current price
            price = current['close']
            
            # Analyze market trend
            ema_fast = current['ema_9']
            ema_slow = current['ema_21']
            trend = "NEUTRAL"
            
            if ema_fast > ema_slow and price > ema_fast:
                trend = "BULLISH"
            elif ema_fast < ema_slow and price < ema_fast:
                trend = "BEARISH"
            
            # Simple ATR for stops
            atr = current.get('atr', price * 0.015)
            
            # Volume check - much looser
            volume_ok = True  # Always pass volume check
            if 'volume' in current and 'volume_ma' in current:
                volume_ok = current['volume'] > (current.get('volume_ma', current['volume']) * self.min_volume_multiplier)
            
            # ENHANCED BUY CONDITIONS with trend
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
            
            # 5. TREND BONUS - Trade with trend for better win rate
            if trend == "BULLISH" and buy_score > 0:
                buy_score += 1
                buy_reasons.append("Bullish trend")
            
            # ENHANCED SELL CONDITIONS with trend
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
            
            # 5. TREND BONUS - Trade with trend for better win rate
            if trend == "BEARISH" and sell_score > 0:
                sell_score += 1
                sell_reasons.append("Bearish trend")
            
            # Generate signal with BALANCED requirements
            # Require higher score for better quality signals
            min_score = self.min_score if hasattr(self, 'min_score') else 3
            if buy_score >= min_score and buy_score > sell_score and volume_ok:
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
            
            elif sell_score >= min_score and sell_score > buy_score and volume_ok:
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