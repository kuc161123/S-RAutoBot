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
        
        # Removed scalping - using unified R/R multipliers
        
        # Trend analysis (from scalping)
        self.trend_ema_fast = 9
        self.trend_ema_slow = 21
        self.use_trend_filter = True
        
        # Support/Resistance settings
        self.sr_lookback = 50  # Candles to look back for S/R
        self.sr_touches = 2  # Min touches to confirm S/R
        self.sr_tolerance = 0.002  # 0.2% tolerance for S/R levels
        
        # Market structure settings
        self.structure_lookback = 20
        
        logger.info(f"Enhanced aggressive strategy with S/R - RSI: {self.rsi_oversold}/{self.rsi_overbought}, Min score: {self.min_score}")
    
    def _find_support_resistance(self, df: pd.DataFrame) -> tuple:
        """Find support and resistance levels"""
        recent_df = df.tail(self.sr_lookback)
        highs = []
        lows = []
        
        for i in range(1, len(recent_df) - 1):
            # Find local highs (resistance)
            if recent_df.iloc[i]['high'] > recent_df.iloc[i-1]['high'] and \
               recent_df.iloc[i]['high'] > recent_df.iloc[i+1]['high']:
                highs.append(recent_df.iloc[i]['high'])
            
            # Find local lows (support)
            if recent_df.iloc[i]['low'] < recent_df.iloc[i-1]['low'] and \
               recent_df.iloc[i]['low'] < recent_df.iloc[i+1]['low']:
                lows.append(recent_df.iloc[i]['low'])
        
        # Cluster similar levels
        support_levels = self._cluster_levels(lows) if lows else []
        resistance_levels = self._cluster_levels(highs) if highs else []
        
        return support_levels, resistance_levels
    
    def _cluster_levels(self, levels: List[float]) -> List[float]:
        """Cluster similar price levels"""
        if not levels:
            return []
        
        levels = sorted(levels)
        clustered = []
        current_cluster = [levels[0]]
        
        for level in levels[1:]:
            if abs(level - current_cluster[-1]) / current_cluster[-1] < self.sr_tolerance:
                current_cluster.append(level)
            else:
                clustered.append(sum(current_cluster) / len(current_cluster))
                current_cluster = [level]
        
        if current_cluster:
            clustered.append(sum(current_cluster) / len(current_cluster))
        
        return clustered
    
    def _analyze_market_structure(self, df: pd.DataFrame) -> dict:
        """Analyze market structure (trend strength, momentum)"""
        recent_df = df.tail(self.structure_lookback)
        
        # Calculate momentum
        momentum = (recent_df['close'].iloc[-1] - recent_df['close'].iloc[0]) / recent_df['close'].iloc[0] * 100
        
        # Identify higher highs/lows (trend strength)
        highs = recent_df['high'].values
        lows = recent_df['low'].values
        
        higher_highs = sum(1 for i in range(1, len(highs)) if highs[i] > highs[i-1])
        higher_lows = sum(1 for i in range(1, len(lows)) if lows[i] > lows[i-1])
        lower_highs = sum(1 for i in range(1, len(highs)) if highs[i] < highs[i-1])
        lower_lows = sum(1 for i in range(1, len(lows)) if lows[i] < lows[i-1])
        
        # Determine structure
        if higher_highs > lower_highs and higher_lows > lower_lows:
            structure = "BULLISH"
            strength = (higher_highs + higher_lows) / (len(highs) * 2)
        elif lower_highs > higher_highs and lower_lows > higher_lows:
            structure = "BEARISH"
            strength = (lower_highs + lower_lows) / (len(highs) * 2)
        else:
            structure = "RANGING"
            strength = 0.5
        
        return {
            'structure': structure,
            'strength': strength,
            'momentum': momentum
        }
    
    def _detect_price_patterns(self, df: pd.DataFrame) -> dict:
        """Detect price action patterns"""
        current = df.iloc[-1]
        prev = df.iloc[-2]
        
        patterns = {'hammer': False, 'shooting_star': False, 'doji': False}
        
        body = abs(current['close'] - current['open'])
        lower_wick = min(current['open'], current['close']) - current['low']
        upper_wick = current['high'] - max(current['open'], current['close'])
        
        # Hammer: small body, long lower wick
        if lower_wick > body * 2 and upper_wick < body * 0.5:
            patterns['hammer'] = True
        
        # Shooting star: small body, long upper wick
        if upper_wick > body * 2 and lower_wick < body * 0.5:
            patterns['shooting_star'] = True
        
        # Doji: very small body
        if body < (current['high'] - current['low']) * 0.1:
            patterns['doji'] = True
        
        return patterns
    
    def analyze(self, symbol: str, df: pd.DataFrame) -> Optional[TradingSignal]:
        """Enhanced analysis with trend, S/R, and market structure"""
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
            
            # Find support and resistance levels
            support_levels, resistance_levels = self._find_support_resistance(df)
            
            # Analyze market structure
            market_structure = self._analyze_market_structure(df)
            
            # Detect price patterns
            patterns = self._detect_price_patterns(df)
            
            # Find nearest S/R levels
            nearest_support = min(support_levels, key=lambda x: abs(x - price)) if support_levels else 0
            nearest_resistance = min(resistance_levels, key=lambda x: abs(x - price)) if resistance_levels else float('inf')
            
            # Distance to S/R as percentage
            dist_to_support = abs(price - nearest_support) / price * 100 if nearest_support else float('inf')
            dist_to_resistance = abs(price - nearest_resistance) / price * 100 if nearest_resistance else float('inf')
            
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
            
            # ADVANCED BUY CONDITIONS with S/R and market structure
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
            
            # 3. NEAR SUPPORT LEVEL - High priority
            if dist_to_support < 0.5:  # Within 0.5% of support
                buy_score += 2
                buy_reasons.append(f"At support ${nearest_support:.2f}")
            elif dist_to_support < 1.0:  # Within 1% of support
                buy_score += 1
                buy_reasons.append("Near support")
            
            # 4. MACD turning bullish
            if 'macd' in current and 'macd_signal' in current:
                if current['macd'] > prev['macd']:  # Just improving
                    buy_score += 1
                    buy_reasons.append("MACD improving")
            
            # 5. BULLISH MARKET STRUCTURE
            if market_structure['structure'] == "BULLISH":
                buy_score += 1
                buy_reasons.append(f"Bullish structure ({market_structure['strength']:.1%})")
            
            # 6. BULLISH PRICE PATTERN
            if patterns['hammer']:
                buy_score += 1
                buy_reasons.append("Hammer pattern")
            
            # 7. TREND BONUS - Trade with trend for better win rate
            if trend == "BULLISH" and buy_score > 0:
                buy_score += 1
                buy_reasons.append("Bullish trend")
            
            # 8. MOMENTUM CONFIRMATION
            if market_structure['momentum'] > 0.5:
                buy_score += 1
                buy_reasons.append("Positive momentum")
            
            # ADVANCED SELL CONDITIONS with S/R and market structure
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
            
            # 3. NEAR RESISTANCE LEVEL - High priority
            if dist_to_resistance < 0.5:  # Within 0.5% of resistance
                sell_score += 2
                sell_reasons.append(f"At resistance ${nearest_resistance:.2f}")
            elif dist_to_resistance < 1.0:  # Within 1% of resistance
                sell_score += 1
                sell_reasons.append("Near resistance")
            
            # 4. MACD turning bearish
            if 'macd' in current and 'macd_signal' in current:
                if current['macd'] < prev['macd']:  # Just declining
                    sell_score += 1
                    sell_reasons.append("MACD declining")
            
            # 5. BEARISH MARKET STRUCTURE
            if market_structure['structure'] == "BEARISH":
                sell_score += 1
                sell_reasons.append(f"Bearish structure ({market_structure['strength']:.1%})")
            
            # 6. BEARISH PRICE PATTERN
            if patterns['shooting_star']:
                sell_score += 1
                sell_reasons.append("Shooting star pattern")
            
            # 7. TREND BONUS - Trade with trend for better win rate
            if trend == "BEARISH" and sell_score > 0:
                sell_score += 1
                sell_reasons.append("Bearish trend")
            
            # 8. MOMENTUM CONFIRMATION
            if market_structure['momentum'] < -0.5:
                sell_score += 1
                sell_reasons.append("Negative momentum")
            
            # Generate signal with BALANCED requirements
            # Require higher score for better quality signals
            min_score = self.min_score if hasattr(self, 'min_score') else 3
            if buy_score >= min_score and buy_score > sell_score and volume_ok:
                # Smart stop loss placement using support
                if nearest_support and dist_to_support < 2.0:
                    stop_loss = nearest_support * 0.998  # Just below support
                else:
                    stop_loss = price - (atr * self.rr_sl_multiplier)
                
                # Smart take profit using resistance
                if nearest_resistance and dist_to_resistance < 5.0:
                    take_profit = nearest_resistance * 0.998  # Just below resistance
                else:
                    take_profit = price + (atr * self.rr_tp_multiplier)
                
                # Ensure minimum R:R ratio
                if (take_profit - price) / (price - stop_loss) < 1.5:
                    take_profit = price + ((price - stop_loss) * 2.0)
                
                confidence = min(buy_score / 8, 1.0)  # Adjusted for more conditions
                
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
                # Smart stop loss placement using resistance
                if nearest_resistance and dist_to_resistance < 2.0:
                    stop_loss = nearest_resistance * 1.002  # Just above resistance
                else:
                    stop_loss = price + (atr * self.rr_sl_multiplier)
                
                # Smart take profit using support
                if nearest_support and dist_to_support < 5.0:
                    take_profit = nearest_support * 1.002  # Just above support
                else:
                    take_profit = price - (atr * self.rr_tp_multiplier)
                
                # Ensure minimum R:R ratio
                if (price - take_profit) / (stop_loss - price) < 1.5:
                    take_profit = price - ((stop_loss - price) * 2.0)
                
                confidence = min(sell_score / 8, 1.0)  # Adjusted for more conditions
                
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
        """Scan all symbols for signals - NO LIMIT"""
        signals = []
        
        for symbol, df in market_data.items():
            signal = self.analyze(symbol, df)
            if signal:
                signals.append(signal)
        
        # Sort by confidence
        signals.sort(key=lambda x: x.confidence, reverse=True)
        
        logger.info(f"Aggressive scan: {len(signals)} signals from {len(market_data)} symbols")
        
        # Return ALL signals - let position manager handle limits
        return signals