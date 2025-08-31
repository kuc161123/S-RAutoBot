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
    """Enhanced aggressive strategy with trend and structure analysis + ML"""
    
    def __init__(self, config: dict):
        self.config = config
        
        # SCALPING CONFIGURATION - Quick in/out trades
        self.rsi_oversold = config.get('rsi_oversold', 30)  # More extreme for scalping
        self.rsi_overbought = config.get('rsi_overbought', 70)  # More extreme for scalping
        
        # Volume filter - higher = fewer but better signals
        self.min_volume_multiplier = config.get('min_volume_multiplier', 1.5)  # Higher volume for liquidity
        
        # Minimum score for signal quality (lower for more trades)
        self.min_score = config.get('min_signal_score', 3)  # Reduced for more scalping opportunities
        
        # SCALPING Risk settings - TIGHT stops, QUICK profits
        self.rr_sl_multiplier = config.get('rr_sl_multiplier', 0.5)  # Tight 0.5x ATR stop
        self.rr_tp_multiplier = config.get('rr_tp_multiplier', 1.0)  # Quick 1.0x ATR profit
        
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
        
        # Initialize ML components
        self._init_ml_components()
        
        logger.info(f"Enhanced aggressive strategy with ML + S/R - RSI: {self.rsi_oversold}/{self.rsi_overbought}, Min score: {self.min_score}")
    
    def _init_ml_components(self):
        """Initialize ML tracking and optimization components"""
        try:
            from ..ml.performance_tracker import PerformanceTracker
            from ..ml.adaptive_optimizer import AdaptiveOptimizer
            
            # Initialize performance tracker
            self.performance_tracker = PerformanceTracker(data_dir="ml_data")
            
            # Initialize adaptive optimizer
            self.optimizer = AdaptiveOptimizer(self.performance_tracker, self.config)
            
            # ML prediction cache
            self.ml_predictions = {}
            self.ml_enabled = True
            
            logger.info("ML components initialized successfully")
        except Exception as e:
            logger.warning(f"ML components not available: {e}")
            self.performance_tracker = None
            self.optimizer = None
            self.ml_enabled = False
    
    def _find_support_resistance(self, df: pd.DataFrame) -> tuple:
        """Find support and resistance levels with strength scoring"""
        recent_df = df.tail(self.sr_lookback)
        highs = {}
        lows = {}
        
        for i in range(1, len(recent_df) - 1):
            # Find local highs (resistance) with touch count
            if recent_df.iloc[i]['high'] > recent_df.iloc[i-1]['high'] and \
               recent_df.iloc[i]['high'] > recent_df.iloc[i+1]['high']:
                level = recent_df.iloc[i]['high']
                # Count touches at this level
                touches = self._count_touches(df, level, is_resistance=True)
                highs[level] = touches
            
            # Find local lows (support) with touch count
            if recent_df.iloc[i]['low'] < recent_df.iloc[i-1]['low'] and \
               recent_df.iloc[i]['low'] < recent_df.iloc[i+1]['low']:
                level = recent_df.iloc[i]['low']
                # Count touches at this level
                touches = self._count_touches(df, level, is_resistance=False)
                lows[level] = touches
        
        # Cluster similar levels with strength
        support_levels = self._cluster_levels_with_strength(lows) if lows else []
        resistance_levels = self._cluster_levels_with_strength(highs) if highs else []
        
        return support_levels, resistance_levels
    
    def _count_touches(self, df: pd.DataFrame, level: float, is_resistance: bool) -> int:
        """Count how many times price touched a level"""
        touches = 0
        tolerance = level * self.sr_tolerance
        
        for _, row in df.iterrows():
            if is_resistance:
                if abs(row['high'] - level) < tolerance:
                    touches += 1
            else:
                if abs(row['low'] - level) < tolerance:
                    touches += 1
        
        return touches
    
    def _cluster_levels_with_strength(self, levels_dict: dict) -> List[tuple]:
        """Cluster similar levels and track their strength"""
        if not levels_dict:
            return []
        
        # Sort levels by price
        sorted_levels = sorted(levels_dict.items())
        clustered = []
        current_cluster = [sorted_levels[0]]
        
        for level, touches in sorted_levels[1:]:
            if abs(level - current_cluster[-1][0]) / current_cluster[-1][0] < self.sr_tolerance:
                current_cluster.append((level, touches))
            else:
                # Average the cluster and sum touches
                avg_level = sum(l for l, _ in current_cluster) / len(current_cluster)
                total_touches = sum(t for _, t in current_cluster)
                clustered.append((avg_level, total_touches))
                current_cluster = [(level, touches)]
        
        if current_cluster:
            avg_level = sum(l for l, _ in current_cluster) / len(current_cluster)
            total_touches = sum(t for _, t in current_cluster)
            clustered.append((avg_level, total_touches))
        
        return clustered
    
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
    
    def _detect_divergence(self, df: pd.DataFrame) -> dict:
        """Detect RSI and MACD divergences"""
        if len(df) < 20:
            return {'rsi_bullish_div': False, 'rsi_bearish_div': False, 
                   'macd_bullish_div': False, 'macd_bearish_div': False}
        
        recent = df.tail(20)
        divergences = {}
        
        # Find recent swing highs and lows
        price_highs = []
        price_lows = []
        
        for i in range(1, len(recent) - 1):
            if recent.iloc[i]['high'] > recent.iloc[i-1]['high'] and recent.iloc[i]['high'] > recent.iloc[i+1]['high']:
                price_highs.append(i)
            if recent.iloc[i]['low'] < recent.iloc[i-1]['low'] and recent.iloc[i]['low'] < recent.iloc[i+1]['low']:
                price_lows.append(i)
        
        # Check RSI divergence
        if 'rsi' in recent.columns and len(price_lows) >= 2:
            # Bullish divergence: price makes lower low, RSI makes higher low
            if recent.iloc[price_lows[-1]]['low'] < recent.iloc[price_lows[-2]]['low']:
                if recent.iloc[price_lows[-1]]['rsi'] > recent.iloc[price_lows[-2]]['rsi']:
                    divergences['rsi_bullish_div'] = True
                else:
                    divergences['rsi_bullish_div'] = False
            else:
                divergences['rsi_bullish_div'] = False
        else:
            divergences['rsi_bullish_div'] = False
        
        if 'rsi' in recent.columns and len(price_highs) >= 2:
            # Bearish divergence: price makes higher high, RSI makes lower high
            if recent.iloc[price_highs[-1]]['high'] > recent.iloc[price_highs[-2]]['high']:
                if recent.iloc[price_highs[-1]]['rsi'] < recent.iloc[price_highs[-2]]['rsi']:
                    divergences['rsi_bearish_div'] = True
                else:
                    divergences['rsi_bearish_div'] = False
            else:
                divergences['rsi_bearish_div'] = False
        else:
            divergences['rsi_bearish_div'] = False
        
        # MACD divergence
        divergences['macd_bullish_div'] = False
        divergences['macd_bearish_div'] = False
        
        return divergences
    
    def _detect_price_patterns(self, df: pd.DataFrame) -> dict:
        """Detect enhanced price action patterns"""
        current = df.iloc[-1]
        prev = df.iloc[-2]
        
        patterns = {'hammer': False, 'shooting_star': False, 'doji': False,
                   'bullish_engulfing': False, 'bearish_engulfing': False,
                   'pin_bar': False}
        
        body = abs(current['close'] - current['open'])
        prev_body = abs(prev['close'] - prev['open'])
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
        
        # Bullish engulfing
        if prev['close'] < prev['open'] and current['close'] > current['open']:
            if current['close'] > prev['open'] and current['open'] < prev['close']:
                patterns['bullish_engulfing'] = True
        
        # Bearish engulfing
        if prev['close'] > prev['open'] and current['close'] < current['open']:
            if current['close'] < prev['open'] and current['open'] > prev['close']:
                patterns['bearish_engulfing'] = True
        
        # Pin bar (rejection candle)
        if (upper_wick > body * 3 or lower_wick > body * 3) and body < (current['high'] - current['low']) * 0.3:
            patterns['pin_bar'] = True
        
        return patterns
    
    def analyze(self, symbol: str, df: pd.DataFrame) -> Optional[TradingSignal]:
        """Enhanced analysis with trend, S/R, market structure, and ML"""
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
            
            # Get ML-optimized parameters if available
            if self.ml_enabled and self.optimizer:
                ml_params = self.optimizer.get_optimized_params(symbol, current.to_dict())
                # Apply ML-optimized thresholds
                self.rsi_oversold = ml_params.get('rsi_oversold', self.rsi_oversold)
                self.rsi_overbought = ml_params.get('rsi_overbought', self.rsi_overbought)
                self.min_score = ml_params.get('min_signal_score', self.min_score)
            
            # Find support and resistance levels
            support_levels, resistance_levels = self._find_support_resistance(df)
            
            # Analyze market structure
            market_structure = self._analyze_market_structure(df)
            
            # Detect price patterns
            patterns = self._detect_price_patterns(df)
            
            # Detect divergences
            divergences = self._detect_divergence(df)
            
            # Find nearest S/R levels with strength
            nearest_support = None
            nearest_support_strength = 0
            if support_levels:
                # Find closest support with its strength
                closest = min(support_levels, key=lambda x: abs(x[0] - price))
                nearest_support = closest[0]
                nearest_support_strength = closest[1]
            
            nearest_resistance = None
            nearest_resistance_strength = 0
            if resistance_levels:
                # Find closest resistance with its strength
                closest = min(resistance_levels, key=lambda x: abs(x[0] - price))
                nearest_resistance = closest[0]
                nearest_resistance_strength = closest[1]
            
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
            
            # Calculate ATR and volatility state
            atr = current.get('atr', price * 0.015)
            atr_percentage = (atr / price) * 100
            
            # Determine volatility state for adaptive stops
            if atr_percentage < 0.5:
                volatility_state = "LOW"
                sl_multiplier = 0.3  # Tighter stops in low volatility
                tp_multiplier = 0.8  # Smaller targets
            elif atr_percentage < 1.0:
                volatility_state = "NORMAL"
                sl_multiplier = 0.5  # Normal scalping stops
                tp_multiplier = 1.0  # Normal targets
            else:
                volatility_state = "HIGH"
                sl_multiplier = 0.7  # Wider stops in high volatility
                tp_multiplier = 1.3  # Larger targets
            
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
            
            # 3. NEAR SUPPORT LEVEL - With strength scoring
            if dist_to_support < 0.3:  # Within 0.3% - VERY close for scalping
                # Add more points for stronger levels
                if nearest_support_strength >= 3:
                    buy_score += 4  # Strong level (3+ touches)
                    buy_reasons.append(f"At strong support ${nearest_support:.2f} ({nearest_support_strength} touches)")
                else:
                    buy_score += 3
                    buy_reasons.append(f"At support ${nearest_support:.2f}")
            elif dist_to_support < 0.5:  # Within 0.5% of support
                buy_score += 2
                buy_reasons.append("Near support zone")
            
            # 4. MACD turning bullish
            if 'macd' in current and 'macd_signal' in current:
                if current['macd'] > prev['macd']:  # Just improving
                    buy_score += 1
                    buy_reasons.append("MACD improving")
            
            # 5. BULLISH MARKET STRUCTURE
            if market_structure['structure'] == "BULLISH":
                buy_score += 1
                buy_reasons.append(f"Bullish structure ({market_structure['strength']:.1%})")
            
            # 6. BULLISH PRICE PATTERNS - Enhanced
            if patterns['bullish_engulfing']:
                buy_score += 2  # Stronger pattern
                buy_reasons.append("Bullish engulfing")
            elif patterns['hammer']:
                buy_score += 1
                buy_reasons.append("Hammer pattern")
            
            if patterns['pin_bar'] and dist_to_support < 1.0:
                buy_score += 1
                buy_reasons.append("Pin bar at support")
            
            # 7. TREND BONUS - Trade with trend for better win rate
            if trend == "BULLISH" and buy_score > 0:
                buy_score += 1
                buy_reasons.append("Bullish trend")
            
            # 8. MOMENTUM CONFIRMATION - KEY for scalping
            if market_structure['momentum'] > 1.0:  # Strong momentum for scalps
                buy_score += 2
                buy_reasons.append("Strong momentum")
            elif market_structure['momentum'] > 0.5:
                buy_score += 1
                buy_reasons.append("Positive momentum")
            
            # 9. DIVERGENCE BONUS - High probability reversal
            if divergences['rsi_bullish_div'] and dist_to_support < 1.0:
                buy_score += 2
                buy_reasons.append("RSI bullish divergence")
            
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
            
            # 3. NEAR RESISTANCE LEVEL - With strength scoring
            if dist_to_resistance < 0.3:  # Within 0.3% - VERY close for scalping
                # Add more points for stronger levels
                if nearest_resistance_strength >= 3:
                    sell_score += 4  # Strong level (3+ touches)
                    sell_reasons.append(f"At strong resistance ${nearest_resistance:.2f} ({nearest_resistance_strength} touches)")
                else:
                    sell_score += 3
                    sell_reasons.append(f"At resistance ${nearest_resistance:.2f}")
            elif dist_to_resistance < 0.5:  # Within 0.5% of resistance
                sell_score += 2
                sell_reasons.append("Near resistance zone")
            
            # 4. MACD turning bearish
            if 'macd' in current and 'macd_signal' in current:
                if current['macd'] < prev['macd']:  # Just declining
                    sell_score += 1
                    sell_reasons.append("MACD declining")
            
            # 5. BEARISH MARKET STRUCTURE
            if market_structure['structure'] == "BEARISH":
                sell_score += 1
                sell_reasons.append(f"Bearish structure ({market_structure['strength']:.1%})")
            
            # 6. BEARISH PRICE PATTERNS - Enhanced
            if patterns['bearish_engulfing']:
                sell_score += 2  # Stronger pattern
                sell_reasons.append("Bearish engulfing")
            elif patterns['shooting_star']:
                sell_score += 1
                sell_reasons.append("Shooting star pattern")
            
            if patterns['pin_bar'] and dist_to_resistance < 1.0:
                sell_score += 1
                sell_reasons.append("Pin bar at resistance")
            
            # 7. TREND BONUS - Trade with trend for better win rate
            if trend == "BEARISH" and sell_score > 0:
                sell_score += 1
                sell_reasons.append("Bearish trend")
            
            # 8. MOMENTUM CONFIRMATION
            if market_structure['momentum'] < -1.0:  # Strong negative momentum
                sell_score += 2
                sell_reasons.append("Strong negative momentum")
            elif market_structure['momentum'] < -0.5:
                sell_score += 1
                sell_reasons.append("Negative momentum")
            
            # 9. DIVERGENCE BONUS - High probability reversal
            if divergences['rsi_bearish_div'] and dist_to_resistance < 1.0:
                sell_score += 2
                sell_reasons.append("RSI bearish divergence")
            
            # Generate signal with BALANCED requirements
            # Require higher score for better quality signals
            min_score = self.min_score if hasattr(self, 'min_score') else 3
            if buy_score >= min_score and buy_score > sell_score and volume_ok:
                # Smart stop loss placement using support with volatility adaptation
                if nearest_support and dist_to_support < 2.0:
                    stop_loss = nearest_support * 0.998  # Just below support
                else:
                    stop_loss = price - (atr * sl_multiplier)  # Adaptive based on volatility
                
                # Smart take profit using resistance with volatility adaptation
                if nearest_resistance and dist_to_resistance < 5.0:
                    take_profit = nearest_resistance * 0.998  # Just below resistance
                else:
                    take_profit = price + (atr * tp_multiplier)  # Adaptive based on volatility
                
                # SCALPING: Accept lower R:R for quick trades
                if (take_profit - price) / (price - stop_loss) < 1.0:
                    take_profit = price + ((price - stop_loss) * 1.2)  # 1.2 R:R minimum for scalps
                
                confidence = min(buy_score / 8, 1.0)  # Adjusted for more conditions
                
                # Track signal in ML if enabled
                trade_id = None
                if self.ml_enabled and self.performance_tracker:
                    market_conditions = {
                        'rsi': current.get('rsi', 0),
                        'macd': current.get('macd', 0),
                        'macd_signal': current.get('macd_signal', 0),
                        'stoch_rsi_k': current.get('stoch_rsi_k', 0),
                        'volume_ratio': current.get('volume', 0) / current.get('volume_ma', 1) if 'volume_ma' in current else 1,
                        'distance_to_support': dist_to_support,
                        'distance_to_resistance': dist_to_resistance,
                        'trend': trend,
                        'volatility': atr_percentage
                    }
                    
                    trade_id = self.performance_tracker.record_entry(
                        symbol=symbol,
                        action="BUY",
                        price=price,
                        market_data=market_conditions,
                        confirmations=buy_reasons,
                        score=buy_score,
                        confidence=confidence,
                        params={
                            'rsi_threshold': self.rsi_oversold,
                            'min_confirmations': 2,
                            'min_score': self.min_score
                        }
                    )
                    logger.debug(f"ML tracking BUY signal: {trade_id}")
                
                logger.info(f"BUY signal for {symbol}: Score {buy_score}, Reasons: {buy_reasons}")
                
                signal = TradingSignal(
                    symbol=symbol,
                    action="BUY",
                    price=price,
                    confidence=confidence,
                    reason=", ".join(buy_reasons[:2]),
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    signal_type="SCALP",
                    risk_reward=(take_profit - price) / (price - stop_loss)
                )
                
                # Store trade ID for tracking
                if trade_id:
                    signal.trade_id = trade_id
                
                return signal
            
            elif sell_score >= min_score and sell_score > buy_score and volume_ok:
                # Smart stop loss placement using resistance with volatility adaptation
                if nearest_resistance and dist_to_resistance < 2.0:
                    stop_loss = nearest_resistance * 1.002  # Just above resistance
                else:
                    stop_loss = price + (atr * sl_multiplier)  # Adaptive based on volatility
                
                # Smart take profit using support with volatility adaptation
                if nearest_support and dist_to_support < 5.0:
                    take_profit = nearest_support * 1.002  # Just above support
                else:
                    take_profit = price - (atr * tp_multiplier)  # Adaptive based on volatility
                
                # SCALPING: Accept lower R:R for quick trades
                if (price - take_profit) / (stop_loss - price) < 1.0:
                    take_profit = price - ((stop_loss - price) * 1.2)  # 1.2 R:R minimum for scalps
                
                confidence = min(sell_score / 8, 1.0)  # Adjusted for more conditions
                
                # Track signal in ML if enabled
                trade_id = None
                if self.ml_enabled and self.performance_tracker:
                    market_conditions = {
                        'rsi': current.get('rsi', 0),
                        'macd': current.get('macd', 0),
                        'macd_signal': current.get('macd_signal', 0),
                        'stoch_rsi_k': current.get('stoch_rsi_k', 0),
                        'volume_ratio': current.get('volume', 0) / current.get('volume_ma', 1) if 'volume_ma' in current else 1,
                        'distance_to_support': dist_to_support,
                        'distance_to_resistance': dist_to_resistance,
                        'trend': trend,
                        'volatility': atr_percentage
                    }
                    
                    trade_id = self.performance_tracker.record_entry(
                        symbol=symbol,
                        action="SELL",
                        price=price,
                        market_data=market_conditions,
                        confirmations=sell_reasons,
                        score=sell_score,
                        confidence=confidence,
                        params={
                            'rsi_threshold': self.rsi_overbought,
                            'min_confirmations': 2,
                            'min_score': self.min_score
                        }
                    )
                    logger.debug(f"ML tracking SELL signal: {trade_id}")
                
                logger.info(f"SELL signal for {symbol}: Score {sell_score}, Reasons: {sell_reasons}")
                
                signal = TradingSignal(
                    symbol=symbol,
                    action="SELL",
                    price=price,
                    confidence=confidence,
                    reason=", ".join(sell_reasons[:2]),
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    signal_type="SCALP",
                    risk_reward=(price - take_profit) / (stop_loss - price)
                )
                
                # Store trade ID for tracking
                if trade_id:
                    signal.trade_id = trade_id
                
                return signal
            
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
    
    def get_ml_status(self) -> dict:
        """Get ML components status"""
        if not self.ml_enabled:
            return {'enabled': False, 'reason': 'ML components not initialized'}
        
        try:
            from ..ml.ml_manager import MLManager
            ml_manager = MLManager(self, self.config)
            return ml_manager.get_status()
        except Exception as e:
            return {'enabled': False, 'error': str(e)}
    
    def update_trade_result(self, trade_id: str, exit_price: float, pnl: float):
        """Update ML tracking with trade result"""
        if self.ml_enabled and self.performance_tracker and trade_id:
            try:
                self.performance_tracker.record_exit(trade_id, exit_price, pnl)
                logger.debug(f"ML updated trade {trade_id}: Exit {exit_price}, PNL {pnl}")
            except Exception as e:
                logger.error(f"Failed to update ML trade result: {e}")