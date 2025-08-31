"""
Advanced Scalping Strategy with Support/Resistance and Market Structure
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import structlog

logger = structlog.get_logger(__name__)

@dataclass
class TradingSignal:
    """Trading signal data for scalping"""
    symbol: str
    action: str  # BUY, SELL, or HOLD
    price: float
    confidence: float  # 0-1 confidence score
    reason: str
    stop_loss: float
    take_profit: float
    signal_type: str  # SCALP, SWING, or TREND
    risk_reward: float
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

class ScalpingStrategy:
    """Advanced scalping strategy with S/R, market structure, and ML optimization"""
    
    def __init__(self, config: dict, ml_enabled: bool = True):
        self.config = config
        self.ml_enabled = ml_enabled
        
        # Initialize ML components if enabled
        self.performance_tracker = None
        self.optimizer = None
        
        if self.ml_enabled:
            try:
                from ml.performance_tracker import PerformanceTracker
                from ml.adaptive_optimizer import AdaptiveOptimizer
                
                self.performance_tracker = PerformanceTracker()
                self.optimizer = AdaptiveOptimizer(self.performance_tracker, config)
                logger.info("ML optimization enabled in shadow mode with auto-activation")
            except Exception as e:
                logger.warning(f"ML components not available: {e}")
                self.ml_enabled = False
        
        # RSI settings (keep existing)
        self.rsi_oversold = config.get('rsi_oversold', 30)
        self.rsi_overbought = config.get('rsi_overbought', 70)
        
        # Scalping specific settings from config
        self.scalp_rsi_oversold = config.get('rsi_oversold', 30) - 5  # More extreme for scalping
        self.scalp_rsi_overbought = config.get('rsi_overbought', 70) + 5
        self.min_volume_multiplier = 1.5  # Reduced from 2.0 to 1.5 for more opportunities
        
        # WIN RATE IMPROVEMENTS - Multiple confirmation requirements
        self.min_confirmations = config.get('min_confirmations', 2)  # Reduced from 3 to 2 for more signals
        self.use_volume_profile = True  # Volume confirmation
        self.use_momentum_filter = True  # Momentum confirmation  
        self.use_trend_filter = True  # Trade with trend only
        
        # Support/Resistance settings
        self.sr_lookback = 50  # Candles to look back for S/R
        self.sr_touches = 2  # Min touches to confirm S/R
        self.sr_tolerance = config.get('scalp_stop_loss', 0.002)  # Use stop loss as tolerance
        
        # Market structure settings - OPTIMIZED FOR WIN RATE
        self.trend_ema_fast = 9
        self.trend_ema_slow = 21
        self.trend_ema_200 = 200  # Long-term trend filter
        self.structure_lookback = 20
        
        # Risk reward from config ONLY
        self.min_risk_reward = config.get('min_risk_reward', 1.2)
        
        # R:R multipliers from config ONLY (no hardcoded values)
        self.rr_sl_multiplier = config.get('rr_sl_multiplier', 1.0)
        self.rr_tp_multiplier = config.get('rr_tp_multiplier', 2.0)
        self.scalp_rr_sl_multiplier = config.get('scalp_rr_sl_multiplier', 1.0)
        self.scalp_rr_tp_multiplier = config.get('scalp_rr_tp_multiplier', 1.5)
        
        logger.info(f"Scalping strategy initialized - RSI: {self.rsi_oversold}/{self.rsi_overbought}")
    
    def analyze(self, symbol: str, df: pd.DataFrame) -> Optional[TradingSignal]:
        """Analyze symbol for scalping signals"""
        try:
            # Need at least 200 candles for proper analysis
            if len(df) < 200:
                return None
            
            # Calculate additional indicators for scalping
            df = self._calculate_scalping_indicators(df)
            
            # Identify support and resistance levels
            support_levels, resistance_levels = self._identify_sr_levels(df)
            
            # Analyze market structure
            market_structure = self._analyze_market_structure(df)
            
            # Get latest values
            current = df.iloc[-1]
            prev = df.iloc[-2]
            
            # Check if indicators are valid
            if pd.isna(current['rsi']) or pd.isna(current['macd']):
                return None
            
            # Get current price
            price = current['close']
            
            # Calculate ATR for stop loss (tighter for scalping)
            atr = current.get('atr', price * 0.01)  # Default 1% for scalping
            
            # Volume check - must be significantly above average for scalping
            volume_ok = current['volume'] > (current.get('volume_ma', current['volume']) * self.min_volume_multiplier)
            
            # Generate scalping signal
            signal = self._generate_scalping_signal(
                current, prev, price, atr, volume_ok,
                support_levels, resistance_levels, market_structure
            )
            
            if signal:
                signal.symbol = symbol
                logger.info(f"Scalp signal for {symbol}: {signal.action} at {price:.4f} ({signal.reason})")
                
            return signal
            
        except Exception as e:
            logger.error(f"Error analyzing {symbol}: {e}")
            return None
    
    def _calculate_scalping_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate additional indicators for scalping - ENHANCED FOR WIN RATE"""
        # Multiple EMAs for trend confirmation
        df['ema_9'] = df['close'].ewm(span=self.trend_ema_fast, adjust=False).mean()
        df['ema_21'] = df['close'].ewm(span=self.trend_ema_slow, adjust=False).mean()
        df['ema_200'] = df['close'].ewm(span=self.trend_ema_200, adjust=False).mean()
        
        # VWAP (Volume Weighted Average Price)
        df['vwap'] = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()
        
        # Rate of Change (momentum)
        df['roc'] = df['close'].pct_change(periods=10) * 100
        
        # Enhanced Stochastic RSI with K and D lines
        rsi = df['rsi']
        rsi_min = rsi.rolling(window=14).min()
        rsi_max = rsi.rolling(window=14).max()
        stoch_rsi = (rsi - rsi_min) / (rsi_max - rsi_min + 0.0001) * 100
        df['stoch_rsi_k'] = stoch_rsi.rolling(window=3).mean()
        df['stoch_rsi_d'] = df['stoch_rsi_k'].rolling(window=3).mean()
        df['stoch_rsi'] = stoch_rsi
        
        # Enhanced Volume Profile Analysis
        df['volume_ma'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma']
        df['high_volume'] = df['volume_ratio'] > 1.5  # Volume spike detection
        
        # Momentum indicators
        df['momentum_positive'] = df['roc'] > 0
        
        # Price action patterns
        df['bullish_hammer'] = self._detect_hammer(df)
        df['bearish_shooting_star'] = self._detect_shooting_star(df)
        
        return df
    
    def _identify_sr_levels(self, df: pd.DataFrame) -> Tuple[List[float], List[float]]:
        """Identify support and resistance levels"""
        recent_df = df.tail(self.sr_lookback)
        
        # Find local highs and lows
        highs = []
        lows = []
        
        for i in range(2, len(recent_df) - 2):
            # Local high
            if (recent_df.iloc[i]['high'] > recent_df.iloc[i-1]['high'] and 
                recent_df.iloc[i]['high'] > recent_df.iloc[i-2]['high'] and
                recent_df.iloc[i]['high'] > recent_df.iloc[i+1]['high'] and 
                recent_df.iloc[i]['high'] > recent_df.iloc[i+2]['high']):
                highs.append(recent_df.iloc[i]['high'])
            
            # Local low
            if (recent_df.iloc[i]['low'] < recent_df.iloc[i-1]['low'] and 
                recent_df.iloc[i]['low'] < recent_df.iloc[i-2]['low'] and
                recent_df.iloc[i]['low'] < recent_df.iloc[i+1]['low'] and 
                recent_df.iloc[i]['low'] < recent_df.iloc[i+2]['low']):
                lows.append(recent_df.iloc[i]['low'])
        
        # Cluster similar levels
        support_levels = self._cluster_levels(lows)
        resistance_levels = self._cluster_levels(highs)
        
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
                # Add average of cluster
                clustered.append(sum(current_cluster) / len(current_cluster))
                current_cluster = [level]
        
        if current_cluster:
            clustered.append(sum(current_cluster) / len(current_cluster))
        
        return clustered
    
    def _analyze_market_structure(self, df: pd.DataFrame) -> dict:
        """Analyze market structure (trend, momentum, volatility)"""
        recent_df = df.tail(self.structure_lookback)
        
        # Determine trend
        ema_fast = recent_df['ema_9'].iloc[-1]
        ema_slow = recent_df['ema_21'].iloc[-1]
        price = recent_df['close'].iloc[-1]
        
        if ema_fast > ema_slow and price > ema_fast:
            trend = "BULLISH"
        elif ema_fast < ema_slow and price < ema_fast:
            trend = "BEARISH"
        else:
            trend = "NEUTRAL"
        
        # Calculate momentum
        momentum = recent_df['roc'].iloc[-1]
        
        # Calculate volatility (using ATR percentage)
        volatility = recent_df['atr'].iloc[-1] / price * 100
        
        # Identify higher highs/lows (trend strength)
        highs = recent_df['high'].values
        lows = recent_df['low'].values
        
        higher_highs = sum(1 for i in range(1, len(highs)) if highs[i] > highs[i-1])
        higher_lows = sum(1 for i in range(1, len(lows)) if lows[i] > lows[i-1])
        
        trend_strength = (higher_highs + higher_lows) / (len(highs) * 2)
        
        return {
            'trend': trend,
            'momentum': momentum,
            'volatility': volatility,
            'trend_strength': trend_strength
        }
    
    def _detect_hammer(self, df: pd.DataFrame) -> pd.Series:
        """Detect bullish hammer pattern"""
        body = abs(df['close'] - df['open'])
        lower_wick = df[['open', 'close']].min(axis=1) - df['low']
        upper_wick = df['high'] - df[['open', 'close']].max(axis=1)
        
        # Hammer: small body, long lower wick, little/no upper wick
        hammer = (lower_wick > body * 2) & (upper_wick < body * 0.5)
        return hammer
    
    def _detect_shooting_star(self, df: pd.DataFrame) -> pd.Series:
        """Detect bearish shooting star pattern"""
        body = abs(df['close'] - df['open'])
        lower_wick = df[['open', 'close']].min(axis=1) - df['low']
        upper_wick = df['high'] - df[['open', 'close']].max(axis=1)
        
        # Shooting star: small body, long upper wick, little/no lower wick
        shooting_star = (upper_wick > body * 2) & (lower_wick < body * 0.5)
        return shooting_star
    
    def _generate_scalping_signal(self, current: pd.Series, prev: pd.Series, 
                                  price: float, atr: float, volume_ok: bool,
                                  support_levels: List[float], 
                                  resistance_levels: List[float],
                                  market_structure: dict) -> Optional[TradingSignal]:
        """Generate scalping signal based on multiple confluences"""
        
        # Find nearest S/R levels
        nearest_support = min(support_levels, key=lambda x: abs(x - price)) if support_levels else 0
        nearest_resistance = min(resistance_levels, key=lambda x: abs(x - price)) if resistance_levels else float('inf')
        
        # Distance to S/R as percentage
        dist_to_support = abs(price - nearest_support) / price * 100 if nearest_support else float('inf')
        dist_to_resistance = abs(price - nearest_resistance) / price * 100 if nearest_resistance else float('inf')
        
        # SCALP BUY Conditions with confirmation tracking
        buy_conditions = []
        scalp_score = 0
        confirmations = 0
        
        # 1. Price near support with bounce confirmation
        if dist_to_support < 0.5:  # Within 0.5% of support
            buy_conditions.append("Near support")
            scalp_score += 2
            confirmations += 1
            # Check for bounce pattern
            if current['low'] < prev['low'] and current['close'] > current['open']:
                scalp_score += 1
                buy_conditions.append("Support bounce")
                confirmations += 1
        
        # 2. RSI oversold for scalping
        if current['rsi'] < self.scalp_rsi_oversold:
            buy_conditions.append("RSI oversold")
            scalp_score += 2
        
        # 3. Stochastic RSI oversold
        if current.get('stoch_rsi', 50) < 20:
            buy_conditions.append("Stoch RSI oversold")
            scalp_score += 1
        
        # 4. Bullish market structure
        if market_structure['trend'] == "BULLISH":
            buy_conditions.append("Bullish trend")
            scalp_score += 2
        
        # 5. MACD bullish crossover
        if (prev['macd'] < prev['macd_signal'] and 
            current['macd'] > current['macd_signal']):
            buy_conditions.append("MACD bullish cross")
            scalp_score += 2
        
        # 6. Bullish hammer pattern
        if current.get('bullish_hammer', False):
            buy_conditions.append("Hammer pattern")
            scalp_score += 1
        
        # 7. Price above VWAP (institutional interest)
        if current['close'] > current.get('vwap', current['close']):
            buy_conditions.append("Above VWAP")
            scalp_score += 1
        
        # 8. Momentum turning positive
        if current.get('roc', 0) > prev.get('roc', 0) and current.get('roc', 0) > -1:
            buy_conditions.append("Momentum improving")
            scalp_score += 1
        
        # SCALP SELL Conditions with confirmation tracking
        sell_conditions = []
        sell_score = 0
        sell_confirmations = 0
        
        # 1. Price near resistance
        if dist_to_resistance < 0.5:  # Within 0.5% of resistance
            sell_conditions.append("Near resistance")
            sell_score += 2
        
        # 2. RSI overbought for scalping
        if current['rsi'] > self.scalp_rsi_overbought:
            sell_conditions.append("RSI overbought")
            sell_score += 2
        
        # 3. Stochastic RSI overbought
        if current.get('stoch_rsi', 50) > 80:
            sell_conditions.append("Stoch RSI overbought")
            sell_score += 1
        
        # 4. Bearish market structure
        if market_structure['trend'] == "BEARISH":
            sell_conditions.append("Bearish trend")
            sell_score += 2
        
        # 5. MACD bearish crossover
        if (prev['macd'] > prev['macd_signal'] and 
            current['macd'] < current['macd_signal']):
            sell_conditions.append("MACD bearish cross")
            sell_score += 2
        
        # 6. Shooting star pattern
        if current.get('bearish_shooting_star', False):
            sell_conditions.append("Shooting star")
            sell_score += 1
        
        # 7. Price below VWAP
        if current['close'] < current.get('vwap', current['close']):
            sell_conditions.append("Below VWAP")
            sell_score += 1
        
        # 8. Momentum turning negative
        if current.get('roc', 0) < prev.get('roc', 0) and current.get('roc', 0) < 1:
            sell_conditions.append("Momentum declining")
            sell_score += 1
        
        # Determine signal type based on volatility and timeframe
        if market_structure['volatility'] < 1.5:
            signal_type = "SCALP"
            sl_multiplier = self.scalp_rr_sl_multiplier  # From config only
            tp_multiplier = self.scalp_rr_tp_multiplier  # From config only
            min_score = 3  # Reduced from 5 to 3 for more scalp signals
        else:
            signal_type = "SWING"
            sl_multiplier = self.rr_sl_multiplier  # From config only
            tp_multiplier = self.rr_tp_multiplier  # From config only
            min_score = 2  # Reduced from 4 to 2 for more swing signals
        
        # Generate BUY signal with balanced requirements  
        if scalp_score >= min_score and scalp_score > sell_score and volume_ok and confirmations >= self.min_confirmations:
            # Calculate scalping stop loss and take profit
            if nearest_support > 0:
                # Use support as stop loss reference
                stop_loss = min(nearest_support * 0.995, price - (atr * sl_multiplier))
            else:
                stop_loss = price - (atr * sl_multiplier)
            
            if nearest_resistance < float('inf'):
                # Use resistance as take profit reference
                take_profit = min(nearest_resistance * 0.995, price + (atr * tp_multiplier))
            else:
                take_profit = price + (atr * tp_multiplier)
            
            # Calculate risk/reward
            risk = price - stop_loss
            reward = take_profit - price
            risk_reward = reward / risk if risk > 0 else 0
            
            # Only take trades with good R:R from config
            if risk_reward >= self.min_risk_reward:
                confidence = min(scalp_score / 12, 1.0)
                
                return TradingSignal(
                    symbol="",
                    action="BUY",
                    price=price,
                    confidence=confidence,
                    reason=", ".join(buy_conditions[:3]),  # Top 3 reasons
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    signal_type=signal_type,
                    risk_reward=risk_reward
                )
        
        # Generate SELL signal with balanced requirements
        elif sell_score >= min_score and sell_score > scalp_score and volume_ok and sell_confirmations >= self.min_confirmations:
            # Calculate scalping stop loss and take profit for shorts
            if nearest_resistance < float('inf'):
                stop_loss = max(nearest_resistance * 1.005, price + (atr * sl_multiplier))
            else:
                stop_loss = price + (atr * sl_multiplier)
            
            if nearest_support > 0:
                take_profit = max(nearest_support * 1.005, price - (atr * tp_multiplier))
            else:
                take_profit = price - (atr * tp_multiplier)
            
            # Calculate risk/reward
            risk = stop_loss - price
            reward = price - take_profit
            risk_reward = reward / risk if risk > 0 else 0
            
            # Only take trades with good R:R from config
            if risk_reward >= self.min_risk_reward:
                confidence = min(sell_score / 12, 1.0)
                
                return TradingSignal(
                    symbol="",
                    action="SELL",
                    price=price,
                    confidence=confidence,
                    reason=", ".join(sell_conditions[:3]),
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    signal_type=signal_type,
                    risk_reward=risk_reward
                )
        
        return None
    
    def scan_symbols(self, market_data: Dict[str, pd.DataFrame]) -> List[TradingSignal]:
        """Scan all symbols for scalping signals"""
        signals = []
        
        for symbol, df in market_data.items():
            signal = self.analyze(symbol, df)
            if signal:
                signals.append(signal)
        
        # Sort by confidence and risk/reward
        signals.sort(key=lambda x: (x.confidence * x.risk_reward), reverse=True)
        
        # Prefer SCALP signals in volatile markets
        scalp_signals = [s for s in signals if s.signal_type == "SCALP"]
        swing_signals = [s for s in signals if s.signal_type == "SWING"]
        
        # Return scalps first, then swings
        final_signals = scalp_signals[:5] + swing_signals[:3]  # Max 5 scalps, 3 swings
        
        logger.info(f"Scan complete: {len(scalp_signals)} scalps, {len(swing_signals)} swings found")
        
        return final_signals
    
    def _get_market_snapshot(self, df: pd.DataFrame) -> Dict:
        """Get current market conditions for ML"""
        current = df.iloc[-1]
        return {
            'rsi': current.get('rsi', 50),
            'macd': current.get('macd', 0),
            'volume_ratio': current.get('volume') / df['volume'].rolling(20).mean().iloc[-1] if 'volume' in df.columns else 1,
            'volatility': current.get('atr', 0) / current['close'] * 100 if 'atr' in current else 1,
            'trend': 'BULLISH' if current['close'] > df['close'].rolling(50).mean().iloc[-1] else 'BEARISH'
        }
    
    def get_ml_report(self) -> Optional[Dict]:
        """Get ML optimization report"""
        if not self.ml_enabled or not self.optimizer:
            return None
        
        return self.optimizer.get_adaptation_report()
    
    def enable_ml_live_mode(self) -> bool:
        """Enable live ML adaptations if performance is good"""
        if not self.ml_enabled or not self.optimizer:
            return False
        
        return self.optimizer.enable_live_mode()