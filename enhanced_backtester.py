"""
Enhanced Backtesting Engine - Accurate ML Feature Generation

This enhanced backtester ensures that ML features are generated exactly the same way
as the live bot for both Trend (Breakout) and Mean Reversion strategies.

Key improvements:
1. Generates ML features exactly like live bot
2. Handles different feature generation per strategy
3. Maintains compatibility with existing backtester interface
"""
import pandas as pd
import numpy as np
import logging
from typing import List, Dict, Callable, Optional

from candle_storage_postgres import CandleStorage
from strategy_trend_breakout import TrendSettings as Settings, Signal

logger = logging.getLogger(__name__)

class EnhancedBacktester:
    """
    Enhanced backtester that generates ML features exactly like the live bot
    """
    
    def __init__(self, strategy_func: Callable, strategy_settings: Settings, 
                 reset_state_func: Optional[Callable] = None, strategy_type: str = "trend"):
        """
        Initialize enhanced backtester
        
        Args:
            strategy_func: The function that detects signals (e.g., get_ml_learning_signals)
            strategy_settings: The settings for the strategy
            reset_state_func: Function to reset strategy state for a symbol
            strategy_type: "trend" or "mean_reversion" - determines feature generation
        """
        self.strategy_func = strategy_func
        self.settings = strategy_settings
        self.reset_state_func = reset_state_func
        self.strategy_type = strategy_type
        self.candle_storage = CandleStorage()
        
        logger.info(f"Enhanced backtester initialized for {strategy_type} strategy")

    def run(self, symbol: str, history_df: Optional[pd.DataFrame] = None) -> List[Dict]:
        """
        Run enhanced backtest with accurate ML feature generation
        
        Args:
            symbol: The trading symbol to backtest
            history_df: Optional pre-loaded dataframe of historical data
            
        Returns:
            List of dictionaries with signal data, features, and outcomes
        """
        if self.reset_state_func:
            self.reset_state_func(symbol)
        
        if history_df is None:
            logger.info(f"[{symbol}] Loading historical data from database...")
            history_df = self.candle_storage.load_candles(symbol, limit=100000)
        
        if history_df is None or len(history_df) < 200:
            logger.warning(f"[{symbol}] Insufficient historical data ({len(history_df) if history_df is not None else 0} candles)")
            return []
        
        logger.info(f"[{symbol}] Enhanced backtesting on {len(history_df)} candles from {history_df.index[0]} to {history_df.index[-1]}...")
        
        results = []
        
        # Initialize feature generation based on strategy type
        if self.strategy_type == "trend":
            feature_generator = self._generate_trend_features
        elif self.strategy_type == "mean_reversion":
            feature_generator = self._generate_mr_features
        else:
            logger.error(f"Unknown strategy type: {self.strategy_type}")
            return []
        
        # Iterate through historical data
        for i in range(200, len(history_df)):
            # Create point-in-time data slice
            current_df_slice = history_df.iloc[i-200:i]
            
            # Run signal detection
            signal_output = self.strategy_func(current_df_slice, self.settings, df_1h=None, symbol=symbol)
            
            # Handle both single signals and lists
            signals_to_process = []
            if isinstance(signal_output, list):
                signals_to_process.extend(signal_output)
            elif signal_output is not None:
                signals_to_process.append(signal_output)
            
            for signal in signals_to_process:
                if signal:
                    # Generate ML features exactly like live bot
                    try:
                        ml_features = feature_generator(current_df_slice, signal, symbol)
                        
                        # Simulate trade outcome
                        outcome = self._simulate_trade(history_df, i, signal)
                        
                        if outcome and ml_features:
                            results.append({
                                "timestamp": history_df.index[i],
                                "features": ml_features,
                                "outcome": outcome,
                                "signal": {
                                    'side': signal.side,
                                    'entry': signal.entry,
                                    'sl': signal.sl,
                                    'tp': signal.tp
                                },
                                "symbol": symbol,
                                "strategy": self.strategy_type
                            })
                            
                    except Exception as e:
                        logger.error(f"[{symbol}] Error generating features for signal: {e}")
        
        logger.info(f"[{symbol}] Enhanced backtest complete. Found {len(results)} signals with ML features.")
        return results
    
    def _generate_trend_features(self, df: pd.DataFrame, signal: Signal, symbol: str) -> Dict:
        """Generate ML features for trend breakout exactly like live bot"""
        try:
            close = df['close']; high = df['high']; low = df['low']
            price = float(close.iloc[-1])
            ys = close.tail(20).values if len(close) >= 20 else close.values
            try:
                slope = np.polyfit(np.arange(len(ys)), ys, 1)[0]
            except Exception:
                slope = 0.0
            trend_slope_pct = float((slope / price) * 100.0) if price else 0.0
            ema20 = float(close.ewm(span=20, adjust=False).mean().iloc[-1])
            ema50 = float(close.ewm(span=50, adjust=False).mean().iloc[-1]) if len(close) >= 50 else ema20
            ema_stack_score = 100.0 if (price > ema20 > ema50 or price < ema20 < ema50) else 50.0 if (ema20 != ema50) else 0.0
            rng_today = float(high.iloc[-1] - low.iloc[-1])
            med_range = float((high - low).rolling(20).median().iloc[-1]) if len(df) >= 20 else rng_today
            range_expansion = float(rng_today / max(1e-9, med_range))
            prev = close.shift(); trarr = np.maximum(high - low, np.maximum((high - prev).abs(), (low - prev).abs()))
            atr = float(trarr.rolling(14).mean().iloc[-1]) if len(trarr) >= 14 else float(trarr.iloc[-1])
            atr_pct = float((atr / max(1e-9, price)) * 100.0) if price else 0.0
            close_vs_ema20_pct = float(((price - ema20) / max(1e-9, ema20)) * 100.0) if ema20 else 0.0
            features = {
                'trend_slope_pct': trend_slope_pct,
                'ema_stack_score': ema_stack_score,
                'atr_pct': atr_pct,
                'range_expansion': range_expansion,
                'breakout_dist_atr': float(signal.meta.get('breakout_dist_atr', 0.0) if getattr(signal, 'meta', None) else 0.0),
                'close_vs_ema20_pct': close_vs_ema20_pct,
                'bb_width_pct': 0.0,
                'session': 'us',
                'symbol_cluster': 3,
                'volatility_regime': 'normal'
            }
            try:
                from symbol_clustering import load_symbol_clusters
                features['symbol_cluster'] = load_symbol_clusters().get(symbol, 3)
            except Exception:
                pass
            logger.debug(f"[{symbol}] Generated {len(features)} trend features")
            return features
        except Exception as e:
            logger.error(f"[{symbol}] Error generating trend features: {e}")
            return {}
    
    def _generate_mr_features(self, df: pd.DataFrame, signal: Signal, symbol: str) -> Dict:
        """Generate ML features for mean reversion strategy exactly like live bot"""
        try:
            # Enhanced MR features removed - using basic features
            
            logger.debug(f"[{symbol}] Generating enhanced MR features")
            
            # Create signal data format expected by enhanced features
            signal_data = {
                'side': signal.side,
                'entry': signal.entry,
                'sl': signal.sl,
                'tp': signal.tp,
                'meta': signal.meta
            }
            
            # Generate enhanced MR features like live bot
            # Use basic MR features from signal meta
            enhanced_features = signal.meta.get('mr_features', {})
            
            logger.debug(f"[{symbol}] Generated {len(enhanced_features)} enhanced MR features")
            return enhanced_features
            
        except Exception as e:
            logger.error(f"[{symbol}] Error generating MR features: {e}")
            # Fallback to basic features if enhanced features fail
            try:
                return self._generate_basic_mr_features(df, signal, symbol)
            except Exception as e2:
                logger.error(f"[{symbol}] Error generating basic MR features: {e2}")
                return {}
    
    def _generate_basic_mr_features(self, df: pd.DataFrame, signal: Signal, symbol: str) -> Dict:
        """Generate basic MR features as fallback"""
        try:
            # Basic features analogous to trend but for ranging markets
            features = {}
            
            # Price position in range
            meta = signal.meta
            range_upper = meta.get('range_upper', signal.entry * 1.02)
            range_lower = meta.get('range_lower', signal.entry * 0.98)
            
            if range_upper > range_lower:
                range_position = (signal.entry - range_lower) / (range_upper - range_lower)
                features['range_position'] = range_position
                features['range_width_pct'] = (range_upper - range_lower) / range_lower
            else:
                features['range_position'] = 0.5
                features['range_width_pct'] = 0.02
            
            # Volume analysis
            recent_volume = df['volume'].rolling(20).mean().iloc[-1]
            current_volume = df['volume'].iloc[-1]
            features['volume_ratio'] = current_volume / recent_volume if recent_volume > 0 else 1.0
            
            # Volatility
            returns = df['close'].pct_change().dropna()
            features['volatility'] = returns.rolling(20).std().iloc[-1] if len(returns) > 20 else 0.01
            
            # Range quality (how well-defined the range is)
            price_touches_upper = sum(1 for price in df['high'].tail(50) if abs(price - range_upper) / range_upper < 0.005)
            price_touches_lower = sum(1 for price in df['low'].tail(50) if abs(price - range_lower) / range_lower < 0.005)
            features['range_quality'] = min(price_touches_upper + price_touches_lower, 10) / 10.0
            
            # Add symbol cluster
            features['symbol_cluster'] = 3  # Default cluster for MR
            
            logger.debug(f"[{symbol}] Generated {len(features)} basic MR features")
            return features
            
        except Exception as e:
            logger.error(f"[{symbol}] Error generating basic MR features: {e}")
            return {}
    
    def _simulate_trade(self, df: pd.DataFrame, entry_index: int, signal: Signal) -> str:
        """
        Simulate trade outcome - same as original backtester
        """
        # Look ahead up to 500 candles for outcome
        for i in range(entry_index + 1, min(entry_index + 500, len(df))):
            future_high = df['high'].iloc[i]
            future_low = df['low'].iloc[i]
            
            if signal.side == "long":
                # Check for TP hit
                if future_high >= signal.tp:
                    return "win"
                # Check for SL hit  
                if future_low <= signal.sl:
                    return "loss"
            elif signal.side == "short":
                # Check for TP hit
                if future_low <= signal.tp:
                    return "win"
                # Check for SL hit
                if future_high >= signal.sl:
                    return "loss"
        
        logger.debug(f"No outcome within 500 candles for {signal.side} signal at {signal.entry}")
        return "no_outcome"  # No outcome within timeframe
