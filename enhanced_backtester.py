"""
Enhanced Backtesting Engine - Accurate ML Feature Generation + Trend variant sweeps

This enhanced backtester ensures that ML features are generated exactly the same way
as the live bot for both Trend (Pullback) and Mean Reversion strategies, and adds a
lightweight Trend variant runner that can be used to sweep key parameters offline.

Key improvements:
1. Generates ML features exactly like live bot
2. Handles different feature generation per strategy
3. Adds a TrendVariantBacktester with 3m microframe support and simple trade simulation
4. Maintains compatibility with existing backtester interface
"""
import pandas as pd
import numpy as np
import logging
from typing import List, Dict, Callable, Optional

from candle_storage_postgres import CandleStorage

# Legacy imports for compatibility
try:
    from strategy_trend_breakout import TrendSettings as Settings, Signal
except Exception:
    Settings = object  # placeholder for legacy
    Signal = object

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


# -------------------------- Trend Variant Runner --------------------------- #
from dataclasses import dataclass
from datetime import datetime

try:
    from strategy_pullback import (
        Settings as TPSettings,
        Signal as TPSignal,
        detect_signal_pullback,
        reset_symbol_state as reset_trend_state,
        set_trend_microframe_provider,
    )
except Exception:
    TPSettings = None
    TPSignal = None
    detect_signal_pullback = None
    reset_trend_state = None
    set_trend_microframe_provider = None


@dataclass
class Variant:
    name: str
    # Core knobs
    div_mode: str = "optional"        # off|optional|strict
    div_require: str = "any"          # any|all
    bos_hold_minutes: int = 300        # BOS armed hold
    sl_mode: str = "breakout"         # breakout|hybrid
    breakout_sl_buffer_atr: float = 0.30
    min_r_pct: float = 0.005
    cancel_on_reentry: bool = True
    invalidation_mode: str = "on_close"   # on_close|on_touch
    invalidation_timeframe: str = "3m"    # 3m|1m


class TrendVariantBacktester:
    """Run Trend Pullback backtests with 3m micro support and simple simulation."""

    def __init__(self, storage: CandleStorage | None = None):
        if storage is None:
            storage = CandleStorage()
        self.storage = storage
        # Provider state
        self._df3 = None
        self._end = None

    def _micro_provider(self, symbol: str):
        try:
            if self._df3 is None:
                return None
            if self._end is None:
                return self._df3
            return self._df3[self._df3.index <= self._end]
        except Exception:
            return self._df3

    def run_symbol(self, symbol: str, variant: Variant, max_bars: int = None) -> list[dict]:
        if TPSettings is None or detect_signal_pullback is None or set_trend_microframe_provider is None:
            logging.error("Trend Pullback strategy not available for backtest")
            return []

        # Load 15m and 3m frames
        df15 = self.storage.load_candles(symbol, limit=100000)
        df3 = self.storage.load_candles_3m(symbol, limit=500000)
        if df15 is None or len(df15) < 200:
            logger.warning(f"[{symbol}] Skipping: insufficient 15m history")
            return []
        if df3 is None or len(df3) < 100:
            logger.warning(f"[{symbol}] Skipping: insufficient 3m history")
            return []

        # Wire 3m provider
        self._df3 = df3
        self._end = None
        set_trend_microframe_provider(self._micro_provider)

        # Build settings and apply overrides
        s = TPSettings()
        try:
            s.use_3m_pullback = True; s.use_3m_confirm = True
            s.div_enabled = True
            s.div_mode = variant.div_mode
            s.div_require = variant.div_require
            s.bos_armed_hold_minutes = int(variant.bos_hold_minutes)
            s.sl_mode = variant.sl_mode
            s.breakout_sl_buffer_atr = float(variant.breakout_sl_buffer_atr)
            s.min_r_pct = float(variant.min_r_pct)
        except Exception:
            pass

        # Reset state once per symbol
        try:
            reset_trend_state(symbol)
        except Exception:
            pass

        results: list[dict] = []
        start = 200
        end = len(df15) if not max_bars else min(len(df15), max_bars)
        for i in range(start, end):
            df_slice = df15.iloc[: i + 1]
            self._end = df_slice.index[-1]
            try:
                sig = detect_signal_pullback(df_slice, s, symbol=symbol)
            except Exception as e:
                logger.debug(f"[{symbol}] detect error: {e}")
                sig = None
            if not sig:
                continue
            # Simulate outcome using 15m + 3m re-entry invalidation
            try:
                res = self._simulate(df15, df3, i, sig, variant)
                if res is not None:
                    results.append({
                        'ts': df_slice.index[-1],
                        'symbol': symbol,
                        'side': sig.side,
                        'entry': float(sig.entry),
                        'sl': float(sig.sl),
                        'tp': float(sig.tp),
                        'r': float(res['r']),
                        'exit_reason': res['reason']
                    })
            except Exception as e:
                logger.debug(f"[{symbol}] simulate error: {e}")
                continue
        return results

    def _simulate(self, df15: pd.DataFrame, df3: pd.DataFrame, entry_idx15: int, sig, variant: Variant) -> Optional[dict]:
        side = sig.side
        entry = float(sig.entry); sl = float(sig.sl); tp = float(sig.tp)
        risk = abs(entry - sl) if abs(entry - sl) > 0 else 1e-9
        # Find the 15m start time
        start_ts = df15.index[entry_idx15]
        # Iterate forward 3m bars for fine-grained event order, up to next N hours
        df3f = df3[df3.index > start_ts]
        if df3f is None or len(df3f) == 0:
            return None
        breakout_level = float(getattr(sig, 'meta', {}).get('breakout_level', 0.0)) if getattr(sig, 'meta', None) else 0.0
        use_reentry = bool(variant.cancel_on_reentry)
        # Sim loop
        for ts, row in df3f.iloc[:4000].iterrows():  # cap horizon
            hi = float(row['high']); lo = float(row['low']); cl = float(row['close'])
            if side == 'long':
                # SL/TP priority within bar extremes
                if lo <= sl:
                    r = -1.0
                    return {'r': r, 'reason': 'sl'}
                if hi >= tp:
                    r = (tp - entry) / risk
                    return {'r': r, 'reason': 'tp'}
                # Re-entry invalidation on close (default)
                if use_reentry and breakout_level > 0.0:
                    if cl <= breakout_level:
                        r = (cl - entry) / risk
                        return {'r': r, 'reason': 'reentry_invalidation'}
            else:
                if hi >= sl:
                    r = -1.0
                    return {'r': r, 'reason': 'sl'}
                if lo <= tp:
                    r = (entry - tp) / risk
                    return {'r': r, 'reason': 'tp'}
                if use_reentry and breakout_level > 0.0:
                    if cl >= breakout_level:
                        r = (entry - cl) / risk
                        return {'r': r, 'reason': 'reentry_invalidation'}
        return None


def summarize_results(all_results: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregate per-variant and per-symbol metrics from results dict."""
    rows_variant = []
    rows_symvar = []
    for vname, sym_map in all_results.items():
        agg = {'variant': vname, 'trades': 0, 'wins': 0, 'losses': 0, 'nores': 0, 'total_R': 0.0}
        for sym, trades in sym_map.items():
            sv = {'variant': vname, 'symbol': sym, 'trades': len(trades), 'wins': 0, 'losses': 0, 'total_R': 0.0}
            for t in trades:
                r = float(t.get('r', 0.0)); reason = t.get('exit_reason','')
                sv['trades'] += 0  # already set
                if reason == 'tp' or r > 0:
                    sv['wins'] += 1
                elif reason == 'sl' or r < 0:
                    sv['losses'] += 1
                sv['total_R'] += r
            rows_symvar.append(sv)
            agg['trades'] += sv['trades']; agg['wins'] += sv['wins']; agg['losses'] += sv['losses']; agg['total_R'] += sv['total_R']
        agg['wr_pct'] = (agg['wins']/agg['trades']*100.0) if agg['trades'] else 0.0
        agg['avg_R'] = (agg['total_R']/agg['trades']) if agg['trades'] else 0.0
        rows_variant.append(agg)
    dfv = pd.DataFrame(rows_variant)
    dfs = pd.DataFrame(rows_symvar)
    return dfv, dfs
    
    def _generate_trend_features(self, df: pd.DataFrame, signal: Signal, symbol: str) -> Dict:
        """Generate ML features for Trend Pullback exactly like live bot"""
        try:
            close = df['close']; high = df['high']; low = df['low']
            price = float(close.iloc[-1]) if len(close) else 0.0
            ema20 = float(close.ewm(span=20, adjust=False).mean().iloc[-1]) if len(close) else price
            ema50 = float(close.ewm(span=50, adjust=False).mean().iloc[-1]) if len(close) >= 50 else ema20
            ema_stack_score = 100.0 if (price > ema20 > ema50 or price < ema20 < ema50) else 50.0 if (ema20 != ema50) else 0.0
            rng_today = float(high.iloc[-1] - low.iloc[-1]) if len(high) else 0.0
            med_range = float((high - low).rolling(20).median().iloc[-1]) if len(df) >= 20 else max(1e-9, rng_today)
            range_expansion = float(rng_today / max(1e-9, med_range))
            prev = close.shift(); trarr = np.maximum(high - low, np.maximum((high - prev).abs(), (low - prev).abs()))
            atr = float(trarr.rolling(14).mean().iloc[-1]) if len(trarr) >= 14 else (float(trarr.iloc[-1]) if len(trarr) else 0.0)
            atr_pct = float((atr / max(1e-9, price)) * 100.0) if price else 0.0
            # Map signal meta to new features
            meta = getattr(signal, 'meta', {}) or {}
            break_dist = float(meta.get('break_dist_atr', 0.0))
            retrace_depth = float(meta.get('retrace_depth_atr', 0.0))
            confirms = int(meta.get('confirm_candles', 0))
            features = {
                'atr_pct': atr_pct,
                'break_dist_atr': break_dist,
                'retrace_depth_atr': retrace_depth,
                'confirm_candles': confirms,
                'ema_stack_score': ema_stack_score,
                'range_expansion': range_expansion,
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
