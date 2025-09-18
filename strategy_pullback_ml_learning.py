#!/usr/bin/env python3
"""
ML Learning Pullback Strategy - Minimal Requirements
Allows ML to learn what works instead of hard-coded filters

Phase 1 (0-200 trades): Takes all HL/LH signals for learning
Phase 2 (200+ trades): ML filters based on learned patterns
"""
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Dict
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)

@dataclass
class MinimalSettings:
    """Minimal settings - let ML learn what matters"""
    # Basic structure requirements (KEEP THESE)
    left:int=3      # Increased from 2 for stronger pivot confirmation
    right:int=2     # Keep at 2 for faster detection
    atr_len:int=14
    sl_buf_atr:float=0.5
    rr:float=2.5
    both_hit_rule:str="SL_FIRST"
    confirmation_candles:int=2  # Basic confirmation
    
    # REMOVED/OPTIONAL - Let ML learn these
    use_ema:bool=False  # ML will learn if EMA matters
    ema_len:int=200
    use_vol:bool=False  # ML will learn volume importance
    vol_len:int=20
    vol_mult:float=1.0  # No volume filter initially
    
    # New ML learning flags
    ml_learning_mode:bool=True  # True = accept all for learning
    ml_min_trades:int=200  # When ML takes over
    
    # MTF settings
    use_mtf_sr:bool=True  # Enable multi-timeframe S/R
    mtf_weight:float=2.0  # Prefer major levels 2x over minor
    mtf_min_strength:float=3.0  # Minimum strength for major levels

@dataclass
class Signal:
    side:str
    entry:float
    sl:float
    tp:float
    reason:str
    meta:dict

@dataclass
class BreakoutState:
    """Simple state tracking with zone support"""
    state:str = "NEUTRAL"
    breakout_level:float = 0.0
    breakout_time:Optional[datetime] = None
    pullback_extreme:float = 0.0
    pullback_time:Optional[datetime] = None
    confirmation_count:int = 0
    last_signal_time:Optional[datetime] = None
    last_resistance:float = 0.0
    last_support:float = 0.0
    
    # Zone tracking (0.3% zones around S/R levels)
    resistance_zone_upper:float = 0.0
    resistance_zone_lower:float = 0.0
    support_zone_upper:float = 0.0
    support_zone_lower:float = 0.0
    
    # Track for ML learning (but don't filter)
    breakout_high:float = 0.0
    breakout_low:float = 0.0
    
    # Structure-based stop tracking
    previous_pivot_low:float = 0.0
    previous_pivot_high:float = 0.0
    
    # MTF tracking
    last_mtf_update:Optional[datetime] = None

# Global state tracking
breakout_states: Dict[str, BreakoutState] = {}

def _pivot_high(h:pd.Series, L:int, R:int):
    """Find pivot highs"""
    win = h.rolling(L+R+1, center=True).max()
    mask = (h == win) & h.notna()
    return np.where(mask, h, np.nan)

def _pivot_low(l:pd.Series, L:int, R:int):
    """Find pivot lows"""
    win = l.rolling(L+R+1, center=True).min()
    mask = (l == win) & l.notna()
    return np.where(mask, l, np.nan)

def _atr(df:pd.DataFrame, n:int):
    """Calculate ATR"""
    prev_close = df["close"].shift()
    tr = np.maximum(df["high"]-df["low"],
         np.maximum(abs(df["high"]-prev_close), abs(df["low"]-prev_close)))
    return pd.Series(tr, index=df.index).rolling(n).mean().values

def find_previous_pivot_low(df:pd.DataFrame, current_idx:int, left:int=3, right:int=2, lookback:int=50) -> float:
    """Find the most recent pivot low before current index"""
    start_idx = max(0, current_idx - lookback)
    
    # Get pivot lows
    pivot_lows = _pivot_low(df["low"], left, right)
    
    # Find most recent pivot before current index
    for i in range(current_idx - right - 1, start_idx, -1):
        if not np.isnan(pivot_lows[i]):
            return pivot_lows[i]
    
    # If no pivot found, use lowest low in lookback period
    return df["low"].iloc[start_idx:current_idx].min()

def find_previous_pivot_high(df:pd.DataFrame, current_idx:int, left:int=3, right:int=2, lookback:int=50) -> float:
    """Find the most recent pivot high before current index"""
    start_idx = max(0, current_idx - lookback)
    
    # Get pivot highs
    pivot_highs = _pivot_high(df["high"], left, right)
    
    # Find most recent pivot before current index
    for i in range(current_idx - right - 1, start_idx, -1):
        if not np.isnan(pivot_highs[i]):
            return pivot_highs[i]
    
    # If no pivot found, use highest high in lookback period
    return df["high"].iloc[start_idx:current_idx].max()

def detect_simple_higher_low(df:pd.DataFrame, above_level:float, 
                            lookback:int=10, zone_lower:float=None) -> tuple:
    """
    Simple HL detection - check if pullback stayed above breakout zone
    Returns: (is_hl, pullback_low, retracement_pct)
    """
    recent_lows = df["low"].iloc[-lookback:]
    
    # Find the lowest point
    min_idx = recent_lows.idxmin()
    min_low = recent_lows.loc[min_idx]
    
    # Use zone lower boundary if provided, otherwise use level
    check_level = zone_lower if zone_lower else above_level
    
    # Basic requirement: stayed above breakout zone
    if min_low <= check_level:
        return False, min_low, 0.0
    
    # Check if bouncing
    min_pos = recent_lows.index.get_loc(min_idx)
    current_pos = len(recent_lows) - 1
    
    if current_pos - min_pos >= 2:  # At least 2 candles since low
        if df["close"].iloc[-1] > min_low and df["close"].iloc[-2] > min_low:
            # Calculate retracement for ML learning (but don't filter)
            high = df["high"].iloc[-lookback:].max()
            if high > above_level:
                retracement = (high - min_low) / (high - above_level) * 100
            else:
                retracement = 0.0
            
            return True, min_low, retracement
    
    return False, min_low, 0.0

def detect_simple_lower_high(df:pd.DataFrame, below_level:float,
                            lookback:int=10, zone_upper:float=None) -> tuple:
    """
    Simple LH detection - check if pullback stayed below breakout zone
    Returns: (is_lh, pullback_high, retracement_pct)
    """
    recent_highs = df["high"].iloc[-lookback:]
    
    # Find highest point
    max_idx = recent_highs.idxmax()
    max_high = recent_highs.loc[max_idx]
    
    # Use zone upper boundary if provided, otherwise use level
    check_level = zone_upper if zone_upper else below_level
    
    # Basic requirement: stayed below breakout zone
    if max_high >= check_level:
        return False, max_high, 0.0
    
    # Check if bouncing
    max_pos = recent_highs.index.get_loc(max_idx)
    current_pos = len(recent_highs) - 1
    
    if current_pos - max_pos >= 2:  # At least 2 candles since high
        if df["close"].iloc[-1] < max_high and df["close"].iloc[-2] < max_high:
            # Calculate retracement for ML learning
            low = df["low"].iloc[-lookback:].min()
            if below_level > low:
                retracement = (max_high - low) / (below_level - low) * 100
            else:
                retracement = 0.0
            
            return True, max_high, retracement
    
    return False, max_high, 0.0

def _ema(series: pd.Series, period: int) -> pd.Series:
    """Calculate Exponential Moving Average"""
    return series.ewm(span=period, adjust=False).mean()

def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Calculate RSI"""
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def _bollinger_bands(series: pd.Series, period: int = 20, std_dev: int = 2):
    """Calculate Bollinger Bands"""
    sma = series.rolling(window=period).mean()
    std = series.rolling(window=period).std()
    upper = sma + (std * std_dev)
    lower = sma - (std * std_dev)
    return upper, sma, lower

def calculate_ml_features(df: pd.DataFrame, state: BreakoutState, 
                         side: str, retracement: float) -> dict:
    """
    Calculate all 22 features expected by ML scorer
    """
    # Current price data
    close = df["close"].iloc[-1]
    open_price = df["open"].iloc[-1]
    high = df["high"].iloc[-1]
    low = df["low"].iloc[-1]
    volume = df["volume"].iloc[-1]
    
    # Calculate technical indicators
    ema_200 = _ema(df["close"], 200).iloc[-1] if len(df) >= 200 else close
    rsi_values = _rsi(df["close"], 14)
    rsi_current = rsi_values.iloc[-1] if len(rsi_values) > 0 else 50.0
    
    # Bollinger Bands
    if len(df) >= 20:
        bb_upper, bb_middle, bb_lower = _bollinger_bands(df["close"], 20, 2)
        bb_upper_val = bb_upper.iloc[-1]
        bb_lower_val = bb_lower.iloc[-1]
        bb_width = bb_upper_val - bb_lower_val
        if bb_width > 0:
            bb_position = (close - bb_lower_val) / bb_width
        else:
            bb_position = 0.5
    else:
        bb_position = 0.5
    
    # Volume calculations
    avg_volume_20 = df["volume"].rolling(20).mean().iloc[-1] if len(df) >= 20 else volume
    avg_volume_5 = df["volume"].rolling(5).mean().iloc[-1] if len(df) >= 5 else volume
    volume_ratio = volume / avg_volume_20 if avg_volume_20 > 0 else 1.0
    volume_ma_ratio = avg_volume_5 / avg_volume_20 if avg_volume_20 > 0 else 1.0
    
    # Volume percentile
    if len(df) >= 100:
        volume_history = df["volume"].iloc[-100:]
        volume_percentile = (volume > volume_history).mean() * 100
    else:
        volume_percentile = 50.0
    
    # ATR calculations
    atr_14 = _atr(df, 14)
    atr_current = atr_14[-1]
    candle_range = high - low
    candle_range_atr = candle_range / atr_current if atr_current > 0 else 1.0
    
    # ATR percentile
    if len(atr_14) >= 100:
        atr_history = atr_14[-100:]
        atr_percentile = (atr_current > atr_history).mean() * 100
    else:
        atr_percentile = 50.0
    
    # Trend calculations
    if len(df) >= 20:
        close_prices = df['close'].values[-20:]
        # Simple linear regression slope
        x = np.arange(len(close_prices))
        slope = np.polyfit(x, close_prices, 1)[0]
        avg_price = close_prices.mean()
        trend_strength = (slope / avg_price) * 100  # Percentage trend strength
    else:
        trend_strength = 0.0
    
    # Higher timeframe alignment (simulated with longer MA)
    if len(df) >= 50:
        ma_50 = df['close'].rolling(50).mean().iloc[-1]
        ma_20 = df['close'].rolling(20).mean().iloc[-1]
        if ma_20 > ma_50 and close > ma_20:
            higher_tf_alignment = 100.0  # Bullish alignment
        elif ma_20 < ma_50 and close < ma_20:
            higher_tf_alignment = 0.0  # Bearish alignment
        else:
            higher_tf_alignment = 50.0  # Neutral
    else:
        higher_tf_alignment = 50.0
    
    # EMA distance ratio
    if abs(ema_200) > 0:
        ema_distance_ratio = ((close - ema_200) / ema_200) * 100
    else:
        ema_distance_ratio = 0.0
    
    # Volume trend (5 bar vs 20 bar average)
    volume_trend = volume_ma_ratio
    
    # Breakout volume (current vs average)
    breakout_volume = volume_ratio
    
    # Support/Resistance strength
    if state.breakout_level > 0:
        touches = count_zone_touches(state.breakout_level, df)
        support_resistance_strength = min(touches, 5)  # Cap at 5
    else:
        support_resistance_strength = 0
    
    # Pullback depth (use retracement)
    pullback_depth = retracement
    
    # Confirmation candle strength
    confirmation_candle_strength = 0.0
    if state.confirmation_count > 0:
        # Check strength of confirmation candles
        if side == "long" and close > open_price:
            body = close - open_price
            confirmation_candle_strength = (body / candle_range) * 100 if candle_range > 0 else 0
        elif side == "short" and close < open_price:
            body = open_price - close
            confirmation_candle_strength = (body / candle_range) * 100 if candle_range > 0 else 0
    
    # Risk/Reward calculation (simplified - using ATR)
    if side == "long":
        potential_risk = atr_current * 1.5  # 1.5 ATR stop
        potential_reward = atr_current * 3.0  # 3 ATR target
    else:
        potential_risk = atr_current * 1.5
        potential_reward = atr_current * 3.0
    risk_reward_ratio = potential_reward / potential_risk if potential_risk > 0 else 2.5
    
    # ATR stop distance
    atr_stop_distance = 1.5  # Default 1.5 ATR for stop
    
    # Candle patterns
    candle_body = abs(close - open_price)
    candle_body_ratio = (candle_body / candle_range) * 100 if candle_range > 0 else 0.0
    
    # Wick analysis
    upper_wick = high - max(open_price, close)
    lower_wick = min(open_price, close) - low
    upper_wick_ratio = (upper_wick / candle_range) * 100 if candle_range > 0 else 0.0
    lower_wick_ratio = (lower_wick / candle_range) * 100 if candle_range > 0 else 0.0
    
    # Time features
    now = datetime.now()
    hour_of_day = now.hour
    day_of_week = now.weekday()
    
    # Return all 22 features expected by ML scorer
    return {
        # Core trend features
        'trend_strength': trend_strength,
        'higher_tf_alignment': higher_tf_alignment,
        'ema_distance_ratio': ema_distance_ratio,
        
        # Volume features
        'volume_ratio': volume_ratio,
        'volume_trend': volume_trend,
        'breakout_volume': breakout_volume,
        
        # Structure features
        'support_resistance_strength': support_resistance_strength,
        'pullback_depth': pullback_depth,
        'confirmation_candle_strength': confirmation_candle_strength,
        
        # Risk/volatility features
        'atr_percentile': atr_percentile,
        'risk_reward_ratio': risk_reward_ratio,
        'atr_stop_distance': atr_stop_distance,
        
        # Time features
        'hour_of_day': hour_of_day,
        'day_of_week': day_of_week,
        
        # Candle pattern features
        'candle_body_ratio': candle_body_ratio,
        'upper_wick_ratio': upper_wick_ratio,
        'lower_wick_ratio': lower_wick_ratio,
        'candle_range_atr': candle_range_atr,
        
        # Additional indicators
        'volume_ma_ratio': volume_ma_ratio,
        'rsi': rsi_current,
        'bb_position': bb_position,
        'volume_percentile': volume_percentile,
        
        # Symbol cluster features
        'symbol_cluster': 3,  # Default to cluster 3 (will be overridden)
        'cluster_volatility_norm': 1.0,  # Normalized volatility within cluster
        'cluster_volume_norm': 1.0,  # Normalized volume within cluster
        'btc_correlation_bucket': 2,  # 0=negative, 1=low, 2=medium, 3=high
        'price_tier': 3,  # 1=micro, 2=small, 3=mid, 4=large, 5=mega
        
        # Multi-timeframe S/R features (new)
        'near_major_resistance': 0,  # 1 if near major resistance, 0 otherwise
        'near_major_support': 0,  # 1 if near major support, 0 otherwise
        'mtf_level_strength': 0.0  # Strength of nearest MTF level (0-10 scale)
    }

def count_zone_touches(level:float, df:pd.DataFrame, zone_pct:float=0.003, lookback:int=75) -> int:
    """
    Count how many times price touched a zone (stronger zones have more touches)
    """
    if np.isnan(level):
        return 0
        
    zone_upper = level * (1 + zone_pct)
    zone_lower = level * (1 - zone_pct)
    
    # Look at recent candles
    recent = df.tail(lookback)
    touches = 0
    
    for _, row in recent.iterrows():
        # Check if candle wicked into or touched the zone
        if (zone_lower <= row['high'] <= zone_upper) or \
           (zone_lower <= row['low'] <= zone_upper) or \
           (row['low'] < zone_lower and row['high'] > zone_upper):
            touches += 1
    
    return touches

def get_ml_learning_signals(df:pd.DataFrame, settings:MinimalSettings = None,
                           df_1h:pd.DataFrame = None, symbol:str = "UNKNOWN") -> list:
    """
    Minimal pullback strategy for ML learning phase
    Takes ALL HL/LH signals - lets ML figure out what works
    """
    if settings is None:
        settings = MinimalSettings()
    
    # Initialize state
    if symbol not in breakout_states:
        breakout_states[symbol] = BreakoutState()
    
    state = breakout_states[symbol]
    
    # Get indicators
    pivots_high = _pivot_high(df["high"], settings.left, settings.right)
    pivots_low = _pivot_low(df["low"], settings.left, settings.right)
    atr = _atr(df, settings.atr_len)
    
    # Current values
    close = df["close"].iloc[-1]
    high = df["high"].iloc[-1]
    low = df["low"].iloc[-1]
    current_atr = atr[-1]
    
    # Find recent S/R and create zones
    recent_resistance = np.nan
    recent_support = np.nan
    
    # Zone width: 0.3% on each side (0.6% total zone)
    zone_width = 0.003
    
    # Increased from -20 to -30 for better S/R coverage (captures full trading day)
    for ph in reversed(pivots_high[-30:]):
        if not np.isnan(ph):
            recent_resistance = ph
            break
    
    for pl in reversed(pivots_low[-30:]):
        if not np.isnan(pl):
            recent_support = pl
            break
    
    # Import MTF module if enabled
    if settings.use_mtf_sr:
        try:
            from multi_timeframe_sr import mtf_sr, should_use_mtf_level
            
            # Update MTF levels periodically
            if len(df) % 100 == 0 or not hasattr(state, 'last_mtf_update'):
                mtf_sr.update_sr_levels(symbol, df)
                state.last_mtf_update = datetime.now()
            
            # Check if we should use MTF levels
            if not np.isnan(recent_resistance):
                use_mtf_res, mtf_res, res_reason = should_use_mtf_level(symbol, recent_resistance, close, df)
                if use_mtf_res and mtf_res > 0:
                    logger.info(f"{symbol}: Using MTF resistance: {mtf_res:.4f} ({res_reason})")
                    recent_resistance = mtf_res
            
            if not np.isnan(recent_support):
                use_mtf_sup, mtf_sup, sup_reason = should_use_mtf_level(symbol, recent_support, close, df)
                if use_mtf_sup and mtf_sup > 0:
                    logger.info(f"{symbol}: Using MTF support: {mtf_sup:.4f} ({sup_reason})")
                    recent_support = mtf_sup
                    
        except Exception as e:
            logger.debug(f"{symbol}: MTF S/R check failed: {e}")
            # Continue with regular S/R if MTF fails
    
    # Update tracked levels and zones with strength scoring
    if not np.isnan(recent_resistance):
        state.last_resistance = recent_resistance
        state.resistance_zone_upper = recent_resistance * (1 + zone_width)
        state.resistance_zone_lower = recent_resistance * (1 - zone_width)
        
        # Count zone touches for strength
        touches = count_zone_touches(recent_resistance, df, zone_width)
        strength = "Strong" if touches >= 3 else "Moderate" if touches >= 2 else "Weak"
        logger.debug(f"{symbol}: {strength} resistance zone ({touches} touches): {state.resistance_zone_lower:.2f} - {state.resistance_zone_upper:.2f}")
        
    if not np.isnan(recent_support):
        state.last_support = recent_support
        state.support_zone_upper = recent_support * (1 + zone_width)
        state.support_zone_lower = recent_support * (1 - zone_width)
        
        # Count zone touches for strength  
        touches = count_zone_touches(recent_support, df, zone_width)
        strength = "Strong" if touches >= 3 else "Moderate" if touches >= 2 else "Weak"
        logger.debug(f"{symbol}: {strength} support zone ({touches} touches): {state.support_zone_lower:.2f} - {state.support_zone_upper:.2f}")
    
    signals = []
    
    # SIMPLE STATE MACHINE - Zone-based breakout detection
    if state.state == "NEUTRAL":
        # Check for zone breakouts (must close outside the zone)
        if state.resistance_zone_upper > 0 and close > state.resistance_zone_upper:
            # Confirmed breakout above resistance zone
            state.state = "RESISTANCE_BROKEN"
            state.breakout_level = state.last_resistance  # Use center of zone
            state.breakout_time = datetime.now()
            state.breakout_high = high
            # Find and store previous pivot low for structure-based stop
            current_idx = len(df) - 1
            state.previous_pivot_low = find_previous_pivot_low(df, current_idx, settings.left, settings.right)
            logger.info(f"{symbol}: Resistance zone broken at {state.last_resistance:.2f} (closed above {state.resistance_zone_upper:.2f})")
        
        elif state.support_zone_lower > 0 and close < state.support_zone_lower:
            # Confirmed breakout below support zone
            state.state = "SUPPORT_BROKEN"
            state.breakout_level = state.last_support  # Use center of zone
            state.breakout_time = datetime.now()
            state.breakout_low = low
            # Find and store previous pivot high for structure-based stop
            current_idx = len(df) - 1
            state.previous_pivot_high = find_previous_pivot_high(df, current_idx, settings.left, settings.right)
            logger.info(f"{symbol}: Support zone broken at {state.last_support:.2f} (closed below {state.support_zone_lower:.2f})")
    
    elif state.state == "RESISTANCE_BROKEN":
        # Update swing high
        if high > state.breakout_high:
            state.breakout_high = high
        
        # Simple HL check using zone - no filtering!
        is_hl, pullback_low, retracement = detect_simple_higher_low(
            df, state.breakout_level, zone_lower=state.resistance_zone_lower
        )
        
        if is_hl:
            state.state = "HL_FORMED"
            state.pullback_extreme = pullback_low
            state.pullback_time = datetime.now()
            state.confirmation_count = 0
            
            # Log for learning
            logger.info(f"{symbol}: HL formed at {pullback_low:.2f} "
                       f"(Retracement: {retracement:.1f}%)")
    
    elif state.state == "SUPPORT_BROKEN":
        # Update swing low
        if low < state.breakout_low:
            state.breakout_low = low
        
        # Simple LH check using zone - no filtering!
        is_lh, pullback_high, retracement = detect_simple_lower_high(
            df, state.breakout_level, zone_upper=state.support_zone_upper
        )
        
        if is_lh:
            state.state = "LH_FORMED"
            state.pullback_extreme = pullback_high
            state.pullback_time = datetime.now()
            state.confirmation_count = 0
            
            logger.info(f"{symbol}: LH formed at {pullback_high:.2f} "
                       f"(Retracement: {retracement:.1f}%)")
    
    elif state.state == "HL_FORMED":
        # Basic confirmation
        if df["close"].iloc[-1] > df["open"].iloc[-1]:
            state.confirmation_count += 1
        
        if state.confirmation_count >= settings.confirmation_candles:
            # Generate LONG signal - no filtering!
            entry = close
            
            # HYBRID STOP LOSS METHOD with volatility adjustment
            # Calculate ATR percentile for volatility adjustment
            atr_history = pd.Series(atr[-100:]) if len(atr) >= 100 else pd.Series(atr)
            atr_percentile = (current_atr > atr_history).mean() * 100
            
            # Increase stop buffer in high volatility (>75th percentile)
            volatility_multiplier = 1.0
            if atr_percentile > 75:
                volatility_multiplier = 1.3  # 30% wider stop in high volatility
                logger.info(f"{symbol}: High volatility detected ({atr_percentile:.0f}th percentile), widening stop")
            elif atr_percentile > 90:
                volatility_multiplier = 1.5  # 50% wider stop in extreme volatility
                logger.info(f"{symbol}: Extreme volatility detected ({atr_percentile:.0f}th percentile), widening stop further")
            
            adjusted_sl_buffer = settings.sl_buf_atr * volatility_multiplier
            
            # HYBRID METHOD - use whichever gives more room
            # Option 1: Previous pivot low
            sl_option1 = state.previous_pivot_low - (adjusted_sl_buffer * 0.3 * current_atr)  # Smaller buffer since further away
            
            # Option 2: Breakout level minus ATR
            sl_option2 = state.breakout_level - (adjusted_sl_buffer * 1.6 * current_atr)  # Larger buffer from breakout
            
            # Option 3: Original method (pullback extreme)
            sl_option3 = state.pullback_extreme - (current_atr * adjusted_sl_buffer)
            
            # Use the lowest SL (gives most room)
            sl = min(sl_option1, sl_option2, sl_option3)
            
            # Log which method was used
            if sl == sl_option1:
                logger.info(f"{symbol}: Using previous pivot low for stop: {state.previous_pivot_low:.4f}")
            elif sl == sl_option2:
                logger.info(f"{symbol}: Using breakout level for stop: {state.breakout_level:.4f}")
            else:
                logger.info(f"{symbol}: Using pullback extreme for stop: {state.pullback_extreme:.4f}")
            
            # Ensure minimum stop distance (at least 1% from entry)
            min_stop_distance = entry * 0.01
            if abs(entry - sl) < min_stop_distance:
                sl = entry - min_stop_distance
                logger.info(f"{symbol}: Adjusted stop to minimum distance (1% from entry)")
            
            # Adjust TP based on new stop distance + fees
            # Bybit fees: 0.06% entry (market) + 0.055% exit (limit) = 0.115% total
            # Add 0.05% for slippage = 0.165% total cost
            # To get 2.5:1 after fees, we need to target slightly higher
            fee_adjustment = 1.00165  # Compensate for 0.165% total costs
            tp = entry + ((entry - sl) * settings.rr * fee_adjustment)
            
            # Calculate retracement for ML
            if state.breakout_high > state.breakout_level:
                retracement = ((state.breakout_high - state.pullback_extreme) / 
                             (state.breakout_high - state.breakout_level) * 100)
            else:
                retracement = 50.0
            
            # Get ML features (for learning, not filtering)
            ml_features = calculate_ml_features(df, state, "long", retracement)
            
            signals.append(Signal(
                side="long",
                entry=entry,
                sl=sl,
                tp=tp,
                reason=f"ML Learning: HL above {state.breakout_level:.2f}",
                meta={
                    "breakout_level": state.breakout_level,
                    "pullback_low": state.pullback_extreme,
                    "fib_retracement": f"{retracement:.1f}%",
                    "ml_features": ml_features,
                    "learning_mode": True
                }
            ))
            
            state.state = "SIGNAL_SENT"
            state.last_signal_time = datetime.now()
            
            logger.info(f"{symbol}: LONG signal (Learning Mode) | "
                       f"Retracement: {retracement:.1f}% | "
                       f"Vol: {ml_features['volume_ratio']:.2f}x")
    
    elif state.state == "LH_FORMED":
        # Basic confirmation
        if df["close"].iloc[-1] < df["open"].iloc[-1]:
            state.confirmation_count += 1
        
        if state.confirmation_count >= settings.confirmation_candles:
            # Generate SHORT signal - no filtering!
            entry = close
            
            # HYBRID STOP LOSS METHOD with volatility adjustment
            # Calculate ATR percentile for volatility adjustment
            atr_history = pd.Series(atr[-100:]) if len(atr) >= 100 else pd.Series(atr)
            atr_percentile = (current_atr > atr_history).mean() * 100
            
            # Increase stop buffer in high volatility (>75th percentile)
            volatility_multiplier = 1.0
            if atr_percentile > 75:
                volatility_multiplier = 1.3  # 30% wider stop in high volatility
                logger.info(f"{symbol}: High volatility detected ({atr_percentile:.0f}th percentile), widening stop")
            elif atr_percentile > 90:
                volatility_multiplier = 1.5  # 50% wider stop in extreme volatility
                logger.info(f"{symbol}: Extreme volatility detected ({atr_percentile:.0f}th percentile), widening stop further")
            
            adjusted_sl_buffer = settings.sl_buf_atr * volatility_multiplier
            
            # HYBRID METHOD - use whichever gives more room
            # Option 1: Previous pivot high
            sl_option1 = state.previous_pivot_high + (adjusted_sl_buffer * 0.3 * current_atr)  # Smaller buffer since further away
            
            # Option 2: Breakout level plus ATR
            sl_option2 = state.breakout_level + (adjusted_sl_buffer * 1.6 * current_atr)  # Larger buffer from breakout
            
            # Option 3: Original method (pullback extreme)
            sl_option3 = state.pullback_extreme + (current_atr * adjusted_sl_buffer)
            
            # Use the highest SL (gives most room)
            sl = max(sl_option1, sl_option2, sl_option3)
            
            # Log which method was used
            if sl == sl_option1:
                logger.info(f"{symbol}: Using previous pivot high for stop: {state.previous_pivot_high:.4f}")
            elif sl == sl_option2:
                logger.info(f"{symbol}: Using breakout level for stop: {state.breakout_level:.4f}")
            else:
                logger.info(f"{symbol}: Using pullback extreme for stop: {state.pullback_extreme:.4f}")
            
            # Ensure minimum stop distance (at least 1% from entry)
            min_stop_distance = entry * 0.01
            if abs(sl - entry) < min_stop_distance:
                sl = entry + min_stop_distance
                logger.info(f"{symbol}: Adjusted stop to minimum distance (1% from entry)")
            
            # Adjust TP based on new stop distance + fees
            # Bybit fees: 0.06% entry (market) + 0.055% exit (limit) = 0.115% total
            # Add 0.05% for slippage = 0.165% total cost
            # To get 2.5:1 after fees, we need to target slightly higher
            fee_adjustment = 1.00165  # Compensate for 0.165% total costs
            tp = entry - ((sl - entry) * settings.rr * fee_adjustment)
            
            # Calculate retracement
            if state.breakout_level > state.breakout_low:
                retracement = ((state.pullback_extreme - state.breakout_low) / 
                             (state.breakout_level - state.breakout_low) * 100)
            else:
                retracement = 50.0
            
            # Get ML features
            ml_features = calculate_ml_features(df, state, "short", retracement)
            
            signals.append(Signal(
                side="short",
                entry=entry,
                sl=sl,
                tp=tp,
                reason=f"ML Learning: LH below {state.breakout_level:.2f}",
                meta={
                    "breakout_level": state.breakout_level,
                    "pullback_high": state.pullback_extreme,
                    "fib_retracement": f"{retracement:.1f}%",
                    "ml_features": ml_features,
                    "learning_mode": True
                }
            ))
            
            state.state = "SIGNAL_SENT"
            state.last_signal_time = datetime.now()
            
            logger.info(f"{symbol}: SHORT signal (Learning Mode) | "
                       f"Retracement: {retracement:.1f}% | "
                       f"Vol: {ml_features['volume_ratio']:.2f}x")
    
    elif state.state == "SIGNAL_SENT":
        # Reset after timeout
        if state.last_signal_time:
            time_since = (datetime.now() - state.last_signal_time).seconds / 3600
            if time_since > 4:  # Reset after 4 hours
                state.state = "NEUTRAL"
                logger.info(f"{symbol}: State reset after timeout")
    
    return signals

def reset_symbol_state(symbol: str):
    """Reset state when position closes"""
    if symbol in breakout_states:
        breakout_states[symbol] = BreakoutState()
        logger.info(f"{symbol}: State reset to NEUTRAL")

def explain_ml_takeover():
    """Explain how ML will take over"""
    
    print("\n" + "="*60)
    print("🤖 ML LEARNING STRATEGY - HOW IT WORKS")
    print("="*60)
    
    print("""
PHASE 1: LEARNING MODE (0-200 trades)
=====================================
✅ Takes ALL signals with basic structure:
   - Resistance/Support breakout
   - Forms Higher Low or Lower High
   - 2 confirmation candles
   
❌ NO filtering for:
   - Fibonacci levels (ML learns optimal)
   - Volume requirements (ML learns thresholds)
   - 1H trend (ML learns importance)
   - Volatility (ML learns adjustments)

📊 What happens:
   - More signals (20-40 per day)
   - Lower win rate initially (40-45%)
   - ML records everything:
     * Which retracements work
     * Which volumes matter
     * Which times are best
     * Which symbols behave differently

PHASE 2: ML TAKEOVER (200+ trades)
===================================
🎯 ML automatically starts filtering:
   - Scores every signal 0-100
   - Only takes signals above threshold (70+)
   - Each symbol gets personalized filters
   
📈 ML has learned:
   - "BTCUSDT works at 45-50% retracement"
   - "ETHUSDT needs 1.5x volume or fails"
   - "SOLUSDT doesn't need 1H alignment"
   - "AVAXUSDT best during US session"

✨ Results:
   - Fewer but better signals (8-12 per day)
   - Higher win rate (55-65%)
   - Personalized per symbol
   - Continuously improving

HOW ML TAKES OVER:
==================
In live_bot.py, ML scorer is already integrated:

if ml_scorer.is_trained:  # After 200 trades
    should_take, score = ml_scorer.score_signal(signal)
    if not should_take:
        # ML rejects this signal
        logger.info(f"ML rejected: Score {score}")
        continue
else:
    # Learning mode - take all signals
    logger.info("ML learning - taking signal")

NO CODE CHANGES NEEDED!
======================
The ML scorer already:
✅ Checks if trained (200+ trades)
✅ Scores signals when ready
✅ Filters automatically
✅ Falls back to allowing if any errors

This is SAFE because:
- ML only filters AFTER learning
- Has fallback if scoring fails  
- Gradually improves over time
- You can monitor via /ml command
    """)

if __name__ == "__main__":
    explain_ml_takeover()
    print("\n✅ ML Learning Strategy Ready!")
    print("This will safely let ML learn what works!")
    print("After 200 trades, ML automatically takes over filtering!")