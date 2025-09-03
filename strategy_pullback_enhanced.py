#!/usr/bin/env python3
"""
Enhanced Pullback Strategy with Multi-Timeframe and Dynamic Volatility
"""
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Dict, Tuple
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)

@dataclass
class Settings:
    left:int=2
    right:int=2
    atr_len:int=14
    sl_buf_atr:float=0.5
    rr:float=2.0
    use_ema:bool=False
    ema_len:int=200
    use_vol:bool=False
    vol_len:int=20
    vol_mult:float=1.2
    both_hit_rule:str="SL_FIRST"
    confirmation_candles:int=2  # Base confirmation candles (adjusted dynamically)
    
    # New enhanced settings
    use_higher_tf:bool=True  # Use 1H timeframe confirmation
    use_fibonacci:bool=True  # Use Fibonacci pullback filtering
    use_dynamic_params:bool=True  # Adjust params based on volatility
    fib_min:float=0.382  # Minimum Fibonacci retracement (38.2%)
    fib_max:float=0.618  # Maximum Fibonacci retracement (61.8%)
    fib_danger:float=0.786  # Danger zone (78.6%)

@dataclass
class Signal:
    side:str             # "long" or "short"
    entry:float
    sl:float
    tp:float
    reason:str
    meta:dict

@dataclass
class BreakoutState:
    """Track the state of a breakout for each symbol"""
    state:str = "NEUTRAL"
    breakout_level:float = 0.0
    breakout_time:Optional[datetime] = None
    pullback_extreme:float = 0.0
    pullback_time:Optional[datetime] = None
    confirmation_count:int = 0
    last_signal_time:Optional[datetime] = None
    last_resistance:float = 0.0
    last_support:float = 0.0
    
    # New tracking
    breakout_high:float = 0.0  # Swing high after breakout (for Fib calc)
    breakout_low:float = 0.0   # Swing low after breakout (for Fib calc)
    higher_tf_trend:str = "NEUTRAL"  # 1H trend direction
    volatility_regime:str = "NORMAL"  # LOW, NORMAL, HIGH
    adjusted_confirmation:int = 2  # Dynamically adjusted

# Global state tracking for each symbol
breakout_states: Dict[str, BreakoutState] = {}

def _pivot_high(h:pd.Series, L:int, R:int):
    """Find pivot highs in price series"""
    win = h.rolling(L+R+1, center=True).max()
    mask = (h == win) & h.notna()
    return np.where(mask, h, np.nan)

def _pivot_low(l:pd.Series, L:int, R:int):
    """Find pivot lows in price series"""
    win = l.rolling(L+R+1, center=True).min()
    mask = (l == win) & l.notna()
    return np.where(mask, l, np.nan)

def _atr(df:pd.DataFrame, n:int):
    """Calculate Average True Range"""
    prev_close = df["close"].shift()
    tr = np.maximum(df["high"]-df["low"],
         np.maximum(abs(df["high"]-prev_close), abs(df["low"]-prev_close)))
    return pd.Series(tr, index=df.index).rolling(n).mean().values

def _ema(s:pd.Series, n:int):
    """Calculate Exponential Moving Average"""
    return s.ewm(span=n, adjust=False).mean().values

def analyze_higher_timeframe_trend(df_1h: pd.DataFrame) -> str:
    """
    Analyze 1H timeframe to determine overall trend
    Returns: "BULLISH", "BEARISH", or "NEUTRAL"
    """
    if len(df_1h) < 50:
        return "NEUTRAL"
    
    # Use 20 and 50 period EMAs on 1H
    ema20 = _ema(df_1h["close"], 20)
    ema50 = _ema(df_1h["close"], 50)
    
    # Get recent values
    current_price = df_1h["close"].iloc[-1]
    current_ema20 = ema20[-1]
    current_ema50 = ema50[-1]
    
    # Trend detection
    if current_price > current_ema20 > current_ema50:
        # Check for higher lows in recent pivots
        pivots_low = _pivot_low(df_1h["low"], 5, 5)
        recent_pivots = [p for p in pivots_low[-20:] if not np.isnan(p)]
        if len(recent_pivots) >= 2 and recent_pivots[-1] > recent_pivots[-2]:
            return "BULLISH"
    
    elif current_price < current_ema20 < current_ema50:
        # Check for lower highs in recent pivots
        pivots_high = _pivot_high(df_1h["high"], 5, 5)
        recent_pivots = [p for p in pivots_high[-20:] if not np.isnan(p)]
        if len(recent_pivots) >= 2 and recent_pivots[-1] < recent_pivots[-2]:
            return "BEARISH"
    
    return "NEUTRAL"

def get_volatility_regime(df: pd.DataFrame, lookback: int = 100) -> Tuple[str, float]:
    """
    Determine current volatility regime based on ATR percentile
    Returns: (regime, atr_percentile)
    """
    if len(df) < lookback:
        return "NORMAL", 50.0
    
    # Calculate ATR
    atr_values = _atr(df.iloc[-lookback:], 14)
    current_atr = atr_values[-1]
    
    # Calculate percentile
    atr_percentile = np.percentile(atr_values[~np.isnan(atr_values)], 
                                   np.searchsorted(np.sort(atr_values[~np.isnan(atr_values)]), current_atr) * 100 / len(atr_values[~np.isnan(atr_values)]))
    
    # Classify regime
    if atr_percentile < 30:
        return "LOW", atr_percentile
    elif atr_percentile > 70:
        return "HIGH", atr_percentile
    else:
        return "NORMAL", atr_percentile

def adjust_parameters_for_volatility(base_settings: Settings, volatility_regime: str) -> Settings:
    """
    Dynamically adjust strategy parameters based on volatility
    """
    settings = base_settings
    
    if volatility_regime == "LOW":
        # Low volatility: Be more selective
        settings.confirmation_candles = 3  # Need more confirmation
        settings.sl_buf_atr = 0.3  # Tighter stops
        settings.rr = 2.5  # Look for better R:R
        settings.vol_mult = 1.5  # Higher volume requirement
        
    elif volatility_regime == "HIGH":
        # High volatility: Be more flexible
        settings.confirmation_candles = 2  # Standard confirmation
        settings.sl_buf_atr = 0.7  # Wider stops for volatility
        settings.rr = 1.8  # Accept lower R:R
        settings.vol_mult = 1.0  # Lower volume requirement
        
    else:  # NORMAL
        # Keep default settings
        settings.confirmation_candles = 2
        settings.sl_buf_atr = 0.5
        settings.rr = 2.0
        settings.vol_mult = 1.2
    
    return settings

def calculate_fibonacci_retracement(breakout_level: float, swing_extreme: float, 
                                   current_price: float, side: str) -> Tuple[float, bool]:
    """
    Calculate Fibonacci retracement percentage and check if it's in golden zone
    Returns: (retracement_pct, is_golden_zone)
    """
    if side == "long":
        # For long: breakout_level is the low, swing_extreme is the high
        move_size = swing_extreme - breakout_level
        if move_size <= 0:
            return 0.0, False
        retracement = (swing_extreme - current_price) / move_size
    else:
        # For short: breakout_level is the high, swing_extreme is the low  
        move_size = breakout_level - swing_extreme
        if move_size <= 0:
            return 0.0, False
        retracement = (current_price - swing_extreme) / move_size
    
    # Check if in golden zone (38.2% - 61.8%)
    is_golden = 0.382 <= retracement <= 0.618
    
    return retracement, is_golden

def detect_enhanced_higher_low(df: pd.DataFrame, state: BreakoutState, 
                              settings: Settings, lookback: int = 10) -> bool:
    """
    Enhanced higher low detection with Fibonacci filtering
    """
    recent_lows = df["low"].iloc[-lookback:]
    
    # Find the lowest point in recent candles
    min_idx = recent_lows.idxmin()
    min_low = recent_lows.loc[min_idx]
    
    # Basic higher low check
    if min_low <= state.breakout_level:
        return False
    
    # Check bounce from low
    min_pos = recent_lows.index.get_loc(min_idx)
    current_pos = len(recent_lows) - 1
    if current_pos - min_pos < 2:  # Need at least 2 candles since low
        return False
    
    # Check upward movement
    if df["close"].iloc[-1] <= min_low or df["close"].iloc[-2] <= min_low:
        return False
    
    # Fibonacci check if enabled
    if settings.use_fibonacci and state.breakout_high > 0:
        current_price = df["close"].iloc[-1]
        retrace_pct, is_golden = calculate_fibonacci_retracement(
            state.breakout_level, state.breakout_high, min_low, "long"
        )
        
        if retrace_pct > settings.fib_danger:
            logger.info(f"Pullback too deep: {retrace_pct:.1%} > {settings.fib_danger:.1%}")
            return False
        
        if not is_golden:
            logger.info(f"Pullback not in golden zone: {retrace_pct:.1%}")
            # Allow but with lower confidence
    
    state.pullback_extreme = min_low
    return True

def detect_enhanced_lower_high(df: pd.DataFrame, state: BreakoutState,
                               settings: Settings, lookback: int = 10) -> bool:
    """
    Enhanced lower high detection with Fibonacci filtering
    """
    recent_highs = df["high"].iloc[-lookback:]
    
    # Find highest point in recent candles
    max_idx = recent_highs.idxmax()
    max_high = recent_highs.loc[max_idx]
    
    # Basic lower high check
    if max_high >= state.breakout_level:
        return False
    
    # Check bounce from high
    max_pos = recent_highs.index.get_loc(max_idx)
    current_pos = len(recent_highs) - 1
    if current_pos - max_pos < 2:  # Need at least 2 candles since high
        return False
    
    # Check downward movement
    if df["close"].iloc[-1] >= max_high or df["close"].iloc[-2] >= max_high:
        return False
    
    # Fibonacci check if enabled
    if settings.use_fibonacci and state.breakout_low > 0:
        current_price = df["close"].iloc[-1]
        retrace_pct, is_golden = calculate_fibonacci_retracement(
            state.breakout_level, state.breakout_low, max_high, "short"
        )
        
        if retrace_pct > settings.fib_danger:
            logger.info(f"Pullback too deep: {retrace_pct:.1%} > {settings.fib_danger:.1%}")
            return False
        
        if not is_golden:
            logger.info(f"Pullback not in golden zone: {retrace_pct:.1%}")
    
    state.pullback_extreme = max_high
    return True

def get_signals(df:pd.DataFrame, settings:Settings = None, 
                df_1h:pd.DataFrame = None, symbol:str = "UNKNOWN") -> list:
    """
    Enhanced pullback strategy with multi-timeframe and dynamic parameters
    """
    if settings is None:
        settings = Settings()
    
    # Initialize state for symbol if not exists
    if symbol not in breakout_states:
        breakout_states[symbol] = BreakoutState()
    
    state = breakout_states[symbol]
    
    # Analyze higher timeframe if available
    if settings.use_higher_tf and df_1h is not None and len(df_1h) > 50:
        state.higher_tf_trend = analyze_higher_timeframe_trend(df_1h)
        logger.info(f"{symbol} 1H Trend: {state.higher_tf_trend}")
    
    # Get volatility regime and adjust parameters
    if settings.use_dynamic_params:
        volatility_regime, atr_pct = get_volatility_regime(df)
        state.volatility_regime = volatility_regime
        state.adjusted_confirmation = adjust_parameters_for_volatility(settings, volatility_regime).confirmation_candles
        logger.info(f"{symbol} Volatility: {volatility_regime} (ATR percentile: {atr_pct:.0f}%), Confirmations needed: {state.adjusted_confirmation}")
    else:
        state.adjusted_confirmation = settings.confirmation_candles
    
    # Calculate indicators
    pivots_high = _pivot_high(df["high"], settings.left, settings.right)
    pivots_low = _pivot_low(df["low"], settings.left, settings.right)
    atr = _atr(df, settings.atr_len)
    
    # Get current values
    close = df["close"].iloc[-1]
    high = df["high"].iloc[-1]
    low = df["low"].iloc[-1]
    current_atr = atr[-1]
    
    # Find recent S/R levels
    recent_resistance = np.nan
    recent_support = np.nan
    
    for ph in reversed(pivots_high[-20:]):
        if not np.isnan(ph):
            recent_resistance = ph
            break
    
    for pl in reversed(pivots_low[-20:]):
        if not np.isnan(pl):
            recent_support = pl
            break
    
    # Update tracked levels
    if not np.isnan(recent_resistance):
        state.last_resistance = recent_resistance
    if not np.isnan(recent_support):
        state.last_support = recent_support
    
    signals = []
    
    # State machine logic with enhancements
    if state.state == "NEUTRAL":
        # Check for breakouts
        if not np.isnan(recent_resistance) and close > recent_resistance:
            # Higher timeframe alignment check
            if settings.use_higher_tf and state.higher_tf_trend == "BEARISH":
                logger.info(f"{symbol}: Ignoring resistance break - 1H trend is bearish")
            else:
                state.state = "RESISTANCE_BROKEN"
                state.breakout_level = recent_resistance
                state.breakout_time = datetime.now()
                state.breakout_high = high  # Track for Fibonacci
                logger.info(f"{symbol}: Resistance broken at {recent_resistance:.2f}")
        
        elif not np.isnan(recent_support) and close < recent_support:
            # Higher timeframe alignment check
            if settings.use_higher_tf and state.higher_tf_trend == "BULLISH":
                logger.info(f"{symbol}: Ignoring support break - 1H trend is bullish")
            else:
                state.state = "SUPPORT_BROKEN"
                state.breakout_level = recent_support
                state.breakout_time = datetime.now()
                state.breakout_low = low  # Track for Fibonacci
                logger.info(f"{symbol}: Support broken at {recent_support:.2f}")
    
    elif state.state == "RESISTANCE_BROKEN":
        # Update swing high
        if high > state.breakout_high:
            state.breakout_high = high
        
        # Look for higher low formation with enhancements
        if detect_enhanced_higher_low(df, state, settings):
            state.state = "HL_FORMED"
            state.pullback_time = datetime.now()
            state.confirmation_count = 0
            logger.info(f"{symbol}: Higher low formed at {state.pullback_extreme:.2f}")
    
    elif state.state == "SUPPORT_BROKEN":
        # Update swing low
        if low < state.breakout_low:
            state.breakout_low = low
        
        # Look for lower high formation with enhancements
        if detect_enhanced_lower_high(df, state, settings):
            state.state = "LH_FORMED"
            state.pullback_time = datetime.now()
            state.confirmation_count = 0
            logger.info(f"{symbol}: Lower high formed at {state.pullback_extreme:.2f}")
    
    elif state.state == "HL_FORMED":
        # Wait for confirmation candles
        if df["close"].iloc[-1] > df["open"].iloc[-1]:  # Bullish candle
            state.confirmation_count += 1
            
        if state.confirmation_count >= state.adjusted_confirmation:
            # Generate long signal
            entry = close
            sl = state.pullback_extreme - (current_atr * settings.sl_buf_atr)
            tp = entry + ((entry - sl) * settings.rr)
            
            # Final checks
            vol_check = True
            if settings.use_vol:
                avg_vol = df["volume"].rolling(settings.vol_len).mean().iloc[-1]
                vol_check = df["volume"].iloc[-1] > avg_vol * settings.vol_mult
            
            if vol_check:
                # Calculate Fibonacci info for metadata
                retrace_pct = 0.0
                if state.breakout_high > state.breakout_level:
                    retrace_pct = (state.breakout_high - state.pullback_extreme) / (state.breakout_high - state.breakout_level)
                
                signals.append(Signal(
                    side="long",
                    entry=entry,
                    sl=sl,
                    tp=tp,
                    reason=f"Pullback long: HL above old R {state.breakout_level:.2f}",
                    meta={
                        "breakout_level": state.breakout_level,
                        "pullback_low": state.pullback_extreme,
                        "confirmation_candles": state.confirmation_count,
                        "higher_tf_trend": state.higher_tf_trend,
                        "volatility": state.volatility_regime,
                        "fib_retracement": f"{retrace_pct:.1%}"
                    }
                ))
                state.state = "SIGNAL_SENT"
                state.last_signal_time = datetime.now()
                logger.info(f"{symbol}: Long signal generated | 1H: {state.higher_tf_trend} | Vol: {state.volatility_regime} | Fib: {retrace_pct:.1%}")
    
    elif state.state == "LH_FORMED":
        # Wait for confirmation candles
        if df["close"].iloc[-1] < df["open"].iloc[-1]:  # Bearish candle
            state.confirmation_count += 1
            
        if state.confirmation_count >= state.adjusted_confirmation:
            # Generate short signal
            entry = close
            sl = state.pullback_extreme + (current_atr * settings.sl_buf_atr)
            tp = entry - ((sl - entry) * settings.rr)
            
            # Final checks
            vol_check = True
            if settings.use_vol:
                avg_vol = df["volume"].rolling(settings.vol_len).mean().iloc[-1]
                vol_check = df["volume"].iloc[-1] > avg_vol * settings.vol_mult
            
            if vol_check:
                # Calculate Fibonacci info
                retrace_pct = 0.0
                if state.breakout_level > state.breakout_low:
                    retrace_pct = (state.pullback_extreme - state.breakout_low) / (state.breakout_level - state.breakout_low)
                
                signals.append(Signal(
                    side="short",
                    entry=entry,
                    sl=sl,
                    tp=tp,
                    reason=f"Pullback short: LH below old S {state.breakout_level:.2f}",
                    meta={
                        "breakout_level": state.breakout_level,
                        "pullback_high": state.pullback_extreme,
                        "confirmation_candles": state.confirmation_count,
                        "higher_tf_trend": state.higher_tf_trend,
                        "volatility": state.volatility_regime,
                        "fib_retracement": f"{retrace_pct:.1%}"
                    }
                ))
                state.state = "SIGNAL_SENT"
                state.last_signal_time = datetime.now()
                logger.info(f"{symbol}: Short signal generated | 1H: {state.higher_tf_trend} | Vol: {state.volatility_regime} | Fib: {retrace_pct:.1%}")
    
    elif state.state == "SIGNAL_SENT":
        # Reset after some time or if price moves significantly
        if state.last_signal_time:
            time_since_signal = (datetime.now() - state.last_signal_time).seconds / 3600
            if time_since_signal > 4:  # Reset after 4 hours
                state.state = "NEUTRAL"
                logger.info(f"{symbol}: State reset to NEUTRAL after timeout")
    
    return signals

def reset_symbol_state(symbol: str):
    """Reset state for a symbol (call when position closes)"""
    if symbol in breakout_states:
        breakout_states[symbol] = BreakoutState()
        logger.info(f"{symbol}: State reset to NEUTRAL")