from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Dict
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
    confirmation_candles:int=2  # Number of confirmation candles required

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
    state:str = "NEUTRAL"  # NEUTRAL, RESISTANCE_BROKEN, SUPPORT_BROKEN, HL_FORMED, LH_FORMED, SIGNAL_SENT
    breakout_level:float = 0.0  # The S/R level that was broken
    breakout_time:Optional[datetime] = None
    pullback_extreme:float = 0.0  # Lowest point in pullback for long, highest for short
    pullback_time:Optional[datetime] = None
    confirmation_count:int = 0
    last_signal_time:Optional[datetime] = None
    last_resistance:float = 0.0  # Track the resistance level
    last_support:float = 0.0  # Track the support level

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

def detect_higher_low(df:pd.DataFrame, above_level:float, lookback:int=10) -> bool:
    """
    Detect if price has made a higher low above a certain level.
    Used after resistance break to confirm pullback completion.
    """
    recent_lows = df["low"].iloc[-lookback:]
    
    # Find the lowest point in recent candles
    min_idx = recent_lows.idxmin()
    min_low = recent_lows.loc[min_idx]
    
    # Check if we've made a higher low (price bounced)
    latest_low = df["low"].iloc[-1]
    prev_low = df["low"].iloc[-2]
    
    # Higher low conditions:
    # 1. The minimum low is above the breakout level
    # 2. Current price is moving up from the minimum
    # 3. We have at least 2 candles since the minimum
    if min_low > above_level:
        min_pos = recent_lows.index.get_loc(min_idx)
        current_pos = len(recent_lows) - 1
        if current_pos - min_pos >= 2:  # At least 2 candles since the low
            if df["close"].iloc[-1] > min_low and df["close"].iloc[-2] > min_low:
                return True
    
    return False

def detect_lower_high(df:pd.DataFrame, below_level:float, lookback:int=10) -> bool:
    """
    Detect if price has made a lower high below a certain level.
    Used after support break to confirm pullback completion.
    """
    recent_highs = df["high"].iloc[-lookback:]
    
    # Find the highest point in recent candles
    max_idx = recent_highs.idxmax()
    max_high = recent_highs.loc[max_idx]
    
    # Check if we've made a lower high (price rejected)
    latest_high = df["high"].iloc[-1]
    prev_high = df["high"].iloc[-2]
    
    # Lower high conditions:
    # 1. The maximum high is below the breakout level
    # 2. Current price is moving down from the maximum
    # 3. We have at least 2 candles since the maximum
    if max_high < below_level:
        max_pos = recent_highs.index.get_loc(max_idx)
        current_pos = len(recent_highs) - 1
        if current_pos - max_pos >= 2:  # At least 2 candles since the high
            if df["close"].iloc[-1] < max_high and df["close"].iloc[-2] < max_high:
                return True
    
    return False

def count_confirmation_candles(df:pd.DataFrame, direction:str, count_needed:int=2) -> int:
    """
    Count consecutive confirmation candles in the specified direction.
    For long: bullish candles (close > open)
    For short: bearish candles (close < open)
    """
    confirmation_count = 0
    
    # Check the last 'count_needed' candles
    for i in range(-count_needed, 0):
        if i >= -len(df):
            candle_open = df["open"].iloc[i]
            candle_close = df["close"].iloc[i]
            
            if direction == "long" and candle_close > candle_open:
                confirmation_count += 1
            elif direction == "short" and candle_close < candle_open:
                confirmation_count += 1
            else:
                # Reset count if we get opposite candle
                confirmation_count = 0
                break
    
    return confirmation_count

def detect_signal_pullback(df:pd.DataFrame, s:Settings, symbol:str="") -> Signal|None:
    """
    Enhanced signal detection with pullback strategy:
    1. Wait for S/R breakout
    2. Wait for pullback (HL for long, LH for short)
    3. Wait for confirmation candles
    4. Then generate signal
    """
    # Initialize state for new symbols
    if symbol not in breakout_states:
        breakout_states[symbol] = BreakoutState()
    
    state = breakout_states[symbol]
    
    # Require minimum candles for reliable S/R detection
    min_candles = 200
    if len(df) < min_candles:
        return None
    
    # Calculate current S/R levels
    high, low, close, vol = df["high"], df["low"], df["close"], df["volume"]
    
    ph = pd.Series(_pivot_high(high, s.left, s.right), index=df.index)
    pl = pd.Series(_pivot_low(low, s.left, s.right), index=df.index)
    
    dh = ph.dropna()
    dl = pl.dropna()
    if len(dh) < 2 or len(dl) < 2:
        return None
    
    lastHigh, prevHigh = float(dh.iloc[-1]), float(dh.iloc[-2])
    lastLow, prevLow = float(dl.iloc[-1]), float(dl.iloc[-2])
    
    # Determine market structure
    trendUp = (lastHigh > prevHigh) and (lastLow > prevLow)
    trendDn = (lastHigh < prevHigh) and (lastLow < prevLow)
    
    nearestRes, nearestSup = lastHigh, lastLow
    
    # Update tracked S/R levels
    state.last_resistance = nearestRes
    state.last_support = nearestSup
    
    # Calculate ATR for stop loss
    atr = float(_atr(df, s.atr_len)[-1])
    
    # EMA filter
    ema_ok_long = True
    ema_ok_short = True
    if s.use_ema and len(df) >= s.ema_len:
        ema_val = float(_ema(close, s.ema_len)[-1])
        ema_ok_long = close.iloc[-1] > ema_val
        ema_ok_short = close.iloc[-1] < ema_val
    
    # Volume filter
    vol_ok = True
    if s.use_vol:
        vol_ok = vol.iloc[-1] > vol.rolling(s.vol_len).mean().iloc[-1] * s.vol_mult
    
    c = float(close.iloc[-1])
    current_time = df.index[-1]
    
    # State machine for pullback strategy
    if state.state == "NEUTRAL":
        # Check for initial breakout
        if trendUp and c > nearestRes and vol_ok and ema_ok_long:
            # Resistance broken - wait for pullback
            state.state = "RESISTANCE_BROKEN"
            state.breakout_level = nearestRes
            state.breakout_time = current_time
            state.confirmation_count = 0
            logger.info(f"[{symbol}] Resistance broken at {nearestRes:.4f}, waiting for pullback and HL")
            
        elif trendDn and c < nearestSup and vol_ok and ema_ok_short:
            # Support broken - wait for pullback
            state.state = "SUPPORT_BROKEN"
            state.breakout_level = nearestSup
            state.breakout_time = current_time
            state.confirmation_count = 0
            logger.info(f"[{symbol}] Support broken at {nearestSup:.4f}, waiting for pullback and LH")
    
    elif state.state == "RESISTANCE_BROKEN":
        # Wait for higher low above old resistance
        if detect_higher_low(df, state.breakout_level):
            state.state = "HL_FORMED"
            state.pullback_extreme = df["low"].iloc[-10:].min()  # Record the pullback low
            state.pullback_time = current_time
            logger.info(f"[{symbol}] Higher Low formed above {state.breakout_level:.4f}, waiting for confirmation")
        
        # Reset if price falls back below breakout level
        elif c < state.breakout_level:
            logger.info(f"[{symbol}] Price fell below breakout level, resetting to neutral")
            state.state = "NEUTRAL"
            state.confirmation_count = 0
    
    elif state.state == "SUPPORT_BROKEN":
        # Wait for lower high below old support
        if detect_lower_high(df, state.breakout_level):
            state.state = "LH_FORMED"
            state.pullback_extreme = df["high"].iloc[-10:].max()  # Record the pullback high
            state.pullback_time = current_time
            logger.info(f"[{symbol}] Lower High formed below {state.breakout_level:.4f}, waiting for confirmation")
        
        # Reset if price rises back above breakout level
        elif c > state.breakout_level:
            logger.info(f"[{symbol}] Price rose above breakout level, resetting to neutral")
            state.state = "NEUTRAL"
            state.confirmation_count = 0
    
    elif state.state == "HL_FORMED":
        # Count confirmation candles for long
        confirmations = count_confirmation_candles(df, "long", s.confirmation_candles)
        
        if confirmations >= s.confirmation_candles:
            # Generate LONG signal
            entry = c
            sl = state.pullback_extreme - s.sl_buf_atr * atr  # SL below the pullback low
            
            if entry <= sl:
                logger.info(f"[{symbol}] Long signal rejected - invalid SL placement")
                state.state = "NEUTRAL"
                return None
            
            R = entry - sl
            tp = entry + s.rr * R
            
            state.state = "SIGNAL_SENT"
            state.last_signal_time = current_time
            
            logger.info(f"[{symbol}] 🟢 LONG SIGNAL (Pullback) - Entry: {entry:.4f}, SL: {sl:.4f}, TP: {tp:.4f}")
            
            return Signal("long", entry, sl, tp, 
                         f"Pullback long: HL above {state.breakout_level:.4f} + {s.confirmation_candles} confirmations",
                         {"atr": atr, "res": nearestRes, "sup": nearestSup, 
                          "breakout_level": state.breakout_level, "pullback_low": state.pullback_extreme})
        
        # Reset if price breaks below pullback low
        elif df["low"].iloc[-1] < state.pullback_extreme:
            logger.info(f"[{symbol}] Pullback low broken, resetting to neutral")
            state.state = "NEUTRAL"
            state.confirmation_count = 0
    
    elif state.state == "LH_FORMED":
        # Count confirmation candles for short
        confirmations = count_confirmation_candles(df, "short", s.confirmation_candles)
        
        if confirmations >= s.confirmation_candles:
            # Generate SHORT signal
            entry = c
            sl = state.pullback_extreme + s.sl_buf_atr * atr  # SL above the pullback high
            
            if sl <= entry:
                logger.info(f"[{symbol}] Short signal rejected - invalid SL placement")
                state.state = "NEUTRAL"
                return None
            
            R = sl - entry
            tp = entry - s.rr * R
            
            state.state = "SIGNAL_SENT"
            state.last_signal_time = current_time
            
            logger.info(f"[{symbol}] 🔴 SHORT SIGNAL (Pullback) - Entry: {entry:.4f}, SL: {sl:.4f}, TP: {tp:.4f}")
            
            return Signal("short", entry, sl, tp,
                         f"Pullback short: LH below {state.breakout_level:.4f} + {s.confirmation_candles} confirmations",
                         {"atr": atr, "res": nearestRes, "sup": nearestSup,
                          "breakout_level": state.breakout_level, "pullback_high": state.pullback_extreme})
        
        # Reset if price breaks above pullback high
        elif df["high"].iloc[-1] > state.pullback_extreme:
            logger.info(f"[{symbol}] Pullback high broken, resetting to neutral")
            state.state = "NEUTRAL"
            state.confirmation_count = 0
    
    elif state.state == "SIGNAL_SENT":
        # Don't send another signal until state is reset
        # State will be reset when position closes (handled in live_bot.py)
        pass
    
    return None

def reset_symbol_state(symbol:str):
    """Reset the breakout state for a symbol (called when position closes)"""
    if symbol in breakout_states:
        logger.info(f"[{symbol}] Resetting breakout state to NEUTRAL")
        breakout_states[symbol] = BreakoutState()