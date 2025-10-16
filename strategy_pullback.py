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
    rr:float=2.5
    use_ema:bool=False
    ema_len:int=200
    use_vol:bool=False
    vol_len:int=20
    vol_mult:float=1.2
    both_hit_rule:str="SL_FIRST"
    confirmation_candles:int=2  # Number of confirmation candles required
    use_mtf_sr:bool=True  # Enable multi-timeframe S/R
    mtf_weight:float=2.0  # Prefer major levels 2x over minor
    mtf_min_strength:float=3.0  # Minimum strength for major levels
    min_candles_between_signals:int=5 # Minimum number of candles between signals for the same symbol
    # Extra breathing room when a pivot-based stop is selected
    # Expressed as fraction of entry price (e.g., 0.01 = 1%)
    extra_pivot_breath_pct: float = 0.01

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
    last_signal_candle_time: Optional[datetime] = None
    last_resistance:float = 0.0  # Track the resistance level
    last_support:float = 0.0  # Track the support level
    previous_pivot_low:float = 0.0  # Track previous pivot low for structure-based stops
    previous_pivot_high:float = 0.0  # Track previous pivot high for structure-based stops

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

def find_previous_pivot_low(df:pd.DataFrame, current_idx:int, left:int=2, right:int=2, lookback:int=50) -> float:
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

def find_previous_pivot_high(df:pd.DataFrame, current_idx:int, left:int=2, right:int=2, lookback:int=50) -> float:
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

def detect_signal_pullback(df:pd.DataFrame, s:Settings, symbol:str="") -> Optional[Signal]:
    """
    Enhanced signal detection with pullback strategy:
    1. Wait for S/R breakout
    2. Wait for pullback (HL for long, LH for short)
    3. Wait for confirmation candles
    4. Then generate signal
    """
    # Import MTF module if enabled
    if s.use_mtf_sr:
        try:
            from multi_timeframe_sr import mtf_sr, should_use_mtf_level
        except ImportError:
            logger.warning("MTF S/R module not available, using standard pivots")
            s.use_mtf_sr = False
    
    # Initialize state for new symbols
    if symbol not in breakout_states:
        breakout_states[symbol] = BreakoutState()
    
    state = breakout_states[symbol]

    # Utility: compact preconditions logger
    def _log_preconditions(reason: str = ""):  # reason: optional context tag
        try:
            # Basic distances to S/R in ATR units (positive means beyond the level)
            dist_res_atr = (c - float(state.last_resistance)) / max(1e-9, atr) if state.last_resistance else 0.0
            dist_sup_atr = (float(state.last_support) - c) / max(1e-9, atr) if state.last_support else 0.0
            # Confirmation progress based on current state
            conf_need = int(s.confirmation_candles)
            conf_have = 0
            st = state.state
            if st in ("HL_FORMED", "RESISTANCE_BROKEN"):
                conf_have = count_confirmation_candles(df, "long", conf_need)
            elif st in ("LH_FORMED", "SUPPORT_BROKEN"):
                conf_have = count_confirmation_candles(df, "short", conf_need)
            # Trend structure snapshot (best-effort from last pivots when available)
            # Recompute minimally to avoid holding external state
            try:
                ph2 = pd.Series(_pivot_high(df["high"], s.left, s.right)).dropna()
                pl2 = pd.Series(_pivot_low(df["low"], s.left, s.right)).dropna()
                trend_up = bool(len(ph2) >= 2 and len(pl2) >= 2 and float(ph2.iloc[-1]) > float(ph2.iloc[-2]) and float(pl2.iloc[-1]) > float(pl2.iloc[-2]))
                trend_dn = bool(len(ph2) >= 2 and len(pl2) >= 2 and float(ph2.iloc[-1]) < float(ph2.iloc[-2]) and float(pl2.iloc[-1]) < float(pl2.iloc[-2]))
            except Exception:
                trend_up = trend_dn = False
            logger.info(
                f"[{symbol}] Preconditions: state={state.state} res={state.last_resistance:.4f} sup={state.last_support:.4f} "
                f"dResATR={dist_res_atr:.2f} dSupATR={dist_sup_atr:.2f} conf={conf_have}/{conf_need} "
                f"trendUp={trend_up} trendDn={trend_dn}{(' reason='+reason) if reason else ''}"
            )
        except Exception:
            pass

    # Cooldown logic to prevent multiple signals in quick succession
    if state.last_signal_candle_time:
        time_since_last_signal = df.index[-1] - state.last_signal_candle_time
        # Assuming 15-minute candles, convert min_candles_between_signals to timedelta
        cooldown_duration = pd.Timedelta(minutes=s.min_candles_between_signals * 15)
        if time_since_last_signal < cooldown_duration:
            _log_preconditions("cooldown")
            return None # Still in cooldown period
    
    # Require minimum candles for reliable S/R detection
    min_candles = 200
    if len(df) < min_candles:
        # Not enough history for stable S/R
        try:
            logger.info(f"[{symbol}] Preconditions: insufficient_history ({len(df)}/{min_candles})")
        except Exception:
            pass
        return None
    
    # Calculate current S/R levels
    high, low, close, vol = df["high"], df["low"], df["close"], df["volume"]
    
    ph = pd.Series(_pivot_high(high, s.left, s.right), index=df.index)
    pl = pd.Series(_pivot_low(low, s.left, s.right), index=df.index)
    
    dh = ph.dropna()
    dl = pl.dropna()
    if len(dh) < 2 or len(dl) < 2:
        _log_preconditions("sr_insufficient")
        return None
    
    lastHigh, prevHigh = float(dh.iloc[-1]), float(dh.iloc[-2])
    lastLow, prevLow = float(dl.iloc[-1]), float(dl.iloc[-2])
    
    # Determine market structure
    trendUp = (lastHigh > prevHigh) and (lastLow > prevLow)
    trendDn = (lastHigh < prevHigh) and (lastLow < prevLow)
    
    nearestRes, nearestSup = lastHigh, lastLow
    
    # Update tracked S/R levels (will be updated after MTF check)
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
    
    # Check for MTF levels if enabled (moved here for current_time)
    if s.use_mtf_sr:
        try:
            # Update MTF levels periodically based on configured interval
            if mtf_sr.should_update(symbol):
                mtf_sr.update_sr_levels(symbol, df)
                logger.debug(f"[{symbol}] Updated HTF S/R levels")
            
            # Check if we should use MTF levels
            original_resistance = nearestRes
            original_support = nearestSup
            
            use_mtf_res, mtf_res, res_reason = should_use_mtf_level(symbol, nearestRes, c, df)
            if use_mtf_res and mtf_res > 0:
                # Validate resistance is above current price
                if mtf_res > c:
                    logger.info(f"[{symbol}] Using MTF resistance: {mtf_res:.4f} ({res_reason})")
                    nearestRes = mtf_res
                else:
                    logger.warning(f"[{symbol}] MTF resistance {mtf_res:.4f} is below price {c:.4f}, keeping original")
            
            use_mtf_sup, mtf_sup, sup_reason = should_use_mtf_level(symbol, nearestSup, c, df)
            if use_mtf_sup and mtf_sup > 0:
                # Validate support is below current price
                if mtf_sup < c:
                    logger.info(f"[{symbol}] Using MTF support: {mtf_sup:.4f} ({sup_reason})")
                    nearestSup = mtf_sup
                else:
                    logger.warning(f"[{symbol}] MTF support {mtf_sup:.4f} is above price {c:.4f}, keeping original")
                
        except Exception as e:
            logger.debug(f"[{symbol}] MTF S/R check failed: {e}")
    
    # Update tracked S/R levels after MTF check
    state.last_resistance = nearestRes
    state.last_support = nearestSup
    
    # State machine for pullback strategy
    if state.state == "NEUTRAL":
        # Check for initial breakout
        if trendUp and c > nearestRes and vol_ok and ema_ok_long:
            # Resistance broken - wait for pullback
            state.state = "RESISTANCE_BROKEN"
            state.breakout_level = nearestRes
            state.breakout_time = current_time
            state.confirmation_count = 0
            # Find and store previous pivot low for structure-based stop
            current_idx = len(df) - 1
            state.previous_pivot_low = find_previous_pivot_low(df, current_idx)
            logger.info(f"[{symbol}] Resistance broken at {nearestRes:.4f}, waiting for pullback and HL")
            
        elif trendDn and c < nearestSup and vol_ok and ema_ok_short:
            # Support broken - wait for pullback
            state.state = "SUPPORT_BROKEN"
            state.breakout_level = nearestSup
            state.breakout_time = current_time
            state.confirmation_count = 0
            # Find and store previous pivot high for structure-based stop
            current_idx = len(df) - 1
            state.previous_pivot_high = find_previous_pivot_high(df, current_idx)
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
        # Verbose per-candle confirmation diagnostics
        try:
            if confirmations > 0 and confirmations < s.confirmation_candles:
                o = float(df["open"].iloc[-1]); cl = float(df["close"].iloc[-1])
                hi = float(df["high"].iloc[-1]); lo = float(df["low"].iloc[-1])
                body = abs(cl - o); rng = max(1e-9, hi - lo)
                body_ratio = body / rng * 100.0
                try:
                    atr_recent = float(_atr(df, s.atr_len)[-1])
                except Exception:
                    atr_recent = rng
                candle_range_atr = (rng / atr_recent * 100.0) if atr_recent > 0 else 0.0
                try:
                    vol_ratio = float(df['volume'].iloc[-1] / df['volume'].rolling(20).mean().iloc[-1])
                except Exception:
                    vol_ratio = 1.0
                logger.info(f"[{symbol}] HL confirmation {confirmations}/{s.confirmation_candles} | Body {body_ratio:.1f}% | Range/ATR {candle_range_atr:.1f}% | Vol× {vol_ratio:.2f}")
        except Exception:
            pass
        
        if confirmations >= s.confirmation_candles:
            # Generate LONG signal
            entry = c
            
            # HYBRID STOP LOSS METHOD - use whichever gives more room
            # Option 1: Previous pivot low
            sl_option1 = state.previous_pivot_low - (s.sl_buf_atr * 0.3 * atr)  # Smaller buffer since further away
            
            # Option 2: Breakout level minus ATR
            sl_option2 = state.breakout_level - (s.sl_buf_atr * 1.6 * atr)  # Larger buffer from breakout
            
            # Option 3: Original method (pullback extreme)
            sl_option3 = state.pullback_extreme - (s.sl_buf_atr * atr)
            
            # Use the lowest SL (gives most room)
            sl = min(sl_option1, sl_option2, sl_option3)

            # Breathing room: apply to any chosen stop method (move SL further by +extra% of entry)
            if s.extra_pivot_breath_pct > 0:
                old_sl = sl
                sl = float(sl) - float(entry) * float(s.extra_pivot_breath_pct)
                logger.info(f"[{symbol}] Stop breathing: -{s.extra_pivot_breath_pct*100:.1f}% of entry ({old_sl:.4f} -> {sl:.4f})")
            
            # Ensure minimum stop distance (at least 1% from entry)
            min_stop_distance = entry * 0.01
            if abs(entry - sl) < min_stop_distance:
                sl = entry - min_stop_distance
                logger.info(f"[{symbol}] Adjusted stop to minimum distance (1% from entry)")
            
            # Log which method was used
            if sl == sl_option1:
                logger.info(f"[{symbol}] Using previous pivot low for stop: {state.previous_pivot_low:.4f}")
            elif sl == sl_option2:
                logger.info(f"[{symbol}] Using breakout level for stop: {state.breakout_level:.4f}")
            else:
                logger.info(f"[{symbol}] Using pullback extreme for stop: {state.pullback_extreme:.4f}")
            
            if entry <= sl:
                logger.info(f"[{symbol}] Long signal rejected - invalid SL placement")
                state.state = "NEUTRAL"
                return None
            
            R = entry - sl
            # Account for fees and slippage in TP calculation
            # Bybit fees: 0.06% entry + 0.055% exit (limit) = 0.115% total
            # Add 0.05% for slippage = 0.165% total cost
            # To get 2.5:1 after fees, we need to target slightly higher
            fee_adjustment = 1.00165  # Compensate for 0.165% total costs
            tp = entry + (s.rr * R * fee_adjustment)
            
            state.state = "SIGNAL_SENT"
            state.last_signal_time = current_time
            state.last_signal_candle_time = df.index[-1]
            
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
        # Verbose per-candle confirmation diagnostics
        try:
            if confirmations > 0 and confirmations < s.confirmation_candles:
                o = float(df["open"].iloc[-1]); cl = float(df["close"].iloc[-1])
                hi = float(df["high"].iloc[-1]); lo = float(df["low"].iloc[-1])
                body = abs(cl - o); rng = max(1e-9, hi - lo)
                body_ratio = body / rng * 100.0
                try:
                    atr_recent = float(_atr(df, s.atr_len)[-1])
                except Exception:
                    atr_recent = rng
                candle_range_atr = (rng / atr_recent * 100.0) if atr_recent > 0 else 0.0
                try:
                    vol_ratio = float(df['volume'].iloc[-1] / df['volume'].rolling(20).mean().iloc[-1])
                except Exception:
                    vol_ratio = 1.0
                logger.info(f"[{symbol}] LH confirmation {confirmations}/{s.confirmation_candles} | Body {body_ratio:.1f}% | Range/ATR {candle_range_atr:.1f}% | Vol× {vol_ratio:.2f}")
        except Exception:
            pass
        # Log each confirmation candle as it accrues (until signal fires)
        try:
            if confirmations > 0 and confirmations < s.confirmation_candles:
                logger.info(f"[{symbol}] LH confirmation {confirmations}/{s.confirmation_candles}")
        except Exception:
            pass
        
        if confirmations >= s.confirmation_candles:
            # Generate SHORT signal
            entry = c
            
            # HYBRID STOP LOSS METHOD - use whichever gives more room
            # Option 1: Previous pivot high
            sl_option1 = state.previous_pivot_high + (s.sl_buf_atr * 0.3 * atr)  # Smaller buffer since further away
            
            # Option 2: Breakout level plus ATR
            sl_option2 = state.breakout_level + (s.sl_buf_atr * 1.6 * atr)  # Larger buffer from breakout
            
            # Option 3: Original method (pullback extreme)
            sl_option3 = state.pullback_extreme + (s.sl_buf_atr * atr)
            
            # Use the highest SL (gives most room)
            sl = max(sl_option1, sl_option2, sl_option3)

            # Breathing room: apply to any chosen stop method (move SL further by +extra% of entry)
            if s.extra_pivot_breath_pct > 0:
                old_sl = sl
                sl = float(sl) + float(entry) * float(s.extra_pivot_breath_pct)
                logger.info(f"[{symbol}] Stop breathing: +{s.extra_pivot_breath_pct*100:.1f}% of entry ({old_sl:.4f} -> {sl:.4f})")
            
            # Ensure minimum stop distance (at least 1% from entry)
            min_stop_distance = entry * 0.01
            if abs(sl - entry) < min_stop_distance:
                sl = entry + min_stop_distance
                logger.info(f"[{symbol}] Adjusted stop to minimum distance (1% from entry)")
            
            # Log which method was used
            if sl == sl_option1:
                logger.info(f"[{symbol}] Using previous pivot high for stop: {state.previous_pivot_high:.4f}")
            elif sl == sl_option2:
                logger.info(f"[{symbol}] Using breakout level for stop: {state.breakout_level:.4f}")
            else:
                logger.info(f"[{symbol}] Using pullback extreme for stop: {state.pullback_extreme:.4f}")
            
            if sl <= entry:
                logger.info(f"[{symbol}] Short signal rejected - invalid SL placement")
                state.state = "NEUTRAL"
                return None
            
            R = sl - entry
            # Account for fees and slippage in TP calculation
            # Bybit fees: 0.06% entry + 0.055% exit (limit) = 0.115% total
            # Add 0.05% for slippage = 0.165% total cost
            # To get 2.5:1 after fees, we need to target slightly higher
            fee_adjustment = 1.00165  # Compensate for 0.165% total costs
            tp = entry - (s.rr * R * fee_adjustment)
            
            state.state = "SIGNAL_SENT"
            state.last_signal_time = current_time
            state.last_signal_candle_time = df.index[-1]
            
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
    
    # No signal this bar: emit snapshot of current preconditions
    _log_preconditions()
    return None

def reset_symbol_state(symbol:str):
    """Reset the breakout state for a symbol (called when position closes)"""
    if symbol in breakout_states:
        logger.info(f"[{symbol}] Resetting breakout state to NEUTRAL")
        breakout_states[symbol] = BreakoutState()
