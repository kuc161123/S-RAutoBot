"""
Enhanced Strategy with Dynamic R:R Support
This is an example showing how to integrate dynamic R:R without breaking existing code
It extends the existing strategy.py with minimal changes
"""
from dataclasses import dataclass
import numpy as np
import pandas as pd
import logging

# Import existing strategy components
from strategy import (
    Settings as BaseSettings, 
    Signal, 
    _pivot_high, 
    _pivot_low, 
    _atr, 
    _ema,
    detect_signal as base_detect_signal
)

# Import new dynamic RR calculator
from dynamic_rr_calculator import get_dynamic_rr_calculator

logger = logging.getLogger(__name__)

@dataclass
class EnhancedSettings(BaseSettings):
    """Extended settings with dynamic RR support"""
    use_dynamic_rr: bool = False
    dynamic_rr_min: float = 1.5
    dynamic_rr_max: float = 4.0

def detect_signal_enhanced(df: pd.DataFrame, s: EnhancedSettings, symbol: str = "") -> Signal | None:
    """
    Enhanced signal detection with dynamic R:R
    Falls back to base implementation if dynamic RR is disabled
    """
    # First, check if we should use dynamic R:R
    if not hasattr(s, 'use_dynamic_rr') or not s.use_dynamic_rr:
        # Use original implementation
        return base_detect_signal(df, s, symbol)
    
    # Otherwise, implement enhanced logic
    # (Most of this is copied from original strategy.py to maintain consistency)
    
    # Require 200 candles minimum for reliable S/R detection
    min_candles = 200
    if len(df) < min_candles:
        return None
    
    high, low, close, vol = df["high"], df["low"], df["close"], df["volume"]
    
    # Calculate pivots (same as original)
    ph = pd.Series(_pivot_high(high, s.left, s.right), index=df.index)
    pl = pd.Series(_pivot_low(low, s.left, s.right), index=df.index)
    
    dh = ph.dropna()
    dl = pl.dropna()
    if len(dh) < 2 or len(dl) < 2:
        return None
    
    lastHigh, prevHigh = float(dh.iloc[-1]), float(dh.iloc[-2])
    lastLow, prevLow = float(dl.iloc[-1]), float(dl.iloc[-2])
    
    # Trend detection (same as original)
    trendUp = (lastHigh > prevHigh) and (lastLow > prevLow)
    trendDn = (lastHigh < prevHigh) and (lastLow < prevLow)
    
    nearestRes, nearestSup = lastHigh, lastLow
    
    atr = float(_atr(df, s.atr_len)[-1])
    
    # EMA filter (same as original)
    ema_ok_long = True
    ema_ok_short = True
    if s.use_ema and len(df) >= s.ema_len:
        ema_val = float(_ema(close, s.ema_len)[-1])
        ema_ok_long = close.iloc[-1] > ema_val
        ema_ok_short = close.iloc[-1] < ema_val
    
    # Volume filter (same as original)
    vol_ok = True
    if s.use_vol:
        vol_ok = vol.iloc[-1] > vol.rolling(s.vol_len).mean().iloc[-1] * s.vol_mult
    
    c = float(close.iloc[-1])
    crossRes = (c > nearestRes)
    crossSup = (c < nearestSup)
    
    # Prepare features for dynamic R:R calculation
    features = {
        'trend_strength': 50,  # Placeholder - would calculate properly
        'higher_tf_alignment': 50,  # Placeholder
        'atr_percentile': 50,  # Placeholder
        'volatility_regime': 'normal',  # Placeholder
        'session': 'us',  # Placeholder
        'hour_of_day': pd.Timestamp.now().hour,
        'day_of_week': pd.Timestamp.now().dayofweek
    }
    
    # Calculate volatility regime (simplified)
    if len(df) >= 100:
        atr_series = pd.Series([_atr(df.iloc[i-14:i], 14)[-1] 
                               for i in range(14, len(df))])
        current_atr_percentile = (atr < atr_series).mean() * 100
        features['atr_percentile'] = current_atr_percentile
        
        if current_atr_percentile < 30:
            features['volatility_regime'] = 'low'
        elif current_atr_percentile > 70:
            features['volatility_regime'] = 'high'
    
    # Get dynamic R:R
    rr_calculator = get_dynamic_rr_calculator(base_rr=s.rr, enabled=s.use_dynamic_rr)
    dynamic_rr = rr_calculator.calculate_rr(features, symbol)
    
    # Long signal
    if trendUp and crossRes and vol_ok and ema_ok_long:
        entry = c
        sl = nearestSup - s.sl_buf_atr * atr
        if entry <= sl:
            logger.info(f"[{symbol}] Long signal rejected - invalid SL placement")
            return None
        
        R = entry - sl
        # Use dynamic R:R instead of fixed
        fee_adjustment = 1.00165  # Same as original
        tp = entry + (dynamic_rr * R * fee_adjustment)
        
        logger.info(f"[{symbol}] 🟢 LONG SIGNAL - Entry: {entry:.4f}, SL: {sl:.4f}, "
                   f"TP: {tp:.4f}, R:R = 1:{dynamic_rr:.2f} (base: {s.rr})")
        
        return Signal("long", entry, sl, tp, 
                     f"Up-structure breakout over resistance (RR: {dynamic_rr:.2f})",
                     {"atr": atr, "res": nearestRes, "sup": nearestSup, 
                      "dynamic_rr": dynamic_rr})
    
    # Short signal
    if trendDn and crossSup and vol_ok and ema_ok_short:
        entry = c
        sl = nearestRes + s.sl_buf_atr * atr
        if sl <= entry:
            logger.info(f"[{symbol}] Short signal rejected - invalid SL placement")
            return None
        
        R = sl - entry
        # Use dynamic R:R instead of fixed
        fee_adjustment = 1.00165  # Same as original
        tp = entry - (dynamic_rr * R * fee_adjustment)
        
        logger.info(f"[{symbol}] 🔴 SHORT SIGNAL - Entry: {entry:.4f}, SL: {sl:.4f}, "
                   f"TP: {tp:.4f}, R:R = 1:{dynamic_rr:.2f} (base: {s.rr})")
        
        return Signal("short", entry, sl, tp, 
                     f"Down-structure breakdown under support (RR: {dynamic_rr:.2f})",
                     {"atr": atr, "res": nearestRes, "sup": nearestSup, 
                      "dynamic_rr": dynamic_rr})
    
    return None

# Wrapper function to maintain compatibility
def detect_signal(df: pd.DataFrame, s: Settings, symbol: str = "") -> Signal | None:
    """
    Wrapper that routes to enhanced or base detection
    Maintains full backward compatibility
    """
    # Check if settings have dynamic RR enabled
    if hasattr(s, 'use_dynamic_rr') and s.use_dynamic_rr:
        return detect_signal_enhanced(df, s, symbol)
    else:
        return base_detect_signal(df, s, symbol)