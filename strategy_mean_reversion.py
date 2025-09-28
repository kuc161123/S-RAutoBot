"""
Mean Reversion Trading Strategy
Designed for ranging markets.

- Buys at strong support levels.
- Sells at strong resistance levels.
- Assumes the price will revert to the mean.
"""
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional, Tuple
from datetime import datetime
import pandas as pd
import numpy as np
import logging

# Import base classes and utilities from existing strategy
from strategy_pullback import Settings, Signal, _pivot_high, _pivot_low, _atr

@dataclass
class BreakoutState:
    """Track the state of a breakout for each symbol (simplified for mean reversion)"""
    state:str = "NEUTRAL" # NEUTRAL, SIGNAL_SENT
    last_signal_candle_time: Optional['datetime.datetime'] = None
    # Add other relevant state variables if needed for more complex mean reversion

logger = logging.getLogger(__name__)

def detect_signal(df: pd.DataFrame, s: Settings, symbol: str = "") -> Optional[Signal]:
    """
    Detects mean-reversion signals in a ranging market.

    Args:
        df (pd.DataFrame): Price data.
        s (Settings): Strategy settings.
        symbol (str): The trading symbol.

    Returns:
        Optional[Signal]: A trading signal if conditions are met.
    """
    # Initialize state for new symbols
    if symbol not in mean_reversion_states:
        mean_reversion_states[symbol] = BreakoutState()
    
    state = mean_reversion_states[symbol]

    # Cooldown logic to prevent multiple signals in quick succession
    if state.last_signal_candle_time:
        time_since_last_signal = df.index[-1] - state.last_signal_candle_time
        # Assuming 15-minute candles, convert min_candles_between_signals to timedelta
        cooldown_duration = pd.Timedelta(minutes=s.min_candles_between_signals * 15)
        if time_since_last_signal < cooldown_duration:
            return None # Still in cooldown period

    min_candles = 100
    if len(df) < min_candles:
        return None

    high, low, close = df["high"], df["low"], df["close"]

    # Use more sensitive pivots for range detection
    ph = pd.Series(_pivot_high(high, 5, 5), index=df.index).dropna()
    pl = pd.Series(_pivot_low(low, 5, 5), index=df.index).dropna()

    if len(ph) < 2 or len(dl) < 2:
        return None

    # Identify the range
    upper_range = ph.iloc[-2:].mean() # Average of last 2 pivot highs
    lower_range = pl.iloc[-2:].mean() # Average of last 2 pivot lows

    if upper_range <= lower_range:
        return None # Invalid range

    current_price = close.iloc[-1]
    current_atr = _atr(df, s.atr_len)[-1]

    # --- LONG SIGNAL (Buy at Support) ---
    # Condition: Price touches the lower part of the range and shows reversal
    if abs(current_price - lower_range) < (current_atr * 0.5):
        # Reversal confirmation: Bullish candle (close > open)
        if close.iloc[-1] > df["open"].iloc[-1]:
            entry = current_price
            sl = lower_range - (current_atr * s.sl_buf_atr)
            tp = upper_range # Target the top of the range

            if entry > sl and tp > entry:
                logger.info(f"[{symbol}] MEAN REVERSION LONG: Bouncing off support {lower_range:.4f}")
                state.last_signal_candle_time = df.index[-1]
                return Signal(
                    side="long",
                    entry=entry,
                    sl=sl,
                    tp=tp,
                    reason=f"Mean Reversion: Bounce from support @ {lower_range:.2f}",
                    meta={"range_upper": upper_range, "range_lower": lower_range}
                )

    # --- SHORT SIGNAL (Sell at Resistance) ---
    # Condition: Price touches the upper part of the range and shows reversal
    if abs(current_price - upper_range) < (current_atr * 0.5):
        # Reversal confirmation: Bearish candle (close < open)
        if close.iloc[-1] < df["open"].iloc[-1]:
            entry = current_price
            sl = upper_range + (current_atr * s.sl_buf_atr)
            tp = lower_range # Target the bottom of the range

            if sl > entry and entry > tp:
                logger.info(f"[{symbol}] MEAN REVERSION SHORT: Rejecting from resistance {upper_range:.4f}")
                state.last_signal_candle_time = df.index[-1]
                return Signal(
                    side="short",
                    entry=entry,
                    sl=sl,
                    tp=tp,
                    reason=f"Mean Reversion: Rejection from resistance @ {upper_range:.2f}",
                    meta={"range_upper": upper_range, "range_lower": lower_range}
                )
    return None

# Global state tracking for each symbol
mean_reversion_states: Dict[str, BreakoutState] = {}

def reset_symbol_state(symbol: str):
    """Reset the state for a symbol (called when position closes or before backtest)."""
    if symbol in mean_reversion_states:
        del mean_reversion_states[symbol]
        logger.debug(f"[{symbol}] Mean Reversion state reset.")
