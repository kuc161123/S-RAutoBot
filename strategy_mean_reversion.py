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

    if len(ph) < 2 or len(pl) < 2:
        return None

    # Identify the range
    upper_range = ph.iloc[-2:].mean() # Average of last 2 pivot highs
    lower_range = pl.iloc[-2:].mean() # Average of last 2 pivot lows

    if upper_range <= lower_range:
        return None # Invalid range

    # Range width filter: Only trade ranges between 2.5% and 6% wide (REVERTED TO ORIGINAL)
    # This ensures fixed 2.5:1 R:R aligns well with natural range boundaries
    range_width = (upper_range - lower_range) / lower_range
    min_range_width = 0.025  # 2.5% minimum (original strict setting)
    max_range_width = 0.06   # 6.0% maximum (original strict setting)

    if range_width < min_range_width:
        logger.debug(f"[{symbol}] Range too narrow: {range_width:.1%} < {min_range_width:.1%} minimum")
        return None

    if range_width > max_range_width:
        logger.debug(f"[{symbol}] Range too wide: {range_width:.1%} > {max_range_width:.1%} maximum")
        return None

    logger.debug(f"[{symbol}] Range width {range_width:.1%} within optimal bounds ({min_range_width:.1%}-{max_range_width:.1%})")

    current_price = close.iloc[-1]
    current_atr = _atr(df, s.atr_len)[-1]

    # --- LONG SIGNAL (Buy at Support) ---
    # Condition: Price touches the lower part of the range and shows reversal (REVERTED TO ORIGINAL)
    if abs(current_price - lower_range) < (current_atr * 0.5):  # Original strict 0.5 ATR
        # Reversal confirmation: Bullish candle (close > open) - ORIGINAL STRICT REQUIREMENT
        if close.iloc[-1] > df["open"].iloc[-1]:
            entry = current_price

            # HYBRID SL METHOD - use whichever gives more room (same as pullback strategy)
            # Volatility adjustment for SL buffer
            volatility_percentile = 0.5  # Default middle volatility
            if len(df) >= 50:
                atr_series = pd.Series([_atr(df.iloc[i-14:i+1], 14)[-1] for i in range(14, len(df))])
                current_atr_rank = (atr_series < current_atr).sum() / len(atr_series)
                volatility_percentile = current_atr_rank

            # Volatility multiplier (higher volatility = wider stops)
            if volatility_percentile > 0.8:
                volatility_multiplier = 1.4  # High volatility
            elif volatility_percentile > 0.6:
                volatility_multiplier = 1.2  # Above average
            elif volatility_percentile < 0.2:
                volatility_multiplier = 0.8  # Low volatility
            else:
                volatility_multiplier = 1.0  # Normal

            adjusted_sl_buffer = s.sl_buf_atr * volatility_multiplier

            # Option 1: Range lower boundary with small buffer (conservative)
            sl_option1 = lower_range - (adjusted_sl_buffer * 0.3 * current_atr)

            # Option 2: ATR-based from entry (standard)
            sl_option2 = entry - (adjusted_sl_buffer * 1.0 * current_atr)

            # Option 3: Support level with larger buffer (range-specific)
            sl_option3 = lower_range - (current_atr * adjusted_sl_buffer)

            # Use the lowest SL (gives most room for LONG)
            sl = min(sl_option1, sl_option2, sl_option3)

            # Log which method was used
            if sl == sl_option1:
                logger.info(f"{symbol}: Using range boundary for stop: {lower_range:.4f}")
            elif sl == sl_option2:
                logger.info(f"{symbol}: Using entry-based stop: {entry:.4f}")
            else:
                logger.info(f"{symbol}: Using support level for stop: {lower_range:.4f}")

            # Ensure minimum stop distance (1% of entry price)
            min_stop_distance = entry * 0.01
            if abs(entry - sl) < min_stop_distance:
                sl = entry - min_stop_distance
                logger.info(f"{symbol}: Adjusted stop to minimum distance (1% from entry)")

            # FIXED RISK:REWARD with fee adjustment (same as pullback)
            # Bybit fees: 0.06% entry (market) + 0.055% exit (limit) = 0.115% total
            # Add 0.05% for slippage = 0.165% total cost
            # To get 2.5:1 after fees, we need to target slightly higher
            fee_adjustment = 1.00165  # Compensate for 0.165% total costs
            tp = entry + ((entry - sl) * s.rr * fee_adjustment)

            if entry > sl and tp > entry:
                logger.info(f"[{symbol}] MEAN REVERSION LONG: Bouncing off support {lower_range:.4f}. Entry: {entry:.4f}, SL: {sl:.4f}, TP: {tp:.4f}, R: {(entry - sl):.4f}, RR: {s.rr}")
                state.last_signal_candle_time = df.index[-1]

                # Calculate mean reversion specific features
                signal_data = {
                    'side': 'long',
                    'entry': entry,
                    'sl': sl,
                    'tp': tp,
                    'meta': {'range_upper': upper_range, 'range_lower': lower_range}
                }

                # Import and calculate ML features specific to mean reversion
                try:
                    from ml_scorer_mean_reversion import calculate_mean_reversion_features
                    mr_features = calculate_mean_reversion_features(df, signal_data, symbol)
                    logger.debug(f"[{symbol}] Mean reversion features calculated: {len(mr_features)} features")
                except Exception as e:
                    logger.warning(f"[{symbol}] Failed to calculate MR features: {e}")
                    mr_features = {}

                return Signal(
                    side="long",
                    entry=entry,
                    sl=sl,
                    tp=tp,
                    reason=f"Mean Reversion: Bounce from support @ {lower_range:.2f}",
                    meta={
                        "range_upper": upper_range,
                        "range_lower": lower_range,
                        "mr_features": mr_features,  # Store MR features for ML
                        "strategy_name": "mean_reversion"
                    }
                )

    # --- SHORT SIGNAL (Sell at Resistance) ---
    # Condition: Price touches the upper part of the range and shows reversal (REVERTED TO ORIGINAL)
    if abs(current_price - upper_range) < (current_atr * 0.5):  # Original strict 0.5 ATR
        # Reversal confirmation: Bearish candle (close < open) - ORIGINAL STRICT REQUIREMENT
        if close.iloc[-1] < df["open"].iloc[-1]:
            entry = current_price

            # HYBRID SL METHOD - use whichever gives more room (same as pullback strategy)
            # Volatility adjustment for SL buffer
            volatility_percentile = 0.5  # Default middle volatility
            if len(df) >= 50:
                atr_series = pd.Series([_atr(df.iloc[i-14:i+1], 14)[-1] for i in range(14, len(df))])
                current_atr_rank = (atr_series < current_atr).sum() / len(atr_series)
                volatility_percentile = current_atr_rank

            # Volatility multiplier (higher volatility = wider stops)
            if volatility_percentile > 0.8:
                volatility_multiplier = 1.4  # High volatility
            elif volatility_percentile > 0.6:
                volatility_multiplier = 1.2  # Above average
            elif volatility_percentile < 0.2:
                volatility_multiplier = 0.8  # Low volatility
            else:
                volatility_multiplier = 1.0  # Normal

            adjusted_sl_buffer = s.sl_buf_atr * volatility_multiplier

            # Option 1: Range upper boundary with small buffer (conservative)
            sl_option1 = upper_range + (adjusted_sl_buffer * 0.3 * current_atr)

            # Option 2: ATR-based from entry (standard)
            sl_option2 = entry + (adjusted_sl_buffer * 1.0 * current_atr)

            # Option 3: Resistance level with larger buffer (range-specific)
            sl_option3 = upper_range + (current_atr * adjusted_sl_buffer)

            # Use the highest SL (gives most room for SHORT)
            sl = max(sl_option1, sl_option2, sl_option3)

            # Log which method was used
            if sl == sl_option1:
                logger.info(f"{symbol}: Using range boundary for stop: {upper_range:.4f}")
            elif sl == sl_option2:
                logger.info(f"{symbol}: Using entry-based stop: {entry:.4f}")
            else:
                logger.info(f"{symbol}: Using resistance level for stop: {upper_range:.4f}")

            # Ensure minimum stop distance (1% of entry price)
            min_stop_distance = entry * 0.01
            if abs(sl - entry) < min_stop_distance:
                sl = entry + min_stop_distance
                logger.info(f"{symbol}: Adjusted stop to minimum distance (1% from entry)")

            # FIXED RISK:REWARD with fee adjustment (same as pullback)
            # Bybit fees: 0.06% entry (market) + 0.055% exit (limit) = 0.115% total
            # Add 0.05% for slippage = 0.165% total cost
            # To get 2.5:1 after fees, we need to target slightly higher
            fee_adjustment = 1.00165  # Compensate for 0.165% total costs
            tp = entry - ((sl - entry) * s.rr * fee_adjustment)

            if sl > entry and entry > tp:
                logger.info(f"[{symbol}] MEAN REVERSION SHORT: Rejecting from resistance {upper_range:.4f}. Entry: {entry:.4f}, SL: {sl:.4f}, TP: {tp:.4f}, R: {(sl - entry):.4f}, RR: {s.rr}")
                state.last_signal_candle_time = df.index[-1]

                # Calculate mean reversion specific features
                signal_data = {
                    'side': 'short',
                    'entry': entry,
                    'sl': sl,
                    'tp': tp,
                    'meta': {'range_upper': upper_range, 'range_lower': lower_range}
                }

                # Import and calculate ML features specific to mean reversion
                try:
                    from ml_scorer_mean_reversion import calculate_mean_reversion_features
                    mr_features = calculate_mean_reversion_features(df, signal_data, symbol)
                    logger.debug(f"[{symbol}] Mean reversion features calculated: {len(mr_features)} features")
                except Exception as e:
                    logger.warning(f"[{symbol}] Failed to calculate MR features: {e}")
                    mr_features = {}

                return Signal(
                    side="short",
                    entry=entry,
                    sl=sl,
                    tp=tp,
                    reason=f"Mean Reversion: Rejection from resistance @ {upper_range:.2f}",
                    meta={
                        "range_upper": upper_range,
                        "range_lower": lower_range,
                        "mr_features": mr_features,  # Store MR features for ML
                        "strategy_name": "mean_reversion"
                    }
                )
    return None

# Global state tracking for each symbol
mean_reversion_states: Dict[str, BreakoutState] = {}

def reset_symbol_state(symbol: str):
    """Reset the state for a symbol (called when position closes or before backtest)."""
    if symbol in mean_reversion_states:
        del mean_reversion_states[symbol]
        logger.debug(f"[{symbol}] Mean Reversion state reset.")
