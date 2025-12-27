"""
RSI Divergence Detector Module
==============================
Detects RSI divergence patterns for trading signals.

Divergence Types:
- Regular Bullish: Price makes Lower Low, RSI makes Higher Low → Long signal
- Regular Bearish: Price makes Higher High, RSI makes Lower High → Short signal
- Hidden Bullish: Price makes Higher Low, RSI makes Lower Low → Long (continuation)
- Hidden Bearish: Price makes Lower High, RSI makes Higher High → Short (continuation)

Based on walk-forward validated backtest:
- 26,850 trades | 61.3% WR | +0.84 EV at 2:1 R:R
- 100% consistent across 6 periods
"""

import pandas as pd
import numpy as np
import logging
from typing import List, Dict, Optional
from dataclasses import dataclass

logger = logging.getLogger("DivergenceDetector")

# =============================================================================
# CONFIGURATION (matches backtest parameters)
# =============================================================================

RSI_PERIOD = 14
LOOKBACK_BARS = 14
MIN_PIVOT_DISTANCE = 5
PIVOT_LEFT = 3
PIVOT_RIGHT = 3

# RSI zones for signal confirmation
RSI_OVERSOLD = 30
RSI_OVERBOUGHT = 70


@dataclass
class DivergenceSignal:
    """Represents a detected divergence signal"""
    symbol: str
    side: str  # 'long' or 'short'
    signal_type: str  # 'regular_bullish', 'regular_bearish', 'hidden_bullish', 'hidden_bearish'
    rsi_value: float
    price: float
    timestamp: pd.Timestamp
    combo: str  # Format: "DIV:regular_bullish"
    
    def __post_init__(self):
        self.combo = f"DIV:{self.signal_type}"


def calculate_rsi(close: pd.Series, period: int = RSI_PERIOD) -> pd.Series:
    """
    Calculate RSI (Relative Strength Index).
    
    Args:
        close: Series of closing prices
        period: RSI period (default 14)
        
    Returns:
        Series of RSI values (0-100)
    """
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    return rsi


def find_pivots(data: np.ndarray, left: int = PIVOT_LEFT, right: int = PIVOT_RIGHT) -> tuple:
    """
    Find pivot highs and lows in a data series.
    
    Args:
        data: numpy array of values
        left: bars to left for pivot confirmation
        right: bars to right for pivot confirmation
        
    Returns:
        (pivot_highs, pivot_lows) - arrays with values at pivots, NaN elsewhere
    """
    n = len(data)
    pivot_highs = np.full(n, np.nan)
    pivot_lows = np.full(n, np.nan)
    
    for i in range(left, n - right):
        # Check for pivot high
        is_high = True
        for j in range(i - left, i + right + 1):
            if j != i and data[j] >= data[i]:
                is_high = False
                break
        if is_high:
            pivot_highs[i] = data[i]
        
        # Check for pivot low
        is_low = True
        for j in range(i - left, i + right + 1):
            if j != i and data[j] <= data[i]:
                is_low = False
                break
        if is_low:
            pivot_lows[i] = data[i]
    
    return pivot_highs, pivot_lows


def detect_divergence(
    df: pd.DataFrame, 
    symbol: str,
    lookback: int = LOOKBACK_BARS,
    min_distance: int = MIN_PIVOT_DISTANCE
) -> List[DivergenceSignal]:
    """
    Detect RSI divergence patterns using ROLLING WINDOW to match Backtest Logic.
    
    Backtest Logic:
    - Uses rolling(14) min/max to identify swing points
    - Compares current price/RSI vs simple rolling window
    - NO complex fractal pivot detection (keeps it simple and robust)
    """
    if len(df) < 50:
        return []
    
    # Ensure RSI is calculated
    if 'rsi' not in df.columns:
        df = df.copy()
        df['rsi'] = calculate_rsi(df['close'], RSI_PERIOD)
    
    # Calculate rolling metrics exactly like backtest
    # df['price_low'] = df['low'].rolling(lookback).min()
    # df['price_high'] = df['high'].rolling(lookback).max()
    # But for detection on ANY candle (not just vector), we look at the last row
    
    # We need history to calculate rolling values
    # Copy relevant columns to avoid affecting source
    data = df[['high', 'low', 'close', 'rsi']].copy()
    
    data['price_low'] = data['low'].rolling(lookback).min()
    data['price_high'] = data['high'].rolling(lookback).max()
    data['rsi_low'] = data['rsi'].rolling(lookback).min()
    data['rsi_high'] = data['rsi'].rolling(lookback).max()
    
    # We only check the LAST confirmed candle (iloc[-1])
    # The 'df' passed here is usually df_closed (the last closed candle is the last row)
    curr = data.iloc[-1]
    
    # BACKTEST MATCHING:
    # Regular Divergence in backtest uses: df['rsi'] > df['rsi_low'].shift(lookback)
    # shift(lookback) means we access the rolling min from 'lookback' bars ago.
    # rolling window at T-14 covers [T-14-13 ... T-14]. i.e. [T-27 ... T-14].
    # So we need to look at the window from 2*lookback ago up to lookback ago.
    
    # Distant window for Regular Divergence (matches shift(lookback))
    distant_start = -2 * lookback - 1
    distant_end = -lookback - 1
    
    # Safety Check: Ensure we have enough data
    if abs(distant_start) > len(data):
        # Not enough history for full match, fallback to what we have or return empty
        return []
        
    distant_window = data.iloc[distant_start:distant_end]
    
    signals = []
    timestamp = df.index[-1]
    price = curr['close']
    rsi = curr['rsi']
    
    # 1. Regular Bullish
    # Backtest: low <= price_low (current window) AND rsi > rsi_low.shift(lookback) (distant window)
    is_new_low = curr['low'] <= curr['price_low'] + 0.0000001
    
    # RSI low from DISTANT window (matches shift(lookback))
    distant_rsi_low = distant_window['rsi'].min()
    
    if is_new_low and curr['rsi'] > distant_rsi_low and curr['rsi'] < 40:
        signals.append(DivergenceSignal(symbol, 'long', 'regular_bullish', rsi, price, timestamp, "DIV:regular_bullish"))

    # 2. Regular Bearish
    # Backtest: high >= price_high (current window) AND rsi < rsi_high.shift(lookback) (distant window)
    is_new_high = curr['high'] >= curr['price_high'] - 0.0000001
    distant_rsi_high = distant_window['rsi'].max()
    
    if is_new_high and curr['rsi'] < distant_rsi_high and curr['rsi'] > 60:
        signals.append(DivergenceSignal(symbol, 'short', 'regular_bearish', rsi, price, timestamp, "DIV:regular_bearish"))

    # 3. Hidden Bullish
    # Price is HIGHER than old low, but RSI is lower (Continuation)
    # Check: Low > Old Low (from 14 bars ago) AND RSI < Old RSI
    # In backtest: df['low'] > df['low'].shift(lookback)
    compare_candle = data.iloc[-lookback-1] if len(data) > lookback else data.iloc[0]
    
    if curr['low'] > compare_candle['low'] and curr['rsi'] < compare_candle['rsi'] and curr['rsi'] < 60:
         signals.append(DivergenceSignal(symbol, 'long', 'hidden_bullish', rsi, price, timestamp, "DIV:hidden_bullish"))

    # 4. Hidden Bearish
    # Price is LOWER than old high, but RSI is higher
    if curr['high'] < compare_candle['high'] and curr['rsi'] > compare_candle['rsi'] and curr['rsi'] > 40:
         signals.append(DivergenceSignal(symbol, 'short', 'hidden_bearish', rsi, price, timestamp, "DIV:hidden_bearish"))
            
    return signals


def prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare DataFrame with all required indicators for divergence detection.
    
    Args:
        df: Raw OHLCV DataFrame
        
    Returns:
        DataFrame with RSI and other indicators added
    """
    df = df.copy()
    
    # RSI
    df['rsi'] = calculate_rsi(df['close'], RSI_PERIOD)
    
    # ATR for SL/TP calculation
    high_low = df['high'] - df['low']
    high_close = abs(df['high'] - df['close'].shift())
    low_close = abs(df['low'] - df['close'].shift())
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr'] = true_range.rolling(20).mean()
    
    # Volume filter (above 50% of 20-period average)
    df['vol_ma'] = df['volume'].rolling(20).mean()
    df['vol_ok'] = df['volume'] > df['vol_ma'] * 0.5
    
    return df.dropna()


def get_divergence_combo_name(signal_type: str) -> str:
    """Get standardized combo name for a divergence type."""
    return f"DIV:{signal_type}"


# Signal type descriptions for notifications
SIGNAL_DESCRIPTIONS = {
    'regular_bullish': '📈 Regular Bullish (Price LL, RSI HL) - Reversal UP',
    'regular_bearish': '📉 Regular Bearish (Price HH, RSI LH) - Reversal DOWN',
    'hidden_bullish': '📈 Hidden Bullish (Price HL, RSI LL) - Trend Continuation UP',
    'hidden_bearish': '📉 Hidden Bearish (Price LH, RSI HH) - Trend Continuation DOWN'
}


def get_signal_description(signal_type: str) -> str:
    """Get human-readable description for a signal type."""
    return SIGNAL_DESCRIPTIONS.get(signal_type, signal_type)
