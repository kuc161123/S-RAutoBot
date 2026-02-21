"""
Multi-Divergence Detector - 4 Divergence Types
===============================================
Detects all 4 RSI divergence types with EMA 200 Trend filter.
Based on comprehensive 500-symbol backtest validation.

Divergence Types:
1. REG_BULL (Regular Bullish): Price LL, RSI HL → LONG (Reversal)
2. REG_BEAR (Regular Bearish): Price HH, RSI LH → SHORT (Reversal)
3. HID_BULL (Hidden Bullish): Price HL, RSI LL → LONG (Continuation)
4. HID_BEAR (Hidden Bearish): Price LH, RSI HH → SHORT (Continuation)

Key Features:
- Pivot-based divergence detection (NO LOOK-AHEAD BIAS)
- EMA 200 trend filter
- Break of Structure (BOS) confirmation
- Symbol-specific divergence type filtering
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple
from datetime import datetime

# Configuration
RSI_PERIOD = 14
LOOKBACK_BARS = 50
MIN_PIVOT_DISTANCE = 3
PIVOT_RIGHT = 3  # Confirmation lag (no look-ahead)
DAILY_EMA_PROXY = 200  # 200 EMA for 1H timeframe

# Divergence type codes
DIV_REG_BULL = 'REG_BULL'  # Regular Bullish
DIV_REG_BEAR = 'REG_BEAR'  # Regular Bearish
DIV_HID_BULL = 'HID_BULL'  # Hidden Bullish
DIV_HID_BEAR = 'HID_BEAR'  # Hidden Bearish


@dataclass
class DivergenceSignal:
    """Represents a divergence signal with type identification"""
    symbol: str
    side: str  # 'long' or 'short'
    signal_type: str  # 'bullish' or 'bearish' (legacy compatibility)
    divergence_code: str  # 'REG_BULL', 'REG_BEAR', 'HID_BULL', 'HID_BEAR'
    divergence_idx: int
    swing_level: float  # BOS trigger level
    rsi_value: float
    price: float
    timestamp: pd.Timestamp
    pivot_timestamp: pd.Timestamp  # NEW: For robust deduplication
    daily_trend_aligned: bool
    
    def to_dict(self):
        return {
            'symbol': self.symbol,
            'side': self.side,
            'type': self.signal_type,
            'divergence_code': self.divergence_code,
            'price': self.price,
            'swing': self.swing_level,
            'timestamp': self.timestamp.isoformat() if isinstance(self.timestamp, pd.Timestamp) else str(self.timestamp),
            'pivot_timestamp': self.pivot_timestamp.isoformat() if isinstance(self.pivot_timestamp, pd.Timestamp) else str(self.pivot_timestamp),
            'trend_aligned': self.daily_trend_aligned
        }
    
    def get_display_name(self) -> str:
        """Get human-readable divergence name"""
        names = {
            'REG_BULL': 'Regular Bullish',
            'REG_BEAR': 'Regular Bearish',
            'HID_BULL': 'Hidden Bullish',
            'HID_BEAR': 'Hidden Bearish'
        }
        return names.get(self.divergence_code, self.divergence_code)
    
    def get_short_name(self) -> str:
        """Get short divergence name for compact display"""
        names = {
            'REG_BULL': 'Reg Bull',
            'REG_BEAR': 'Reg Bear',
            'HID_BULL': 'Hid Bull',
            'HID_BEAR': 'Hid Bear'
        }
        return names.get(self.divergence_code, self.divergence_code)


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
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    return rsi


def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate Average True Range"""
    hl = df['high'] - df['low']
    hc = abs(df['high'] - df['close'].shift())
    lc = abs(df['low'] - df['close'].shift())
    
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    
    return atr


def calculate_daily_ema(close: pd.Series, period: int = DAILY_EMA_PROXY) -> pd.Series:
    """Calculate EMA 200 for trend filtering"""
    return close.ewm(span=period, adjust=False).mean()


def find_pivots(data: np.ndarray, left: int = 3, right: int = 3) -> Tuple[np.ndarray, np.ndarray]:
    """
    Find pivot highs and lows in price data.
    
    A pivot at index i is confirmed at i+right.
    This matches the validated backtest logic - NO LOOK-AHEAD BIAS.
    
    Args:
        data: Array of values (typically close prices)
        left: Bars to left for pivot
        right: Bars to right for pivot
        
    Returns:
        (pivot_highs, pivot_lows) - arrays with values at pivots, NaN elsewhere
    """
    n = len(data)
    pivot_highs = np.full(n, np.nan)
    pivot_lows = np.full(n, np.nan)
    
    for i in range(left, n - right):
        # Check if local high
        is_high = all(data[j] < data[i] for j in range(i - left, i + right + 1) if j != i)
        if is_high:
            pivot_highs[i] = data[i]
        
        # Check if local low
        is_low = all(data[j] > data[i] for j in range(i - left, i + right + 1) if j != i)
        if is_low:
            pivot_lows[i] = data[i]
    
    return pivot_highs, pivot_lows


def detect_divergences(df: pd.DataFrame, symbol: str, allowed_types: List[str] = None) -> List[DivergenceSignal]:
    """
    Detect all 4 RSI divergence types with Daily Trend filter.
    
    Divergence Types:
    - REG_BULL: Price Lower Low, RSI Higher Low (Reversal → Long)
    - REG_BEAR: Price Higher High, RSI Lower High (Reversal → Short)
    - HID_BULL: Price Higher Low, RSI Lower Low (Continuation → Long)
    - HID_BEAR: Price Lower High, RSI Higher High (Continuation → Short)
    
    Args:
        df: DataFrame with OHLC + indicators
        symbol: Trading pair symbol
        allowed_types: List of allowed divergence codes (None = all types)
        
    Returns:
        List of DivergenceSignal objects
    """
    if len(df) < 100:
        return []
    
    # Default to all types if not specified
    if allowed_types is None:
        allowed_types = [DIV_REG_BULL, DIV_REG_BEAR, DIV_HID_BULL, DIV_HID_BEAR]
    
    close = df['close'].values
    high = df['high'].values
    low = df['low'].values
    rsi = df['rsi'].values
    daily_ema = df['daily_ema'].values
    
    n = len(df)
    price_high_pivots, price_low_pivots = find_pivots(close, 3, 3)

    signals = []
    # [BACKTEST ALIGNMENT] Track used pivot pairs to prevent the same pair from
    # generating signals at multiple scan positions. Matches backtest dedup logic.
    used_pivots = set()
    
    # Scan for divergences
    # Start after EMA warmup to have valid indicators, end at n - PIVOT_RIGHT to ensure pivot confirmation
    scan_start = max(30, DAILY_EMA_PROXY + 10)  # Need EMA 200 warmup
    for i in range(scan_start, n - PIVOT_RIGHT):
        current_price = close[i]
        current_ema = daily_ema[i]

        # [BACKTEST ALIGNMENT] Skip rows with NaN indicators (from preserved warmup rows)
        if np.isnan(current_ema) or np.isnan(rsi[i]):
            continue
        
        # ========== BULLISH DIVERGENCES (LONG) ==========
        # Find current and previous pivot lows
        curr_pl = curr_pli = prev_pl = prev_pli = None
        
        # Start searching from i - PIVOT_RIGHT to avoid look-ahead
        for j in range(i - PIVOT_RIGHT, max(i - LOOKBACK_BARS - PIVOT_RIGHT, 0), -1):
            if not np.isnan(price_low_pivots[j]):
                if curr_pl is None:
                    curr_pl, curr_pli = price_low_pivots[j], j
                elif prev_pl is None and j < curr_pli - MIN_PIVOT_DISTANCE:
                    prev_pl, prev_pli = price_low_pivots[j], j
                    break
        
        if curr_pl is not None and prev_pl is not None:
            # [BACKTEST ALIGNMENT] Skip if this pivot pair already generated a signal
            dedup_key_bull = (curr_pli, prev_pli, 'BULL')
            if dedup_key_bull in used_pivots:
                pass  # Skip to bearish check below
            else:
                swing_high = max(high[max(0, i-LOOKBACK_BARS):i+1])
                trend_aligned = current_price > current_ema

                # REG_BULL: Price LL (curr < prev), RSI HL (curr > prev)
                if DIV_REG_BULL in allowed_types:
                    if curr_pl < prev_pl and rsi[curr_pli] > rsi[prev_pli]:
                        signals.append(DivergenceSignal(
                            symbol=symbol,
                            side='long',
                            signal_type='bullish',
                            divergence_code=DIV_REG_BULL,
                            divergence_idx=i,
                            swing_level=swing_high,
                            rsi_value=rsi[i],
                            price=current_price,
                            timestamp=df.index[i],
                            pivot_timestamp=df.index[curr_pli],
                            daily_trend_aligned=trend_aligned
                        ))
                        used_pivots.add(dedup_key_bull)

                # HID_BULL: Price HL (curr > prev), RSI LL (curr < prev)
                if DIV_HID_BULL in allowed_types and dedup_key_bull not in used_pivots:
                    if curr_pl > prev_pl and rsi[curr_pli] < rsi[prev_pli]:
                        signals.append(DivergenceSignal(
                            symbol=symbol,
                            side='long',
                            signal_type='bullish',
                            divergence_code=DIV_HID_BULL,
                            divergence_idx=i,
                            swing_level=swing_high,
                            rsi_value=rsi[i],
                            price=current_price,
                            timestamp=df.index[i],
                            pivot_timestamp=df.index[curr_pli],
                            daily_trend_aligned=trend_aligned
                        ))
                        used_pivots.add(dedup_key_bull)
        
        # ========== BEARISH DIVERGENCES (SHORT) ==========
        # Find current and previous pivot highs
        curr_ph = curr_phi = prev_ph = prev_phi = None
        
        for j in range(i - PIVOT_RIGHT, max(i - LOOKBACK_BARS - PIVOT_RIGHT, 0), -1):
            if not np.isnan(price_high_pivots[j]):
                if curr_ph is None:
                    curr_ph, curr_phi = price_high_pivots[j], j
                elif prev_ph is None and j < curr_phi - MIN_PIVOT_DISTANCE:
                    prev_ph, prev_phi = price_high_pivots[j], j
                    break
        
        if curr_ph is not None and prev_ph is not None:
            # [BACKTEST ALIGNMENT] Skip if this pivot pair already generated a signal
            dedup_key_bear = (curr_phi, prev_phi, 'BEAR')
            if dedup_key_bear in used_pivots:
                pass  # Skip - already used
            else:
                swing_low = min(low[max(0, i-LOOKBACK_BARS):i+1])
                trend_aligned = current_price < current_ema

                # REG_BEAR: Price HH (curr > prev), RSI LH (curr < prev)
                if DIV_REG_BEAR in allowed_types:
                    if curr_ph > prev_ph and rsi[curr_phi] < rsi[prev_phi]:
                        signals.append(DivergenceSignal(
                            symbol=symbol,
                            side='short',
                            signal_type='bearish',
                            divergence_code=DIV_REG_BEAR,
                            divergence_idx=i,
                            swing_level=swing_low,
                            rsi_value=rsi[i],
                            price=current_price,
                            timestamp=df.index[i],
                            pivot_timestamp=df.index[curr_phi],
                            daily_trend_aligned=trend_aligned
                        ))
                        used_pivots.add(dedup_key_bear)

                # HID_BEAR: Price LH (curr < prev), RSI HH (curr > prev)
                if DIV_HID_BEAR in allowed_types and dedup_key_bear not in used_pivots:
                    if curr_ph < prev_ph and rsi[curr_phi] > rsi[prev_phi]:
                        signals.append(DivergenceSignal(
                            symbol=symbol,
                            side='short',
                            signal_type='bearish',
                            divergence_code=DIV_HID_BEAR,
                            divergence_idx=i,
                            swing_level=swing_low,
                            rsi_value=rsi[i],
                            price=current_price,
                            timestamp=df.index[i],
                            pivot_timestamp=df.index[curr_phi],
                            daily_trend_aligned=trend_aligned
                        ))
                        used_pivots.add(dedup_key_bear)
    
    return signals


def prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare DataFrame with all required indicators.

    NOTE: We do NOT drop NaN rows. This preserves all candles so that
    pivot indices and divergence detection align with the backtest.
    NaN values are handled explicitly in detect_divergences().

    Args:
        df: Raw OHLCV DataFrame

    Returns:
        DataFrame with RSI, ATR, Daily EMA added (NaN rows preserved)
    """
    df = df.copy()

    # Calculate indicators
    df['rsi'] = calculate_rsi(df['close'])
    df['atr'] = calculate_atr(df)
    df['daily_ema'] = calculate_daily_ema(df['close'])

    # [BACKTEST ALIGNMENT] Do NOT dropna() - keep all rows to preserve indices.
    # The detect_divergences() function skips rows with NaN indicators.

    return df


def check_bos(df: pd.DataFrame, signal: DivergenceSignal, current_idx: int) -> bool:
    """
    Check if Break of Structure (BOS) has occurred.
    
    Args:
        df: DataFrame with OHLC data
        signal: Original divergence signal
        current_idx: Current candle index
        
    Returns:
        True if BOS confirmed, False otherwise
    """
    if current_idx >= len(df):
        return False
    
    current_close = df.iloc[current_idx]['close']
    
    if signal.side == 'long':
        return current_close > signal.swing_level
    else:
        return current_close < signal.swing_level


def is_trend_aligned(df: pd.DataFrame, signal: DivergenceSignal, current_idx: int) -> bool:
    """
    Check if current price is still aligned with daily trend.
    
    Args:
        df: DataFrame with indicators
        signal: Divergence signal
        current_idx: Current candle index
        
    Returns:
        True if trend still aligned, False otherwise
    """
    if current_idx >= len(df):
        return False
    
    current_price = df.iloc[current_idx]['close']
    current_ema = df.iloc[current_idx]['daily_ema']
    
    if signal.side == 'long':
        return current_price > current_ema
    else:
        return current_price < current_ema


# Signal descriptions for notifications
SIGNAL_DESCRIPTIONS = {
    'REG_BULL': '📈 Regular Bullish (Price LL → RSI HL) → LONG',
    'REG_BEAR': '📉 Regular Bearish (Price HH → RSI LH) → SHORT',
    'HID_BULL': '📈 Hidden Bullish (Price HL → RSI LL) → LONG',
    'HID_BEAR': '📉 Hidden Bearish (Price LH → RSI HH) → SHORT',
    # Legacy compatibility
    'bullish': '📈 Bullish Divergence → LONG',
    'bearish': '📉 Bearish Divergence → SHORT'
}

SIGNAL_EMOJIS = {
    'REG_BULL': '🔵',  # Regular Bullish
    'REG_BEAR': '🔴',  # Regular Bearish
    'HID_BULL': '🟢',  # Hidden Bullish
    'HID_BEAR': '🟠',  # Hidden Bearish
}


def get_signal_description(code: str) -> str:
    """Get human-readable description for a divergence code"""
    return SIGNAL_DESCRIPTIONS.get(code, f"Unknown: {code}")


def get_signal_emoji(code: str) -> str:
    """Get emoji for a divergence code"""
    return SIGNAL_EMOJIS.get(code, '⚪')
