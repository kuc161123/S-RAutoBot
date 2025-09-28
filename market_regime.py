"""
Market Regime Detection Model
Classifies the market into Trending, Ranging, or Volatile states.
"""
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

def get_market_regime(df: pd.DataFrame, adx_period: int = 14, bb_period: int = 20) -> str:
    """
    Determines the current market regime.

    Args:
        df (pd.DataFrame): DataFrame with columns ['high', 'low', 'close'].
                           Should contain at least 50 periods of data.
        adx_period (int): The period for calculating ADX.
        bb_period (int): The period for calculating Bollinger Bands.

    Returns:
        str: "Trending", "Ranging", or "Volatile".
    """
    if len(df) < adx_period + bb_period:
        return "Ranging"  # Default to ranging if not enough data

    try:
        # 1. Calculate ADX for Trend Strength
        high = df['high']
        low = df['low']
        close = df['close']

        plus_dm = high.diff()
        minus_dm = low.diff().mul(-1)
        plus_dm[plus_dm < 0] = 0
        plus_dm[plus_dm < minus_dm] = 0
        minus_dm[minus_dm < 0] = 0
        minus_dm[minus_dm < plus_dm] = 0

        tr1 = pd.DataFrame(high - low)
        tr2 = pd.DataFrame(abs(high - close.shift(1)))
        tr3 = pd.DataFrame(abs(low - close.shift(1)))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.ewm(alpha=1/adx_period, adjust=False).mean()

        plus_di = 100 * (plus_dm.ewm(alpha=1/adx_period, adjust=False).mean() / atr)
        minus_di = 100 * (minus_dm.ewm(alpha=1/adx_period, adjust=False).mean() / atr)
        
        dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
        adx = dx.ewm(alpha=1/adx_period, adjust=False).mean()
        
        current_adx = adx.iloc[-1]

        # 2. Calculate Bollinger Band Width for Volatility
        sma = close.rolling(window=bb_period).mean()
        std = close.rolling(window=bb_period).std()
        upper_bb = sma + (std * 2)
        lower_bb = sma + (std * 2)
        bb_width = ((upper_bb - lower_bb) / sma).iloc[-1]

        # 3. Classify Regime
        # These thresholds can be tuned
        adx_threshold = 25
        bb_width_threshold = bb_width > (df['close'].pct_change().rolling(bb_period).std().iloc[-1] * 3) # Volatility expansion

        if current_adx > adx_threshold:
            return "Trending"
        elif bb_width_threshold:
            return "Volatile"
        else:
            return "Ranging"

    except Exception as e:
        logger.error(f"Error calculating market regime: {e}")
        return "Ranging" # Default to a safe regime
