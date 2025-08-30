"""
Technical indicators for trading signals
"""
import pandas as pd
import numpy as np
from ta import momentum, trend
import structlog

logger = structlog.get_logger(__name__)

def calculate_rsi(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate RSI"""
    try:
        return momentum.RSIIndicator(close=df['close'], window=period).rsi()
    except Exception as e:
        logger.error(f"Error calculating RSI: {e}")
        return pd.Series()

def calculate_macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> dict:
    """Calculate MACD"""
    try:
        macd_indicator = trend.MACD(close=df['close'], window_slow=slow, window_fast=fast, window_sign=signal)
        return {
            'macd': macd_indicator.macd(),
            'signal': macd_indicator.macd_signal(),
            'histogram': macd_indicator.macd_diff()
        }
    except Exception as e:
        logger.error(f"Error calculating MACD: {e}")
        return {'macd': pd.Series(), 'signal': pd.Series(), 'histogram': pd.Series()}

def calculate_bollinger_bands(df: pd.DataFrame, period: int = 20, std_dev: int = 2) -> dict:
    """Calculate Bollinger Bands"""
    try:
        sma = df['close'].rolling(window=period).mean()
        std = df['close'].rolling(window=period).std()
        
        return {
            'upper': sma + (std * std_dev),
            'middle': sma,
            'lower': sma - (std * std_dev)
        }
    except Exception as e:
        logger.error(f"Error calculating Bollinger Bands: {e}")
        return {'upper': pd.Series(), 'middle': pd.Series(), 'lower': pd.Series()}

def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate Average True Range"""
    try:
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        
        return true_range.rolling(period).mean()
    except Exception as e:
        logger.error(f"Error calculating ATR: {e}")
        return pd.Series()

def calculate_volume_profile(df: pd.DataFrame, bins: int = 20) -> dict:
    """Calculate volume profile"""
    try:
        price_range = df['high'].max() - df['low'].min()
        bin_size = price_range / bins
        
        volume_profile = {}
        
        for i in range(bins):
            price_level = df['low'].min() + (i * bin_size)
            mask = (df['close'] >= price_level) & (df['close'] < price_level + bin_size)
            volume_profile[price_level] = df.loc[mask, 'volume'].sum()
        
        return volume_profile
    except Exception as e:
        logger.error(f"Error calculating volume profile: {e}")
        return {}

def add_all_indicators(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Add all indicators to dataframe"""
    try:
        # Make a copy to avoid modifying original
        df = df.copy()
        
        # RSI
        df['rsi'] = calculate_rsi(df, config.get('rsi_period', 14))
        
        # MACD
        macd = calculate_macd(df, 
                             config.get('macd_fast', 12),
                             config.get('macd_slow', 26),
                             config.get('macd_signal', 9))
        df['macd'] = macd['macd']
        df['macd_signal'] = macd['signal']
        df['macd_histogram'] = macd['histogram']
        
        # Bollinger Bands
        bb = calculate_bollinger_bands(df)
        df['bb_upper'] = bb['upper']
        df['bb_middle'] = bb['middle']
        df['bb_lower'] = bb['lower']
        
        # ATR for stop loss
        df['atr'] = calculate_atr(df)
        
        # Simple moving averages
        df['sma_20'] = df['close'].rolling(20).mean()
        df['sma_50'] = df['close'].rolling(50).mean()
        df['sma_200'] = df['close'].rolling(200).mean()
        
        # Volume moving average
        df['volume_ma'] = df['volume'].rolling(20).mean()
        
        return df
        
    except Exception as e:
        logger.error(f"Error adding indicators: {e}")
        return df