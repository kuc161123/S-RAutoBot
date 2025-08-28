"""
Technical indicators wrapper that uses 'ta' library instead of 'talib'
Provides same interface but uses the already-installed ta library
"""
import pandas as pd
import numpy as np
from ta import trend, momentum, volatility

def ATR(high, low, close, timeperiod=14):
    """Calculate Average True Range"""
    df = pd.DataFrame({'high': high, 'low': low, 'close': close})
    indicator = volatility.AverageTrueRange(high=df['high'], low=df['low'], close=df['close'], window=timeperiod)
    return indicator.average_true_range().values

def BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0):
    """Calculate Bollinger Bands"""
    df = pd.DataFrame({'close': close})
    indicator = volatility.BollingerBands(close=df['close'], window=timeperiod, window_dev=nbdevup)
    upper = indicator.bollinger_hband().values
    middle = indicator.bollinger_mavg().values
    lower = indicator.bollinger_lband().values
    return upper, middle, lower

def RSI(close, timeperiod=14):
    """Calculate Relative Strength Index"""
    df = pd.DataFrame({'close': close})
    indicator = momentum.RSIIndicator(close=df['close'], window=timeperiod)
    return indicator.rsi().values

def EMA(close, timeperiod):
    """Calculate Exponential Moving Average"""
    df = pd.DataFrame({'close': close})
    ema = df['close'].ewm(span=timeperiod, adjust=False).mean()
    return ema.values

def MACD(close, fastperiod=12, slowperiod=26, signalperiod=9):
    """Calculate MACD"""
    df = pd.DataFrame({'close': close})
    indicator = trend.MACD(close=df['close'], window_slow=slowperiod, window_fast=fastperiod, window_sign=signalperiod)
    macd = indicator.macd().values
    macd_signal = indicator.macd_signal().values
    macd_hist = indicator.macd_diff().values
    return macd, macd_signal, macd_hist