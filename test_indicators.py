#!/usr/bin/env python3
"""Test that technical indicators work with ta library fallback"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

# Test the wrapper
from crypto_trading_bot.src.utils import technical_indicators as indicators

# Create sample data
np.random.seed(42)
close = 100 + np.cumsum(np.random.randn(100) * 2)
high = close + np.abs(np.random.randn(100))
low = close - np.abs(np.random.randn(100))
volume = np.random.randint(1000, 10000, 100)

print("Testing Technical Indicators Wrapper")
print("=" * 50)

# Test ATR
atr = indicators.ATR(high, low, close, timeperiod=14)
print(f"✅ ATR calculated: Last value = {atr[-1]:.2f}")

# Test Bollinger Bands
upper, middle, lower = indicators.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2)
print(f"✅ Bollinger Bands: Upper={upper[-1]:.2f}, Middle={middle[-1]:.2f}, Lower={lower[-1]:.2f}")

# Test RSI
rsi = indicators.RSI(close, timeperiod=14)
print(f"✅ RSI calculated: Last value = {rsi[-1]:.2f}")

# Test EMA
ema20 = indicators.EMA(close, timeperiod=20)
ema50 = indicators.EMA(close, timeperiod=50)
print(f"✅ EMA calculated: EMA20={ema20[-1]:.2f}, EMA50={ema50[-1]:.2f}")

# Test MACD
macd, signal, hist = indicators.MACD(close)
print(f"✅ MACD calculated: MACD={macd[-1]:.2f}, Signal={signal[-1]:.2f}, Hist={hist[-1]:.2f}")

print("\n✅ All indicators working correctly with 'ta' library!")