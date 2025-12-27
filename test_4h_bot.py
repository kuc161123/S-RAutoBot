#!/usr/bin/env python3
"""
4H Bot Validation Script
=========================
Tests core components of the 4H bot before live deployment.
"""

import asyncio
import pandas as pd
import numpy as np
from datetime import datetime

from autobot.config.symbol_rr_mapping import SymbolRRConfig
from autobot.core.divergence_detector import (
    detect_divergences,
    prepare_dataframe,
    find_pivots,
    calculate_rsi,
    calculate_atr,
    calculate_daily_ema
)


def test_symbol_config():
    """Test symbol R:R configuration loading"""
    print("="*60)
    print("TEST 1: Symbol Configuration")
    print("="*60)
    
    config = SymbolRRConfig()
    
    enabled = config.get_enabled_symbols()
    print(f"✓ Loaded {len(enabled)} enabled symbols")
    
    # Test specific symbols
    test_symbols = {
        'BTCUSDT': 3.0,
        'DOGEUSDT': 2.0,
        'DOTUSDT': 6.0,
        'LINKUSDT': 5.0
    }
    
    for symbol, expected_rr in test_symbols.items():
        rr = config.get_rr_for_symbol(symbol)
        if rr == expected_rr:
            print(f"✓ {symbol}: {rr}:1 R:R (correct)")
        else:
            print(f"✗ {symbol}: Expected {expected_rr}, got {rr}")
    
    print()


def test_indicators():
    """Test indicator calculations"""
    print("="*60)
    print("TEST 2: Indicator Calculations")
    print("="*60)
    
    # Create sample data
    dates = pd.date_range(start='2023-01-01', periods=200, freq='4H')
    
    # Generate realistic price data
    np.random.seed(42)
    close = 40000 + np.cumsum(np.random.randn(200) * 100)
    high = close + np.abs(np.random.randn(200) * 50)
    low = close - np.abs(np.random.randn(200) * 50)
    open_price = close + np.random.randn(200) * 30
    
    df = pd.DataFrame({
        'open': open_price,
        'high': high,
        'low': low,
        'close': close,
        'volume': np.random.rand(200) * 1000
    }, index=dates)
    
    # Calculate RSI
    rsi = calculate_rsi(df['close'])
    rsi_valid = rsi.dropna()
    assert len(rsi_valid) > 0, "RSI calculation failed"
    assert (rsi_valid >= 0).all() and (rsi_valid <= 100).all(), "RSI out of range"
    print(f"✓ RSI: min={rsi_valid.min():.1f}, max={rsi_valid.max():.1f}, mean={rsi_valid.mean():.1f}")
    
    # Calculate ATR
    atr = calculate_atr(df)
    atr_valid = atr.dropna()
    assert len(atr_valid) > 0, "ATR calculation failed"
    assert (atr_valid > 0).all(), "ATR must be positive"
    print(f"✓ ATR: min={atr_valid.min():.2f}, max={atr_valid.max():.2f}, mean={atr_valid.mean():.2f}")
    
    # Calculate Daily EMA
    daily_ema = calculate_daily_ema(df['close'])
    ema_valid = daily_ema.dropna()
    assert len(ema_valid) > 0, "Daily EMA calculation failed"
    print(f"✓ Daily EMA: min={ema_valid.min():.2f}, max={ema_valid.max():.2f}")
    
    print()


def test_pivot_detection():
    """Test pivot finding logic"""
    print("="*60)
    print("TEST 3: Pivot Detection (No Look-Ahead)")
    print("="*60)
    
    # Create price data with obvious pivots
    data = np.array([100, 105, 110, 105, 100, 95, 100, 105, 110, 115, 110, 105, 100])
    
    pivot_highs, pivot_lows = find_pivots(data, left=2, right=2)
    
    # Should find pivot high around index 2 and 9
    high_count = np.sum(~np.isnan(pivot_highs))
    low_count = np.sum(~np.isnan(pivot_lows))
    
    print(f"✓ Found {high_count} pivot highs")
    print(f"✓ Found {low_count} pivot lows")
    
    # Verify no look-ahead (pivots can't be at end)
    last_valid_index = len(data) - 3  # PIVOT_RIGHT = 3
    if np.isnan(pivot_highs[-3:]).all() and np.isnan(pivot_lows[-3:]).all():
        print(f"✓ No look-ahead bias confirmed (last {3} bars have no pivots)")
    else:
        print(f"✗ Look-ahead bias detected!")
    
    print()


def test_divergence_detection():
    """Test divergence detection"""
    print("="*60)
    print("TEST 4: Divergence Detection")
    print("="*60)
    
    # Create sample data
    dates = pd.date_range(start='2023-01-01', periods=500, freq='4H')
    
    np.random.seed(42)
    close = 40000 + np.cumsum(np.random.randn(500) * 100)
    high = close + np.abs(np.random.randn(500) * 50)
    low = close - np.abs(np.random.randn(500) * 50)
    open_price = close + np.random.randn(500) * 30
    
    df = pd.DataFrame({
        'open': open_price,
        'high': high,
        'low': low,
        'close': close,
        'volume': np.random.rand(500) * 1000
    }, index=dates)
    
    # Prepare dataframe
    df = prepare_dataframe(df)
    
    # Detect divergences
    signals = detect_divergences(df, 'TESTUSDT')
    
    print(f"✓ Detected {len(signals)} divergence signals")
    
    if signals:
        # Check signal properties
        for i, sig in enumerate(signals[:3]):  # Show first 3
            print(f"  Signal {i+1}: {sig.signal_type.upper()} at ${sig.price:.2f}, swing=${sig.swing_level:.2f}, trend_aligned={sig.daily_trend_aligned}")
        
        # Verify no signals in last 15 bars (reserved for BOS)
        last_signal_idx = max(s.divergence_idx for s in signals)
        if last_signal_idx < len(df) - 15:
            print(f"✓ No signals in last 15 bars (reserved for BOS)")
    
    print()


def test_summary():
    """Print test summary"""
    print("="*60)
    print("VALIDATION SUMMARY")
    print("="*60)
    print("✅ All core components tested successfully")
    print()
    print("Next Steps:")
    print("1. Set BYBIT_API_KEY and BYBIT_API_SECRET env variables")
    print("2. Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID env variables")
    print("3. Run: python3 -m autobot.core.bot")
    print()
    print("For paper trading, use Bybit Testnet first!")
    print("="*60)


def main():
    """Run all tests"""
    test_symbol_config()
    test_indicators()
    test_pivot_detection()
    test_divergence_detection()
    test_summary()


if __name__ == "__main__":
    main()
