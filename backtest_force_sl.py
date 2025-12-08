#!/usr/bin/env python3
"""
Quick backtest to compare:
1. SKIP: Skip trades if SL < 0.5%
2. FORCE: Force minimum 0.5% SL distance

Uses cached data from previous backtest.
"""

import requests
import pandas as pd
import numpy as np
import time
import math
from datetime import datetime, timedelta

BYBIT_BASE = "https://api.bybit.com"

# Symbols that had low SL distance
LOW_VOL_SYMBOLS = ['POLUSDT', 'ALGOUSDT', 'CRVUSDT', 'SANDUSDT', 'MANAUSDT', 'GALAUSDT']

def wilson_lower_bound(wins: int, total: int, z: float = 1.96) -> float:
    if total == 0:
        return 0.0
    p = wins / total
    denominator = 1 + z*z / total
    centre = p + z*z / (2*total)
    spread = z * math.sqrt((p*(1-p) + z*z/(4*total)) / total)
    lower = (centre - spread) / denominator
    return max(0, lower * 100)

def fetch_klines(symbol: str, interval: str = '3', days: int = 30) -> pd.DataFrame:
    all_data = []
    end_time = int(datetime.utcnow().timestamp() * 1000)
    start_time = int((datetime.utcnow() - timedelta(days=days)).timestamp() * 1000)
    
    url = f"{BYBIT_BASE}/v5/market/kline"
    limit = 1000
    current_end = end_time
    
    while current_end > start_time:
        params = {
            'category': 'linear',
            'symbol': symbol,
            'interval': interval,
            'limit': limit,
            'end': current_end
        }
        
        try:
            resp = requests.get(url, params=params, timeout=10)
            data = resp.json()
            
            if data.get('retCode') != 0 or not data.get('result', {}).get('list'):
                break
                
            klines = data['result']['list']
            all_data.extend(klines)
            earliest = int(klines[-1][0])
            if earliest <= start_time:
                break
            current_end = earliest - 1
            time.sleep(0.1)
        except:
            break
    
    if not all_data:
        return pd.DataFrame()
    
    df = pd.DataFrame(all_data, columns=['start', 'open', 'high', 'low', 'close', 'volume', 'turnover'])
    df = df.astype({'start': int, 'open': float, 'high': float, 'low': float, 'close': float, 'volume': float, 'turnover': float})
    df['start'] = pd.to_datetime(df['start'], unit='ms')
    df = df.sort_values('start').drop_duplicates('start').reset_index(drop=True)
    return df

def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if len(df) < 50:
        return df
    
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    ema12 = df['close'].ewm(span=12).mean()
    ema26 = df['close'].ewm(span=26).mean()
    df['macd'] = ema12 - ema26
    df['macd_signal'] = df['macd'].ewm(span=9).mean()
    
    high_low = df['high'] - df['low']
    high_close = abs(df['high'] - df['close'].shift())
    low_close = abs(df['low'] - df['close'].shift())
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr'] = true_range.rolling(14).mean()
    
    df['roll_high'] = df['high'].rolling(50).max()
    df['roll_low'] = df['low'].rolling(50).min()
    
    return df.dropna()

def get_combo(row) -> str:
    rsi = row['rsi']
    if rsi < 30: r_bin = '<30'
    elif rsi < 40: r_bin = '30-40'
    elif rsi < 60: r_bin = '40-60'
    elif rsi < 70: r_bin = '60-70'
    else: r_bin = '70+'
    
    m_bin = 'bull' if row['macd'] > row['macd_signal'] else 'bear'
    
    high, low, close = row['roll_high'], row['roll_low'], row['close']
    if high == low:
        f_bin = '0-23'
    else:
        fib = (high - close) / (high - low) * 100
        if fib < 23.6: f_bin = '0-23'
        elif fib < 38.2: f_bin = '23-38'
        elif fib < 50.0: f_bin = '38-50'
        elif fib < 61.8: f_bin = '50-61'
        elif fib < 78.6: f_bin = '61-78'
        elif fib < 100: f_bin = '78-100'
        else: f_bin = '100+'
    
    return f"RSI:{r_bin} MACD:{m_bin} Fib:{f_bin}"

def simulate_trade_with_sl_mode(df, entry_idx, side, atr, entry_price, mode='skip', rr_ratio=2.0, max_candles=80):
    """
    Simulate trade with different SL modes.
    mode='skip': Return None if SL < 0.5%
    mode='force': Force SL to be at least 0.5%
    """
    MIN_SL_PCT = 0.5
    
    sl_distance_atr = atr
    sl_distance_min = entry_price * (MIN_SL_PCT / 100)
    
    # Calculate SL distance percentage based on ATR
    sl_pct = (sl_distance_atr / entry_price) * 100
    
    if mode == 'skip' and sl_pct < MIN_SL_PCT:
        return 'SKIPPED', sl_pct
    
    # For 'force' mode, use the larger of ATR or min distance
    if mode == 'force':
        sl_distance = max(sl_distance_atr, sl_distance_min)
    else:
        sl_distance = sl_distance_atr
    
    tp_distance = sl_distance * rr_ratio
    
    if side == 'long':
        sl = entry_price - sl_distance
        tp = entry_price + tp_distance
    else:
        sl = entry_price + sl_distance
        tp = entry_price - tp_distance
    
    # Check future candles
    for i in range(entry_idx + 1, min(entry_idx + max_candles, len(df))):
        candle = df.iloc[i]
        high = candle['high']
        low = candle['low']
        
        if side == 'long':
            if low <= sl:
                return 'loss', sl_pct
            elif high >= tp:
                return 'win', sl_pct
        else:
            if high >= sl:
                return 'loss', sl_pct
            elif low <= tp:
                return 'win', sl_pct
    
    return 'timeout', sl_pct

def run_comparison():
    print("🔬 Force Minimum SL Comparison Backtest")
    print("="*60)
    
    # Fetch data for low-vol symbols
    print(f"\n📥 Fetching data for {len(LOW_VOL_SYMBOLS)} symbols...")
    symbol_data = {}
    
    for symbol in LOW_VOL_SYMBOLS:
        print(f"   {symbol}...", end=' ')
        df = fetch_klines(symbol, '3', 14)
        if len(df) > 100:
            df = calculate_indicators(df)
            symbol_data[symbol] = df
            print(f"✅ {len(df)} candles")
        else:
            print(f"❌")
        time.sleep(0.2)
    
    # Run simulations
    results = {
        'skip': {'total_signals': 0, 'skipped': 0, 'trades': 0, 'wins': 0},
        'force': {'total_signals': 0, 'skipped': 0, 'trades': 0, 'wins': 0}
    }
    
    print(f"\n🧪 Running simulations...")
    
    for symbol, df in symbol_data.items():
        prev_row = None
        for idx in range(len(df) - 40):
            row = df.iloc[idx]
            
            # Simple signal detection every N candles
            if idx % 20 == 0:  # Sample signals
                for side in ['long', 'short']:
                    entry = row['close']
                    atr = row['atr']
                    
                    for mode in ['skip', 'force']:
                        results[mode]['total_signals'] += 1
                        outcome, sl_pct = simulate_trade_with_sl_mode(df, idx, side, atr, entry, mode)
                        
                        if outcome == 'SKIPPED':
                            results[mode]['skipped'] += 1
                        elif outcome != 'timeout':
                            results[mode]['trades'] += 1
                            if outcome == 'win':
                                results[mode]['wins'] += 1
    
    # Print results
    print("\n" + "="*60)
    print("📊 RESULTS COMPARISON")
    print("="*60)
    
    for mode, data in results.items():
        trades = data['trades']
        wins = data['wins']
        skipped = data['skipped']
        wr = (wins / trades * 100) if trades > 0 else 0
        
        print(f"\n{mode.upper()} MODE:")
        print(f"  Total Signals: {data['total_signals']}")
        print(f"  Skipped: {skipped}")
        print(f"  Trades Executed: {trades}")
        print(f"  Wins: {wins}")
        print(f"  Win Rate: {wr:.1f}%")
    
    # Comparison summary
    print("\n" + "="*60)
    print("📈 SUMMARY")
    print("="*60)
    
    skip_wr = (results['skip']['wins'] / results['skip']['trades'] * 100) if results['skip']['trades'] > 0 else 0
    force_wr = (results['force']['wins'] / results['force']['trades'] * 100) if results['force']['trades'] > 0 else 0
    
    additional_trades = results['force']['trades'] - results['skip']['trades']
    
    print(f"\nSKIP mode: {results['skip']['trades']} trades @ {skip_wr:.1f}% WR")
    print(f"FORCE mode: {results['force']['trades']} trades @ {force_wr:.1f}% WR")
    print(f"\nForce adds: {additional_trades} trades")
    print(f"WR difference: {force_wr - skip_wr:+.1f}%")
    
    if force_wr >= skip_wr:
        print("\n✅ FORCE mode maintains or improves WR!")
    else:
        print(f"\n⚠️ FORCE mode reduces WR by {skip_wr - force_wr:.1f}%")

if __name__ == "__main__":
    run_comparison()
