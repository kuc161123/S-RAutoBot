#!/usr/bin/env python3
"""
Comprehensive Combo Discovery Backtest

Uses 60 days of data on top 20 symbols with walk-forward validation
to discover truly winning combos that hold up over time.
"""

import requests
import pandas as pd
import numpy as np
import time
import math
from datetime import datetime, timedelta
from collections import defaultdict

BYBIT_BASE = "https://api.bybit.com"

# Top 20 most liquid symbols
TOP_20_SYMBOLS = [
    'BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'XRPUSDT', 'DOGEUSDT',
    'ADAUSDT', 'AVAXUSDT', 'DOTUSDT', 'LINKUSDT', 'LTCUSDT',
    'UNIUSDT', 'ATOMUSDT', 'NEARUSDT', 'APTUSDT', 'ARBUSDT',
    'OPUSDT', 'SUIUSDT', 'INJUSDT', 'SEIUSDT', 'ORDIUSDT',
]

def wilson_lower_bound(wins: int, total: int, z: float = 1.96) -> float:
    if total == 0:
        return 0.0
    p = wins / total
    denominator = 1 + z*z / total
    centre = p + z*z / (2*total)
    spread = z * math.sqrt((p*(1-p) + z*z/(4*total)) / total)
    lower = (centre - spread) / denominator
    return max(0, lower * 100)

def bayesian_wr(wins: int, total: int, prior_wins: int = 4, prior_total: int = 10) -> float:
    """Bayesian win rate with skeptical prior (40% baseline)."""
    return (wins + prior_wins) / (total + prior_total) * 100

def fetch_klines(symbol: str, interval: str = '3', days: int = 60) -> pd.DataFrame:
    all_data = []
    end_time = int(datetime.utcnow().timestamp() * 1000)
    start_time = int((datetime.utcnow() - timedelta(days=days)).timestamp() * 1000)
    
    url = f"{BYBIT_BASE}/v5/market/kline"
    current_end = end_time
    
    while current_end > start_time:
        params = {
            'category': 'linear',
            'symbol': symbol,
            'interval': interval,
            'limit': 1000,
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
            time.sleep(0.05)
        except:
            break
    
    if not all_data:
        return pd.DataFrame()
    
    df = pd.DataFrame(all_data, columns=['start', 'open', 'high', 'low', 'close', 'volume', 'turnover'])
    df = df.astype({'start': int, 'open': float, 'high': float, 'low': float, 'close': float})
    df['start'] = pd.to_datetime(df['start'], unit='ms')
    df = df.sort_values('start').drop_duplicates('start').reset_index(drop=True)
    return df

def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if len(df) < 50:
        return df
    
    # RSI
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # MACD
    ema12 = df['close'].ewm(span=12).mean()
    ema26 = df['close'].ewm(span=26).mean()
    df['macd'] = ema12 - ema26
    df['macd_signal'] = df['macd'].ewm(span=9).mean()
    
    # ATR
    high_low = df['high'] - df['low']
    high_close = abs(df['high'] - df['close'].shift())
    low_close = abs(df['low'] - df['close'].shift())
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr'] = true_range.rolling(14).mean()
    
    # Fib
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

def simulate_trade(df, entry_idx, side, entry, atr, rr_ratio=2.0, max_candles=80):
    """Simulate with force minimum SL."""
    MIN_SL_PCT = 0.5
    
    min_sl_dist = entry * (MIN_SL_PCT / 100)
    sl_dist = max(atr, min_sl_dist)
    tp_dist = sl_dist * rr_ratio
    
    if side == 'long':
        sl = entry - sl_dist
        tp = entry + tp_dist
    else:
        sl = entry + sl_dist
        tp = entry - tp_dist
    
    for i in range(entry_idx + 1, min(entry_idx + max_candles, len(df))):
        candle = df.iloc[i]
        if side == 'long':
            if candle['low'] <= sl:
                return 'loss'
            elif candle['high'] >= tp:
                return 'win'
        else:
            if candle['high'] >= sl:
                return 'loss'
            elif candle['low'] <= tp:
                return 'win'
    
    return 'timeout'

def run_backtest():
    print("🔬 Comprehensive Combo Discovery (60 Days, Top 20 Symbols)")
    print("="*70)
    
    # Fetch data
    symbol_data = {}
    print(f"\n📥 Fetching 60-day data for {len(TOP_20_SYMBOLS)} symbols...")
    
    for i, symbol in enumerate(TOP_20_SYMBOLS):
        print(f"   [{i+1}/{len(TOP_20_SYMBOLS)}] {symbol}...", end=' ', flush=True)
        df = fetch_klines(symbol, '3', 60)
        if len(df) > 1000:
            df = calculate_indicators(df)
            symbol_data[symbol] = df
            print(f"✅ {len(df)} candles")
        else:
            print(f"❌ Insufficient data")
        time.sleep(0.3)
    
    print(f"\n✅ Loaded data for {len(symbol_data)} symbols")
    
    # Walk-forward: Train on first 40 days, test on last 20 days
    train_days = 40
    test_days = 20
    
    all_combos = defaultdict(lambda: {
        'train_wins': 0, 'train_losses': 0,
        'test_wins': 0, 'test_losses': 0
    })
    
    print(f"\n📊 Walk-Forward Validation")
    print(f"   Train: First {train_days} days")
    print(f"   Test: Last {test_days} days")
    
    for symbol, df in symbol_data.items():
        # Split by time
        total_candles = len(df)
        train_end = int(total_candles * (train_days / 60))
        
        df_train = df.iloc[:train_end]
        df_test = df.iloc[train_end:]
        
        # Train phase - collect combo stats
        for side in ['long', 'short']:
            for idx in range(50, len(df_train) - 80, 10):  # Sample every 10 candles
                row = df_train.iloc[idx]
                combo = get_combo(row)
                key = f"{symbol}|{side}|{combo}"
                
                outcome = simulate_trade(df_train, idx, side, row['close'], row['atr'])
                if outcome == 'win':
                    all_combos[key]['train_wins'] += 1
                elif outcome == 'loss':
                    all_combos[key]['train_losses'] += 1
        
        # Test phase - validate combos
        for side in ['long', 'short']:
            for idx in range(0, len(df_test) - 80, 10):
                row = df_test.iloc[idx]
                combo = get_combo(row)
                key = f"{symbol}|{side}|{combo}"
                
                outcome = simulate_trade(df_test, idx, side, row['close'], row['atr'])
                if outcome == 'win':
                    all_combos[key]['test_wins'] += 1
                elif outcome == 'loss':
                    all_combos[key]['test_losses'] += 1
    
    # Filter combos that pass criteria
    MIN_TRAIN_TRADES = 20
    MIN_TRAIN_WR = 45
    MIN_TEST_WR = 40
    
    winning_combos = []
    
    for key, stats in all_combos.items():
        train_total = stats['train_wins'] + stats['train_losses']
        test_total = stats['test_wins'] + stats['test_losses']
        
        if train_total < MIN_TRAIN_TRADES or test_total < 5:
            continue
        
        train_wr = stats['train_wins'] / train_total * 100
        test_wr = stats['test_wins'] / test_total * 100 if test_total > 0 else 0
        bayesian = bayesian_wr(stats['train_wins'], train_total)
        
        # Must pass both train and test
        if train_wr >= MIN_TRAIN_WR and test_wr >= MIN_TEST_WR:
            symbol, side, combo = key.split('|')
            winning_combos.append({
                'symbol': symbol,
                'side': side,
                'combo': combo,
                'train_n': train_total,
                'train_wr': train_wr,
                'test_n': test_total,
                'test_wr': test_wr,
                'bayesian_wr': bayesian
            })
    
    # Sort by test WR
    winning_combos.sort(key=lambda x: x['test_wr'], reverse=True)
    
    # Print results
    print("\n" + "="*70)
    print(f"🏆 WINNING COMBOS (Train WR ≥ {MIN_TRAIN_WR}% AND Test WR ≥ {MIN_TEST_WR}%)")
    print("="*70)
    
    if not winning_combos:
        print("\n⚠️ No combos passed both train and test criteria!")
    else:
        print(f"\nFound {len(winning_combos)} reliable combos:\n")
        
        for i, c in enumerate(winning_combos[:30], 1):  # Top 30
            print(f"{i:2}. {c['symbol']} {c['side']} {c['combo']}")
            print(f"    Train: {c['train_wr']:.1f}% (N={c['train_n']}) | Test: {c['test_wr']:.1f}% (N={c['test_n']})")
    
    # Summary by symbol
    print("\n" + "="*70)
    print("📊 SUMMARY BY SYMBOL")
    print("="*70)
    
    symbol_counts = defaultdict(int)
    for c in winning_combos:
        symbol_counts[c['symbol']] += 1
    
    for symbol, count in sorted(symbol_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"   {symbol}: {count} winning combos")
    
    # Overall stats
    print("\n" + "="*70)
    print("📈 OVERALL STATISTICS")
    print("="*70)
    
    print(f"\nTotal combos analyzed: {len(all_combos)}")
    print(f"Combos passing criteria: {len(winning_combos)}")
    print(f"Pass rate: {len(winning_combos)/len(all_combos)*100:.1f}%")
    
    if winning_combos:
        avg_train_wr = sum(c['train_wr'] for c in winning_combos) / len(winning_combos)
        avg_test_wr = sum(c['test_wr'] for c in winning_combos) / len(winning_combos)
        print(f"Avg Train WR: {avg_train_wr:.1f}%")
        print(f"Avg Test WR: {avg_test_wr:.1f}%")
    
    print("\n✅ Backtest complete!")
    
    return winning_combos

if __name__ == "__main__":
    run_backtest()
