#!/usr/bin/env python3
"""
Comprehensive 60-Day Backtest for Composite Scoring

Fetches actual 3m candle data from Bybit, generates signals using combo logic,
simulates trades (TP/SL hits), and compares execution paths.
"""

import requests
import pandas as pd
import numpy as np
import time
import math
from datetime import datetime, timedelta
from collections import defaultdict

# Bybit public API
BYBIT_BASE = "https://api.bybit.com"

# Top symbols to test (reduce from 400 to speed up)
TOP_SYMBOLS = [
    'BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'XRPUSDT', 'DOGEUSDT',
    'ADAUSDT', 'AVAXUSDT', 'DOTUSDT', 'LINKUSDT', 'MATICUSDT',
    'LTCUSDT', 'UNIUSDT', 'ATOMUSDT', 'NEARUSDT', 'APTUSDT',
    'ARBUSDT', 'OPUSDT', 'SUIUSDT', 'SEIUSDT', 'INJUSDT',
    'FTMUSDT', 'SANDUSDT', 'MANAUSDT', 'GALAUSDT', 'AXSUSDT',
    'AAVEUSDT', 'MKRUSDT', 'CRVUSDT', 'LDOUSDT', 'RNDRUSDT',
]

def wilson_lower_bound(wins: int, total: int, z: float = 1.96) -> float:
    """Calculate Wilson score lower bound (95% confidence)."""
    if total == 0:
        return 0.0
    p = wins / total
    denominator = 1 + z*z / total
    centre = p + z*z / (2*total)
    spread = z * math.sqrt((p*(1-p) + z*z/(4*total)) / total)
    lower = (centre - spread) / denominator
    return max(0, lower * 100)

def fetch_klines(symbol: str, interval: str = '3', days: int = 30) -> pd.DataFrame:
    """Fetch klines from Bybit."""
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
            
            # Get the earliest timestamp
            earliest = int(klines[-1][0])
            if earliest <= start_time:
                break
            current_end = earliest - 1
            
            time.sleep(0.1)  # Rate limit
            
        except Exception as e:
            print(f"Error fetching {symbol}: {e}")
            break
    
    if not all_data:
        return pd.DataFrame()
    
    # Convert to DataFrame
    df = pd.DataFrame(all_data, columns=['start', 'open', 'high', 'low', 'close', 'volume', 'turnover'])
    df = df.astype({
        'start': int,
        'open': float,
        'high': float,
        'low': float,
        'close': float,
        'volume': float,
        'turnover': float
    })
    df['start'] = pd.to_datetime(df['start'], unit='ms')
    df = df.sort_values('start').drop_duplicates('start').reset_index(drop=True)
    
    return df

def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate RSI, MACD, and Fib levels."""
    if len(df) < 50:
        return df
    
    # RSI (14)
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
    
    # ATR (14)
    high_low = df['high'] - df['low']
    high_close = abs(df['high'] - df['close'].shift())
    low_close = abs(df['low'] - df['close'].shift())
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr'] = true_range.rolling(14).mean()
    
    # Rolling high/low for Fib
    df['roll_high'] = df['high'].rolling(50).max()
    df['roll_low'] = df['low'].rolling(50).min()
    
    return df.dropna()

def get_combo(row) -> str:
    """Generate combo string from row."""
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

def detect_signal(row, prev_row) -> tuple:
    """Detect long/short signal based on combo change."""
    if prev_row is None:
        return None, None
    
    combo = get_combo(row)
    prev_combo = get_combo(prev_row)
    
    # Signal: new combo formed (different from previous)
    if combo != prev_combo:
        # Long signals: RSI oversold or Fib high
        rsi = row['rsi']
        if rsi < 40 or 'Fib:61-78' in combo or 'Fib:78-100' in combo:
            return 'long', combo
        # Short signals: RSI overbought or Fib low
        elif rsi > 60 or 'Fib:0-23' in combo or 'Fib:23-38' in combo:
            return 'short', combo
    
    return None, None

def simulate_trade(df: pd.DataFrame, entry_idx: int, side: str, atr: float, rr_ratio: float = 2.0, max_candles: int = 80) -> str:
    """Simulate trade outcome based on TP/SL hit."""
    entry = df.iloc[entry_idx]['close']
    
    if side == 'long':
        sl = entry - atr
        tp = entry + (rr_ratio * atr)
    else:
        sl = entry + atr
        tp = entry - (rr_ratio * atr)
    
    # Check future candles for TP/SL hit
    for i in range(entry_idx + 1, min(entry_idx + max_candles, len(df))):
        candle = df.iloc[i]
        high = candle['high']
        low = candle['low']
        
        if side == 'long':
            if low <= sl:
                return 'loss'  # SL hit first
            elif high >= tp:
                return 'win'
        else:
            if high >= sl:
                return 'loss'  # SL hit first
            elif low <= tp:
                return 'win'
    
    return 'timeout'  # No TP/SL hit within max_candles

def get_session(hour: int) -> str:
    """Get trading session from hour UTC."""
    if 0 <= hour < 8:
        return 'asia'
    elif 8 <= hour < 16:
        return 'london'
    else:
        return 'newyork'

def calculate_composite_score(combo_wr, symbol_wr, session_wr, btc_aligned, vol_wr):
    """Calculate composite score."""
    W_COMBO = 0.40
    W_SYMBOL = 0.25
    W_SESSION = 0.15
    W_BTC = 0.10
    W_VOL = 0.10
    
    combo_wr = combo_wr if combo_wr is not None else 40
    symbol_wr = symbol_wr if symbol_wr is not None else 40
    session_wr = session_wr if session_wr is not None else 50
    vol_wr = vol_wr if vol_wr is not None else 50
    btc_score = 100 if btc_aligned else 0
    
    return (
        W_COMBO * combo_wr +
        W_SYMBOL * symbol_wr +
        W_SESSION * session_wr +
        W_BTC * btc_score +
        W_VOL * vol_wr
    )

def run_backtest(days: int = 30, train_days: int = 14):
    """Run comprehensive backtest."""
    print(f"🔬 Comprehensive {days}-Day Backtest")
    print("="*60)
    print(f"Symbols: {len(TOP_SYMBOLS)}")
    print(f"Train period: {train_days} days")
    print(f"Test period: {days - train_days} days")
    
    # Fetch data for all symbols
    symbol_data = {}
    print(f"\n📥 Fetching candle data...")
    
    for i, symbol in enumerate(TOP_SYMBOLS):
        print(f"   [{i+1}/{len(TOP_SYMBOLS)}] {symbol}...", end=' ')
        df = fetch_klines(symbol, '3', days)
        if len(df) > 100:
            df = calculate_indicators(df)
            symbol_data[symbol] = df
            print(f"✅ {len(df)} candles")
        else:
            print(f"❌ Insufficient data")
        time.sleep(0.2)
    
    print(f"\n✅ Loaded data for {len(symbol_data)} symbols")
    
    # Split into train/test periods
    cutoff_date = datetime.utcnow() - timedelta(days=days - train_days)
    
    # Collect training statistics
    print(f"\n📊 Training on first {train_days} days...")
    combo_stats = defaultdict(lambda: {'wins': 0, 'total': 0})
    symbol_stats = defaultdict(lambda: {'wins': 0, 'total': 0})
    session_stats = defaultdict(lambda: {'wins': 0, 'total': 0})
    
    # Generate training signals
    train_count = 0
    for symbol, df in symbol_data.items():
        df_train = df[df['start'] < cutoff_date]
        if len(df_train) < 100:
            continue
            
        prev_row = None
        for idx, row in df_train.iterrows():
            side, combo = detect_signal(row, prev_row)
            prev_row = row
            
            if side and combo:
                # Check if we have enough future data
                future_idx = df_train.index.get_loc(idx)
                if future_idx + 40 > len(df_train):
                    continue
                
                outcome = simulate_trade(df_train, future_idx, side, row['atr'])
                if outcome == 'timeout':
                    continue
                
                train_count += 1
                key = f"{symbol}:{side}:{combo}"
                combo_stats[key]['total'] += 1
                symbol_stats[symbol]['total'] += 1
                session = get_session(row['start'].hour)
                session_stats[session]['total'] += 1
                
                if outcome == 'win':
                    combo_stats[key]['wins'] += 1
                    symbol_stats[symbol]['wins'] += 1
                    session_stats[session]['wins'] += 1
    
    print(f"   Training signals: {train_count}")
    
    # Test on remaining period
    print(f"\n🧪 Testing on last {days - train_days} days...")
    
    results = {
        'combo_only': {'trades': 0, 'wins': 0},
        'composite': {'trades': 0, 'wins': 0},
        'combined': {'trades': 0, 'wins': 0},
    }
    
    test_count = 0
    for symbol, df in symbol_data.items():
        df_test = df[df['start'] >= cutoff_date]
        if len(df_test) < 50:
            continue
        
        prev_row = None
        for idx, row in df_test.iterrows():
            side, combo = detect_signal(row, prev_row)
            prev_row = row
            
            if side and combo:
                future_idx = df_test.index.get_loc(idx)
                if future_idx + 40 > len(df_test):
                    continue
                
                outcome = simulate_trade(df_test, future_idx, side, row['atr'])
                if outcome == 'timeout':
                    continue
                
                test_count += 1
                
                # Calculate stats from training data
                key = f"{symbol}:{side}:{combo}"
                combo_data = combo_stats.get(key, {'wins': 0, 'total': 0})
                combo_wr = wilson_lower_bound(combo_data['wins'], combo_data['total']) if combo_data['total'] >= 5 else None
                
                symbol_data_stats = symbol_stats.get(symbol, {'wins': 0, 'total': 0})
                symbol_wr = wilson_lower_bound(symbol_data_stats['wins'], symbol_data_stats['total']) if symbol_data_stats['total'] >= 5 else None
                
                session = get_session(row['start'].hour)
                session_data = session_stats.get(session, {'wins': 0, 'total': 0})
                session_wr = wilson_lower_bound(session_data['wins'], session_data['total']) if session_data['total'] >= 5 else None
                
                # Path 1: Combo-only (LB WR >= 42%)
                combo_pass = combo_wr is not None and combo_wr >= 42
                
                # Path 2: Composite Score (>= 50)
                comp_score = calculate_composite_score(combo_wr, symbol_wr, session_wr, True, 50)
                composite_pass = comp_score >= 50
                
                # Record results
                if combo_pass:
                    results['combo_only']['trades'] += 1
                    if outcome == 'win':
                        results['combo_only']['wins'] += 1
                
                if composite_pass:
                    results['composite']['trades'] += 1
                    if outcome == 'win':
                        results['composite']['wins'] += 1
                
                if combo_pass or composite_pass:
                    results['combined']['trades'] += 1
                    if outcome == 'win':
                        results['combined']['wins'] += 1
    
    print(f"   Test signals: {test_count}")
    
    # Print results
    print("\n" + "="*60)
    print("📊 BACKTEST RESULTS")
    print("="*60)
    
    for path_name, data in results.items():
        trades = data['trades']
        wins = data['wins']
        wr = (wins / trades * 100) if trades > 0 else 0
        lb_wr = wilson_lower_bound(wins, trades) if trades > 0 else 0
        
        print(f"\n{path_name.upper()}:")
        print(f"  Trades: {trades}")
        print(f"  Wins: {wins}")
        print(f"  Win Rate: {wr:.1f}%")
        print(f"  LB Win Rate: {lb_wr:.1f}%")
    
    # Comparison
    print("\n" + "="*60)
    print("📈 COMPARISON")
    print("="*60)
    
    combo_trades = results['combo_only']['trades']
    combined_trades = results['combined']['trades']
    
    if combo_trades > 0:
        additional = combined_trades - combo_trades
        print(f"\n✅ Composite adds {additional} additional trades (+{additional/combo_trades*100:.0f}%)")
    
    print("\n✅ Backtest complete!")

if __name__ == "__main__":
    run_backtest(days=30, train_days=14)
