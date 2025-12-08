#!/usr/bin/env python3
"""
Backtest the specifically promoted combos from analytics.
Uses force minimum SL and simulates on historical data.
"""

import requests
import pandas as pd
import numpy as np
import time
import math
from datetime import datetime, timedelta

BYBIT_BASE = "https://api.bybit.com"

# Promoted combos from your analytics (symbol, side, combo_str)
PROMOTED_COMBOS = [
    ('BLASTUSDT', 'short', 'RSI:40-60 MACD:bull Fib:50-61', 77),
    ('AAVEUSDT', 'long', 'RSI:40-60 MACD:bull Fib:38-50', 76),
    ('10000ELONUSDT', 'long', 'RSI:40-60 MACD:bull Fib:38-50', 74),
    ('GPSUSDT', 'long', 'RSI:40-60 MACD:bear Fib:50-61', 74),
    ('GRTUSDT', 'long', 'RSI:40-60 MACD:bull Fib:38-50', 72),
    ('API3USDT', 'long', 'RSI:40-60 MACD:bull Fib:23-38', 72),
    ('RDNTUSDT', 'long', 'RSI:40-60 MACD:bull Fib:0-23', 72),
    ('LUNA2USDT', 'long', 'RSI:40-60 MACD:bull Fib:50-61', 72),
    ('MANTAUSDT', 'long', 'RSI:40-60 MACD:bear Fib:61-78', 72),
    ('GMTUSDT', 'long', 'RSI:60-70 MACD:bull Fib:38-50', 72),
    ('ORDIUSDT', 'long', 'RSI:40-60 MACD:bull Fib:61-78', 70),
    ('PROVEUSDT', 'long', 'RSI:40-60 MACD:bear Fib:50-61', 70),
    ('MBOXUSDT', 'long', 'RSI:40-60 MACD:bear Fib:50-61', 70),
    ('1000000MOGUSDT', 'long', 'RSI:40-60 MACD:bull Fib:23-38', 70),
    ('ORDERUSDT', 'long', 'RSI:40-60 MACD:bear Fib:38-50', 70),
    ('NFPUSDT', 'long', 'RSI:40-60 MACD:bear Fib:38-50', 70),
    ('POLYXUSDT', 'long', 'RSI:70+ MACD:bull Fib:0-23', 70),
    ('GLMUSDT', 'long', 'RSI:40-60 MACD:bull Fib:61-78', 68),
    ('POLUSDT', 'long', 'RSI:40-60 MACD:bear Fib:38-50', 65),
    ('AEROUSDT', 'long', 'RSI:40-60 MACD:bear Fib:23-38', 65),
    ('POPCATUSDT', 'long', 'RSI:40-60 MACD:bull Fib:50-61', 62),
    ('BANANAUSDT', 'long', 'RSI:40-60 MACD:bull Fib:50-61', 62),
    ('ACXUSDT', 'long', 'RSI:40-60 MACD:bull Fib:38-50', 62),
    ('DOGEUSDT', 'long', 'RSI:60-70 MACD:bull Fib:0-23', 61),
    ('ALPINEUSDT', 'short', 'RSI:40-60 MACD:bear Fib:38-50', 61),
    ('MOODENGUSDT', 'long', 'RSI:40-60 MACD:bull Fib:23-38', 60),
    ('LABUSDT', 'long', 'RSI:40-60 MACD:bear Fib:61-78', 60),
    ('CHILLGUYUSDT', 'long', 'RSI:40-60 MACD:bear Fib:38-50', 58),
    ('OPUSDT', 'long', 'RSI:40-60 MACD:bull Fib:50-61', 57),
    ('AERGOUSDT', 'long', 'RSI:60-70 MACD:bull Fib:0-23', 57),
    ('ADAUSDT', 'long', 'RSI:60-70 MACD:bull Fib:0-23', 57),
    ('CVXUSDT', 'long', 'RSI:40-60 MACD:bear Fib:23-38', 56),
    ('PHBUSDT', 'long', 'RSI:40-60 MACD:bull Fib:61-78', 55),
    ('MTLUSDT', 'long', 'RSI:40-60 MACD:bull Fib:38-50', 55),
    ('DOODUSDT', 'long', 'RSI:40-60 MACD:bear Fib:38-50', 53),
    ('OPENUSDT', 'long', 'RSI:40-60 MACD:bull Fib:0-23', 53),
    ('ALGOUSDT', 'long', 'RSI:40-60 MACD:bull Fib:23-38', 52),
    ('CLOUSDT', 'long', 'RSI:40-60 MACD:bear Fib:61-78', 50),
    ('ICXUSDT', 'long', 'RSI:40-60 MACD:bull Fib:23-38', 50),
    ('PUMPFUNUSDT', 'short', 'RSI:60-70 MACD:bull Fib:0-23', 49),
    ('HNTUSDT', 'long', 'RSI:40-60 MACD:bear Fib:0-23', 49),
    ('MOVEUSDT', 'long', 'RSI:40-60 MACD:bear Fib:23-38', 45),
    ('MOVRUSDT', 'long', 'RSI:40-60 MACD:bear Fib:0-23', 45),
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

def fetch_klines(symbol: str, interval: str = '3', days: int = 14) -> pd.DataFrame:
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
            time.sleep(0.1)
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

def simulate_trade_force_sl(df, entry_idx, side, atr, entry, rr_ratio=2.0, max_candles=80):
    """Simulate with force minimum SL."""
    MIN_SL_PCT = 0.5
    MIN_TP_PCT = 1.0
    
    min_sl_dist = entry * (MIN_SL_PCT / 100)
    min_tp_dist = entry * (MIN_TP_PCT / 100)
    
    sl_dist = max(atr, min_sl_dist)
    tp_dist = max(rr_ratio * atr, min_tp_dist)
    
    if side == 'long':
        sl = entry - sl_dist
        tp = entry + tp_dist
    else:
        sl = entry + sl_dist
        tp = entry - tp_dist
    
    for i in range(entry_idx + 1, min(entry_idx + max_candles, len(df))):
        candle = df.iloc[i]
        high = candle['high']
        low = candle['low']
        
        if side == 'long':
            if low <= sl:
                return 'loss'
            elif high >= tp:
                return 'win'
        else:
            if high >= sl:
                return 'loss'
            elif low <= tp:
                return 'win'
    
    return 'timeout'

def run_backtest():
    print("🔬 Backtest on 43 Promoted Combos")
    print("="*60)
    
    # Group by symbol
    symbols = list(set([c[0] for c in PROMOTED_COMBOS]))
    print(f"Symbols: {len(symbols)}")
    
    # Fetch data
    symbol_data = {}
    print(f"\n📥 Fetching 14-day data for {len(symbols)} symbols...")
    
    for i, symbol in enumerate(symbols):
        print(f"   [{i+1}/{len(symbols)}] {symbol}...", end=' ')
        df = fetch_klines(symbol, '3', 14)
        if len(df) > 100:
            df = calculate_indicators(df)
            symbol_data[symbol] = df
            print(f"✅ {len(df)} candles")
        else:
            print(f"❌ No data")
        time.sleep(0.2)
    
    print(f"\n✅ Loaded data for {len(symbol_data)} symbols")
    
    # Run simulations
    results = {'total': 0, 'wins': 0, 'losses': 0, 'timeouts': 0, 'signals': 0}
    combo_results = {}
    
    print(f"\n🧪 Simulating trades on promoted combos...")
    
    for symbol, target_side, target_combo, expected_wr in PROMOTED_COMBOS:
        if symbol not in symbol_data:
            print(f"   ⚠️ {symbol}: No data")
            continue
        
        df = symbol_data[symbol]
        signals = 0
        wins = 0
        losses = 0
        
        # Find signals matching this combo
        for idx in range(50, len(df) - 80):
            row = df.iloc[idx]
            combo = get_combo(row)
            
            if combo == target_combo:
                signals += 1
                entry = row['close']
                atr = row['atr']
                
                outcome = simulate_trade_force_sl(df, idx, target_side, atr, entry)
                
                results['signals'] += 1
                
                if outcome == 'win':
                    wins += 1
                    results['wins'] += 1
                elif outcome == 'loss':
                    losses += 1
                    results['losses'] += 1
                else:
                    results['timeouts'] += 1
        
        if signals > 0:
            actual_wr = wins / (wins + losses) * 100 if (wins + losses) > 0 else 0
            combo_results[f"{symbol} {target_side} {target_combo}"] = {
                'signals': signals,
                'wins': wins,
                'losses': losses,
                'expected_wr': expected_wr,
                'actual_wr': actual_wr
            }
    
    # Print results
    print("\n" + "="*60)
    print("📊 RESULTS BY COMBO")
    print("="*60)
    
    for combo, data in sorted(combo_results.items(), key=lambda x: x[1]['actual_wr'], reverse=True):
        if data['wins'] + data['losses'] > 0:
            print(f"\n{combo}")
            print(f"   Signals: {data['signals']}, W/L: {data['wins']}/{data['losses']}")
            print(f"   Expected: {data['expected_wr']}% | Actual: {data['actual_wr']:.1f}%")
    
    print("\n" + "="*60)
    print("📈 OVERALL SUMMARY")
    print("="*60)
    
    total_trades = results['wins'] + results['losses']
    if total_trades > 0:
        overall_wr = results['wins'] / total_trades * 100
        lb_wr = wilson_lower_bound(results['wins'], total_trades)
        
        print(f"\nTotal Signals Found: {results['signals']}")
        print(f"Trades Completed: {total_trades}")
        print(f"Timeouts: {results['timeouts']}")
        print(f"Wins: {results['wins']}, Losses: {results['losses']}")
        print(f"Win Rate: {overall_wr:.1f}%")
        print(f"LB Win Rate (95% CI): {lb_wr:.1f}%")
    
    print("\n✅ Backtest complete!")

if __name__ == "__main__":
    run_backtest()
