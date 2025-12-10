#!/usr/bin/env python3
"""
R:R Comparison Backtest (1:1 vs 2:1)

SIMPLIFIED BINS + 50 SYMBOLS for faster analysis.
This is for COMPARISON ONLY - does not affect the live bot.

Features:
- Simplified indicator bins (fewer categories)
- 50 top symbols only
- Side-by-side 1:1 vs 2:1 R:R comparison
- Walk-forward validation (60% train, 40% test)
"""

import requests
import pandas as pd
import numpy as np
import time
import math
import yaml
from datetime import datetime, timedelta
from collections import defaultdict

BYBIT_BASE = "https://api.bybit.com"

# Execution costs
SPREAD_PCT = 0.01
SLIPPAGE_PCT = 0.02
TOTAL_COST_PCT = SPREAD_PCT + SLIPPAGE_PCT

def wilson_lower_bound(wins: int, total: int, z: float = 1.96) -> float:
    if total == 0:
        return 0.0
    p = wins / total
    denominator = 1 + z*z / total
    centre = p + z*z / (2*total)
    spread = z * math.sqrt((p*(1-p) + z*z/(4*total)) / total)
    lower = (centre - spread) / denominator
    return max(0, lower * 100)

def get_all_symbols(limit=50) -> list:
    """Fetch top 50 symbols by volume."""
    url = f"{BYBIT_BASE}/v5/market/tickers"
    params = {'category': 'linear'}
    
    try:
        resp = requests.get(url, params=params, timeout=10)
        data = resp.json()
        
        if data.get('retCode') != 0:
            return []
        
        tickers = data.get('result', {}).get('list', [])
        usdt_tickers = [t for t in tickers if t['symbol'].endswith('USDT')]
        usdt_tickers.sort(key=lambda x: float(x.get('turnover24h', 0)), reverse=True)
        
        return [t['symbol'] for t in usdt_tickers[:limit]]
    except Exception as e:
        print(f"Error fetching symbols: {e}")
        return []

def fetch_klines(symbol: str, interval: str = '3', days: int = 60) -> pd.DataFrame:
    """Fetch historical klines."""
    all_data = []
    end_time = int(datetime.utcnow().timestamp() * 1000)
    start_time = int((datetime.utcnow() - timedelta(days=days)).timestamp() * 1000)
    
    url = f"{BYBIT_BASE}/v5/market/kline"
    current_end = end_time
    retries = 0
    
    while current_end > start_time:
        params = {
            'category': 'linear',
            'symbol': symbol,
            'interval': interval,
            'limit': 1000,
            'end': current_end
        }
        
        try:
            resp = requests.get(url, params=params, timeout=30)
            data = resp.json()
            
            if data.get('retCode') != 0 or not data.get('result', {}).get('list'):
                retries += 1
                if retries >= 3:
                    break
                time.sleep(0.5)
                continue
                
            klines = data['result']['list']
            all_data.extend(klines)
            earliest = int(klines[-1][0])
            if earliest <= start_time:
                break
            current_end = earliest - 1
            retries = 0
            time.sleep(0.05)
        except Exception as e:
            retries += 1
            if retries >= 3:
                break
            time.sleep(1)
    
    if not all_data:
        return pd.DataFrame()
    
    df = pd.DataFrame(all_data, columns=['start', 'open', 'high', 'low', 'close', 'volume', 'turnover'])
    df = df.astype({'start': int, 'open': float, 'high': float, 'low': float, 'close': float})
    df['start'] = pd.to_datetime(df['start'], unit='ms')
    df = df.sort_values('start').drop_duplicates('start').reset_index(drop=True)
    return df

def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate technical indicators."""
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
    
    # Fib levels
    df['roll_high'] = df['high'].rolling(50).max()
    df['roll_low'] = df['low'].rolling(50).min()
    
    # VWAP
    df['tp'] = (df['high'] + df['low'] + df['close']) / 3
    df['tp_v'] = df['tp'] * df['volume'].astype(float)
    df['cum_tp_v'] = df['tp_v'].cumsum()
    df['cum_v'] = df['volume'].astype(float).cumsum()
    df['vwap'] = df['cum_tp_v'] / df['cum_v']
    
    return df.dropna()

def get_combo_simplified(row) -> str:
    """
    SIMPLIFIED combo string with fewer bins:
    - RSI: 3 bins (oversold, neutral, overbought)
    - MACD: 2 bins (bull, bear)
    - Fib: 3 bins (low, mid, high)
    """
    rsi = row['rsi']
    if rsi < 40:
        r_bin = 'oversold'
    elif rsi > 60:
        r_bin = 'overbought'
    else:
        r_bin = 'neutral'
    
    m_bin = 'bull' if row['macd'] > row['macd_signal'] else 'bear'
    
    high, low, close = row['roll_high'], row['roll_low'], row['close']
    if high == low:
        f_bin = 'low'
    else:
        fib = (high - close) / (high - low) * 100
        if fib < 38:
            f_bin = 'low'
        elif fib < 62:
            f_bin = 'mid'
        else:
            f_bin = 'high'
    
    return f"RSI:{r_bin} MACD:{m_bin} Fib:{f_bin}"

def check_vwap_signal(row, prev_row=None) -> str:
    if prev_row is None:
        return None
    
    if row['low'] <= row['vwap'] and row['close'] > row['vwap']:
        return 'long'
    
    if row['high'] >= row['vwap'] and row['close'] < row['vwap']:
        return 'short'
    
    return None

def simulate_trade(df, entry_idx, side, entry, atr, rr_ratio, max_candles=80):
    """Simulate trade with given R:R ratio."""
    MIN_SL_PCT = 0.5
    min_sl_dist = entry * (MIN_SL_PCT / 100)
    sl_dist = max(atr, min_sl_dist)
    tp_dist = sl_dist * rr_ratio
    
    cost = entry * (TOTAL_COST_PCT / 100)
    
    if side == 'long':
        adjusted_entry = entry + cost
        sl = adjusted_entry - sl_dist
        tp = adjusted_entry + tp_dist
    else:
        adjusted_entry = entry - cost
        sl = adjusted_entry + sl_dist
        tp = adjusted_entry - tp_dist
    
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

def run_comparison_backtest(num_symbols=50, days=60, train_pct=0.6):
    """Run side-by-side comparison of 1:1 vs 2:1 R:R."""
    print("="*70)
    print("🔬 R:R COMPARISON BACKTEST (1:1 vs 2:1)")
    print("   SIMPLIFIED BINS | 50 SYMBOLS | COMPARISON ONLY")
    print("="*70)
    
    # Get symbols
    print(f"\n📋 Fetching top {num_symbols} symbols...")
    symbols = get_all_symbols(num_symbols)
    print(f"   Found {len(symbols)} symbols")
    
    # Fetch data
    symbol_data = {}
    print(f"\n📥 Fetching {days}-day data...")
    
    for i, symbol in enumerate(symbols):
        if (i+1) % 10 == 0 or i == 0:
            print(f"   [{i+1}/{len(symbols)}] {symbol}...", flush=True)
        df = fetch_klines(symbol, '3', days)
        if len(df) > 500:
            df = calculate_indicators(df)
            symbol_data[symbol] = df
        time.sleep(0.05)
    
    print(f"\n✅ Loaded data for {len(symbol_data)} symbols")
    
    # Run both R:R ratios
    rr_ratios = [1.0, 2.0]
    results = {}
    
    for rr in rr_ratios:
        print(f"\n📊 Testing R:R {rr}:1...")
        
        combo_stats = defaultdict(lambda: {
            'train_wins': 0, 'train_losses': 0,
            'test_wins': 0, 'test_losses': 0
        })
        
        total_signals = 0
        for symbol, df in symbol_data.items():
            total_candles = len(df)
            train_end = int(total_candles * train_pct)
            
            df_train = df.iloc[:train_end]
            df_test = df.iloc[train_end:]
            
            for phase, df_phase in [('train', df_train), ('test', df_test)]:
                for idx in range(51, len(df_phase) - 80):
                    row = df_phase.iloc[idx]
                    prev_row = df_phase.iloc[idx - 1]
                    
                    side = check_vwap_signal(row, prev_row)
                    if not side:
                        continue
                    
                    total_signals += 1
                    combo = get_combo_simplified(row)
                    key = f"{symbol}|{side}|{combo}"
                    
                    outcome = simulate_trade(df_phase, idx, side, row['close'], row['atr'], rr)
                    
                    if outcome == 'win':
                        combo_stats[key][f'{phase}_wins'] += 1
                    elif outcome in ['loss', 'timeout']:
                        combo_stats[key][f'{phase}_losses'] += 1
        
        # Filter winning combos
        MIN_TRAIN_TRADES = 10
        MIN_TRAIN_WR = 50
        MIN_TEST_WR = 50
        
        winning_combos = []
        
        for key, stats in combo_stats.items():
            train_total = stats['train_wins'] + stats['train_losses']
            test_total = stats['test_wins'] + stats['test_losses']
            
            if train_total < MIN_TRAIN_TRADES:
                continue
            
            train_wr = stats['train_wins'] / train_total * 100
            test_wr = stats['test_wins'] / test_total * 100 if test_total > 0 else 0
            
            if train_wr >= MIN_TRAIN_WR and test_wr >= MIN_TEST_WR:
                symbol, side, combo = key.split('|')
                winning_combos.append({
                    'symbol': symbol,
                    'side': side,
                    'combo': combo,
                    'train_n': train_total,
                    'train_wr': train_wr,
                    'test_n': test_total,
                    'test_wr': test_wr
                })
        
        results[rr] = {
            'total_signals': total_signals,
            'combos': winning_combos,
            'count': len(winning_combos)
        }
        
        print(f"   Signals: {total_signals} | Valid Combos: {len(winning_combos)}")
    
    # Print comparison
    print("\n" + "="*70)
    print("📈 COMPARISON RESULTS")
    print("="*70)
    
    print(f"\n{'Metric':<30} {'1:1 R:R':<20} {'2:1 R:R':<20}")
    print("-"*70)
    print(f"{'Total Signals':<30} {results[1.0]['total_signals']:<20} {results[2.0]['total_signals']:<20}")
    print(f"{'Valid Combos':<30} {results[1.0]['count']:<20} {results[2.0]['count']:<20}")
    
    # Calculate expected value per trade
    # 1:1 R:R: Win = +1R, Loss = -1R -> EV = WR - (1-WR) = 2*WR - 1
    # 2:1 R:R: Win = +2R, Loss = -1R -> EV = 2*WR - (1-WR) = 3*WR - 1
    
    avg_train_wr_1 = np.mean([c['train_wr'] for c in results[1.0]['combos']]) if results[1.0]['combos'] else 0
    avg_train_wr_2 = np.mean([c['train_wr'] for c in results[2.0]['combos']]) if results[2.0]['combos'] else 0
    
    ev_1 = 2 * (avg_train_wr_1/100) - 1 if avg_train_wr_1 else 0
    ev_2 = 3 * (avg_train_wr_2/100) - 1 if avg_train_wr_2 else 0
    
    print(f"{'Avg Train WR':<30} {avg_train_wr_1:.1f}%{'':<13} {avg_train_wr_2:.1f}%")
    print(f"{'Expected Value (R)':<30} {ev_1:+.2f}R{'':<15} {ev_2:+.2f}R")
    
    # Winner
    print("\n" + "="*70)
    if ev_2 > ev_1:
        print("🏆 WINNER: 2:1 R:R (Higher Expected Value)")
    elif ev_1 > ev_2:
        print("🏆 WINNER: 1:1 R:R (Higher Expected Value)")
    else:
        print("🤝 TIE: Both R:R ratios have similar Expected Value")
    print("="*70)
    
    # Top combos for each
    print("\n📊 TOP 5 COMBOS (1:1 R:R):")
    print("-"*70)
    for i, c in enumerate(sorted(results[1.0]['combos'], key=lambda x: x['train_wr'], reverse=True)[:5], 1):
        print(f"{i}. {c['symbol']} {c['side']} {c['combo']} | Train: {c['train_wr']:.0f}% (N={c['train_n']}) | Test: {c['test_wr']:.0f}%")
    
    print("\n📊 TOP 5 COMBOS (2:1 R:R):")
    print("-"*70)
    for i, c in enumerate(sorted(results[2.0]['combos'], key=lambda x: x['train_wr'], reverse=True)[:5], 1):
        print(f"{i}. {c['symbol']} {c['side']} {c['combo']} | Train: {c['train_wr']:.0f}% (N={c['train_n']}) | Test: {c['test_wr']:.0f}%")
    
    # Save results
    output = {
        '_metadata': {
            'generated': datetime.utcnow().isoformat(),
            'symbols_tested': len(symbol_data),
            'bins': 'simplified (3 RSI, 2 MACD, 3 Fib)',
            'note': 'COMPARISON ONLY - Does not affect live bot'
        },
        'comparison': {
            '1_to_1': {
                'total_signals': results[1.0]['total_signals'],
                'valid_combos': results[1.0]['count'],
                'avg_train_wr': round(avg_train_wr_1, 1),
                'expected_value_r': round(ev_1, 3)
            },
            '2_to_1': {
                'total_signals': results[2.0]['total_signals'],
                'valid_combos': results[2.0]['count'],
                'avg_train_wr': round(avg_train_wr_2, 1),
                'expected_value_r': round(ev_2, 3)
            },
            'winner': '2:1' if ev_2 > ev_1 else ('1:1' if ev_1 > ev_2 else 'tie')
        }
    }
    
    with open('rr_comparison_RESULTS.yaml', 'w') as f:
        yaml.dump(output, f, default_flow_style=False, sort_keys=False)
    
    print(f"\n✅ Results saved to: rr_comparison_RESULTS.yaml")
    print("⚠️ NOTE: This is for comparison only. Bot settings unchanged.")
    
    return results

if __name__ == "__main__":
    run_comparison_backtest(num_symbols=50, days=60, train_pct=0.6)
