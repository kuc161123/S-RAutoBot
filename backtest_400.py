#!/usr/bin/env python3
"""
400-Symbol Comprehensive Backtest

Fetches actual symbols from Bybit, runs walk-forward validation,
and exports winning combos to backtest_golden_combos.yaml
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

def get_all_symbols(limit=400) -> list:
    """Fetch top symbols from Bybit by volume."""
    url = f"{BYBIT_BASE}/v5/market/tickers"
    params = {'category': 'linear'}
    
    try:
        resp = requests.get(url, params=params, timeout=10)
        data = resp.json()
        
        if data.get('retCode') != 0:
            return []
        
        tickers = data.get('result', {}).get('list', [])
        
        # Filter USDT perpetuals and sort by volume
        usdt_tickers = [t for t in tickers if t['symbol'].endswith('USDT')]
        usdt_tickers.sort(key=lambda x: float(x.get('turnover24h', 0)), reverse=True)
        
        return [t['symbol'] for t in usdt_tickers[:limit]]
    except Exception as e:
        print(f"Error fetching symbols: {e}")
        return []

def fetch_klines(symbol: str, interval: str = '3', days: int = 30) -> pd.DataFrame:
    all_data = []
    end_time = int(datetime.utcnow().timestamp() * 1000)
    start_time = int((datetime.utcnow() - timedelta(days=days)).timestamp() * 1000)
    
    url = f"{BYBIT_BASE}/v5/market/kline"
    current_end = end_time
    retries = 0
    max_retries = 3
    
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
                if retries >= max_retries:
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
            if retries >= max_retries:
                print(f"Failed to fetch {symbol}: {e}", flush=True)
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

def simulate_trade(df, entry_idx, side, entry, atr, rr_ratio=2.0, max_candles=80):
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

def run_backtest(num_symbols=100, days=30, train_pct=0.6):
    print(f"🔬 {num_symbols}-Symbol Comprehensive Backtest")
    print("="*70)
    
    # Get symbols
    print(f"\n📋 Fetching top {num_symbols} symbols...")
    symbols = get_all_symbols(num_symbols)
    print(f"   Found {len(symbols)} symbols")
    
    # Fetch data
    symbol_data = {}
    print(f"\n📥 Fetching {days}-day data...")
    
    for i, symbol in enumerate(symbols):
        if (i+1) % 20 == 0 or i == 0:
            print(f"   [{i+1}/{len(symbols)}] {symbol}...", flush=True)
        df = fetch_klines(symbol, '3', days)
        if len(df) > 500:
            df = calculate_indicators(df)
            symbol_data[symbol] = df
        time.sleep(0.05)
    
    print(f"\n✅ Loaded data for {len(symbol_data)} symbols")
    
    # Walk-forward split
    combo_stats = defaultdict(lambda: {
        'train_wins': 0, 'train_losses': 0,
        'test_wins': 0, 'test_losses': 0
    })
    
    print(f"\n📊 Walk-Forward Validation (Train: {train_pct*100:.0f}%, Test: {(1-train_pct)*100:.0f}%)")
    
    for symbol, df in symbol_data.items():
        total_candles = len(df)
        train_end = int(total_candles * train_pct)
        
        df_train = df.iloc[:train_end]
        df_test = df.iloc[train_end:]
        
        for phase, df_phase in [('train', df_train), ('test', df_test)]:
            for side in ['long', 'short']:
                for idx in range(50, len(df_phase) - 80, 20):
                    row = df_phase.iloc[idx]
                    combo = get_combo(row)
                    key = f"{symbol}|{side}|{combo}"
                    
                    outcome = simulate_trade(df_phase, idx, side, row['close'], row['atr'])
                    if outcome == 'win':
                        combo_stats[key][f'{phase}_wins'] += 1
                    elif outcome == 'loss':
                        combo_stats[key][f'{phase}_losses'] += 1
    
    # Filter winning combos - User requested 50%+ WR
    MIN_TRAIN_TRADES = 10
    MIN_TRAIN_WR = 50  # 50% minimum on training
    MIN_TEST_WR = 50   # 50% minimum on testing
    
    winning_combos = []
    
    for key, stats in combo_stats.items():
        train_total = stats['train_wins'] + stats['train_losses']
        test_total = stats['test_wins'] + stats['test_losses']
        
        if train_total < MIN_TRAIN_TRADES or test_total < 3:
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
    
    winning_combos.sort(key=lambda x: (x['test_wr'], x['train_wr']), reverse=True)
    
    # Print results
    print("\n" + "="*70)
    print(f"🏆 WINNING COMBOS (Train WR ≥ {MIN_TRAIN_WR}% AND Test WR ≥ {MIN_TEST_WR}%)")
    print("="*70)
    print(f"\nFound {len(winning_combos)} reliable combos")
    
    for i, c in enumerate(winning_combos[:30], 1):
        print(f"{i:2}. {c['symbol']} {c['side']} {c['combo']}")
        print(f"    Train: {c['train_wr']:.1f}% (N={c['train_n']}) | Test: {c['test_wr']:.1f}% (N={c['test_n']})")
    
    # Export to YAML
    yaml_output = {}
    for c in winning_combos:
        sym = c['symbol']
        side = c['side']
        combo = c['combo']
        
        if sym not in yaml_output:
            yaml_output[sym] = {'allowed_combos_long': [], 'allowed_combos_short': []}
        
        if side == 'long':
            if combo not in yaml_output[sym]['allowed_combos_long']:
                yaml_output[sym]['allowed_combos_long'].append(combo)
        else:
            if combo not in yaml_output[sym]['allowed_combos_short']:
                yaml_output[sym]['allowed_combos_short'].append(combo)
    
    # Save to PENDING file (not the active bot file)
    output_file = 'backtest_golden_combos_PENDING.yaml'
    with open(output_file, 'w') as f:
        f.write("# Backtest-Validated Golden Combos (PENDING REVIEW)\n")
        f.write(f"# Generated: {datetime.utcnow().isoformat()}\n")
        f.write(f"# Symbols tested: {len(symbol_data)}\n")
        f.write(f"# Combos found: {len(winning_combos)}\n")
        f.write(f"# Criteria: Train WR >= {MIN_TRAIN_WR}%, Test WR >= {MIN_TEST_WR}%, Min N >= {MIN_TRAIN_TRADES}\n")
        f.write("# \n")
        f.write("# To apply: cp backtest_golden_combos_PENDING.yaml backtest_golden_combos.yaml\n\n")
        yaml.dump(yaml_output, f, default_flow_style=False)
    
    print(f"\n✅ Exported {len(yaml_output)} symbols to {output_file}")
    print(f"   Total combos: {len(winning_combos)}")
    print(f"\n📌 To apply these results to the bot:")
    print(f"   cp {output_file} backtest_golden_combos.yaml")
    
    return winning_combos

if __name__ == "__main__":
    # Full 400-symbol backtest with 120 days of data
    # 50%+ WR threshold on both train and test
    run_backtest(num_symbols=400, days=120, train_pct=0.6)
