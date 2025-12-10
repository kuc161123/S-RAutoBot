#!/usr/bin/env python3
"""
1:1 R:R Backtest with ORIGINAL 70 Indicator Bins (400 Symbols)

Using the ORIGINAL indicator bins (5 RSI x 2 MACD x 7 Fib = 70 combos)
to compare against simplified 18-combo results.

Features:
- Walk-forward validation (60% train, 40% test)
- Realistic fees (0.055% maker + 0.055% taker = 0.11% total)
- Spread + slippage simulation
- 1:1 R:R ratio
- ORIGINAL 70 indicator bins (same as golden combos)
- Filter for WR >= 50% in BOTH train and test
- Min 10 trades in train, 3 in test

COMPARISON ONLY - Does not affect live bot.
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

# REALISTIC EXECUTION COSTS
MAKER_FEE_PCT = 0.055
TAKER_FEE_PCT = 0.055
TOTAL_FEE_PCT = MAKER_FEE_PCT + TAKER_FEE_PCT
SPREAD_PCT = 0.01
SLIPPAGE_PCT = 0.02
TOTAL_COST_PCT = TOTAL_FEE_PCT + SPREAD_PCT + SLIPPAGE_PCT

# R:R RATIO - 1:1 for comparison
RR_RATIO = 1.0

# MINIMUM REQUIREMENTS
MIN_TRAIN_TRADES = 10
MIN_TEST_TRADES = 3
MIN_TRAIN_WR = 50
MIN_TEST_WR = 50

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
        except:
            retries += 1
            if retries >= max_retries:
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
    
    df['tp'] = (df['high'] + df['low'] + df['close']) / 3
    df['tp_v'] = df['tp'] * df['volume'].astype(float)
    df['cum_tp_v'] = df['tp_v'].cumsum()
    df['cum_v'] = df['volume'].astype(float).cumsum()
    df['vwap'] = df['cum_tp_v'] / df['cum_v']
    
    return df.dropna()

def get_combo_original(row) -> str:
    """
    ORIGINAL 70-combo bins (5 RSI x 2 MACD x 7 Fib)
    Same as the original golden combos backtest.
    """
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

def check_vwap_signal(row, prev_row=None) -> str:
    if prev_row is None:
        return None
    
    if row['low'] <= row['vwap'] and row['close'] > row['vwap']:
        return 'long'
    
    if row['high'] >= row['vwap'] and row['close'] < row['vwap']:
        return 'short'
    
    return None

def simulate_trade_realistic(df, entry_idx, side, entry, atr, max_candles=80):
    MIN_SL_PCT = 0.5
    min_sl_dist = entry * (MIN_SL_PCT / 100)
    sl_dist = max(atr, min_sl_dist)
    tp_dist = sl_dist * RR_RATIO  # 1:1 R:R
    
    entry_cost = entry * (TOTAL_COST_PCT / 100)
    
    if side == 'long':
        adjusted_entry = entry + entry_cost
        sl = adjusted_entry - sl_dist
        tp = adjusted_entry + tp_dist
    else:
        adjusted_entry = entry - entry_cost
        sl = adjusted_entry + sl_dist
        tp = adjusted_entry - tp_dist
    
    for i in range(entry_idx + 1, min(entry_idx + max_candles, len(df))):
        candle = df.iloc[i]
        
        if side == 'long':
            if candle['low'] <= sl:
                return 'loss', i - entry_idx
            elif candle['high'] >= tp:
                return 'win', i - entry_idx
        else:
            if candle['high'] >= sl:
                return 'loss', i - entry_idx
            elif candle['low'] <= tp:
                return 'win', i - entry_idx
    
    return 'timeout', max_candles

def process_single_symbol(symbol: str, days: int = 60, train_pct: float = 0.6) -> dict:
    df = fetch_klines(symbol, '3', days)
    
    if len(df) < 500:
        return None, 0, 0
    
    df = calculate_indicators(df)
    
    if len(df) < 200:
        return None, 0, 0
    
    total_candles = len(df)
    train_end = int(total_candles * train_pct)
    
    df_train = df.iloc[:train_end]
    df_test = df.iloc[train_end:]
    
    combo_stats = defaultdict(lambda: {
        'train_wins': 0, 'train_losses': 0,
        'test_wins': 0, 'test_losses': 0,
        'train_candles': [],
        'test_candles': []
    })
    
    total_signals = 0
    total_wins = 0
    
    for idx in range(51, len(df_train) - 80):
        row = df_train.iloc[idx]
        prev_row = df_train.iloc[idx - 1]
        
        side = check_vwap_signal(row, prev_row)
        if not side:
            continue
        
        combo = get_combo_original(row)
        key = f"{side}|{combo}"
        
        outcome, candles = simulate_trade_realistic(df_train, idx, side, row['close'], row['atr'])
        total_signals += 1
        
        if outcome == 'win':
            combo_stats[key]['train_wins'] += 1
            total_wins += 1
        else:
            combo_stats[key]['train_losses'] += 1
        combo_stats[key]['train_candles'].append(candles)
    
    for idx in range(51, len(df_test) - 80):
        row = df_test.iloc[idx]
        prev_row = df_test.iloc[idx - 1]
        
        side = check_vwap_signal(row, prev_row)
        if not side:
            continue
        
        combo = get_combo_original(row)
        key = f"{side}|{combo}"
        
        outcome, candles = simulate_trade_realistic(df_test, idx, side, row['close'], row['atr'])
        total_signals += 1
        
        if outcome == 'win':
            combo_stats[key]['test_wins'] += 1
            total_wins += 1
        else:
            combo_stats[key]['test_losses'] += 1
        combo_stats[key]['test_candles'].append(candles)
    
    overall_wr = (total_wins / total_signals * 100) if total_signals > 0 else 0
    
    winning_combos = []
    
    for key, stats in combo_stats.items():
        train_total = stats['train_wins'] + stats['train_losses']
        test_total = stats['test_wins'] + stats['test_losses']
        
        if train_total < MIN_TRAIN_TRADES:
            continue
        if test_total < MIN_TEST_TRADES:
            continue
        
        train_wr = stats['train_wins'] / train_total * 100
        test_wr = stats['test_wins'] / test_total * 100
        train_lb = wilson_lower_bound(stats['train_wins'], train_total)
        test_lb = wilson_lower_bound(stats['test_wins'], test_total)
        
        if train_wr >= MIN_TRAIN_WR and test_wr >= MIN_TEST_WR:
            side, combo = key.split('|')
            winning_combos.append({
                'symbol': symbol,
                'side': side,
                'combo': combo,
                'train_n': train_total,
                'train_wr': round(train_wr, 1),
                'train_lb': round(train_lb, 1),
                'test_n': test_total,
                'test_wr': round(test_wr, 1),
                'test_lb': round(test_lb, 1),
                'avg_hold_candles': int(np.mean(stats['train_candles'])) if stats['train_candles'] else 0
            })
    
    return winning_combos, total_signals, overall_wr

def run_backtest(num_symbols=400, days=60, train_pct=0.6):
    print("="*70)
    print("🔬 1:1 R:R BACKTEST with ORIGINAL 70 INDICATOR BINS")
    print("="*70)
    print(f"   R:R Ratio: {RR_RATIO}:1")
    print(f"   Days: {days}")
    print(f"   Walk-forward: {train_pct*100:.0f}% train / {(1-train_pct)*100:.0f}% test")
    print(f"   Fees: {TOTAL_FEE_PCT}% | Spread: {SPREAD_PCT}% | Slip: {SLIPPAGE_PCT}%")
    print(f"   Total Cost: {TOTAL_COST_PCT:.2f}%")
    print(f"   Min Train WR: {MIN_TRAIN_WR}% | Min Test WR: {MIN_TEST_WR}%")
    print(f"   Min Train N: {MIN_TRAIN_TRADES} | Min Test N: {MIN_TEST_TRADES}")
    print(f"   Indicator Bins: ORIGINAL (5 RSI x 2 MACD x 7 Fib = 70 combos)")
    print("="*70)
    
    print(f"\n📋 Fetching top {num_symbols} symbols...")
    symbols = get_all_symbols(num_symbols)
    print(f"   Found {len(symbols)} symbols")
    
    all_combos = []
    symbols_with_combos = 0
    
    print(f"\n📊 Processing symbols (showing WR & N for each)...")
    print("-" * 70)
    
    for i, symbol in enumerate(symbols):
        try:
            result = process_single_symbol(symbol, days, train_pct)
            combos, total_signals, overall_wr = result
            
            if combos and len(combos) > 0:
                all_combos.extend(combos)
                symbols_with_combos += 1
                combo_summary = f"✅ {len(combos)} golden combo(s)"
                print(f"[{i+1:3}/{len(symbols)}] {symbol:20} | N={total_signals:4} | WR={overall_wr:5.1f}% | {combo_summary}")
                for c in combos:
                    print(f"         └─ {c['side']:5} {c['combo']} | Train: {c['train_wr']:.0f}% (N={c['train_n']}) | Test: {c['test_wr']:.0f}% (N={c['test_n']})")
            else:
                if total_signals > 0:
                    print(f"[{i+1:3}/{len(symbols)}] {symbol:20} | N={total_signals:4} | WR={overall_wr:5.1f}% | ❌ No qualifying combos")
                else:
                    print(f"[{i+1:3}/{len(symbols)}] {symbol:20} | ⚠️  Insufficient data")
                    
        except Exception as e:
            print(f"[{i+1:3}/{len(symbols)}] {symbol:20} | ❌ Error: {e}")
        
        time.sleep(0.02)
    
    all_combos.sort(key=lambda x: (x['train_wr'], x['test_wr']), reverse=True)
    
    print(f"\n✅ Processing complete!")
    print(f"   Symbols processed: {len(symbols)}")
    print(f"   Symbols with combos: {symbols_with_combos}")
    print(f"   Total winning combos: {len(all_combos)}")
    
    print("\n" + "="*70)
    print(f"🏆 1:1 R:R RESULTS with ORIGINAL 70 BINS")
    print(f"   Criteria: Train WR ≥ {MIN_TRAIN_WR}%, Test WR ≥ {MIN_TEST_WR}%")
    print(f"   Min Samples: Train N ≥ {MIN_TRAIN_TRADES}, Test N ≥ {MIN_TEST_TRADES}")
    print("="*70)
    
    print(f"\n📊 TOP 30 COMBOS:")
    print("-"*70)
    for i, c in enumerate(all_combos[:30], 1):
        print(f"{i:2}. {c['symbol']} {c['side']} {c['combo']}")
        print(f"    Train: WR={c['train_wr']:.1f}% LB={c['train_lb']:.1f}% (N={c['train_n']})")
        print(f"    Test:  WR={c['test_wr']:.1f}% LB={c['test_lb']:.1f}% (N={c['test_n']})")
    
    output_file = 'backtest_1to1_ORIGINAL_70_BINS_RESULTS.yaml'
    yaml_output = {
        '_metadata': {
            'rr_ratio': '1:1',
            'indicator_bins': 'ORIGINAL (5x2x7 = 70 combos)',
            'generated': datetime.utcnow().isoformat(),
            'symbols_tested': len(symbols),
            'symbols_with_combos': symbols_with_combos,
            'total_combos': len(all_combos),
            'criteria': {
                'train_wr': f'>= {MIN_TRAIN_WR}%',
                'test_wr': f'>= {MIN_TEST_WR}%',
                'min_train_n': MIN_TRAIN_TRADES,
                'min_test_n': MIN_TEST_TRADES
            },
            'execution_costs': {
                'fees_pct': TOTAL_FEE_PCT,
                'spread_pct': SPREAD_PCT,
                'slippage_pct': SLIPPAGE_PCT,
                'total_pct': TOTAL_COST_PCT
            },
            'note': 'COMPARISON ONLY - Does not affect live bot'
        }
    }
    
    for c in all_combos:
        sym = c['symbol']
        side = c['side']
        combo = c['combo']
        
        if sym not in yaml_output:
            yaml_output[sym] = {
                'allowed_combos_long': [],
                'allowed_combos_short': [],
                'stats_long': {},
                'stats_short': {}
            }
        
        if side == 'long':
            yaml_output[sym]['allowed_combos_long'].append(combo)
            yaml_output[sym]['stats_long'][combo] = {
                'train_wr': c['train_wr'],
                'train_lb': c['train_lb'],
                'train_n': c['train_n'],
                'test_wr': c['test_wr'],
                'test_lb': c['test_lb'],
                'test_n': c['test_n'],
                'avg_hold_candles': c['avg_hold_candles']
            }
        else:
            yaml_output[sym]['allowed_combos_short'].append(combo)
            yaml_output[sym]['stats_short'][combo] = {
                'train_wr': c['train_wr'],
                'train_lb': c['train_lb'],
                'train_n': c['train_n'],
                'test_wr': c['test_wr'],
                'test_lb': c['test_lb'],
                'test_n': c['test_n'],
                'avg_hold_candles': c['avg_hold_candles']
            }
    
    with open(output_file, 'w') as f:
        f.write("# 1:1 R:R Backtest Results with ORIGINAL 70 INDICATOR BINS\n")
        f.write(f"# Generated: {datetime.utcnow().isoformat()}\n")
        f.write(f"# R:R Ratio: 1:1\n")
        f.write(f"# Indicator Bins: ORIGINAL (5 RSI x 2 MACD x 7 Fib = 70 combos)\n")
        f.write(f"# Symbols tested: {len(symbols)}\n")
        f.write(f"# Total costs: {TOTAL_COST_PCT:.2f}% (fees + spread + slippage)\n")
        f.write(f"# Criteria: Train WR >= {MIN_TRAIN_WR}%, Test WR >= {MIN_TEST_WR}%\n")
        f.write(f"# Min Samples: Train N >= {MIN_TRAIN_TRADES}, Test N >= {MIN_TEST_TRADES}\n")
        f.write("#\n")
        f.write("# NOTE: This file is for COMPARISON ONLY.\n\n")
        yaml.dump(yaml_output, f, default_flow_style=False, sort_keys=False)
    
    print(f"\n✅ Results saved to: {output_file}")
    print(f"   Symbols with combos: {symbols_with_combos}")
    print(f"   Total winning combos: {len(all_combos)}")
    print(f"\n⚠️ NOTE: This is for comparison only. Bot settings unchanged.")
    
    return all_combos

if __name__ == "__main__":
    run_backtest(num_symbols=400, days=60, train_pct=0.6)
