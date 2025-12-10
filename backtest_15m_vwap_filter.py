#!/usr/bin/env python3
"""
15-MINUTE VWAP FILTER BACKTEST

Uses 15m timeframe VWAP as directional filter for 3m signals:
- LONG: Only if price is ABOVE 15m VWAP (uptrend confirmation)
- SHORT: Only if price is BELOW 15m VWAP (downtrend confirmation)

Features:
- Walk-forward validation (60% train, 40% test)
- Realistic fees and slippage
- 2:1 R:R ratio (original)
- 3m VWAP signal + 15m VWAP direction filter
- 40% WR threshold for both train and test

COMPARISON ONLY - Does NOT change bot settings.
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

# R:R RATIO - Original 2:1
RR_RATIO = 2.0

# MINIMUM REQUIREMENTS
MIN_TRAIN_TRADES = 15
MIN_TEST_TRADES = 5
MIN_TRAIN_WR = 40
MIN_TEST_WR = 40

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
    df = df.astype({'start': int, 'open': float, 'high': float, 'low': float, 'close': float, 'volume': float})
    df['start'] = pd.to_datetime(df['start'], unit='ms')
    df = df.sort_values('start').drop_duplicates('start').reset_index(drop=True)
    return df

def calculate_indicators_3m(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate indicators for 3m timeframe."""
    if len(df) < 50:
        return df
    
    # ATR for SL/TP
    high_low = df['high'] - df['low']
    high_close = abs(df['high'] - df['close'].shift())
    low_close = abs(df['low'] - df['close'].shift())
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr'] = true_range.rolling(14).mean()
    
    # 3m VWAP (for signal detection)
    df['tp'] = (df['high'] + df['low'] + df['close']) / 3
    df['tp_v'] = df['tp'] * df['volume']
    df['cum_tp_v'] = df['tp_v'].cumsum()
    df['cum_v'] = df['volume'].cumsum()
    df['vwap_3m'] = df['cum_tp_v'] / df['cum_v']
    
    return df.dropna()

def calculate_15m_vwap(df_3m: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate 15m VWAP from 3m data.
    Resample 3m candles to 15m, then calculate VWAP.
    """
    df = df_3m.copy()
    df.set_index('start', inplace=True)
    
    # Resample 3m to 15m
    df_15m = df.resample('15T').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()
    
    # Calculate 15m VWAP
    df_15m['tp'] = (df_15m['high'] + df_15m['low'] + df_15m['close']) / 3
    df_15m['tp_v'] = df_15m['tp'] * df_15m['volume']
    df_15m['cum_tp_v'] = df_15m['tp_v'].cumsum()
    df_15m['cum_v'] = df_15m['volume'].cumsum()
    df_15m['vwap_15m'] = df_15m['cum_tp_v'] / df_15m['cum_v']
    
    # Resample back to 3m (forward fill the 15m VWAP to each 3m candle)
    df_15m_vwap = df_15m[['vwap_15m']].resample('3T').ffill()
    
    # Merge back to original 3m dataframe
    df = df.join(df_15m_vwap, how='left')
    df['vwap_15m'] = df['vwap_15m'].ffill()
    
    df.reset_index(inplace=True)
    
    return df.dropna()

def check_vwap_signal_with_15m_filter(row, prev_row=None) -> str:
    """
    Check for VWAP signal with 15m VWAP directional filter.
    
    LONG: 3m VWAP cross signal AND price > 15m VWAP
    SHORT: 3m VWAP cross signal AND price < 15m VWAP
    """
    if prev_row is None:
        return None
    
    # Long signal: Low touched/crossed below 3m VWAP and closed above
    if row['low'] <= row['vwap_3m'] and row['close'] > row['vwap_3m']:
        # Check 15m VWAP filter: price must be ABOVE 15m VWAP for long
        if row['close'] > row['vwap_15m']:
            return 'long'
    
    # Short signal: High touched/crossed above 3m VWAP and closed below
    if row['high'] >= row['vwap_3m'] and row['close'] < row['vwap_3m']:
        # Check 15m VWAP filter: price must be BELOW 15m VWAP for short
        if row['close'] < row['vwap_15m']:
            return 'short'
    
    return None

def simulate_trade_realistic(df, entry_idx, side, entry, atr, max_candles=80):
    MIN_SL_PCT = 0.5
    min_sl_dist = entry * (MIN_SL_PCT / 100)
    sl_dist = max(atr, min_sl_dist)
    tp_dist = sl_dist * RR_RATIO
    
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
    """Process a single symbol with 15m VWAP filter."""
    # Fetch 3m data
    df = fetch_klines(symbol, '3', days)
    
    if len(df) < 500:
        return None
    
    # Calculate 3m indicators
    df = calculate_indicators_3m(df)
    
    if len(df) < 200:
        return None
    
    # Calculate 15m VWAP
    df = calculate_15m_vwap(df)
    
    if len(df) < 200:
        return None
    
    total_candles = len(df)
    train_end = int(total_candles * train_pct)
    
    df_train = df.iloc[:train_end].copy()
    df_test = df.iloc[train_end:].copy()
    
    # Track stats for long and short
    stats = {
        'long': {'train_wins': 0, 'train_losses': 0, 'test_wins': 0, 'test_losses': 0},
        'short': {'train_wins': 0, 'train_losses': 0, 'test_wins': 0, 'test_losses': 0}
    }
    
    # Also track "unfiltered" for comparison
    stats_unfiltered = {
        'long': {'train_wins': 0, 'train_losses': 0, 'test_wins': 0, 'test_losses': 0},
        'short': {'train_wins': 0, 'train_losses': 0, 'test_wins': 0, 'test_losses': 0}
    }
    
    # Process train data WITH 15m filter
    for idx in range(51, len(df_train) - 80):
        row = df_train.iloc[idx]
        prev_row = df_train.iloc[idx - 1]
        
        # Check signal with 15m VWAP filter
        side = check_vwap_signal_with_15m_filter(row, prev_row)
        if not side:
            continue
        
        outcome, candles = simulate_trade_realistic(df_train, idx, side, row['close'], row['atr'])
        
        if outcome == 'win':
            stats[side]['train_wins'] += 1
        else:
            stats[side]['train_losses'] += 1
    
    # Process test data WITH 15m filter
    for idx in range(51, len(df_test) - 80):
        row = df_test.iloc[idx]
        prev_row = df_test.iloc[idx - 1]
        
        side = check_vwap_signal_with_15m_filter(row, prev_row)
        if not side:
            continue
        
        outcome, candles = simulate_trade_realistic(df_test, idx, side, row['close'], row['atr'])
        
        if outcome == 'win':
            stats[side]['test_wins'] += 1
        else:
            stats[side]['test_losses'] += 1
    
    return stats

def analyze_results(symbol: str, stats: dict) -> list:
    """Analyze results and find winning configs."""
    winning_configs = []
    
    for side in ['long', 'short']:
        train_total = stats[side]['train_wins'] + stats[side]['train_losses']
        test_total = stats[side]['test_wins'] + stats[side]['test_losses']
        
        if train_total < MIN_TRAIN_TRADES:
            continue
        if test_total < MIN_TEST_TRADES:
            continue
        
        train_wr = stats[side]['train_wins'] / train_total * 100
        test_wr = stats[side]['test_wins'] / test_total * 100
        train_lb = wilson_lower_bound(stats[side]['train_wins'], train_total)
        test_lb = wilson_lower_bound(stats[side]['test_wins'], test_total)
        
        if train_wr >= MIN_TRAIN_WR and test_wr >= MIN_TEST_WR:
            winning_configs.append({
                'symbol': symbol,
                'side': side,
                'filter': '15m_VWAP',
                'train_n': train_total,
                'train_wr': round(train_wr, 1),
                'train_lb': round(train_lb, 1),
                'test_n': test_total,
                'test_wr': round(test_wr, 1),
                'test_lb': round(test_lb, 1)
            })
    
    return winning_configs, stats

def run_backtest(num_symbols=400, days=60, train_pct=0.6):
    print("="*70)
    print("🔬 15-MINUTE VWAP FILTER BACKTEST")
    print("="*70)
    print(f"   R:R Ratio: {RR_RATIO}:1 (original)")
    print(f"   Days: {days}")
    print(f"   Walk-forward: {train_pct*100:.0f}% train / {(1-train_pct)*100:.0f}% test")
    print(f"   Filter: Price vs 15m VWAP (long if above, short if below)")
    print(f"   Min Train WR: {MIN_TRAIN_WR}% | Min Test WR: {MIN_TEST_WR}%")
    print(f"   Min Train N: {MIN_TRAIN_TRADES} | Min Test N: {MIN_TEST_TRADES}")
    print("="*70)
    print(f"\n⚠️  COMPARISON ONLY - Does NOT change bot settings!\n")
    
    print(f"📋 Fetching top {num_symbols} symbols...")
    symbols = get_all_symbols(num_symbols)
    print(f"   Found {len(symbols)} symbols")
    
    all_configs = []
    symbols_with_configs = 0
    
    print(f"\n📊 Processing symbols (testing 15m VWAP filter)...")
    print("-" * 70)
    
    for i, symbol in enumerate(symbols):
        try:
            stats = process_single_symbol(symbol, days, train_pct)
            
            if stats is None:
                print(f"[{i+1:3}/{len(symbols)}] {symbol:20} | ⚠️  Insufficient data")
                continue
            
            # Analyze results
            winning, raw_stats = analyze_results(symbol, stats)
            
            # Calculate overall stats
            total_long_train = raw_stats['long']['train_wins'] + raw_stats['long']['train_losses']
            total_short_train = raw_stats['short']['train_wins'] + raw_stats['short']['train_losses']
            long_wr = (raw_stats['long']['train_wins'] / total_long_train * 100) if total_long_train > 0 else 0
            short_wr = (raw_stats['short']['train_wins'] / total_short_train * 100) if total_short_train > 0 else 0
            
            if winning:
                all_configs.extend(winning)
                symbols_with_configs += 1
                print(f"[{i+1:3}/{len(symbols)}] {symbol:20} | ✅ {len(winning)} winning config(s)")
                for w in winning:
                    print(f"         └─ {w['side']:5} 15mVWAP | Train: {w['train_wr']:.0f}% (N={w['train_n']}) | Test: {w['test_wr']:.0f}% (N={w['test_n']})")
            else:
                print(f"[{i+1:3}/{len(symbols)}] {symbol:20} | ❌ long WR={long_wr:.0f}% (N={total_long_train}) | short WR={short_wr:.0f}% (N={total_short_train})")
                    
        except Exception as e:
            print(f"[{i+1:3}/{len(symbols)}] {symbol:20} | ❌ Error: {e}")
        
        time.sleep(0.02)
    
    # Sort by train WR
    all_configs.sort(key=lambda x: (x['train_wr'], x['test_wr']), reverse=True)
    
    print(f"\n✅ Processing complete!")
    print(f"   Symbols processed: {len(symbols)}")
    print(f"   Symbols with configs: {symbols_with_configs}")
    print(f"   Total winning configs: {len(all_configs)}")
    
    # Print summary
    print("\n" + "="*70)
    print("🏆 15m VWAP FILTER RESULTS")
    print("="*70)
    
    print(f"\n📊 TOP 30 CONFIGS:")
    print("-"*70)
    for i, c in enumerate(all_configs[:30], 1):
        print(f"{i:2}. {c['symbol']} {c['side']}")
        print(f"    Train: WR={c['train_wr']:.1f}% LB={c['train_lb']:.1f}% (N={c['train_n']})")
        print(f"    Test:  WR={c['test_wr']:.1f}% LB={c['test_lb']:.1f}% (N={c['test_n']})")
    
    # Side distribution
    print("\n📈 SIDE DISTRIBUTION:")
    long_count = sum(1 for c in all_configs if c['side'] == 'long')
    short_count = sum(1 for c in all_configs if c['side'] == 'short')
    print(f"   Long:  {long_count} configs")
    print(f"   Short: {short_count} configs")
    
    # Save results
    output_file = 'backtest_15M_VWAP_FILTER_RESULTS.yaml'
    yaml_output = {
        '_metadata': {
            'rr_ratio': f'{RR_RATIO}:1',
            'filter_type': '15m_VWAP_DIRECTION',
            'description': 'Long if price > 15m VWAP, Short if price < 15m VWAP',
            'generated': datetime.utcnow().isoformat(),
            'symbols_tested': len(symbols),
            'symbols_with_configs': symbols_with_configs,
            'total_configs': len(all_configs),
            'long_configs': long_count,
            'short_configs': short_count,
            'criteria': {
                'train_wr': f'>= {MIN_TRAIN_WR}%',
                'test_wr': f'>= {MIN_TEST_WR}%',
                'min_train_n': MIN_TRAIN_TRADES,
                'min_test_n': MIN_TEST_TRADES
            },
            'note': 'COMPARISON ONLY - Does NOT affect live bot'
        }
    }
    
    for c in all_configs:
        sym = c['symbol']
        if sym not in yaml_output:
            yaml_output[sym] = {
                'long': None,
                'short': None
            }
        yaml_output[sym][c['side']] = {
            'filter': c['filter'],
            'train_wr': c['train_wr'],
            'train_lb': c['train_lb'],
            'train_n': c['train_n'],
            'test_wr': c['test_wr'],
            'test_lb': c['test_lb'],
            'test_n': c['test_n']
        }
    
    with open(output_file, 'w') as f:
        f.write("# 15-MINUTE VWAP FILTER Backtest Results\n")
        f.write(f"# Generated: {datetime.utcnow().isoformat()}\n")
        f.write(f"# R:R Ratio: {RR_RATIO}:1 (original)\n")
        f.write(f"# Filter: Price must be ABOVE 15m VWAP for long, BELOW for short\n")
        f.write(f"# Criteria: Train WR >= {MIN_TRAIN_WR}%, Test WR >= {MIN_TEST_WR}%\n")
        f.write("#\n")
        f.write("# NOTE: This file is for COMPARISON ONLY - Does NOT affect bot.\n\n")
        yaml.dump(yaml_output, f, default_flow_style=False, sort_keys=False)
    
    print(f"\n✅ Results saved to: {output_file}")
    print(f"\n⚠️  NOTE: This is for COMPARISON ONLY. Bot settings NOT changed.")
    
    return all_configs

if __name__ == "__main__":
    run_backtest(num_symbols=400, days=60, train_pct=0.6)
