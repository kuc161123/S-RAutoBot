#!/usr/bin/env python3
"""
VOLUME + ATR FILTER BACKTEST

Combines volume and ATR filters to find high-quality setups.
Uses 1:1 R:R for higher win rate.

Filters:
- Volume: Current candle volume >= X * 20-period average
- ATR: Current ATR >= Y% of price (ensures sufficient volatility)

Features:
- Walk-forward validation (60% train, 40% test)
- Realistic fees and slippage
- 1:1 R:R ratio (changed from 2:1)
- Auto-commits passing symbols to volume_atr_combos.yaml

COMPARISON ONLY - Does NOT change bot settings.
"""

import requests
import pandas as pd
import numpy as np
import time
import math
import yaml
import subprocess
import os
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

# R:R RATIO - 1:1 (higher win rate target)
RR_RATIO = 1.0

# VOLUME MULTIPLIERS TO TEST
VOLUME_MULTIPLIERS = [1.0, 1.5, 2.0, 2.5, 3.0]

# ATR PERCENTAGES TO TEST (min ATR as % of price)
ATR_PERCENTAGES = [0.3, 0.5, 0.7, 1.0]

# MINIMUM REQUIREMENTS
MIN_TRAIN_TRADES = 15
MIN_TEST_TRADES = 3
MIN_TRAIN_WR = 50  # Higher WR threshold for 1:1 RR
MIN_TEST_WR = 50

# OUTPUT FILE
OUTPUT_FILE = 'volume_atr_combos.yaml'

def auto_commit_symbol(symbol: str, side: str, config: dict):
    """Auto-commit passing symbol to yaml and push to git."""
    try:
        existing = {'_metadata': {
            'generated': datetime.utcnow().isoformat(),
            'rr_ratio': f'{RR_RATIO}:1',
            'filter_type': 'volume+atr',
            'min_train_wr': MIN_TRAIN_WR,
            'min_train_n': MIN_TRAIN_TRADES
        }}
        
        if os.path.exists(OUTPUT_FILE):
            with open(OUTPUT_FILE, 'r') as f:
                loaded = yaml.safe_load(f)
                if loaded:
                    existing = loaded
        
        if symbol not in existing:
            existing[symbol] = {}
        
        if side == 'long':
            existing[symbol]['allowed_long'] = True
            existing[symbol]['volume_mult_long'] = config['volume_mult']
            existing[symbol]['atr_min_long'] = config['atr_min']
            existing[symbol]['stats_long'] = {
                'train_wr': config['train_wr'],
                'train_n': config['train_n'],
                'test_wr': config.get('test_wr', 0),
                'test_n': config.get('test_n', 0)
            }
        else:
            existing[symbol]['allowed_short'] = True
            existing[symbol]['volume_mult_short'] = config['volume_mult']
            existing[symbol]['atr_min_short'] = config['atr_min']
            existing[symbol]['stats_short'] = {
                'train_wr': config['train_wr'],
                'train_n': config['train_n'],
                'test_wr': config.get('test_wr', 0),
                'test_n': config.get('test_n', 0)
            }
        
        with open(OUTPUT_FILE, 'w') as f:
            f.write('# Volume + ATR Filter Combos - AUTO-GENERATED\n')
            f.write(f'# Last Updated: {datetime.utcnow().isoformat()}\n')
            f.write(f'# R:R Ratio: {RR_RATIO}:1\n\n')
            yaml.dump(existing, f, default_flow_style=False, sort_keys=False)
        
        commit_msg = f'Add {symbol} {side} Vol>{config["volume_mult"]}x ATR>{config["atr_min"]}% WR={config["train_wr"]}%'
        subprocess.run(['git', 'add', OUTPUT_FILE], capture_output=True)
        subprocess.run(['git', 'commit', '-m', commit_msg], capture_output=True)
        subprocess.run(['git', 'push'], capture_output=True)
        
        print(f"         📤 AUTO-COMMITTED: {symbol} {side} to git")
        return True
    except Exception as e:
        print(f"         ⚠️ Commit failed: {e}")
        return False

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

def fetch_klines(symbol: str, interval: str = '3', days: int = 120) -> pd.DataFrame:
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

def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if len(df) < 50:
        return df
    
    # ATR for SL/TP
    high_low = df['high'] - df['low']
    high_close = abs(df['high'] - df['close'].shift())
    low_close = abs(df['low'] - df['close'].shift())
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr'] = true_range.rolling(14).mean()
    
    # ATR as percentage of price
    df['atr_pct'] = (df['atr'] / df['close']) * 100
    
    # VWAP
    df['tp'] = (df['high'] + df['low'] + df['close']) / 3
    df['tp_v'] = df['tp'] * df['volume']
    df['cum_tp_v'] = df['tp_v'].cumsum()
    df['cum_v'] = df['volume'].cumsum()
    df['vwap'] = df['cum_tp_v'] / df['cum_v']
    
    # Rolling average volume (20 candles)
    df['avg_volume'] = df['volume'].rolling(20).mean()
    
    # Volume ratio (current / average)
    df['volume_ratio'] = df['volume'] / df['avg_volume']
    
    return df.dropna()

def check_vwap_signal(row, prev_row=None) -> str:
    if prev_row is None:
        return None
    
    if row['low'] <= row['vwap'] and row['close'] > row['vwap']:
        return 'long'
    
    if row['high'] >= row['vwap'] and row['close'] < row['vwap']:
        return 'short'
    
    return None

def simulate_trade_realistic(df, entry_idx, side, entry, atr, max_candles=80):
    """Simulate trade with 1:1 R:R"""
    MIN_SL_PCT = 0.5
    min_sl_dist = entry * (MIN_SL_PCT / 100)
    sl_dist = max(atr, min_sl_dist)
    tp_dist = sl_dist * RR_RATIO  # 1:1 R:R
    
    entry_cost = entry * (TOTAL_COST_PCT / 100)
    
    if side == 'long':
        sl_price = entry - sl_dist
        tp_price = entry + tp_dist - entry_cost
    else:
        sl_price = entry + sl_dist
        tp_price = entry - tp_dist + entry_cost
    
    for i in range(entry_idx + 1, min(entry_idx + max_candles, len(df))):
        candle = df.iloc[i]
        high, low = candle['high'], candle['low']
        
        if side == 'long':
            if low <= sl_price:
                return 'loss', i - entry_idx
            if high >= tp_price:
                return 'win', i - entry_idx
        else:
            if high >= sl_price:
                return 'loss', i - entry_idx
            if low <= tp_price:
                return 'win', i - entry_idx
    
    return 'timeout', max_candles

def process_single_symbol(symbol: str, df: pd.DataFrame) -> dict:
    """Process symbol with Volume + ATR filter combinations."""
    df = calculate_indicators(df)
    if len(df) < 100:
        return None
    
    # Walk-forward split (60% train, 40% test)
    split_idx = int(len(df) * 0.6)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]
    
    # Test all combinations of volume_mult and atr_min
    results = {'long': [], 'short': []}
    
    for vol_mult in VOLUME_MULTIPLIERS:
        for atr_min in ATR_PERCENTAGES:
            for side in ['long', 'short']:
                train_trades = []
                test_trades = []
                
                # Train period signals
                for i in range(1, len(train_df)):
                    row = train_df.iloc[i]
                    prev_row = train_df.iloc[i-1]
                    
                    signal = check_vwap_signal(row, prev_row)
                    if signal != side:
                        continue
                    
                    # Check volume filter
                    if row['volume_ratio'] < vol_mult:
                        continue
                    
                    # Check ATR filter
                    if row['atr_pct'] < atr_min:
                        continue
                    
                    # Simulate trade
                    outcome, candles = simulate_trade_realistic(
                        train_df, i, side, row['close'], row['atr']
                    )
                    if outcome != 'timeout':
                        train_trades.append(1 if outcome == 'win' else 0)
                
                # Test period signals
                for i in range(1, len(test_df)):
                    row = test_df.iloc[i]
                    prev_row = test_df.iloc[i-1]
                    
                    signal = check_vwap_signal(row, prev_row)
                    if signal != side:
                        continue
                    
                    if row['volume_ratio'] < vol_mult:
                        continue
                    
                    if row['atr_pct'] < atr_min:
                        continue
                    
                    outcome, candles = simulate_trade_realistic(
                        test_df, i, side, row['close'], row['atr']
                    )
                    if outcome != 'timeout':
                        test_trades.append(1 if outcome == 'win' else 0)
                
                # Calculate stats
                train_n = len(train_trades)
                train_wins = sum(train_trades)
                train_wr = (train_wins / train_n * 100) if train_n > 0 else 0
                
                test_n = len(test_trades)
                test_wins = sum(test_trades)
                test_wr = (test_wins / test_n * 100) if test_n > 0 else 0
                
                results[side].append({
                    'volume_mult': vol_mult,
                    'atr_min': atr_min,
                    'train_wr': round(train_wr, 1),
                    'train_n': train_n,
                    'test_wr': round(test_wr, 1),
                    'test_n': test_n
                })
    
    return results

def analyze_results(symbol: str, results: dict):
    """Find best configs and auto-commit passing ones."""
    winning_configs = []
    committed_configs = []
    
    for side in ['long', 'short']:
        configs = results.get(side, [])
        
        for cfg in configs:
            # Check if passes TRAIN validation
            if cfg['train_n'] >= MIN_TRAIN_TRADES and cfg['train_wr'] >= MIN_TRAIN_WR:
                # Auto-commit on train pass
                commit_cfg = {
                    'volume_mult': cfg['volume_mult'],
                    'atr_min': cfg['atr_min'],
                    'train_wr': cfg['train_wr'],
                    'train_n': cfg['train_n'],
                    'test_wr': cfg['test_wr'],
                    'test_n': cfg['test_n']
                }
                auto_commit_symbol(symbol, side, commit_cfg)
                committed_configs.append((side, cfg))
                
                # Check if also passes TEST
                if cfg['test_n'] >= MIN_TEST_TRADES and cfg['test_wr'] >= MIN_TEST_WR:
                    winning_configs.append((side, cfg))
    
    return winning_configs, committed_configs

def run_backtest(days=120):
    print("=" * 70)
    print("🔬 VOLUME + ATR FILTER BACKTEST (AUTO-COMMIT MODE)")
    print("=" * 70)
    print(f"   R:R Ratio: {RR_RATIO}:1")
    print(f"   Days: {days}")
    print(f"   Walk-forward: 60% train / 40% test")
    print(f"   Volume Multipliers: {VOLUME_MULTIPLIERS}")
    print(f"   ATR Minimums: {ATR_PERCENTAGES}%")
    print(f"   Min Train WR: {MIN_TRAIN_WR}% | Min Test WR: {MIN_TEST_WR}%")
    print(f"   Min Train N: {MIN_TRAIN_TRADES} | Min Test N: {MIN_TEST_TRADES}")
    print(f"   📤 AUTO-COMMIT: Enabled (on TRAIN pass)")
    print("=" * 70)
    print()
    print("⚠️  Symbols passing TRAIN will be auto-committed to volume_atr_combos.yaml")
    print()
    
    print("📋 Fetching top 400 symbols...")
    symbols = get_all_symbols(400)
    print(f"   Found {len(symbols)} symbols")
    print()
    
    print("📊 Processing symbols (testing volume + ATR filters)...")
    print("-" * 70)
    
    all_winning = []
    all_committed = []
    
    for idx, sym in enumerate(symbols, 1):
        df = fetch_klines(sym, '3', days)
        
        if df.empty or len(df) < 100:
            print(f"[{idx:3d}/{len(symbols)}] {sym:20s} | ⚠️  Insufficient data")
            continue
        
        results = process_single_symbol(sym, df)
        if results is None:
            print(f"[{idx:3d}/{len(symbols)}] {sym:20s} | ⚠️  Processing failed")
            continue
        
        winning, committed = analyze_results(sym, results)
        
        if winning:
            all_winning.extend([(sym, s, c) for s, c in winning])
            print(f"[{idx:3d}/{len(symbols)}] {sym:20s} | ✅ {len(winning)} winning (train+test)")
            for side, cfg in winning:
                print(f"         └─ {side:5s} Vol>{cfg['volume_mult']}x ATR>{cfg['atr_min']}% | Train: {cfg['train_wr']}% (N={cfg['train_n']}) | Test: {cfg['test_wr']}% (N={cfg['test_n']})")
        elif committed:
            all_committed.extend([(sym, s, c) for s, c in committed])
            print(f"[{idx:3d}/{len(symbols)}] {sym:20s} | 💾 {len(committed)} train-only (committed)")
            for side, cfg in committed:
                print(f"         └─ {side:5s} Vol>{cfg['volume_mult']}x ATR>{cfg['atr_min']}% | Train: {cfg['train_wr']}% (N={cfg['train_n']}) | Test: {cfg['test_wr']}% (N={cfg['test_n']})")
        else:
            # Find best config to show why it failed
            best = None
            best_score = 0
            for side in ['long', 'short']:
                for cfg in results.get(side, []):
                    score = cfg['train_wr'] * min(cfg['train_n'] / MIN_TRAIN_TRADES, 1)
                    if score > best_score:
                        best_score = score
                        best = (side, cfg)
            
            if best:
                side, cfg = best
                reasons = []
                if cfg['train_wr'] < MIN_TRAIN_WR:
                    reasons.append(f"TrainWR<{MIN_TRAIN_WR}%")
                if cfg['train_n'] < MIN_TRAIN_TRADES:
                    reasons.append(f"TrainN<{MIN_TRAIN_TRADES}")
                if cfg['test_wr'] < MIN_TEST_WR:
                    reasons.append(f"TestWR<{MIN_TEST_WR}%")
                if cfg['test_n'] < MIN_TEST_TRADES:
                    reasons.append(f"TestN<{MIN_TEST_TRADES}")
                
                reason_str = ", ".join(reasons) if reasons else "No valid combos"
                print(f"[{idx:3d}/{len(symbols)}] {sym:20s} | Vol>{cfg['volume_mult']}x ATR>{cfg['atr_min']}% {side} | Train: {cfg['train_wr']:.0f}% N={cfg['train_n']} | Test: {cfg['test_wr']:.0f}% N={cfg['test_n']} | ❌{reason_str}")
            else:
                print(f"[{idx:3d}/{len(symbols)}] {sym:20s} | ❌ No signals found")
        
        time.sleep(0.1)
    
    # Summary
    print()
    print("=" * 70)
    print("📊 BACKTEST COMPLETE")
    print("=" * 70)
    print(f"   Total Symbols: {len(symbols)}")
    print(f"   Passing Train+Test: {len(all_winning)} configs")
    print(f"   Auto-Committed (Train only): {len(all_committed)} configs")
    print(f"   Output: {OUTPUT_FILE}")
    print("=" * 70)

if __name__ == "__main__":
    run_backtest(days=120)
