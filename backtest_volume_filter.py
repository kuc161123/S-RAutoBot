#!/usr/bin/env python3
"""
VOLUME FILTER BACKTEST

Tests different volume filters per symbol to find optimal thresholds.
Instead of indicator combos, uses ONLY volume as the signal filter.

For each symbol:
1. Get historical data
2. Detect VWAP signals (same as before)
3. Test different volume filters (1x, 1.5x, 2x, 2.5x, 3x average volume)
4. Find which volume filter gives best WR with adequate samples
5. Report results per symbol

Features:
- Walk-forward validation (60% train, 40% test)
- Realistic fees and slippage
- 2:1 R:R ratio
- Volume multipliers: 1.0x, 1.5x, 2.0x, 2.5x, 3.0x avg volume
- Process one symbol at a time

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

# R:R RATIO - 2:1 (current bot setting)
RR_RATIO = 2.0

# VOLUME MULTIPLIERS TO TEST
VOLUME_MULTIPLIERS = [1.0, 1.5, 2.0, 2.5, 3.0]

# MINIMUM REQUIREMENTS
MIN_TRAIN_TRADES = 15
MIN_TEST_TRADES = 3
MIN_TRAIN_WR = 40
MIN_TEST_WR = 40

# OUTPUT FILE
VOLUME_COMBOS_FILE = 'volume_filter_combos.yaml'

def auto_commit_symbol(symbol: str, side: str, config: dict):
    """
    Auto-commit a passing symbol to volume_filter_combos.yaml and push to git.
    Called immediately when a symbol passes TRAIN validation.
    """
    try:
        # Load existing file
        existing = {'_metadata': {
            'generated': datetime.utcnow().isoformat(),
            'rr_ratio': f'{RR_RATIO}:1',
            'min_train_wr': MIN_TRAIN_WR,
            'min_train_n': MIN_TRAIN_TRADES,
            'min_test_n': MIN_TEST_TRADES
        }}
        
        if os.path.exists(VOLUME_COMBOS_FILE):
            with open(VOLUME_COMBOS_FILE, 'r') as f:
                loaded = yaml.safe_load(f)
                if loaded:
                    existing = loaded
        
        # Add/update symbol
        if symbol not in existing:
            existing[symbol] = {}
        
        if side == 'long':
            existing[symbol]['allowed_long'] = True
            existing[symbol]['volume_mult_long'] = config['volume_mult']
            existing[symbol]['stats_long'] = {
                'train_wr': config['train_wr'],
                'train_n': config['train_n'],
                'test_wr': config.get('test_wr', 0),
                'test_n': config.get('test_n', 0)
            }
        else:
            existing[symbol]['allowed_short'] = True
            existing[symbol]['volume_mult_short'] = config['volume_mult']
            existing[symbol]['stats_short'] = {
                'train_wr': config['train_wr'],
                'train_n': config['train_n'],
                'test_wr': config.get('test_wr', 0),
                'test_n': config.get('test_n', 0)
            }
        
        # Write file
        with open(VOLUME_COMBOS_FILE, 'w') as f:
            f.write('# Volume Filter Golden Combos - AUTO-GENERATED\n')
            f.write(f'# Last Updated: {datetime.utcnow().isoformat()}\n\n')
            yaml.dump(existing, f, default_flow_style=False, sort_keys=False)
        
        # Git commit and push
        commit_msg = f'Add {symbol} {side} Vol>{config["volume_mult"]}x TrainWR={config["train_wr"]}% N={config["train_n"]}'
        subprocess.run(['git', 'add', VOLUME_COMBOS_FILE], capture_output=True)
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

def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if len(df) < 50:
        return df
    
    # ATR for SL/TP
    high_low = df['high'] - df['low']
    high_close = abs(df['high'] - df['close'].shift())
    low_close = abs(df['low'] - df['close'].shift())
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr'] = true_range.rolling(14).mean()
    
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
    """Process a single symbol, testing different volume filters."""
    df = fetch_klines(symbol, '3', days)
    
    if len(df) < 500:
        return None
    
    df = calculate_indicators(df)
    
    if len(df) < 200:
        return None
    
    total_candles = len(df)
    train_end = int(total_candles * train_pct)
    
    df_train = df.iloc[:train_end].copy()
    df_test = df.iloc[train_end:].copy()
    
    # Test each volume multiplier
    results = {}
    
    for vol_mult in VOLUME_MULTIPLIERS:
        # Track stats for this volume filter
        stats = {
            'long': {'train_wins': 0, 'train_losses': 0, 'test_wins': 0, 'test_losses': 0},
            'short': {'train_wins': 0, 'train_losses': 0, 'test_wins': 0, 'test_losses': 0}
        }
        
        # Process train data
        for idx in range(51, len(df_train) - 80):
            row = df_train.iloc[idx]
            prev_row = df_train.iloc[idx - 1]
            
            # Check VWAP signal
            side = check_vwap_signal(row, prev_row)
            if not side:
                continue
            
            # Apply volume filter
            if row['volume_ratio'] < vol_mult:
                continue  # Skip if volume is below threshold
            
            outcome, candles = simulate_trade_realistic(df_train, idx, side, row['close'], row['atr'])
            
            if outcome == 'win':
                stats[side]['train_wins'] += 1
            else:
                stats[side]['train_losses'] += 1
        
        # Process test data
        for idx in range(51, len(df_test) - 80):
            row = df_test.iloc[idx]
            prev_row = df_test.iloc[idx - 1]
            
            side = check_vwap_signal(row, prev_row)
            if not side:
                continue
            
            if row['volume_ratio'] < vol_mult:
                continue
            
            outcome, candles = simulate_trade_realistic(df_test, idx, side, row['close'], row['atr'])
            
            if outcome == 'win':
                stats[side]['test_wins'] += 1
            else:
                stats[side]['test_losses'] += 1
        
        results[vol_mult] = stats
    
    return results

def analyze_results(symbol: str, results: dict) -> list:
    """
    Analyze volume filter results and find best filters.
    AUTO-COMMITS to git when TRAIN validation passes (before requiring test).
    """
    winning_configs = []
    committed_configs = []
    
    for vol_mult, stats in results.items():
        for side in ['long', 'short']:
            train_total = stats[side]['train_wins'] + stats[side]['train_losses']
            test_total = stats[side]['test_wins'] + stats[side]['test_losses']
            
            if train_total < MIN_TRAIN_TRADES:
                continue
            
            train_wr = stats[side]['train_wins'] / train_total * 100
            test_wr = (stats[side]['test_wins'] / test_total * 100) if test_total > 0 else 0
            train_lb = wilson_lower_bound(stats[side]['train_wins'], train_total)
            test_lb = wilson_lower_bound(stats[side]['test_wins'], test_total) if test_total > 0 else 0
            
            # AUTO-COMMIT ON TRAIN PASS (per user request)
            if train_wr >= MIN_TRAIN_WR:
                config = {
                    'symbol': symbol,
                    'side': side,
                    'volume_mult': vol_mult,
                    'train_n': train_total,
                    'train_wr': round(train_wr, 1),
                    'train_lb': round(train_lb, 1),
                    'test_n': test_total,
                    'test_wr': round(test_wr, 1),
                    'test_lb': round(test_lb, 1)
                }
                
                # Auto-commit to git immediately on train pass
                auto_commit_symbol(symbol, side, config)
                committed_configs.append(config)
                
                # Also add to winning_configs if test passes
                if test_total >= MIN_TEST_TRADES and test_wr >= MIN_TEST_WR:
                    winning_configs.append(config)
    
    return winning_configs, committed_configs

def run_backtest(num_symbols=400, days=60, train_pct=0.6):
    print("="*70)
    print("🔬 VOLUME FILTER BACKTEST (AUTO-COMMIT MODE)")
    print("="*70)
    print(f"   R:R Ratio: {RR_RATIO}:1")
    print(f"   Days: {days}")
    print(f"   Walk-forward: {train_pct*100:.0f}% train / {(1-train_pct)*100:.0f}% test")
    print(f"   Volume Multipliers: {VOLUME_MULTIPLIERS}")
    print(f"   Min Train WR: {MIN_TRAIN_WR}% | Min Test WR: {MIN_TEST_WR}%")
    print(f"   Min Train N: {MIN_TRAIN_TRADES} | Min Test N: {MIN_TEST_TRADES}")
    print(f"   📤 AUTO-COMMIT: Enabled (on TRAIN pass)")
    print("="*70)
    print(f"\n⚠️  Symbols passing TRAIN will be auto-committed to {VOLUME_COMBOS_FILE}\n")
    
    print(f"📋 Fetching top {num_symbols} symbols...")
    symbols = get_all_symbols(num_symbols)
    print(f"   Found {len(symbols)} symbols")
    
    all_configs = []
    all_committed = []
    symbols_with_configs = 0
    
    print(f"\n📊 Processing symbols (testing volume filters)...")
    print("-" * 70)
    
    for i, symbol in enumerate(symbols):
        try:
            results = process_single_symbol(symbol, days, train_pct)
            
            if results is None:
                print(f"[{i+1:3}/{len(symbols)}] {symbol:20} | ⚠️  Insufficient data")
                continue
            
            # Analyze - returns (winning_configs, committed_configs)
            winning, committed = analyze_results(symbol, results)
            
            if committed:
                all_committed.extend(committed)
            
            if winning:
                all_configs.extend(winning)
                symbols_with_configs += 1
                print(f"[{i+1:3}/{len(symbols)}] {symbol:20} | ✅ {len(winning)} winning (train+test)")
                for w in winning:
                    print(f"         └─ {w['side']:5} Vol>{w['volume_mult']}x | Train: {w['train_wr']:.0f}% (N={w['train_n']}) | Test: {w['test_wr']:.0f}% (N={w['test_n']})")
            elif committed:
                print(f"[{i+1:3}/{len(symbols)}] {symbol:20} | 💾 {len(committed)} train-only (committed)")
                for c in committed:
                    print(f"         └─ {c['side']:5} Vol>{c['volume_mult']}x | Train: {c['train_wr']:.0f}% (N={c['train_n']}) | Test: {c['test_wr']:.0f}% (N={c['test_n']})")
            else:
                # Show best result with BOTH train and test details
                best_score = -1
                best_info = ""
                best_fail_reason = ""
                for vol_mult, stats in results.items():
                    for side in ['long', 'short']:
                        train_total = stats[side]['train_wins'] + stats[side]['train_losses']
                        test_total = stats[side]['test_wins'] + stats[side]['test_losses']
                        if train_total > 0:
                            train_wr = stats[side]['train_wins'] / train_total * 100
                            test_wr = (stats[side]['test_wins'] / test_total * 100) if test_total > 0 else 0
                            score = train_wr + test_wr  # Combined score to find best
                            if score > best_score:
                                best_score = score
                                # Determine why it failed
                                reasons = []
                                if train_total < MIN_TRAIN_TRADES:
                                    reasons.append(f"TrainN<{MIN_TRAIN_TRADES}")
                                elif train_wr < MIN_TRAIN_WR:
                                    reasons.append(f"TrainWR<{MIN_TRAIN_WR}%")
                                if test_total < MIN_TEST_TRADES:
                                    reasons.append(f"TestN<{MIN_TEST_TRADES}")
                                elif test_wr < MIN_TEST_WR:
                                    reasons.append(f"TestWR<{MIN_TEST_WR}%")
                                best_fail_reason = ", ".join(reasons) if reasons else "Unknown"
                                best_info = f"Vol>{vol_mult}x {side} | Train: {train_wr:.0f}% N={train_total} | Test: {test_wr:.0f}% N={test_total} | ❌{best_fail_reason}"
                
                print(f"[{i+1:3}/{len(symbols)}] {symbol:20} | {best_info}")
                    
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
    print("🏆 VOLUME FILTER RESULTS")
    print("="*70)
    
    print(f"\n📊 TOP 30 CONFIGS:")
    print("-"*70)
    for i, c in enumerate(all_configs[:30], 1):
        print(f"{i:2}. {c['symbol']} {c['side']} Vol>{c['volume_mult']}x")
        print(f"    Train: WR={c['train_wr']:.1f}% LB={c['train_lb']:.1f}% (N={c['train_n']})")
        print(f"    Test:  WR={c['test_wr']:.1f}% LB={c['test_lb']:.1f}% (N={c['test_n']})")
    
    # Volume distribution analysis
    print("\n📈 VOLUME MULTIPLIER DISTRIBUTION:")
    vol_counts = defaultdict(int)
    for c in all_configs:
        vol_counts[c['volume_mult']] += 1
    for vol_mult in VOLUME_MULTIPLIERS:
        print(f"   Vol>{vol_mult}x: {vol_counts[vol_mult]} configs")
    
    # Save results
    output_file = 'backtest_VOLUME_FILTER_RESULTS.yaml'
    yaml_output = {
        '_metadata': {
            'rr_ratio': f'{RR_RATIO}:1',
            'filter_type': 'VOLUME',
            'volume_multipliers_tested': VOLUME_MULTIPLIERS,
            'generated': datetime.utcnow().isoformat(),
            'symbols_tested': len(symbols),
            'symbols_with_configs': symbols_with_configs,
            'total_configs': len(all_configs),
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
                'configs': []
            }
        yaml_output[sym]['configs'].append({
            'side': c['side'],
            'volume_mult': c['volume_mult'],
            'train_wr': c['train_wr'],
            'train_lb': c['train_lb'],
            'train_n': c['train_n'],
            'test_wr': c['test_wr'],
            'test_lb': c['test_lb'],
            'test_n': c['test_n']
        })
    
    with open(output_file, 'w') as f:
        f.write("# VOLUME FILTER Backtest Results\n")
        f.write(f"# Generated: {datetime.utcnow().isoformat()}\n")
        f.write(f"# R:R Ratio: {RR_RATIO}:1\n")
        f.write(f"# Volume Multipliers Tested: {VOLUME_MULTIPLIERS}\n")
        f.write(f"# Criteria: Train WR >= {MIN_TRAIN_WR}%, Test WR >= {MIN_TEST_WR}%\n")
        f.write("#\n")
        f.write("# NOTE: This file is for COMPARISON ONLY - Does NOT affect bot.\n\n")
        yaml.dump(yaml_output, f, default_flow_style=False, sort_keys=False)
    
    print(f"\n✅ Results saved to: {output_file}")
    print(f"\n⚠️  NOTE: This is for COMPARISON ONLY. Bot settings NOT changed.")
    
    return all_configs

if __name__ == "__main__":
    run_backtest(num_symbols=400, days=120, train_pct=0.6)
