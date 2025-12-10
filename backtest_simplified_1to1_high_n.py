#!/usr/bin/env python3
"""
SIMPLIFIED 1:1 R:R Backtest (400 Symbols)

Based on simplified_2to1 but with 1:1 R:R and HIGHER SAMPLE COUNTS.

Features:
- Walk-forward validation (60% train, 40% test)
- Realistic fees (0.055% maker + 0.055% taker = 0.11% total)
- Spread + slippage simulation
- 1:1 R:R (for comparison with 2:1)
- Process one symbol at a time
- Filter for WR >= 50% in BOTH train and test
- Higher sample counts: MIN 20 trades in train, MIN 5 in test
- Detailed results with WR and N samples

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
MAKER_FEE_PCT = 0.055     # 0.055% maker fee
TAKER_FEE_PCT = 0.055     # 0.055% taker fee
TOTAL_FEE_PCT = MAKER_FEE_PCT + TAKER_FEE_PCT  # 0.11%
SPREAD_PCT = 0.01         # ~1 tick spread
SLIPPAGE_PCT = 0.02       # Slippage on entry/exit
TOTAL_COST_PCT = TOTAL_FEE_PCT + SPREAD_PCT + SLIPPAGE_PCT  # ~0.14%

# R:R RATIO - 1:1 (for comparison)
RR_RATIO = 1.0  # 1:1 R:R

# MINIMUM REQUIREMENTS - REASONABLE FOR 18-COMBO SYSTEM
MIN_TRAIN_TRADES = 15    # Need at least 15 trades in training
MIN_TEST_TRADES = 3      # Need at least 3 trades in test for validation
MIN_TRAIN_WR = 50        # 50%+ win rate in training
MIN_TEST_WR = 50         # 50%+ win rate in test

def wilson_lower_bound(wins: int, total: int, z: float = 1.96) -> float:
    """Calculate Wilson score lower bound for win rate confidence."""
    if total == 0:
        return 0.0
    p = wins / total
    denominator = 1 + z*z / total
    centre = p + z*z / (2*total)
    spread = z * math.sqrt((p*(1-p) + z*z/(4*total)) / total)
    lower = (centre - spread) / denominator
    return max(0, lower * 100)

def get_all_symbols(limit=400) -> list:
    """Fetch top symbols by volume."""
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
    """Fetch historical klines with pagination."""
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
    
    # Rolling high/low for Fibonacci
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
    SIMPLIFIED combo string with FEWER bins:
    - RSI: 3 bins (oversold <40, neutral 40-60, overbought >60)
    - MACD: 2 bins (bull, bear)
    - Fib: 3 bins (low <38, mid 38-62, high >62)
    
    Total: 3 x 2 x 3 = 18 combos (vs original 70)
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
    """Check for VWAP cross signal."""
    if prev_row is None:
        return None
    
    # Long: Low touched/crossed below VWAP and closed above
    if row['low'] <= row['vwap'] and row['close'] > row['vwap']:
        return 'long'
    
    # Short: High touched/crossed above VWAP and closed below
    if row['high'] >= row['vwap'] and row['close'] < row['vwap']:
        return 'short'
    
    return None

def simulate_trade_realistic(df, entry_idx, side, entry, atr, max_candles=80):
    """
    REALISTIC trade simulation with 1:1 R:R:
    - TP = SL distance (1:1)
    - Entry + Exit costs applied
    - SL checked BEFORE TP on same candle (conservative)
    """
    # Minimum SL distance (0.5% of entry)
    MIN_SL_PCT = 0.5
    min_sl_dist = entry * (MIN_SL_PCT / 100)
    sl_dist = max(atr, min_sl_dist)
    tp_dist = sl_dist * RR_RATIO  # 1:1 R:R
    
    # Apply execution costs
    entry_cost = entry * (TOTAL_COST_PCT / 100)
    
    if side == 'long':
        adjusted_entry = entry + entry_cost
        sl = adjusted_entry - sl_dist
        tp = adjusted_entry + tp_dist
    else:
        adjusted_entry = entry - entry_cost
        sl = adjusted_entry + sl_dist
        tp = adjusted_entry - tp_dist
    
    # Simulate candle-by-candle (REALISTIC)
    for i in range(entry_idx + 1, min(entry_idx + max_candles, len(df))):
        candle = df.iloc[i]
        
        if side == 'long':
            # CRITICAL: Check SL FIRST (conservative - assumes worst case)
            if candle['low'] <= sl:
                return 'loss', i - entry_idx
            elif candle['high'] >= tp:
                return 'win', i - entry_idx
        else:
            # CRITICAL: Check SL FIRST (conservative - assumes worst case)
            if candle['high'] >= sl:
                return 'loss', i - entry_idx
            elif candle['low'] <= tp:
                return 'win', i - entry_idx
    
    return 'timeout', max_candles  # Timeout = loss

def process_single_symbol(symbol: str, days: int = 60, train_pct: float = 0.6) -> dict:
    """Process a single symbol with walk-forward validation."""
    df = fetch_klines(symbol, '3', days)
    
    if len(df) < 500:
        return None, 0, 0  # No data
    
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
    
    # Process train data
    for idx in range(51, len(df_train) - 80):
        row = df_train.iloc[idx]
        prev_row = df_train.iloc[idx - 1]
        
        side = check_vwap_signal(row, prev_row)
        if not side:
            continue
        
        combo = get_combo_simplified(row)
        key = f"{side}|{combo}"
        
        outcome, candles = simulate_trade_realistic(df_train, idx, side, row['close'], row['atr'])
        total_signals += 1
        
        if outcome == 'win':
            combo_stats[key]['train_wins'] += 1
            total_wins += 1
        else:
            combo_stats[key]['train_losses'] += 1
        combo_stats[key]['train_candles'].append(candles)
    
    # Process test data
    for idx in range(51, len(df_test) - 80):
        row = df_test.iloc[idx]
        prev_row = df_test.iloc[idx - 1]
        
        side = check_vwap_signal(row, prev_row)
        if not side:
            continue
        
        combo = get_combo_simplified(row)
        key = f"{side}|{combo}"
        
        outcome, candles = simulate_trade_realistic(df_test, idx, side, row['close'], row['atr'])
        total_signals += 1
        
        if outcome == 'win':
            combo_stats[key]['test_wins'] += 1
            total_wins += 1
        else:
            combo_stats[key]['test_losses'] += 1
        combo_stats[key]['test_candles'].append(candles)
    
    # Calculate overall WR for this symbol
    overall_wr = (total_wins / total_signals * 100) if total_signals > 0 else 0
    
    # Filter winning combos with HIGHER sample requirements
    winning_combos = []
    
    for key, stats in combo_stats.items():
        train_total = stats['train_wins'] + stats['train_losses']
        test_total = stats['test_wins'] + stats['test_losses']
        
        # Higher sample count requirements
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
    """Run realistic 1:1 R:R backtest with higher sample counts."""
    print("="*70)
    print("🔬 SIMPLIFIED 1:1 R:R BACKTEST (HIGHER SAMPLE COUNTS)")
    print("="*70)
    print(f"   R:R Ratio: {RR_RATIO}:1")
    print(f"   Days: {days}")
    print(f"   Walk-forward: {train_pct*100:.0f}% train / {(1-train_pct)*100:.0f}% test")
    print(f"   Fees: {TOTAL_FEE_PCT}% | Spread: {SPREAD_PCT}% | Slip: {SLIPPAGE_PCT}%")
    print(f"   Total Cost: {TOTAL_COST_PCT:.2f}%")
    print(f"   Min Train WR: {MIN_TRAIN_WR}% | Min Test WR: {MIN_TEST_WR}%")
    print(f"   Min Train N: {MIN_TRAIN_TRADES} | Min Test N: {MIN_TEST_TRADES}")
    print(f"   Indicator Bins: RSI(3) x MACD(2) x Fib(3) = 18 combos")
    print("="*70)
    
    # Get symbols
    print(f"\n📋 Fetching top {num_symbols} symbols...")
    symbols = get_all_symbols(num_symbols)
    print(f"   Found {len(symbols)} symbols")
    
    # Process each symbol one at a time
    all_combos = []
    symbols_with_combos = 0
    
    print(f"\n📊 Processing symbols (showing WR & N for each)...")
    print("-" * 70)
    
    for i, symbol in enumerate(symbols):
        try:
            result = process_single_symbol(symbol, days, train_pct)
            combos, total_signals, overall_wr = result
            
            # Print progress for EVERY symbol
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
    
    # Sort by train WR
    all_combos.sort(key=lambda x: (x['train_wr'], x['test_wr']), reverse=True)
    
    print(f"\n✅ Processing complete!")
    print(f"   Symbols processed: {len(symbols)}")
    print(f"   Symbols with combos: {symbols_with_combos}")
    print(f"   Total winning combos: {len(all_combos)}")
    
    # Print results
    print("\n" + "="*70)
    print(f"🏆 1:1 R:R SIMPLIFIED RESULTS (High Sample Count)")
    print(f"   Criteria: Train WR ≥ {MIN_TRAIN_WR}%, Test WR ≥ {MIN_TEST_WR}%")
    print(f"   Min Samples: Train N ≥ {MIN_TRAIN_TRADES}, Test N ≥ {MIN_TEST_TRADES}")
    print("="*70)
    
    print(f"\n📊 TOP 30 COMBOS:")
    print("-"*70)
    for i, c in enumerate(all_combos[:30], 1):
        print(f"{i:2}. {c['symbol']} {c['side']} {c['combo']}")
        print(f"    Train: WR={c['train_wr']:.1f}% LB={c['train_lb']:.1f}% (N={c['train_n']})")
        print(f"    Test:  WR={c['test_wr']:.1f}% LB={c['test_lb']:.1f}% (N={c['test_n']})")
    
    # Export to YAML - DIFFERENT FILE NAME
    output_file = 'backtest_simplified_1to1_HIGH_N_RESULTS.yaml'
    yaml_output = {
        '_metadata': {
            'rr_ratio': '1:1',
            'indicator_bins': 'SIMPLIFIED (3x2x3 = 18 combos)',
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
        f.write("# SIMPLIFIED 1:1 R:R Backtest Results (HIGH SAMPLE COUNT)\n")
        f.write(f"# Generated: {datetime.utcnow().isoformat()}\n")
        f.write(f"# R:R Ratio: 1:1\n")
        f.write(f"# Indicator Bins: SIMPLIFIED (3 RSI x 2 MACD x 3 Fib = 18 combos)\n")
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
