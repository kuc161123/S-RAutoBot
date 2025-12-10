#!/usr/bin/env python3
"""
1:1 R:R Backtest Analysis (COMPARISON ONLY)

This script runs a realistic backtest with 1:1 R:R ratio to compare against
the current 2:1 R:R strategy. Results are saved to a separate file and
DO NOT affect the live bot.

Features:
- Walk-forward validation (60% train, 40% test)
- Realistic execution costs (spread + slippage)
- Detailed results with WR and N samples
- SL checked before TP on same-candle hits
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

# Execution costs (realistic)
SPREAD_PCT = 0.01
SLIPPAGE_PCT = 0.02
TOTAL_COST_PCT = SPREAD_PCT + SLIPPAGE_PCT

# R:R RATIO FOR THIS BACKTEST
RR_RATIO = 1.0  # 1:1 R:R

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
    """Fetch top symbols from Bybit by volume."""
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
    """Calculate technical indicators matching bot.py logic."""
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
    
    # VWAP approximation (cumulative)
    df['tp'] = (df['high'] + df['low'] + df['close']) / 3
    df['tp_v'] = df['tp'] * df['volume'].astype(float)
    df['cum_tp_v'] = df['tp_v'].cumsum()
    df['cum_v'] = df['volume'].astype(float).cumsum()
    df['vwap'] = df['cum_tp_v'] / df['cum_v']
    
    return df.dropna()

def get_combo(row) -> str:
    """Generate combo string matching bot.py get_combo()."""
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

def simulate_trade(df, entry_idx, side, entry, atr, rr_ratio=1.0, max_candles=80):
    """
    Simulate a trade with realistic execution.
    
    CRITICAL: Check SL before TP if both hit in same candle (conservative).
    """
    MIN_SL_PCT = 0.5
    min_sl_dist = entry * (MIN_SL_PCT / 100)
    sl_dist = max(atr, min_sl_dist)
    tp_dist = sl_dist * rr_ratio  # 1:1 R:R
    
    # Add execution costs
    cost = entry * (TOTAL_COST_PCT / 100)
    
    if side == 'long':
        adjusted_entry = entry + cost
        sl = adjusted_entry - sl_dist
        tp = adjusted_entry + tp_dist
    else:
        adjusted_entry = entry - cost
        sl = adjusted_entry + sl_dist
        tp = adjusted_entry - tp_dist
    
    # Simulate candle-by-candle
    for i in range(entry_idx + 1, min(entry_idx + max_candles, len(df))):
        candle = df.iloc[i]
        
        if side == 'long':
            # CRITICAL: Check SL FIRST (conservative approach)
            if candle['low'] <= sl:
                return 'loss'
            elif candle['high'] >= tp:
                return 'win'
        else:
            # CRITICAL: Check SL FIRST (conservative approach)
            if candle['high'] >= sl:
                return 'loss'
            elif candle['low'] <= tp:
                return 'win'
    
    return 'timeout'  # Count as loss

def run_backtest(num_symbols=400, days=60, train_pct=0.6):
    """Run backtest with 1:1 R:R and walk-forward validation."""
    print(f"🔬 1:1 R:R Backtest Analysis")
    print(f"   (Comparison only - does not affect live bot)")
    print("="*70)
    print(f"   R:R Ratio: {RR_RATIO}:1")
    print(f"   Days: {days}")
    print(f"   Walk-forward: {train_pct*100:.0f}% train / {(1-train_pct)*100:.0f}% test")
    print("="*70)
    
    # Get symbols
    print(f"\n📋 Fetching top {num_symbols} symbols...")
    symbols = get_all_symbols(num_symbols)
    print(f"   Found {len(symbols)} symbols")
    
    # Fetch data
    symbol_data = {}
    print(f"\n📥 Fetching {days}-day data...")
    
    for i, symbol in enumerate(symbols):
        if (i+1) % 50 == 0 or i == 0:
            print(f"   [{i+1}/{len(symbols)}] {symbol}...", flush=True)
        df = fetch_klines(symbol, '3', days)
        if len(df) > 500:
            df = calculate_indicators(df)
            symbol_data[symbol] = df
        time.sleep(0.05)
    
    print(f"\n✅ Loaded data for {len(symbol_data)} symbols")
    
    # Walk-forward validation with VWAP signal detection
    combo_stats = defaultdict(lambda: {
        'train_wins': 0, 'train_losses': 0,
        'test_wins': 0, 'test_losses': 0
    })
    
    print(f"\n📊 Running Walk-Forward Backtest with VWAP signals...")
    
    total_signals = 0
    for symbol, df in symbol_data.items():
        total_candles = len(df)
        train_end = int(total_candles * train_pct)
        
        df_train = df.iloc[:train_end]
        df_test = df.iloc[train_end:]
        
        for phase, df_phase in [('train', df_train), ('test', df_test)]:
            # Process every candle looking for VWAP signals
            for idx in range(51, len(df_phase) - 80):
                row = df_phase.iloc[idx]
                prev_row = df_phase.iloc[idx - 1]
                
                # Check for VWAP signal (matching bot logic)
                side = check_vwap_signal(row, prev_row)
                if not side:
                    continue
                
                total_signals += 1
                combo = get_combo(row)
                key = f"{symbol}|{side}|{combo}"
                
                # Simulate trade with 1:1 R:R
                outcome = simulate_trade(df_phase, idx, side, row['close'], row['atr'], RR_RATIO)
                
                if outcome == 'win':
                    combo_stats[key][f'{phase}_wins'] += 1
                elif outcome in ['loss', 'timeout']:
                    combo_stats[key][f'{phase}_losses'] += 1
    
    print(f"   Total signals processed: {total_signals}")
    
    # Filter winning combos
    MIN_TRAIN_TRADES = 10
    MIN_TRAIN_WR = 50
    MIN_TEST_WR = 50
    
    winning_combos = []
    all_combos = []  # Track all combos for comparison
    
    for key, stats in combo_stats.items():
        train_total = stats['train_wins'] + stats['train_losses']
        test_total = stats['test_wins'] + stats['test_losses']
        
        if train_total < 5:  # Minimum sample
            continue
        
        train_wr = stats['train_wins'] / train_total * 100
        test_wr = stats['test_wins'] / test_total * 100 if test_total > 0 else 0
        train_lb = wilson_lower_bound(stats['train_wins'], train_total)
        test_lb = wilson_lower_bound(stats['test_wins'], test_total) if test_total > 0 else 0
        
        symbol, side, combo = key.split('|')
        
        combo_data = {
            'symbol': symbol,
            'side': side,
            'combo': combo,
            'train_n': train_total,
            'train_wr': train_wr,
            'train_lb': train_lb,
            'test_n': test_total,
            'test_wr': test_wr,
            'test_lb': test_lb
        }
        
        all_combos.append(combo_data)
        
        if train_total >= MIN_TRAIN_TRADES and train_wr >= MIN_TRAIN_WR and test_wr >= MIN_TEST_WR:
            winning_combos.append(combo_data)
    
    winning_combos.sort(key=lambda x: (x['test_wr'], x['train_wr']), reverse=True)
    
    # Print results
    print("\n" + "="*70)
    print(f"🏆 1:1 R:R RESULTS (Train WR ≥ {MIN_TRAIN_WR}% AND Test WR ≥ {MIN_TEST_WR}%)")
    print("="*70)
    print(f"\nFound {len(winning_combos)} reliable combos with 1:1 R:R")
    
    print("\n📊 TOP 30 COMBOS:")
    print("-"*70)
    for i, c in enumerate(winning_combos[:30], 1):
        print(f"{i:2}. {c['symbol']} {c['side']} {c['combo']}")
        print(f"    Train: WR={c['train_wr']:.1f}% LB={c['train_lb']:.1f}% (N={c['train_n']})")
        print(f"    Test:  WR={c['test_wr']:.1f}% LB={c['test_lb']:.1f}% (N={c['test_n']})")
    
    # Export to YAML (SEPARATE FILE - does not affect bot)
    output_file = 'backtest_1to1_rr_RESULTS.yaml'
    yaml_output = {
        '_metadata': {
            'rr_ratio': '1:1',
            'generated': datetime.utcnow().isoformat(),
            'symbols_tested': len(symbol_data),
            'total_signals': total_signals,
            'winning_combos': len(winning_combos),
            'criteria': f'Train WR >= {MIN_TRAIN_WR}%, Test WR >= {MIN_TEST_WR}%, Min N >= {MIN_TRAIN_TRADES}',
            'note': 'COMPARISON ONLY - Does not affect live bot'
        }
    }
    
    for c in winning_combos:
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
        
        # Store combo with full stats
        combo_with_stats = f"{combo} [WR:{c['train_wr']:.0f}%/{c['test_wr']:.0f}% N:{c['train_n']}/{c['test_n']}]"
        
        if side == 'long':
            yaml_output[sym]['allowed_combos_long'].append(combo)
            yaml_output[sym]['stats_long'][combo] = {
                'train_wr': round(c['train_wr'], 1),
                'train_lb': round(c['train_lb'], 1),
                'train_n': c['train_n'],
                'test_wr': round(c['test_wr'], 1),
                'test_lb': round(c['test_lb'], 1),
                'test_n': c['test_n']
            }
        else:
            yaml_output[sym]['allowed_combos_short'].append(combo)
            yaml_output[sym]['stats_short'][combo] = {
                'train_wr': round(c['train_wr'], 1),
                'train_lb': round(c['train_lb'], 1),
                'train_n': c['train_n'],
                'test_wr': round(c['test_wr'], 1),
                'test_lb': round(c['test_lb'], 1),
                'test_n': c['test_n']
            }
    
    with open(output_file, 'w') as f:
        f.write("# 1:1 R:R Backtest Results (COMPARISON ONLY)\n")
        f.write(f"# Generated: {datetime.utcnow().isoformat()}\n")
        f.write(f"# R:R Ratio: 1:1\n")
        f.write(f"# Symbols tested: {len(symbol_data)}\n")
        f.write(f"# Total signals: {total_signals}\n")
        f.write(f"# Winning combos: {len(winning_combos)}\n")
        f.write(f"# Criteria: Train WR >= {MIN_TRAIN_WR}%, Test WR >= {MIN_TEST_WR}%, Min N >= {MIN_TRAIN_TRADES}\n")
        f.write("#\n")
        f.write("# NOTE: This file is for COMPARISON ONLY.\n")
        f.write("# The live bot uses backtest_golden_combos.yaml with 2:1 R:R.\n\n")
        yaml.dump(yaml_output, f, default_flow_style=False, sort_keys=False)
    
    print(f"\n✅ Results saved to: {output_file}")
    print(f"   Symbols with combos: {len([k for k in yaml_output.keys() if k != '_metadata'])}")
    print(f"   Total winning combos: {len(winning_combos)}")
    print(f"\n⚠️ NOTE: This is for comparison only. Bot still uses 2:1 R:R.")
    
    # Summary comparison hint
    print("\n" + "="*70)
    print("📈 To compare with current 2:1 R:R results:")
    print("   - Current bot has 164 combos at 2:1 R:R")
    print(f"   - This 1:1 R:R test found {len(winning_combos)} combos")
    print("="*70)
    
    return winning_combos

if __name__ == "__main__":
    run_backtest(num_symbols=400, days=60, train_pct=0.6)
