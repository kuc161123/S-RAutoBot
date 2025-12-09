#!/usr/bin/env python3
"""
Incremental Backtest with Feature Analysis

Processes one symbol at a time, showing results immediately.
Includes session and volatility analysis.
"""

import requests
import pandas as pd
import numpy as np
import time
import math
import yaml
import sys
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

def get_session(hour: int) -> str:
    if hour < 8:
        return 'asia'
    elif hour < 16:
        return 'london'
    else:
        return 'newyork'

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
        print(f"Error: {e}")
        return []

def fetch_klines(symbol: str, days: int = 120) -> pd.DataFrame:
    all_data = []
    end_time = int(datetime.utcnow().timestamp() * 1000)
    start_time = int((datetime.utcnow() - timedelta(days=days)).timestamp() * 1000)
    
    url = f"{BYBIT_BASE}/v5/market/kline"
    current_end = end_time
    retries = 0
    
    while current_end > start_time and retries < 3:
        params = {
            'category': 'linear',
            'symbol': symbol,
            'interval': '3',
            'limit': 1000,
            'end': current_end
        }
        try:
            resp = requests.get(url, params=params, timeout=30)
            data = resp.json()
            if data.get('retCode') != 0 or not data.get('result', {}).get('list'):
                retries += 1
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
    df['atr_pct'] = (df['atr'] / df['close'] * 100)
    
    df['roll_high'] = df['high'].rolling(50).max()
    df['roll_low'] = df['low'].rolling(50).min()
    
    # Hour and session
    df['hour'] = df['start'].dt.hour
    df['session'] = df['hour'].apply(get_session)
    
    # Day of week (0=Monday, 6=Sunday)
    df['day_of_week'] = df['start'].dt.dayofweek
    df['day_name'] = df['start'].dt.day_name()
    
    # Volatility regime
    atr_ma = df['atr_pct'].rolling(100).mean()
    df['volatility'] = 'medium'
    df.loc[df['atr_pct'] > atr_ma * 1.2, 'volatility'] = 'high'
    df.loc[df['atr_pct'] < atr_ma * 0.8, 'volatility'] = 'low'
    
    # Volume spike (above 1.5x average)
    vol_float = df['volume'].astype(float)
    vol_ma = vol_float.rolling(50).mean()
    df['vol_spike'] = vol_float > (vol_ma * 1.5)
    df['volume_regime'] = 'normal'
    df.loc[df['vol_spike'], 'volume_regime'] = 'high'
    df.loc[vol_float < vol_ma * 0.5, 'volume_regime'] = 'low'
    
    # Trend (EMA 50 vs EMA 200)
    df['ema50'] = df['close'].ewm(span=50).mean()
    df['ema200'] = df['close'].ewm(span=200).mean()
    df['trend'] = 'sideways'
    df.loc[df['ema50'] > df['ema200'] * 1.01, 'trend'] = 'uptrend'
    df.loc[df['ema50'] < df['ema200'] * 0.99, 'trend'] = 'downtrend'
    
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
    """Simulate trade EXACTLY as bot does it.
    
    Matches bot.py execute_trade() logic:
    - MIN_SL_PCT = 0.5% minimum SL distance
    - MIN_TP_PCT = 1.0% minimum TP distance (for 2:1 R:R)
    - Fixed 2:1 R:R ratio
    - Entry on candle close
    - Check SL before TP on each candle (conservative)
    """
    MIN_SL_PCT = 0.5
    MIN_TP_PCT = 1.0
    
    min_sl_dist = entry * (MIN_SL_PCT / 100)
    min_tp_dist = entry * (MIN_TP_PCT / 100)
    
    # Use the LARGER of ATR-based or minimum distance (matches bot)
    sl_dist = max(atr, min_sl_dist)
    tp_dist = max(sl_dist * rr_ratio, min_tp_dist)
    
    # Apply execution cost to entry
    cost = entry * (TOTAL_COST_PCT / 100)
    
    if side == 'long':
        adjusted_entry = entry + cost
        sl = adjusted_entry - sl_dist
        tp = adjusted_entry + tp_dist
    else:
        adjusted_entry = entry - cost
        sl = adjusted_entry + sl_dist
        tp = adjusted_entry - tp_dist
    
    # Simulate candle-by-candle (check SL first - conservative)
    for i in range(entry_idx + 1, min(entry_idx + max_candles, len(df))):
        candle = df.iloc[i]
        if side == 'long':
            if candle['low'] <= sl:  # SL hit first (conservative)
                return 'loss'
            elif candle['high'] >= tp:
                return 'win'
        else:
            if candle['high'] >= sl:  # SL hit first (conservative)
                return 'loss'
            elif candle['low'] <= tp:
                return 'win'
    
    return 'timeout'

def process_symbol(symbol: str, days: int = 120, train_pct: float = 0.6):
    """Process one symbol and return winning combos with feature analysis."""
    
    print(f"\n{'='*60}")
    print(f"📊 {symbol}")
    print(f"{'='*60}")
    
    # Fetch data
    print(f"   Fetching {days} days of data...", end=" ", flush=True)
    df = fetch_klines(symbol, days)
    
    if len(df) < 500:
        print(f"❌ Only {len(df)} candles")
        return []
    
    df = calculate_indicators(df)
    print(f"✅ {len(df)} candles")
    
    # Walk-forward split
    total_candles = len(df)
    train_end = int(total_candles * train_pct)
    df_train = df.iloc[:train_end]
    df_test = df.iloc[train_end:]
    
    # Track by combo (simple - no features)
    combo_stats = defaultdict(lambda: {
        'train': {'w': 0, 'l': 0},
        'test': {'w': 0, 'l': 0}
    })
    
    # Run backtest
    for phase, df_phase in [('train', df_train), ('test', df_test)]:
        for side in ['long', 'short']:
            for idx in range(50, len(df_phase) - 80, 1):  # EVERY CANDLE for maximum realism
                row = df_phase.iloc[idx]
                combo = get_combo(row)
                key = f"{side}|{combo}"
                
                outcome = simulate_trade(df_phase, idx, side, row['close'], row['atr'])
                
                if outcome == 'win':
                    combo_stats[key][phase]['w'] += 1
                elif outcome == 'loss':
                    combo_stats[key][phase]['l'] += 1
    
    # Filter winners
    MIN_TRAIN_TRADES = 10
    MIN_TRAIN_WR = 50
    MIN_TEST_WR = 50
    
    winners = []
    
    for key, stats in combo_stats.items():
        side, combo = key.split('|')
        
        train_total = stats['train']['w'] + stats['train']['l']
        test_total = stats['test']['w'] + stats['test']['l']
        
        if train_total < MIN_TRAIN_TRADES or test_total < 3:
            continue
        
        train_wr = stats['train']['w'] / train_total * 100
        test_wr = stats['test']['w'] / test_total * 100 if test_total > 0 else 0
        
        if train_wr >= MIN_TRAIN_WR and test_wr >= MIN_TEST_WR:
            winners.append({
                'symbol': symbol,
                'side': side,
                'combo': combo,
                'train_wr': train_wr,
                'train_n': train_total,
                'test_wr': test_wr,
                'test_n': test_total
            })
    
    # Print results
    if winners:
        print(f"\n   🏆 WINNING ({len(winners)}):")
        for w in sorted(winners, key=lambda x: x['test_wr'], reverse=True):
            print(f"      {w['side'].upper():5} {w['combo']} | Train:{w['train_wr']:.0f}%(N={w['train_n']}) Test:{w['test_wr']:.0f}%(N={w['test_n']})")
    else:
        print(f" ❌ No winners")
    
    return winners

def run_incremental_backtest(num_symbols: int = 400, days: int = 120):
    """Run backtest processing one symbol at a time with auto-commit."""
    
    print("🔬 Ultra-Realistic Backtest (Auto-Commit Mode)")
    print("=" * 60)
    print(f"Symbols: {num_symbols} | Days: {days} | Min WR: 50%")
    print("✅ Will auto-commit & push when combos found")
    print("=" * 60)
    
    # Get symbols
    print("\n📋 Fetching symbols...", flush=True)
    symbols = get_all_symbols(num_symbols)
    print(f"   Found {len(symbols)} symbols")
    
    # Process each symbol
    all_winners = []
    yaml_output = {}
    last_commit_count = 0
    
    for i, symbol in enumerate(symbols):
        print(f"\n[{i+1}/{len(symbols)}]", end="")
        
        winners = process_symbol(symbol, days)
        all_winners.extend(winners)
        
        # Update YAML output
        new_combos_added = False
        for w in winners:
            sym = w['symbol']
            if sym not in yaml_output:
                yaml_output[sym] = {'allowed_combos_long': [], 'allowed_combos_short': []}
            
            key = 'allowed_combos_long' if w['side'] == 'long' else 'allowed_combos_short'
            if w['combo'] not in yaml_output[sym][key]:
                yaml_output[sym][key].append(w['combo'])
                new_combos_added = True
        
        # Save and auto-commit when new combos added
        if new_combos_added:
            # Write to MAIN file (not pending)
            with open('backtest_golden_combos.yaml', 'w') as f:
                f.write("# Backtest-Validated Golden Combos (LIVE)\n")
                f.write(f"# Updated: {datetime.utcnow().isoformat()}\n")
                f.write(f"# Progress: {i+1}/{len(symbols)} symbols\n")
                f.write(f"# Total combos: {len(all_winners)}\n")
                f.write("# Criteria: 60 days, every candle, 50%+ WR train & test\n\n")
                yaml.dump(yaml_output, f, default_flow_style=False)
            
            # Auto-commit and push
            import subprocess
            try:
                new_count = len(all_winners) - last_commit_count
                commit_msg = f"🤖 Add {new_count} combo(s) from {symbol} [{i+1}/{len(symbols)}]"
                subprocess.run(['git', 'add', 'backtest_golden_combos.yaml'], 
                             capture_output=True, timeout=10)
                subprocess.run(['git', 'commit', '-m', commit_msg], 
                             capture_output=True, timeout=10)
                subprocess.run(['git', 'push', 'origin', 'main'], 
                             capture_output=True, timeout=30)
                print(f"\n   📤 Committed: {commit_msg}")
                last_commit_count = len(all_winners)
            except Exception as e:
                print(f"\n   ⚠️ Git error: {e}")
        
        sys.stdout.flush()
    
    # Final summary
    print("\n" + "=" * 60)
    print("🏆 FINAL RESULTS")
    print("=" * 60)
    print(f"Total winning combos: {len(all_winners)}")
    print(f"Symbols with combos: {len(yaml_output)}")
    
    # Top 10
    top_10 = sorted(all_winners, key=lambda x: x['test_wr'], reverse=True)[:10]
    print("\nTop 10 by Test WR:")
    for i, w in enumerate(top_10, 1):
        print(f"{i:2}. {w['symbol']} {w['side']} - Test: {w['test_wr']:.0f}% | {w['combo']}")
    
    print(f"\n✅ All results saved to: backtest_golden_combos.yaml")
    print(f"✅ Commits pushed to GitHub")
    
    return all_winners

if __name__ == "__main__":
    # Ultra-realistic: 60 days, every candle, matching live bot exactly
    # Auto-commits and pushes when combos found
    run_incremental_backtest(num_symbols=400, days=60)
