#!/usr/bin/env python3
"""
GRID SEARCH: Find Optimal Trailing Configuration

Tests all combinations of:
- BE level: 0.3R, 0.5R, 0.7R
- Trail start: 0.5R, 0.7R, 1.0R, 1.5R
- Trail distance: 0.3R, 0.5R, 0.7R, 1.0R

STRICT NO-LOOKAHEAD + WALK-FORWARD VALIDATION
Only out-of-sample results shown for decision making
"""

import pandas as pd
import numpy as np
import requests
import yaml
from itertools import product

def fetch_klines(symbol, interval='60', limit=1000):
    """Fetch klines from Bybit."""
    url = "https://api.bybit.com/v5/market/kline"
    params = {'category': 'linear', 'symbol': symbol, 'interval': interval, 'limit': limit}
    try:
        resp = requests.get(url, params=params, timeout=15)
        data = resp.json()
        if data.get('retCode') == 0 and data.get('result', {}).get('list'):
            klines = data['result']['list']
            klines.reverse()
            return klines
    except:
        pass
    return None

def add_indicators(df):
    """Add RSI and ATR."""
    df = df.copy()
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    df['rsi'] = 100 - (100 / (1 + rs))
    tr = pd.concat([
        df['high'] - df['low'],
        abs(df['high'] - df['close'].shift(1)),
        abs(df['low'] - df['close'].shift(1))
    ], axis=1).max(axis=1)
    df['atr'] = tr.rolling(14).mean()
    return df

def detect_divergence(df_up_to_now, lookback=14):
    """Detect divergence - NO LOOKAHEAD."""
    if len(df_up_to_now) < lookback + 5:
        return None
    current_idx = len(df_up_to_now) - 1
    close = df_up_to_now['close'].values
    rsi = df_up_to_now['rsi'].values
    low = df_up_to_now['low'].values
    high = df_up_to_now['high'].values
    
    if np.isnan(rsi[current_idx]):
        return None
    
    current_rsi = rsi[current_idx]
    current_low = low[current_idx]
    current_high = high[current_idx]
    
    for j in range(max(5, current_idx - lookback), current_idx - 2):
        if np.isnan(rsi[j]):
            continue
        if current_low < low[j] and current_rsi > rsi[j] and current_rsi < 40:
            return {'type': 'regular_bullish', 'side': 'long'}
        if current_low > low[j] and current_rsi < rsi[j] and current_rsi < 50:
            return {'type': 'hidden_bullish', 'side': 'long'}
        if current_high > high[j] and current_rsi < rsi[j] and current_rsi > 60:
            return {'type': 'regular_bearish', 'side': 'short'}
        if current_high < high[j] and current_rsi > rsi[j] and current_rsi > 50:
            return {'type': 'hidden_bearish', 'side': 'short'}
    return None

def calculate_sl(df_up_to_now, side, atr):
    """Calculate SL - NO LOOKAHEAD."""
    lookback = min(15, len(df_up_to_now) - 1)
    recent = df_up_to_now.iloc[-lookback-1:-1]
    if side == 'long':
        return recent['low'].min()
    else:
        return recent['high'].max()

def simulate_trade(df, entry_idx, side, entry_price, sl_price, be_r, trail_start_r, trail_dist_r):
    """Simulate trade candle-by-candle - NO LOOKAHEAD."""
    if entry_idx >= len(df) - 1:
        return None
    
    sl_distance = abs(entry_price - sl_price)
    if sl_distance <= 0:
        return None
    
    current_sl = sl_price
    max_r = 0.0
    sl_at_be = False
    trailing = False
    
    for i in range(entry_idx + 1, min(entry_idx + 200, len(df))):
        candle = df.iloc[i]
        h, l = candle['high'], candle['low']
        
        if side == 'long':
            candle_max_r = (h - entry_price) / sl_distance
            sl_hit = l <= current_sl
        else:
            candle_max_r = (entry_price - l) / sl_distance
            sl_hit = h >= current_sl
        
        max_r = max(max_r, candle_max_r)
        
        # SL hit
        if sl_hit:
            if trailing:
                if side == 'long':
                    exit_r = (current_sl - entry_price) / sl_distance
                else:
                    exit_r = (entry_price - current_sl) / sl_distance
                total_r = exit_r
            elif sl_at_be:
                total_r = 0
            else:
                total_r = -1.0
            return {'total_r': total_r, 'exit': 'trail_sl' if trailing else ('be' if sl_at_be else 'sl'), 'max_r': max_r}
        
        # Move to BE
        if not sl_at_be and max_r >= be_r:
            current_sl = entry_price
            sl_at_be = True
        
        # Start trailing
        if sl_at_be and max_r >= trail_start_r:
            trailing = True
            trail_level = max_r - trail_dist_r
            if trail_level > 0:
                if side == 'long':
                    new_sl = entry_price + (trail_level * sl_distance)
                    current_sl = max(current_sl, new_sl)
                else:
                    new_sl = entry_price - (trail_level * sl_distance)
                    current_sl = min(current_sl, new_sl)
        
        # TP at 3R
        if candle_max_r >= 3.0:
            return {'total_r': 3.0, 'exit': 'tp3r', 'max_r': max_r}
    
    # Timeout
    exit_r = max(0, max_r - 0.5) if sl_at_be else -1
    return {'total_r': exit_r, 'exit': 'timeout', 'max_r': max_r}

def run_backtest(symbols, be_r, trail_start_r, trail_dist_r, train_pct=0.7):
    """Run walk-forward backtest for one configuration."""
    in_sample = []
    out_sample = []
    
    for sym in symbols:
        try:
            klines = fetch_klines(sym, '60', 1000)
            if not klines or len(klines) < 200:
                continue
            
            df = pd.DataFrame(klines, columns=['start', 'open', 'high', 'low', 'close', 'volume', 'turnover'])
            for c in ['open', 'high', 'low', 'close', 'volume']:
                df[c] = df[c].astype(float)
            
            df = add_indicators(df)
            df = df.dropna().reset_index(drop=True)
            
            if len(df) < 100:
                continue
            
            split_idx = int(len(df) * train_pct)
            cooldown = 0
            
            for i in range(30, len(df) - 10):
                if cooldown > 0:
                    cooldown -= 1
                    continue
                
                df_available = df.iloc[:i+1]
                signal = detect_divergence(df_available)
                if not signal:
                    continue
                
                entry_idx = i + 1
                if entry_idx >= len(df):
                    continue
                
                entry_price = df.iloc[entry_idx]['open']
                side = signal['side']
                atr = df_available.iloc[-1]['atr']
                sl_price = calculate_sl(df_available, side, atr)
                
                sl_distance = abs(entry_price - sl_price)
                min_sl = 0.3 * atr
                max_sl = 2.0 * atr
                
                if sl_distance < min_sl:
                    sl_price = entry_price - min_sl if side == 'long' else entry_price + min_sl
                elif sl_distance > max_sl:
                    sl_price = entry_price - max_sl if side == 'long' else entry_price + max_sl
                
                result = simulate_trade(df, entry_idx, side, entry_price, sl_price, be_r, trail_start_r, trail_dist_r)
                
                if result:
                    result['symbol'] = sym
                    if entry_idx < split_idx:
                        in_sample.append(result)
                    else:
                        out_sample.append(result)
                    cooldown = 10
        except:
            continue
    
    return in_sample, out_sample

def analyze(results):
    """Analyze results."""
    if not results:
        return None
    df = pd.DataFrame(results)
    total = len(df)
    wins = len(df[df['total_r'] > 0])
    wr = wins / total * 100
    total_r = df['total_r'].sum()
    avg_r = df['total_r'].mean()
    return {'trades': total, 'wr': wr, 'total_r': total_r, 'avg_r': avg_r}

def main():
    print("=" * 80)
    print("GRID SEARCH: Optimal Trailing Configuration")
    print("=" * 80)
    print("\nTesting 150 symbols with walk-forward validation...")
    print("Only OUT-OF-SAMPLE results shown (robust!)\n")
    
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    symbols = config.get('trade', {}).get('divergence_symbols', [])[:150]
    
    # Grid search parameters
    be_levels = [0.3, 0.5, 0.7]
    trail_starts = [0.5, 0.7, 1.0, 1.5]
    trail_distances = [0.3, 0.5, 0.7, 1.0]
    
    results = []
    total_configs = len(be_levels) * len(trail_starts) * len(trail_distances)
    current = 0
    
    print(f"Testing {total_configs} configurations...\n")
    
    for be_r in be_levels:
        for trail_start_r in trail_starts:
            for trail_dist_r in trail_distances:
                current += 1
                
                # Skip invalid configs (trail start must be >= BE)
                if trail_start_r < be_r:
                    continue
                
                # Skip configs where trail distance >= trail start (doesn't make sense)
                if trail_dist_r >= trail_start_r:
                    continue
                
                config_name = f"BE:{be_r}R Start:{trail_start_r}R Dist:{trail_dist_r}R"
                print(f"[{current}/{total_configs}] Testing {config_name}...", end="\r")
                
                in_sample, out_sample = run_backtest(symbols, be_r, trail_start_r, trail_dist_r)
                
                in_stats = analyze(in_sample)
                out_stats = analyze(out_sample)
                
                if in_stats and out_stats and out_stats['trades'] >= 1000:
                    # Check robustness
                    wr_drop = in_stats['wr'] - out_stats['wr']
                    avgr_drop = in_stats['avg_r'] - out_stats['avg_r']
                    
                    # Robust if WR drop < 10% and AvgR drop < 0.1
                    is_robust = abs(wr_drop) < 10 and abs(avgr_drop) < 0.1
                    
                    results.append({
                        'be_r': be_r,
                        'trail_start_r': trail_start_r,
                        'trail_dist_r': trail_dist_r,
                        'config': config_name,
                        'oos_trades': out_stats['trades'],
                        'oos_wr': out_stats['wr'],
                        'oos_total_r': out_stats['total_r'],
                        'oos_avg_r': out_stats['avg_r'],
                        'wr_drop': wr_drop,
                        'avgr_drop': avgr_drop,
                        'robust': is_robust
                    })
    
    print("\n" + "=" * 80)
    print("RESULTS (Out-of-Sample Only)")
    print("=" * 80)
    
    # Sort by out-of-sample total R
    results.sort(key=lambda x: x['oos_total_r'], reverse=True)
    
    print(f"\n{'Rank':<5} {'Config':<30} {'Trades':>7} {'WR':>7} {'Total R':>9} {'Avg R':>8} {'Robust':>7}")
    print("-" * 85)
    
    for i, r in enumerate(results[:20], 1):
        robust_icon = "✅" if r['robust'] else "❌"
        print(f"{i:<5} {r['config']:<30} {r['oos_trades']:>7} {r['oos_wr']:>6.1f}% {r['oos_total_r']:>+9.1f} {r['oos_avg_r']:>+8.3f} {robust_icon:>7}")
    
    # Show only robust results
    robust_results = [r for r in results if r['robust']]
    
    if robust_results:
        print("\n" + "=" * 80)
        print("TOP 10 ROBUST CONFIGURATIONS (Recommended)")
        print("=" * 80)
        print(f"\n{'Rank':<5} {'Config':<30} {'Trades':>7} {'WR':>7} {'Total R':>9} {'Avg R':>8}")
        print("-" * 80)
        
        for i, r in enumerate(robust_results[:10], 1):
            print(f"{i:<5} {r['config']:<30} {r['oos_trades']:>7} {r['oos_wr']:>6.1f}% {r['oos_total_r']:>+9.1f} {r['oos_avg_r']:>+8.3f}")
        
        best = robust_results[0]
        print(f"\n🏆 BEST ROBUST CONFIG:")
        print(f"   BE at: {best['be_r']}R")
        print(f"   Trail Start: {best['trail_start_r']}R")
        print(f"   Trail Distance: {best['trail_dist_r']}R")
        print(f"   Out-of-Sample: {best['oos_total_r']:+.1f}R ({best['oos_trades']} trades, {best['oos_wr']:.1f}% WR)")
        print(f"   Robustness: WR drop {best['wr_drop']:+.1f}%, AvgR drop {best['avgr_drop']:+.3f}")
    else:
        print("\n⚠️ No robust configurations found!")

if __name__ == '__main__':
    main()
