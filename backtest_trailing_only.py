#!/usr/bin/env python3
"""
Backtest: Compare Partial TP vs Trailing Only

Strategy 1: Current (50% at 0.5R + trail from 1R)
Strategy 2: No partial TP, just trail from 0.5R (BE at 0.5R)
"""

import pandas as pd
import numpy as np
import requests
import yaml

STRATEGIES = {
    'current_partial': {
        'name': 'Current (50% @ 0.5R + Trail)',
        'partial_pct': 0.5,
        'partial_r': 0.5,
        'sl_to_be_at_r': 0.5,
        'trail_start_r': 1.0,
        'trail_distance_r': 0.5,
    },
    'trailing_only': {
        'name': 'Trailing Only (No Partial TP)',
        'partial_pct': 0,  # No partial
        'partial_r': 0,
        'sl_to_be_at_r': 0.5,  # Still move to BE at 0.5R
        'trail_start_r': 1.0,  # Trail from 1R
        'trail_distance_r': 0.5,  # Trail 0.5R behind
    },
    'aggressive_trail': {
        'name': 'Aggressive Trail (BE 0.5R, Trail 0.5R)',
        'partial_pct': 0,
        'partial_r': 0,
        'sl_to_be_at_r': 0.5,
        'trail_start_r': 0.5,  # Start trailing immediately at 0.5R
        'trail_distance_r': 0.5,
    },
}

def fetch_klines(symbol, interval='60', limit=1000):
    """Fetch klines from Bybit public API."""
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

def prepare_df(df):
    """Add RSI and ATR."""
    df = df.copy()
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    df['rsi'] = 100 - (100 / (1 + rs))
    tr = pd.concat([
        df['high'] - df['low'],
        abs(df['high'] - df['close'].shift()),
        abs(df['low'] - df['close'].shift())
    ], axis=1).max(axis=1)
    df['atr'] = tr.rolling(14).mean()
    return df.dropna()

def find_divergence(df, lookback=14):
    """Simple divergence detection."""
    if len(df) < lookback + 5:
        return []
    signals = []
    close = df['close'].values
    rsi = df['rsi'].values
    low = df['low'].values
    high = df['high'].values
    i = len(df) - 1
    
    for j in range(max(5, i - lookback), i - 2):
        if low[i] < low[j] and rsi[i] > rsi[j] and rsi[i] < 40:
            signals.append({'type': 'regular_bullish', 'side': 'long'})
            break
        if low[i] > low[j] and rsi[i] < rsi[j] and rsi[i] < 50:
            signals.append({'type': 'hidden_bullish', 'side': 'long'})
            break
        if high[i] > high[j] and rsi[i] < rsi[j] and rsi[i] > 60:
            signals.append({'type': 'regular_bearish', 'side': 'short'})
            break
        if high[i] < high[j] and rsi[i] > rsi[j] and rsi[i] > 50:
            signals.append({'type': 'hidden_bearish', 'side': 'short'})
            break
    return signals

def simulate_trade(df, idx, side, sl_distance, cfg):
    """Simulate trade with strategy config."""
    if idx + 2 >= len(df):
        return None
    
    entry = df.iloc[idx + 1]['open']
    current_sl = entry - sl_distance if side == 'long' else entry + sl_distance
    partial_filled = False
    partial_r = 0
    remaining = 1.0
    max_r = 0
    sl_at_be = False
    trailing = False
    
    for j in range(idx + 2, min(idx + 200, len(df))):
        candle = df.iloc[j]
        h, l = candle['high'], candle['low']
        
        if side == 'long':
            r_high = (h - entry) / sl_distance
            r_low = (l - entry) / sl_distance
            sl_hit = l <= current_sl
        else:
            r_high = (entry - l) / sl_distance
            r_low = (entry - h) / sl_distance
            sl_hit = h >= current_sl
        
        max_r = max(max_r, r_high)
        
        # SL hit
        if sl_hit:
            if trailing:
                # Calculate exit R based on where SL was
                if side == 'long':
                    exit_r = (current_sl - entry) / sl_distance
                else:
                    exit_r = (entry - current_sl) / sl_distance
                total_r = partial_r + (exit_r * remaining)
            elif sl_at_be:
                total_r = partial_r  # Exit at BE = 0 for remaining
            else:
                total_r = partial_r - remaining  # Full loss on remaining
            
            return {
                'total_r': total_r, 
                'exit': 'trail_sl' if trailing else ('be' if sl_at_be else 'sl'),
                'partial_filled': partial_filled, 
                'max_r': max_r
            }
        
        # Partial TP (only if enabled)
        if cfg['partial_pct'] > 0 and not partial_filled and r_high >= cfg['partial_r']:
            partial_filled = True
            partial_r = cfg['partial_pct'] * cfg['partial_r']
            remaining -= cfg['partial_pct']
        
        # SL to BE
        if not sl_at_be and max_r >= cfg['sl_to_be_at_r']:
            current_sl = entry
            sl_at_be = True
        
        # Trailing
        if sl_at_be and max_r >= cfg['trail_start_r']:
            trailing = True
            trail = max_r - cfg['trail_distance_r']
            new_sl = entry + trail * sl_distance if side == 'long' else entry - trail * sl_distance
            if (side == 'long' and new_sl > current_sl) or (side == 'short' and new_sl < current_sl):
                current_sl = new_sl
        
        # Full TP at 3R
        if r_high >= 3.0:
            total_r = partial_r + (3.0 * remaining)
            return {'total_r': total_r, 'exit': 'tp3r', 'partial_filled': partial_filled, 'max_r': max_r}
    
    # Timeout - exit at current level
    return {'total_r': partial_r + (max_r * remaining * 0.5), 'exit': 'timeout', 'partial_filled': partial_filled, 'max_r': max_r}

def run_backtest(symbols, strategy_key):
    """Run backtest for one strategy."""
    cfg = STRATEGIES[strategy_key]
    results = []
    
    print(f"\n{'='*50}")
    print(f"Testing: {cfg['name']}")
    print(f"{'='*50}")
    
    for sym in symbols:
        try:
            klines = fetch_klines(sym, '60', 1000)
            if not klines or len(klines) < 100:
                continue
            
            df = pd.DataFrame(klines, columns=['start', 'open', 'high', 'low', 'close', 'volume', 'turnover'])
            for c in ['open', 'high', 'low', 'close', 'volume']:
                df[c] = df[c].astype(float)
            
            df = prepare_df(df)
            if len(df) < 50:
                continue
            
            cooldown = 0
            for i in range(30, len(df) - 10):
                if cooldown > 0:
                    cooldown -= 1
                    continue
                
                signals = find_divergence(df.iloc[:i+1])
                if not signals:
                    continue
                
                sig = signals[0]
                side = sig['side']
                atr = df.iloc[i]['atr']
                swing_low = df.iloc[max(0,i-15):i+1]['low'].min()
                swing_high = df.iloc[max(0,i-15):i+1]['high'].max()
                entry = df.iloc[i+1]['open']
                
                sl_dist = entry - swing_low if side == 'long' else swing_high - entry
                sl_dist = max(0.3 * atr, min(sl_dist, 2.0 * atr))
                
                if sl_dist <= 0:
                    continue
                
                result = simulate_trade(df, i, side, sl_dist, cfg)
                if result:
                    result['symbol'] = sym
                    result['side'] = side
                    results.append(result)
                    cooldown = 10
        except Exception as e:
            continue
    
    return results

def analyze(results, name):
    """Analyze results."""
    if not results:
        print(f"No trades for {name}")
        return None
    
    df = pd.DataFrame(results)
    total = len(df)
    wins = len(df[df['total_r'] > 0])
    wr = wins / total * 100
    total_r = df['total_r'].sum()
    avg_r = df['total_r'].mean()
    
    # Exit type breakdown
    exits = df['exit'].value_counts()
    trail_exits = exits.get('trail_sl', 0)
    be_exits = exits.get('be', 0)
    sl_exits = exits.get('sl', 0)
    tp_exits = exits.get('tp3r', 0)
    
    print(f"\n📊 {name}")
    print(f"   Trades: {total} | WR: {wr:.1f}% | Total R: {total_r:+.1f}")
    print(f"   Avg R: {avg_r:+.3f}")
    print(f"   Exits: Trail={trail_exits} | BE={be_exits} | SL={sl_exits} | TP3R={tp_exits}")
    
    return {'name': name, 'trades': total, 'wr': wr, 'total_r': total_r, 'avg_r': avg_r,
            'trail_exits': trail_exits, 'be_exits': be_exits, 'sl_exits': sl_exits, 'tp_exits': tp_exits}

def main():
    print("=" * 60)
    print("BACKTEST: Partial TP vs Trailing Only")
    print("=" * 60)
    
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    symbols = config.get('trade', {}).get('divergence_symbols', [])[:30]
    print(f"\nTesting {len(symbols)} symbols...")
    
    all_stats = []
    for key in STRATEGIES:
        results = run_backtest(symbols, key)
        stats = analyze(results, STRATEGIES[key]['name'])
        if stats:
            all_stats.append(stats)
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY COMPARISON")
    print("=" * 60)
    print(f"\n{'Strategy':<35} {'Trades':>6} {'WR':>7} {'Total R':>9} {'Avg R':>8}")
    print("-" * 70)
    
    for s in all_stats:
        print(f"{s['name']:<35} {s['trades']:>6} {s['wr']:>6.1f}% {s['total_r']:>+9.1f} {s['avg_r']:>+8.3f}")
    
    if all_stats:
        best = max(all_stats, key=lambda x: x['total_r'])
        print(f"\n🏆 BEST: {best['name']} with {best['total_r']:+.1f}R")
        
        current = next((s for s in all_stats if 'Current' in s['name']), None)
        trail_only = next((s for s in all_stats if 'Trailing Only' in s['name']), None)
        
        if current and trail_only:
            diff = trail_only['total_r'] - current['total_r']
            print(f"\n{'✅' if diff > 0 else '❌'} Trailing Only vs Partial TP: {diff:+.1f}R difference")
            print(f"\n   Current (50% @ 0.5R):")
            print(f"   → Total R: {current['total_r']:+.1f} | WR: {current['wr']:.1f}%")
            print(f"   → Trail exits: {current['trail_exits']} | BE exits: {current['be_exits']}")
            print(f"\n   Trailing Only (No Partial):")
            print(f"   → Total R: {trail_only['total_r']:+.1f} | WR: {trail_only['wr']:.1f}%")
            print(f"   → Trail exits: {trail_only['trail_exits']} | BE exits: {trail_only['be_exits']}")

if __name__ == '__main__':
    main()
