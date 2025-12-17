#!/usr/bin/env python3
"""
Backtest: Compare Partial TP at 0.5R vs 1R

Compares exit strategies:
1. Current: 50% at 1R, SL to BE at 1R, trail from 2R
2. Proposed: 50% at 0.5R, SL to BE at 1R, trail from 2R  
3. Aggressive: 50% at 0.5R, SL to BE at 0.5R, trail from 1.5R
4. Hybrid: 25% at 0.5R, 25% at 1R, SL to BE at 1R, trail from 2R
"""

import pandas as pd
import numpy as np
import requests
import yaml

# Strategy configurations
STRATEGIES = {
    'current_1r': {
        'name': 'Current (50% @ 1R)',
        'partial_1_pct': 0.5,
        'partial_1_r': 1.0,
        'sl_to_be_at_r': 1.0,
        'trail_start_r': 2.0,
        'trail_distance_r': 1.0,
    },
    'proposed_05r': {
        'name': 'Proposed (50% @ 0.5R)',
        'partial_1_pct': 0.5,
        'partial_1_r': 0.5,
        'sl_to_be_at_r': 1.0,
        'trail_start_r': 2.0,
        'trail_distance_r': 1.0,
    },
    'aggressive_05r': {
        'name': 'Aggressive (50% @ 0.5R, BE @ 0.5R)',
        'partial_1_pct': 0.5,
        'partial_1_r': 0.5,
        'sl_to_be_at_r': 0.5,
        'trail_start_r': 1.5,
        'trail_distance_r': 1.0,
    },
    'hybrid': {
        'name': 'Hybrid (25% @ 0.5R, 25% @ 1R)',
        'partial_1_pct': 0.25,
        'partial_1_r': 0.5,
        'partial_2_pct': 0.25,
        'partial_2_r': 1.0,
        'sl_to_be_at_r': 1.0,
        'trail_start_r': 2.0,
        'trail_distance_r': 1.0,
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
            klines.reverse()  # Oldest first
            return klines
    except:
        pass
    return None

def prepare_df(df):
    """Add indicators."""
    df = df.copy()
    # RSI
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    df['rsi'] = 100 - (100 / (1 + rs))
    # ATR
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
    
    # Check last bar for divergence
    i = len(df) - 1
    
    # Find recent swing lows/highs
    for j in range(max(5, i - lookback), i - 2):
        # Regular bullish: price LL, RSI HL
        if low[i] < low[j] and rsi[i] > rsi[j] and rsi[i] < 40:
            signals.append({'type': 'regular_bullish', 'side': 'long'})
            break
        # Hidden bullish: price HL, RSI LL  
        if low[i] > low[j] and rsi[i] < rsi[j] and rsi[i] < 50:
            signals.append({'type': 'hidden_bullish', 'side': 'long'})
            break
        # Regular bearish: price HH, RSI LH
        if high[i] > high[j] and rsi[i] < rsi[j] and rsi[i] > 60:
            signals.append({'type': 'regular_bearish', 'side': 'short'})
            break
        # Hidden bearish: price LH, RSI HH
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
    partial_1_filled = False
    partial_2_filled = False
    partial_1_r = 0
    partial_2_r = 0
    remaining = 1.0
    max_r = 0
    sl_at_be = False
    
    for j in range(idx + 2, min(idx + 200, len(df))):
        candle = df.iloc[j]
        h, l = candle['high'], candle['low']
        
        if side == 'long':
            r_high = (h - entry) / sl_distance
            sl_hit = l <= current_sl
        else:
            r_high = (entry - l) / sl_distance
            sl_hit = h >= current_sl
        
        max_r = max(max_r, r_high)
        
        # SL hit
        if sl_hit:
            exit_r = 0 if sl_at_be else -1.0
            total_r = partial_1_r + partial_2_r + (exit_r * remaining)
            return {'total_r': total_r, 'exit': 'sl' if not sl_at_be else 'be', 
                    'partial_1': partial_1_filled, 'max_r': max_r}
        
        # Partial 1
        if not partial_1_filled and r_high >= cfg['partial_1_r']:
            partial_1_filled = True
            partial_1_r = cfg['partial_1_pct'] * cfg['partial_1_r']
            remaining -= cfg['partial_1_pct']
        
        # Partial 2 (hybrid)
        if 'partial_2_r' in cfg and not partial_2_filled and r_high >= cfg['partial_2_r']:
            partial_2_filled = True
            partial_2_r = cfg['partial_2_pct'] * cfg['partial_2_r']
            remaining -= cfg['partial_2_pct']
        
        # SL to BE
        if not sl_at_be and max_r >= cfg['sl_to_be_at_r']:
            current_sl = entry
            sl_at_be = True
        
        # Trailing
        if sl_at_be and max_r >= cfg['trail_start_r']:
            trail = max_r - cfg['trail_distance_r']
            new_sl = entry + trail * sl_distance if side == 'long' else entry - trail * sl_distance
            if (side == 'long' and new_sl > current_sl) or (side == 'short' and new_sl < current_sl):
                current_sl = new_sl
        
        # Full TP at 3R
        if r_high >= 3.0:
            total_r = partial_1_r + partial_2_r + (3.0 * remaining)
            return {'total_r': total_r, 'exit': 'tp3r', 'partial_1': partial_1_filled, 'max_r': max_r}
    
    # Timeout
    return {'total_r': partial_1_r + partial_2_r, 'exit': 'timeout', 'partial_1': partial_1_filled, 'max_r': max_r}

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
            
            # Scan for signals
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
                
                # Calc SL
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
                    result['type'] = sig['type']
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
    partial_rate = df['partial_1'].sum() / total * 100
    
    print(f"\n📊 {name}")
    print(f"   Trades: {total} | WR: {wr:.1f}% | Total R: {total_r:+.1f}")
    print(f"   Avg R: {avg_r:+.3f} | Partial Fill: {partial_rate:.1f}%")
    
    return {'name': name, 'trades': total, 'wr': wr, 'total_r': total_r, 'avg_r': avg_r, 'partial_rate': partial_rate}

def main():
    print("=" * 60)
    print("BACKTEST: Comparing Partial TP at 0.5R vs 1R")
    print("=" * 60)
    
    # Load symbols
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
    print(f"\n{'Strategy':<40} {'Trades':>6} {'WR':>7} {'Total R':>9} {'Avg R':>8}")
    print("-" * 70)
    
    for s in all_stats:
        print(f"{s['name']:<40} {s['trades']:>6} {s['wr']:>6.1f}% {s['total_r']:>+9.1f} {s['avg_r']:>+8.3f}")
    
    if all_stats:
        best = max(all_stats, key=lambda x: x['total_r'])
        print(f"\n🏆 BEST: {best['name']} with {best['total_r']:+.1f}R")
        
        current = next((s for s in all_stats if 'Current' in s['name']), None)
        proposed = next((s for s in all_stats if 'Proposed' in s['name']), None)
        
        if current and proposed:
            diff = proposed['total_r'] - current['total_r']
            print(f"\n{'✅' if diff > 0 else '❌'} 0.5R vs 1R: {diff:+.1f}R difference")
            print(f"   1R (current):   {current['total_r']:+.1f}R | {current['wr']:.1f}% WR | {current['partial_rate']:.0f}% partial fill")
            print(f"   0.5R (proposed): {proposed['total_r']:+.1f}R | {proposed['wr']:.1f}% WR | {proposed['partial_rate']:.0f}% partial fill")

if __name__ == '__main__':
    main()
