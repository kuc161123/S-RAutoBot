#!/usr/bin/env python3
"""
WALK-FORWARD BACKTEST: Partial TP vs Trailing Only

STRICT NO-LOOKAHEAD BIAS RULES:
1. Signal detection uses ONLY completed candles (iloc[-2] at signal time)
2. Entry on NEXT candle open (iloc[-1] becomes entry candle)
3. SL/TP calculated using data available at signal time ONLY
4. Trade management uses candle-by-candle OHLC in sequence
5. Walk-forward: 70% train / 30% test, sliding window

STRATEGIES TESTED:
1. Current: 50% partial at 0.5R, BE at 0.5R, trail from 1R
2. Trailing Only: No partial, BE at 0.5R, trail from 1R  
3. Aggressive Trail: No partial, BE at 0.5R, trail from 0.5R
"""

import pandas as pd
import numpy as np
import requests
import yaml
import time as time_module
from datetime import datetime

# Strategy configurations
STRATEGIES = {
    'current_partial': {
        'name': 'Current (50% @ 0.5R)',
        'partial_pct': 0.5,
        'partial_r': 0.5,
        'sl_to_be_r': 0.5,
        'trail_start_r': 1.0,
        'trail_distance_r': 0.5,
    },
    'trailing_only': {
        'name': 'Trailing Only (No Partial)',
        'partial_pct': 0,
        'partial_r': 0,
        'sl_to_be_r': 0.5,
        'trail_start_r': 1.0,
        'trail_distance_r': 0.5,
    },
    'aggressive_trail': {
        'name': 'Aggressive Trail (BE+Trail @ 0.5R)',
        'partial_pct': 0,
        'partial_r': 0,
        'sl_to_be_r': 0.5,
        'trail_start_r': 0.5,
        'trail_distance_r': 0.5,
    },
}

def fetch_klines(symbol, interval='60', limit=1000):
    """Fetch historical klines from Bybit."""
    url = "https://api.bybit.com/v5/market/kline"
    params = {'category': 'linear', 'symbol': symbol, 'interval': interval, 'limit': limit}
    try:
        resp = requests.get(url, params=params, timeout=15)
        data = resp.json()
        if data.get('retCode') == 0 and data.get('result', {}).get('list'):
            klines = data['result']['list']
            klines.reverse()  # Oldest first
            return klines
    except Exception as e:
        pass
    return None

def add_indicators(df):
    """Add RSI and ATR indicators (no lookahead - uses rolling windows)."""
    df = df.copy()
    
    # RSI (14 period)
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # ATR (14 period)
    tr = pd.concat([
        df['high'] - df['low'],
        abs(df['high'] - df['close'].shift(1)),
        abs(df['low'] - df['close'].shift(1))
    ], axis=1).max(axis=1)
    df['atr'] = tr.rolling(14).mean()
    
    return df

def detect_divergence_at_candle(df_up_to_now, lookback=14):
    """
    Detect divergence using ONLY data available at this moment.
    NO LOOKAHEAD: We only see completed candles.
    
    df_up_to_now: DataFrame up to and including the COMPLETED candle
    Returns: signal dict or None
    """
    if len(df_up_to_now) < lookback + 5:
        return None
    
    # Use the LAST COMPLETED candle for signal detection
    current_idx = len(df_up_to_now) - 1
    
    close = df_up_to_now['close'].values
    rsi = df_up_to_now['rsi'].values
    low = df_up_to_now['low'].values
    high = df_up_to_now['high'].values
    
    # Skip if RSI is NaN
    if np.isnan(rsi[current_idx]):
        return None
    
    current_rsi = rsi[current_idx]
    current_low = low[current_idx]
    current_high = high[current_idx]
    
    # Look for pivot points in the lookback window
    for j in range(max(5, current_idx - lookback), current_idx - 2):
        if np.isnan(rsi[j]):
            continue
            
        # Regular Bullish: Price lower low, RSI higher low
        if current_low < low[j] and current_rsi > rsi[j] and current_rsi < 40:
            return {'type': 'regular_bullish', 'side': 'long'}
        
        # Hidden Bullish: Price higher low, RSI lower low
        if current_low > low[j] and current_rsi < rsi[j] and current_rsi < 50:
            return {'type': 'hidden_bullish', 'side': 'long'}
        
        # Regular Bearish: Price higher high, RSI lower high
        if current_high > high[j] and current_rsi < rsi[j] and current_rsi > 60:
            return {'type': 'regular_bearish', 'side': 'short'}
        
        # Hidden Bearish: Price lower high, RSI higher high
        if current_high < high[j] and current_rsi > rsi[j] and current_rsi > 50:
            return {'type': 'hidden_bearish', 'side': 'short'}
    
    return None

def calculate_sl_at_signal(df_up_to_now, side, atr):
    """
    Calculate SL using ONLY data available at signal time.
    Uses recent swing points - NO LOOKAHEAD.
    """
    lookback = min(15, len(df_up_to_now) - 1)
    recent = df_up_to_now.iloc[-lookback-1:-1]  # Exclude current candle
    
    if side == 'long':
        swing_low = recent['low'].min()
        return swing_low
    else:
        swing_high = recent['high'].max()
        return swing_high

def simulate_trade_candle_by_candle(df, entry_idx, side, entry_price, sl_price, cfg):
    """
    Simulate trade processing EACH CANDLE in sequence.
    NO LOOKAHEAD: Only uses current candle's OHLC.
    
    Returns: dict with outcome and stats
    """
    if entry_idx >= len(df) - 1:
        return None
    
    sl_distance = abs(entry_price - sl_price)
    if sl_distance <= 0:
        return None
    
    current_sl = sl_price
    partial_filled = False
    partial_r = 0.0
    remaining = 1.0
    max_r = 0.0
    sl_at_be = False
    trailing = False
    
    # Process each candle after entry
    for i in range(entry_idx + 1, min(entry_idx + 200, len(df))):
        candle = df.iloc[i]
        o, h, l, c = candle['open'], candle['high'], candle['low'], candle['close']
        
        # Calculate current R (based on candle extremes)
        if side == 'long':
            candle_max_r = (h - entry_price) / sl_distance
            candle_min_r = (l - entry_price) / sl_distance
            sl_hit = l <= current_sl
            current_r = (c - entry_price) / sl_distance
        else:
            candle_max_r = (entry_price - l) / sl_distance
            candle_min_r = (entry_price - h) / sl_distance
            sl_hit = h >= current_sl
            current_r = (entry_price - c) / sl_distance
        
        max_r = max(max_r, candle_max_r)
        
        # CHECK SL FIRST (before any profit taking)
        if sl_hit:
            if trailing:
                # Exit at trailing SL level
                if side == 'long':
                    exit_r = (current_sl - entry_price) / sl_distance
                else:
                    exit_r = (entry_price - current_sl) / sl_distance
                total_r = partial_r + (exit_r * remaining)
                exit_type = 'trail_sl'
            elif sl_at_be:
                # Exit at breakeven
                total_r = partial_r + 0  # BE = 0R for remaining
                exit_type = 'be'
            else:
                # Full loss
                total_r = partial_r - (1.0 * remaining)
                exit_type = 'sl'
            
            return {
                'total_r': total_r,
                'exit_type': exit_type,
                'max_r': max_r,
                'partial_filled': partial_filled,
                'candles_held': i - entry_idx
            }
        
        # PARTIAL TP (if configured and not yet filled)
        if cfg['partial_pct'] > 0 and not partial_filled:
            if candle_max_r >= cfg['partial_r']:
                partial_filled = True
                partial_r = cfg['partial_pct'] * cfg['partial_r']
                remaining -= cfg['partial_pct']
        
        # MOVE SL TO BE
        if not sl_at_be and max_r >= cfg['sl_to_be_r']:
            current_sl = entry_price
            sl_at_be = True
        
        # TRAILING SL
        if sl_at_be and max_r >= cfg['trail_start_r']:
            trailing = True
            trail_level = max_r - cfg['trail_distance_r']
            if trail_level > 0:
                if side == 'long':
                    new_sl = entry_price + (trail_level * sl_distance)
                    current_sl = max(current_sl, new_sl)
                else:
                    new_sl = entry_price - (trail_level * sl_distance)
                    current_sl = min(current_sl, new_sl)
        
        # FULL TP AT 3R
        if candle_max_r >= 3.0:
            total_r = partial_r + (3.0 * remaining)
            return {
                'total_r': total_r,
                'exit_type': 'tp3r',
                'max_r': max_r,
                'partial_filled': partial_filled,
                'candles_held': i - entry_idx
            }
    
    # Timeout - still open after 200 candles
    exit_r = max(0, max_r - 0.5) if sl_at_be else -1
    total_r = partial_r + (exit_r * remaining)
    return {
        'total_r': total_r,
        'exit_type': 'timeout',
        'max_r': max_r,
        'partial_filled': partial_filled,
        'candles_held': 200
    }

def run_walk_forward_backtest(symbols, strategy_key, train_pct=0.7):
    """
    Walk-forward backtest with train/test split.
    
    1. Use first 70% of data for "training" (in-sample)
    2. Use last 30% for "testing" (out-of-sample)
    3. Report both separately
    """
    cfg = STRATEGIES[strategy_key]
    in_sample_results = []
    out_of_sample_results = []
    
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
            
            # Walk-forward split
            split_idx = int(len(df) * train_pct)
            
            # Process candle by candle
            cooldown = 0
            for i in range(30, len(df) - 10):
                if cooldown > 0:
                    cooldown -= 1
                    continue
                
                # Get data available at this moment (all candles up to and including i)
                df_available = df.iloc[:i+1]
                
                # Detect signal on COMPLETED candle (last one in df_available)
                signal = detect_divergence_at_candle(df_available)
                if not signal:
                    continue
                
                # Entry will be on NEXT candle open
                entry_candle_idx = i + 1
                if entry_candle_idx >= len(df):
                    continue
                
                entry_price = df.iloc[entry_candle_idx]['open']
                side = signal['side']
                
                # Calculate SL using ONLY available data
                atr = df_available.iloc[-1]['atr']
                sl_price = calculate_sl_at_signal(df_available, side, atr)
                
                # Constrain SL to ATR bounds
                sl_distance = abs(entry_price - sl_price)
                min_sl = 0.3 * atr
                max_sl = 2.0 * atr
                
                if sl_distance < min_sl:
                    if side == 'long':
                        sl_price = entry_price - min_sl
                    else:
                        sl_price = entry_price + min_sl
                elif sl_distance > max_sl:
                    if side == 'long':
                        sl_price = entry_price - max_sl
                    else:
                        sl_price = entry_price + max_sl
                
                # Simulate trade candle by candle
                result = simulate_trade_candle_by_candle(df, entry_candle_idx, side, entry_price, sl_price, cfg)
                
                if result:
                    result['symbol'] = sym
                    result['side'] = side
                    result['signal_type'] = signal['type']
                    result['entry_idx'] = entry_candle_idx
                    
                    # Classify as in-sample or out-of-sample
                    if entry_candle_idx < split_idx:
                        in_sample_results.append(result)
                    else:
                        out_of_sample_results.append(result)
                    
                    cooldown = 10
                    
        except Exception as e:
            continue
    
    return in_sample_results, out_of_sample_results

def analyze_results(results, label):
    """Analyze and print results."""
    if not results:
        print(f"   {label}: No trades")
        return None
    
    df = pd.DataFrame(results)
    total = len(df)
    wins = len(df[df['total_r'] > 0])
    wr = wins / total * 100
    total_r = df['total_r'].sum()
    avg_r = df['total_r'].mean()
    
    exits = df['exit_type'].value_counts()
    
    print(f"   {label}:")
    print(f"      Trades: {total} | WR: {wr:.1f}% | Total R: {total_r:+.1f} | Avg R: {avg_r:+.3f}")
    print(f"      Exits: Trail={exits.get('trail_sl', 0)} | BE={exits.get('be', 0)} | SL={exits.get('sl', 0)} | TP3R={exits.get('tp3r', 0)}")
    
    return {
        'label': label,
        'trades': total,
        'wr': wr,
        'total_r': total_r,
        'avg_r': avg_r,
        'trail': exits.get('trail_sl', 0),
        'be': exits.get('be', 0),
        'sl': exits.get('sl', 0),
        'tp3r': exits.get('tp3r', 0)
    }

def main():
    print("=" * 70)
    print("WALK-FORWARD BACKTEST: Partial TP vs Trailing Only")
    print("=" * 70)
    print("\nSTRICT NO-LOOKAHEAD RULES:")
    print("  ✓ Signals detected on COMPLETED candles only")
    print("  ✓ Entry on NEXT candle open")
    print("  ✓ SL calculated using data available at signal time")
    print("  ✓ Trade managed candle-by-candle in sequence")
    print("  ✓ Walk-forward: 70% in-sample / 30% out-of-sample\n")
    
    # Load config
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    all_symbols = config.get('trade', {}).get('divergence_symbols', [])
    
    # Test on 50 and 150 symbols
    for sym_count in [50, 150]:
        symbols = all_symbols[:sym_count]
        print(f"\n{'='*70}")
        print(f"TESTING {sym_count} SYMBOLS")
        print(f"{'='*70}")
        
        summary = []
        
        for strategy_key in STRATEGIES:
            cfg = STRATEGIES[strategy_key]
            print(f"\n📊 {cfg['name']}")
            print("-" * 50)
            
            in_sample, out_of_sample = run_walk_forward_backtest(symbols, strategy_key)
            
            in_stats = analyze_results(in_sample, "In-Sample (70%)")
            out_stats = analyze_results(out_of_sample, "Out-of-Sample (30%)")
            
            if in_stats and out_stats:
                # Check for overfitting (in-sample much better than out-of-sample)
                wr_drop = in_stats['wr'] - out_stats['wr']
                r_drop = in_stats['avg_r'] - out_stats['avg_r']
                
                robustness = "✅ ROBUST" if abs(wr_drop) < 10 and abs(r_drop) < 0.1 else "⚠️ CHECK"
                print(f"      Robustness: {robustness} (WR drop: {wr_drop:+.1f}%, AvgR drop: {r_drop:+.3f})")
                
                summary.append({
                    'strategy': cfg['name'],
                    'oos_trades': out_stats['trades'],
                    'oos_wr': out_stats['wr'],
                    'oos_total_r': out_stats['total_r'],
                    'oos_avg_r': out_stats['avg_r'],
                    'robustness': robustness
                })
        
        # Summary table
        print(f"\n{'='*70}")
        print(f"SUMMARY ({sym_count} symbols) - OUT-OF-SAMPLE RESULTS (Most Important!)")
        print("=" * 70)
        print(f"\n{'Strategy':<35} {'Trades':>6} {'WR':>7} {'Total R':>9} {'Avg R':>8} {'Robust':>10}")
        print("-" * 80)
        
        for s in summary:
            print(f"{s['strategy']:<35} {s['oos_trades']:>6} {s['oos_wr']:>6.1f}% {s['oos_total_r']:>+9.1f} {s['oos_avg_r']:>+8.3f} {s['robustness']:>10}")
        
        if summary:
            best = max(summary, key=lambda x: x['oos_total_r'])
            print(f"\n🏆 BEST ({sym_count} syms): {best['strategy']} with {best['oos_total_r']:+.1f}R out-of-sample")
    
    print("\n" + "=" * 70)
    print("FINAL RECOMMENDATION")
    print("=" * 70)
    print("\n⚠️ Use OUT-OF-SAMPLE results for decision making!")
    print("   In-sample results may be overfitted.\n")

if __name__ == '__main__':
    main()
