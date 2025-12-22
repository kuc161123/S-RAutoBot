#!/usr/bin/env python3
"""
ROBUST BACKTEST: Aggressive Trail Strategy
==========================================

Tests the Aggressive Trail configuration with FULL robustness:
- Walk-Forward Validation (70% IS / 30% OOS)
- 50+ symbols
- StochRSI filter (K>80 shorts, K<20 longs)
- ATR-based SL (1.0x ATR)
- Commission: 0.055% per leg
- NO lookahead bias
- Multiple timeframes: 3M and 1H

AGGRESSIVE TRAIL CONFIG:
- BE at: 0.5R
- Trail start: 0.5R (immediately after BE)
- Trail distance: 0.5R behind max price
"""

import pandas as pd
import numpy as np
import requests
import yaml
from datetime import datetime

# ============================================
# CONFIGURATION
# ============================================
NUM_SYMBOLS = 50
DAYS_DATA = 60
COMMISSION_PER_LEG = 0.00055  # 0.055%

# Aggressive Trail Config
BE_R = 0.5
TRAIL_START_R = 0.5
TRAIL_DISTANCE_R = 0.5

# Comparison: Current Bot Config
CURRENT_BE_R = 0.7
CURRENT_TRAIL_START_R = 0.7
CURRENT_TRAIL_DISTANCE_R = 0.3

# StochRSI Filter
STOCH_K_PERIOD = 14
OVERBOUGHT = 80
OVERSOLD = 20

# ============================================
# HELPER FUNCTIONS
# ============================================
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

def get_top_symbols(n=50):
    """Get top N symbols by 24h volume."""
    url = "https://api.bybit.com/v5/market/tickers"
    params = {'category': 'linear'}
    try:
        resp = requests.get(url, params=params, timeout=15)
        data = resp.json()
        if data.get('retCode') == 0:
            tickers = data.get('result', {}).get('list', [])
            usdt = [t for t in tickers if t['symbol'].endswith('USDT')]
            usdt.sort(key=lambda x: float(x.get('volume24h', 0)), reverse=True)
            return [t['symbol'] for t in usdt[:n]]
    except:
        pass
    return []

def add_indicators(df):
    """Add RSI, ATR, and StochRSI."""
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
        abs(df['high'] - df['close'].shift(1)),
        abs(df['low'] - df['close'].shift(1))
    ], axis=1).max(axis=1)
    df['atr'] = tr.rolling(14).mean()
    
    # StochRSI
    rsi = df['rsi']
    rsi_min = rsi.rolling(STOCH_K_PERIOD).min()
    rsi_max = rsi.rolling(STOCH_K_PERIOD).max()
    df['stoch_k'] = ((rsi - rsi_min) / (rsi_max - rsi_min + 1e-10)) * 100
    
    # Volume filter
    df['vol_ma'] = df['volume'].rolling(20).mean()
    df['vol_ok'] = df['volume'] > df['vol_ma'] * 0.5
    
    return df

def detect_divergence(df_up_to_now, lookback=14):
    """Detect divergence - NO LOOKAHEAD."""
    if len(df_up_to_now) < lookback + 5:
        return None
    
    i = len(df_up_to_now) - 1
    close = df_up_to_now['close'].values
    rsi = df_up_to_now['rsi'].values
    low = df_up_to_now['low'].values
    high = df_up_to_now['high'].values
    
    if np.isnan(rsi[i]):
        return None
    
    for j in range(max(5, i - lookback), i - 2):
        if np.isnan(rsi[j]):
            continue
        
        # Regular Bullish: Price LL, RSI HL
        if low[i] < low[j] and rsi[i] > rsi[j] and rsi[i] < 45:
            return {'type': 'regular_bullish', 'side': 'long'}
        
        # Hidden Bullish: Price HL, RSI LL
        if low[i] > low[j] and rsi[i] < rsi[j] and rsi[i] < 50:
            return {'type': 'hidden_bullish', 'side': 'long'}
        
        # Regular Bearish: Price HH, RSI LH
        if high[i] > high[j] and rsi[i] < rsi[j] and rsi[i] > 55:
            return {'type': 'regular_bearish', 'side': 'short'}
        
        # Hidden Bearish: Price LH, RSI HH
        if high[i] < high[j] and rsi[i] > rsi[j] and rsi[i] > 50:
            return {'type': 'hidden_bearish', 'side': 'short'}
    
    return None

def simulate_trade(df, entry_idx, side, entry_price, sl_distance, be_r, trail_start_r, trail_dist_r):
    """Simulate trade with trailing strategy - NO LOOKAHEAD."""
    if entry_idx >= len(df) - 1:
        return None
    
    if sl_distance <= 0:
        return None
    
    # Initial SL
    if side == 'long':
        current_sl = entry_price - sl_distance
    else:
        current_sl = entry_price + sl_distance
    
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
            elif sl_at_be:
                exit_r = 0
            else:
                exit_r = -1.0
            
            # Subtract commission
            exit_r -= 2 * COMMISSION_PER_LEG / (sl_distance / entry_price)
            
            return {
                'total_r': exit_r,
                'exit': 'trail_sl' if trailing else ('be' if sl_at_be else 'sl'),
                'max_r': max_r
            }
        
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
            exit_r = 3.0 - 2 * COMMISSION_PER_LEG / (sl_distance / entry_price)
            return {'total_r': exit_r, 'exit': 'tp3r', 'max_r': max_r}
    
    # Timeout = LOSS
    return {'total_r': -1.0, 'exit': 'timeout', 'max_r': max_r}

def run_backtest(symbols, interval, be_r, trail_start_r, trail_dist_r, use_stoch_filter=True):
    """Run walk-forward backtest."""
    in_sample = []
    out_sample = []
    filtered_signals = 0
    total_signals = 0
    
    for sym in symbols:
        try:
            klines = fetch_klines(sym, interval, 1000)
            if not klines or len(klines) < 200:
                continue
            
            df = pd.DataFrame(klines, columns=['start', 'open', 'high', 'low', 'close', 'volume', 'turnover'])
            for c in ['open', 'high', 'low', 'close', 'volume']:
                df[c] = df[c].astype(float)
            
            df = add_indicators(df)
            df = df.dropna().reset_index(drop=True)
            
            if len(df) < 100:
                continue
            
            # Walk-forward split (70/30)
            split_idx = int(len(df) * 0.7)
            cooldown = 0
            
            for i in range(30, len(df) - 10):
                if cooldown > 0:
                    cooldown -= 1
                    continue
                
                df_available = df.iloc[:i+1]
                signal = detect_divergence(df_available)
                
                if not signal:
                    continue
                
                total_signals += 1
                
                # Volume filter
                if not df_available.iloc[-1]['vol_ok']:
                    continue
                
                # StochRSI filter
                if use_stoch_filter:
                    stoch_k = df_available.iloc[-1]['stoch_k']
                    if signal['side'] == 'short' and stoch_k <= OVERBOUGHT:
                        filtered_signals += 1
                        continue
                    if signal['side'] == 'long' and stoch_k >= OVERSOLD:
                        filtered_signals += 1
                        continue
                
                entry_idx = i + 1
                if entry_idx >= len(df):
                    continue
                
                entry_price = df.iloc[entry_idx]['open']
                side = signal['side']
                atr = df_available.iloc[-1]['atr']
                
                # ATR-based SL (1.0x ATR)
                sl_distance = 1.0 * atr
                
                result = simulate_trade(df, entry_idx, side, entry_price, sl_distance, be_r, trail_start_r, trail_dist_r)
                
                if result:
                    result['symbol'] = sym
                    result['side'] = side
                    
                    if entry_idx < split_idx:
                        in_sample.append(result)
                    else:
                        out_sample.append(result)
                    
                    cooldown = 10
        except Exception as e:
            continue
    
    return in_sample, out_sample, filtered_signals, total_signals

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
    
    # Wilson lower bound
    if total > 0:
        p = wins / total
        z = 1.96
        lb = (p + z*z/(2*total) - z*np.sqrt((p*(1-p) + z*z/(4*total))/total)) / (1 + z*z/total)
        lb_wr = lb * 100
    else:
        lb_wr = 0
    
    # Exit breakdown
    exits = df['exit'].value_counts()
    
    return {
        'trades': total,
        'wr': wr,
        'lb_wr': lb_wr,
        'total_r': total_r,
        'avg_r': avg_r,
        'trail_exits': exits.get('trail_sl', 0),
        'be_exits': exits.get('be', 0),
        'sl_exits': exits.get('sl', 0),
        'tp_exits': exits.get('tp3r', 0),
        'timeout': exits.get('timeout', 0)
    }

def main():
    print("=" * 80)
    print("🔬 ROBUST BACKTEST: Aggressive Trail vs Current Config")
    print("=" * 80)
    
    print("\nFEATURES:")
    print("  ✓ Walk-Forward Validation (70% IS / 30% OOS)")
    print(f"  ✓ {NUM_SYMBOLS} symbols (top by volume)")
    print("  ✓ StochRSI Filter (K>80 shorts, K<20 longs)")
    print("  ✓ ATR-based SL (1.0x ATR)")
    print(f"  ✓ Commission: {COMMISSION_PER_LEG*100:.3f}% per leg")
    print("  ✓ NO lookahead bias")
    print("  ✓ Timeout = LOSS")
    print("=" * 80)
    
    # Fetch symbols
    print(f"\n📦 Fetching top {NUM_SYMBOLS} symbols by volume...")
    symbols = get_top_symbols(NUM_SYMBOLS)
    print(f"   Got {len(symbols)} symbols\n")
    
    # Test configurations
    configs = [
        ("AGGRESSIVE (BE:0.5R Trail:0.5R Dist:0.5R)", BE_R, TRAIL_START_R, TRAIL_DISTANCE_R),
        ("CURRENT BOT (BE:0.7R Trail:0.7R Dist:0.3R)", CURRENT_BE_R, CURRENT_TRAIL_START_R, CURRENT_TRAIL_DISTANCE_R),
    ]
    
    all_results = {}
    
    for interval, tf_name in [('60', '1H'), ('3', '3M')]:
        print(f"\n{'='*60}")
        print(f"📊 TIMEFRAME: {tf_name}")
        print(f"{'='*60}")
        
        for name, be_r, trail_start_r, trail_dist_r in configs:
            print(f"\n  Testing: {name}...")
            
            in_sample, out_sample, filtered, total = run_backtest(
                symbols, interval, be_r, trail_start_r, trail_dist_r, use_stoch_filter=True
            )
            
            oos_stats = analyze(out_sample)
            
            if oos_stats:
                key = f"{tf_name}_{name}"
                all_results[key] = {
                    'tf': tf_name,
                    'config': name,
                    **oos_stats,
                    'filtered': filtered,
                    'total_signals': total
                }
                
                print(f"    OOS: {oos_stats['trades']} trades | {oos_stats['wr']:.1f}% WR | {oos_stats['total_r']:+.1f}R | {oos_stats['avg_r']:+.3f} Avg R")
    
    # Final comparison
    print("\n" + "=" * 80)
    print("📊 OUT-OF-SAMPLE COMPARISON")
    print("=" * 80)
    
    print(f"\n{'Timeframe':<6} {'Config':<40} {'Trades':>7} {'WR%':>6} {'LB WR':>6} {'Total R':>9} {'Avg R':>8}")
    print("-" * 90)
    
    for key, r in all_results.items():
        print(f"{r['tf']:<6} {r['config']:<40} {r['trades']:>7} {r['wr']:>5.1f}% {r['lb_wr']:>5.1f}% {r['total_r']:>+9.1f} {r['avg_r']:>+8.3f}")
    
    # Recommendations
    print("\n" + "=" * 80)
    print("💡 RECOMMENDATIONS")
    print("=" * 80)
    
    # Compare by timeframe
    for tf in ['1H', '3M']:
        agg_key = f"{tf}_AGGRESSIVE (BE:0.5R Trail:0.5R Dist:0.5R)"
        cur_key = f"{tf}_CURRENT BOT (BE:0.7R Trail:0.7R Dist:0.3R)"
        
        if agg_key in all_results and cur_key in all_results:
            agg = all_results[agg_key]
            cur = all_results[cur_key]
            
            diff_r = agg['total_r'] - cur['total_r']
            diff_avg = agg['avg_r'] - cur['avg_r']
            
            winner = "AGGRESSIVE" if agg['total_r'] > cur['total_r'] else "CURRENT"
            icon = "🏆" if winner == "AGGRESSIVE" else "📊"
            
            print(f"\n{tf} TIMEFRAME:")
            print(f"  {icon} Winner: {winner}")
            print(f"     Difference: {diff_r:+.1f}R total, {diff_avg:+.4f} Avg R per trade")

if __name__ == '__main__':
    main()
