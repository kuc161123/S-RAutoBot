#!/usr/bin/env python3
"""
MULTI-TIMEFRAME BACKTEST COMPARISON
===================================
Tests the Hidden Bearish + Optimal Trailing strategy across:
- 2 minute
- 3 minute  
- 5 minute
- 15 minute

Uses the SAME rigorous methodology:
- Strict no-lookahead (pivot right=0)
- Walk-forward validation (70/30)
- Slippage & commission
- Pessimistic assumptions
"""

import requests
import pandas as pd
import numpy as np
import math
import yaml
from collections import defaultdict
from datetime import datetime
import time
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

TIMEFRAMES = ['3', '5', '15']  # Bybit intervals (2, 3, 5, 15 min)
DATA_DAYS = 30                  # More data per timeframe (adjust for lower TFs)
NUM_SYMBOLS = 50                # Fewer symbols for faster multi-TF test

# === TRAILING STRATEGY (OPTIMAL) ===
BE_THRESHOLD_R = 0.7
TRAIL_START_R = 0.7
TRAIL_DISTANCE_R = 0.3
MAX_PROFIT_R = 3.0

# === COSTS ===
SLIPPAGE_PCT = 0.05
COMMISSION_PCT = 0.055

# === DIVERGENCE ===
RSI_PERIOD = 14
LOOKBACK_BARS = 14
PIVOT_LEFT = 3
PIVOT_RIGHT = 0  # No lookahead

# === SL CONSTRAINTS ===
MIN_SL_ATR = 0.3
MAX_SL_ATR = 2.0
SWING_LOOKBACK = 15

# === FILTERS ===
VOLUME_FILTER = True
MIN_VOL_RATIO = 0.5

# === TIMING ===
COOLDOWN_BARS = 10
TRAIN_PCT = 0.70

BASE_URL = "https://api.bybit.com"

# =============================================================================
# HELPERS
# =============================================================================

def wilson_lb(wins, n, z=1.96):
    if n == 0: return 0.0
    p = wins / n
    denom = 1 + z*z/n
    centre = p + z*z/(2*n)
    spread = z * math.sqrt((p*(1-p) + z*z/(4*n)) / n)
    return max(0, (centre - spread) / denom)

def fetch_klines(symbol, interval, days):
    end_ts = int(datetime.now().timestamp() * 1000)
    all_candles = []
    current_end = end_ts
    
    # Calculate candles needed based on timeframe
    mins_per_candle = int(interval)
    candles_per_day = 24 * 60 // mins_per_candle
    candles_needed = days * candles_per_day
    
    while len(all_candles) < candles_needed:
        params = {'category': 'linear', 'symbol': symbol, 'interval': interval, 'limit': 1000, 'end': current_end}
        try:
            resp = requests.get(f"{BASE_URL}/v5/market/kline", params=params, timeout=15)
            data = resp.json().get('result', {}).get('list', [])
            if not data: break
            all_candles.extend(data)
            current_end = int(data[-1][0]) - 1
            if len(data) < 1000: break
            time.sleep(0.05)
        except: break
    
    if not all_candles: return pd.DataFrame()
    
    df = pd.DataFrame(all_candles, columns=['start', 'open', 'high', 'low', 'close', 'volume', 'turnover'])
    df['start'] = pd.to_datetime(df['start'].astype(int), unit='ms')
    for col in ['open', 'high', 'low', 'close', 'volume']: 
        df[col] = df[col].astype(float)
    df.set_index('start', inplace=True)
    df.sort_index(inplace=True)
    return df

def calculate_rsi_at_bar(closes, period=14):
    if len(closes) < period + 1:
        return np.nan
    delta = np.diff(closes)
    gains = np.where(delta > 0, delta, 0)
    losses = np.where(delta < 0, -delta, 0)
    avg_gain = np.mean(gains[-period:])
    avg_loss = np.mean(losses[-period:])
    if avg_loss == 0:
        return 100.0
    return 100 - (100 / (1 + avg_gain / avg_loss))

def calculate_atr_at_bar(highs, lows, closes, period=14):
    if len(closes) < period + 2:
        return np.nan
    tr_list = []
    for i in range(1, len(closes)):
        tr = max(highs[i] - lows[i], abs(highs[i] - closes[i-1]), abs(lows[i] - closes[i-1]))
        tr_list.append(tr)
    if len(tr_list) < period:
        return np.nan
    return np.mean(tr_list[-period:])

def find_pivot_highs_no_lookahead(highs, left=3):
    pivots = []
    n = len(highs)
    for i in range(left, n):
        is_pivot = all(highs[j] < highs[i] for j in range(i-left, i))
        if is_pivot:
            pivots.append((i, highs[i]))
    return pivots

def detect_hidden_bearish_strict(closes, highs, lows, rsis, current_idx, lookback=14):
    if current_idx < 30:
        return None
    
    highs_available = highs[:current_idx+1]
    rsis_available = rsis[:current_idx+1]
    
    current_rsi = rsis_available[-1]
    if np.isnan(current_rsi) or current_rsi < 40:
        return None
    
    pivot_highs = find_pivot_highs_no_lookahead(highs_available, left=PIVOT_LEFT)
    
    if len(pivot_highs) < 2:
        return None
    
    recent_pivots = [(idx, val) for idx, val in pivot_highs 
                     if idx >= current_idx - lookback and idx < current_idx - 2]
    
    if len(recent_pivots) < 2:
        return None
    
    prev_pivot = recent_pivots[-2]
    curr_pivot = recent_pivots[-1]
    
    prev_idx, prev_high = prev_pivot
    curr_idx_p, curr_high = curr_pivot
    
    if curr_idx_p - prev_idx < 5:
        return None
    
    price_lower_high = curr_high < prev_high
    rsi_higher_high = rsis_available[curr_idx_p] > rsis_available[prev_idx]
    
    if price_lower_high and rsi_higher_high:
        return {'type': 'hidden_bearish', 'side': 'short'}
    
    return None

def calculate_sl_strict(highs, lows, side, atr, entry_price, lookback=15):
    if len(highs) < lookback:
        lookback = len(highs)
    
    if side == 'short':
        swing_high = max(highs[-lookback:])
        sl = swing_high
        sl_distance = sl - entry_price
    else:
        swing_low = min(lows[-lookback:])
        sl = swing_low
        sl_distance = entry_price - sl
    
    if sl_distance <= 0:
        sl_distance = atr * 1.0
        sl = entry_price + sl_distance if side == 'short' else entry_price - sl_distance
    
    min_sl = MIN_SL_ATR * atr
    max_sl = MAX_SL_ATR * atr
    
    if sl_distance < min_sl:
        sl_distance = min_sl
    elif sl_distance > max_sl:
        sl_distance = max_sl
    
    sl = entry_price + sl_distance if side == 'short' else entry_price - sl_distance
    
    return sl, sl_distance

def simulate_trade(opens, highs, lows, closes, entry_idx, side, entry_price, sl_price, sl_distance, timeout_bars):
    if side == 'short':
        entry_adj = entry_price * (1 - SLIPPAGE_PCT / 100)
    else:
        entry_adj = entry_price * (1 + SLIPPAGE_PCT / 100)
    
    current_sl = sl_price
    max_r = 0.0
    sl_at_be = False
    trailing = False
    
    for bar_offset in range(1, timeout_bars + 1):
        bar_idx = entry_idx + bar_offset
        if bar_idx >= len(opens):
            break
        
        h, l = highs[bar_idx], lows[bar_idx]
        
        if side == 'short':
            candle_max_r = (entry_adj - l) / sl_distance
            sl_hit = h >= current_sl
        else:
            candle_max_r = (h - entry_adj) / sl_distance
            sl_hit = l <= current_sl
        
        max_r = max(max_r, candle_max_r)
        
        if sl_hit:
            if trailing:
                if side == 'short':
                    exit_r = (entry_adj - current_sl) / sl_distance
                else:
                    exit_r = (current_sl - entry_adj) / sl_distance
            elif sl_at_be:
                exit_r = 0
            else:
                exit_r = -1.0
            
            exit_r -= (COMMISSION_PCT * 2 / 100) * abs(exit_r + 1)
            return {'result': 'sl' if exit_r < 0 else 'trailed', 'r': exit_r}
        
        if not sl_at_be and max_r >= BE_THRESHOLD_R:
            current_sl = entry_adj
            sl_at_be = True
        
        if sl_at_be and max_r >= TRAIL_START_R:
            trailing = True
            trail_level = max_r - TRAIL_DISTANCE_R
            if trail_level > 0:
                if side == 'short':
                    new_sl = entry_adj - (trail_level * sl_distance)
                    current_sl = min(current_sl, new_sl) if current_sl != entry_adj else new_sl
                else:
                    new_sl = entry_adj + (trail_level * sl_distance)
                    current_sl = max(current_sl, new_sl)
        
        if candle_max_r >= MAX_PROFIT_R:
            exit_r = MAX_PROFIT_R - (COMMISSION_PCT * 2 / 100) * MAX_PROFIT_R
            return {'result': 'tp', 'r': exit_r}
    
    return {'result': 'timeout', 'r': -1.0}

# =============================================================================
# BACKTEST ONE TIMEFRAME
# =============================================================================

def backtest_timeframe(symbols, interval, data_days):
    """Run backtest for a single timeframe"""
    
    # Adjust timeout based on timeframe
    mins = int(interval)
    timeout_bars = max(30, 300 // mins)  # ~5 hours worth of bars
    
    oos_trades = []
    
    for sym in symbols:
        try:
            df = fetch_klines(sym, interval, data_days)
            if df.empty or len(df) < 500:
                continue
            
            df = df.reset_index()
            opens = df['open'].values
            highs = df['high'].values
            lows = df['low'].values
            closes = df['close'].values
            volumes = df['volume'].values
            
            n = len(df)
            split_idx = int(n * TRAIN_PCT)
            
            vol_ma = np.full(n, np.nan)
            for i in range(20, n):
                vol_ma[i] = np.mean(volumes[i-20:i])
            
            rsis = np.full(n, np.nan)
            for i in range(RSI_PERIOD + 1, n):
                rsis[i] = calculate_rsi_at_bar(closes[:i+1], RSI_PERIOD)
            
            last_trade_bar = -COOLDOWN_BARS - 1
            
            for i in range(50, n - timeout_bars - 1):
                if i - last_trade_bar < COOLDOWN_BARS:
                    continue
                
                if VOLUME_FILTER and not np.isnan(vol_ma[i]):
                    if volumes[i] < vol_ma[i] * MIN_VOL_RATIO:
                        continue
                
                atr = calculate_atr_at_bar(highs[:i+1], lows[:i+1], closes[:i+1], RSI_PERIOD)
                if np.isnan(atr) or atr <= 0:
                    continue
                
                signal = detect_hidden_bearish_strict(closes, highs, lows, rsis, i, LOOKBACK_BARS)
                
                if not signal:
                    continue
                
                entry_idx = i + 1
                if entry_idx >= n:
                    continue
                
                entry_price = opens[entry_idx]
                side = signal['side']
                
                sl, sl_distance = calculate_sl_strict(
                    highs[:entry_idx], lows[:entry_idx], 
                    side, atr, entry_price, SWING_LOOKBACK
                )
                
                if sl_distance <= 0:
                    continue
                
                result = simulate_trade(
                    opens, highs, lows, closes,
                    entry_idx, side, entry_price, sl, sl_distance, timeout_bars
                )
                
                # Only count OOS trades
                if i >= split_idx:
                    oos_trades.append(result['r'])
                
                last_trade_bar = i
            
            time.sleep(0.02)
        except:
            continue
    
    return oos_trades

# =============================================================================
# MAIN
# =============================================================================

def run_multi_tf_backtest():
    print("=" * 80)
    print("🔬 MULTI-TIMEFRAME BACKTEST COMPARISON")
    print("=" * 80)
    print("\nTesting Hidden Bearish + Optimal Trailing across timeframes:")
    print(f"  Timeframes: {', '.join([f'{tf}min' for tf in TIMEFRAMES])}")
    print(f"  Symbols: {NUM_SYMBOLS}")
    print(f"  Data: {DATA_DAYS} days each")
    print(f"  Methodology: Walk-forward (70/30), no-lookahead")
    print("=" * 80)
    
    # Load symbols
    try:
        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        symbols = config.get('trade', {}).get('divergence_symbols', [])[:NUM_SYMBOLS]
    except:
        symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT']
    
    results = {}
    
    for tf in TIMEFRAMES:
        print(f"\n⏳ Testing {tf}-minute timeframe...")
        start = time.time()
        
        oos_trades = backtest_timeframe(symbols, tf, DATA_DAYS)
        
        elapsed = time.time() - start
        
        if oos_trades:
            n = len(oos_trades)
            wins = len([r for r in oos_trades if r > 0])
            losses = len([r for r in oos_trades if r < 0])
            wr = wins / (wins + losses) * 100 if (wins + losses) > 0 else 0
            total_r = sum(oos_trades)
            avg_r = total_r / n if n > 0 else 0
            lb_wr = wilson_lb(wins, wins + losses) * 100
            
            results[tf] = {
                'trades': n,
                'wins': wins,
                'losses': losses,
                'wr': wr,
                'lb_wr': lb_wr,
                'total_r': total_r,
                'avg_r': avg_r,
                'trades_per_day': n / DATA_DAYS
            }
            
            print(f"   ✅ {n} trades | WR: {wr:.1f}% | Total R: {total_r:+.1f} | Avg: {avg_r:+.3f} ({elapsed:.1f}s)")
        else:
            print(f"   ❌ No trades found")
    
    # Summary
    print("\n" + "=" * 80)
    print("📊 SUMMARY: OUT-OF-SAMPLE RESULTS")
    print("=" * 80)
    
    print(f"\n| TF    | Trades | WR%     | LB WR%  | Total R  | Avg R   | Trades/Day |")
    print(f"|-------|--------|---------|---------|----------|---------|------------|")
    
    for tf in TIMEFRAMES:
        if tf in results:
            r = results[tf]
            print(f"| {tf:>3}m  | {r['trades']:<6} | {r['wr']:<7.1f} | {r['lb_wr']:<7.1f} | {r['total_r']:>+8.1f} | {r['avg_r']:>+7.3f} | {r['trades_per_day']:<10.1f} |")
    
    # Find best
    if results:
        best_by_avgr = max(results.items(), key=lambda x: x[1]['avg_r'])
        best_by_totalr = max(results.items(), key=lambda x: x[1]['total_r'])
        best_by_wr = max(results.items(), key=lambda x: x[1]['wr'])
        
        print(f"\n🏆 **BEST TIMEFRAMES**")
        print(f"├ Highest Win Rate: {best_by_wr[0]}min ({best_by_wr[1]['wr']:.1f}%)")
        print(f"├ Highest Avg R: {best_by_avgr[0]}min ({best_by_avgr[1]['avg_r']:+.3f})")
        print(f"└ Highest Total R: {best_by_totalr[0]}min ({best_by_totalr[1]['total_r']:+.1f}R)")
        
        # Recommendation
        print(f"\n💡 **RECOMMENDATION**")
        
        # Overall best = highest avg_r with decent sample size
        viable = {k: v for k, v in results.items() if v['trades'] >= 100}
        if viable:
            best = max(viable.items(), key=lambda x: x[1]['avg_r'])
            print(f"   Best overall: **{best[0]}-minute** timeframe")
            print(f"   • {best[1]['trades']} trades over {DATA_DAYS} days")
            print(f"   • {best[1]['wr']:.1f}% win rate")
            print(f"   • {best[1]['avg_r']:+.3f} R per trade")
            print(f"   • ~{best[1]['trades_per_day']:.0f} trades/day potential")
        else:
            print("   Insufficient data to recommend. Run with more symbols/days.")
    
    return results

if __name__ == "__main__":
    run_multi_tf_backtest()
