#!/usr/bin/env python3
"""
STOCHASTIC RSI FILTER BACKTEST (3M)
===================================
Tests RSI Divergence with Stochastic RSI Extremes Filter

ROBUSTNESS FEATURES:
1. 50 symbols (top by volume)
2. 30 days of data
3. Walk-forward: 70% train / 30% test
4. NO lookahead (pivot right=0)
5. Slippage: 0.1%
6. Commission: 0.055% per leg
7. Pessimistic timeout (assume loss)
8. SL checked FIRST on every candle
9. Volume filter active

STOCHASTIC RSI FILTER:
- For SHORTS: StochRSI K > 80 (overbought)
- For LONGS: StochRSI K < 20 (oversold)
"""

import requests
import pandas as pd
import numpy as np
import math
from collections import defaultdict
from datetime import datetime
import time
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

DATA_DAYS = 30
NUM_SYMBOLS = 50

# === TRAILING STRATEGY (OPTIMAL) ===
BE_THRESHOLD_R = 0.7
TRAIL_START_R = 0.7
TRAIL_DISTANCE_R = 0.3
MAX_PROFIT_R = 3.0

# === COSTS ===
SLIPPAGE_PCT = 0.10
COMMISSION_PCT = 0.055

# === DIVERGENCE ===
RSI_PERIOD = 14
LOOKBACK_BARS = 14
PIVOT_LEFT = 3
PIVOT_RIGHT = 0  # ZERO lookahead

# === STOCHASTIC RSI ===
STOCH_RSI_K = 14
STOCH_RSI_D = 3
STOCH_RSI_SMOOTH = 3
OVERBOUGHT = 80  # For shorts
OVERSOLD = 20    # For longs

# === SL CONSTRAINTS ===
MIN_SL_ATR = 0.3
MAX_SL_ATR = 2.0
SWING_LOOKBACK = 15

# === FILTERS ===
VOLUME_FILTER = True
MIN_VOL_RATIO = 0.5

# === TIMING ===
COOLDOWN_BARS = 10
TIMEOUT_BARS = 100
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

def get_top_symbols(limit=50):
    """Get top symbols by 24h turnover from Bybit"""
    try:
        resp = requests.get(f"{BASE_URL}/v5/market/tickers?category=linear", timeout=15)
        tickers = resp.json().get('result', {}).get('list', [])
        usdt = [t for t in tickers if t['symbol'].endswith('USDT')]
        usdt.sort(key=lambda x: float(x.get('turnover24h', 0)), reverse=True)
        return [t['symbol'] for t in usdt[:limit]]
    except:
        return []

def fetch_klines(symbol, interval, days):
    end_ts = int(datetime.now().timestamp() * 1000)
    all_candles = []
    current_end = end_ts
    
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
            time.sleep(0.03)
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

def calculate_stoch_rsi(rsi_values, k_period=14, d_period=3, smooth=3):
    """
    Calculate Stochastic RSI
    Returns (K, D) tuple
    """
    if len(rsi_values) < k_period + smooth:
        return np.nan, np.nan
    
    # Get recent RSI values for stochastic calculation
    recent_rsi = rsi_values[-k_period:]
    
    # Handle NaN values
    valid_rsi = [r for r in recent_rsi if not np.isnan(r)]
    if len(valid_rsi) < k_period:
        return np.nan, np.nan
    
    min_rsi = min(valid_rsi)
    max_rsi = max(valid_rsi)
    current_rsi = rsi_values[-1]
    
    if np.isnan(current_rsi):
        return np.nan, np.nan
    
    # Stochastic RSI calculation
    if max_rsi == min_rsi:
        stoch_rsi = 50.0  # Midpoint if no range
    else:
        stoch_rsi = ((current_rsi - min_rsi) / (max_rsi - min_rsi)) * 100
    
    # K = smoothed stoch RSI (for simplicity, we use the raw value)
    k = stoch_rsi
    
    # D = SMA of K (simplified)
    d = k  # In practice, you'd smooth over multiple K values
    
    return k, d

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

def find_pivot_lows_no_lookahead(lows, left=3):
    pivots = []
    n = len(lows)
    for i in range(left, n):
        is_pivot = all(lows[j] > lows[i] for j in range(i-left, i))
        if is_pivot:
            pivots.append((i, lows[i]))
    return pivots

def detect_all_divergences(closes, highs, lows, rsis, current_idx, lookback=14):
    """Detect all types of RSI divergences"""
    if current_idx < 30:
        return None
    
    current_rsi = rsis[current_idx]
    if np.isnan(current_rsi):
        return None
    
    # Get available data up to current bar
    highs_available = highs[:current_idx+1]
    lows_available = lows[:current_idx+1]
    rsis_available = rsis[:current_idx+1]
    
    # Find pivot highs for bearish divergences
    pivot_highs = find_pivot_highs_no_lookahead(highs_available, left=PIVOT_LEFT)
    recent_highs = [(idx, val) for idx, val in pivot_highs 
                    if idx >= current_idx - lookback and idx < current_idx - 2]
    
    # Find pivot lows for bullish divergences
    pivot_lows = find_pivot_lows_no_lookahead(lows_available, left=PIVOT_LEFT)
    recent_lows = [(idx, val) for idx, val in pivot_lows 
                   if idx >= current_idx - lookback and idx < current_idx - 2]
    
    # === BEARISH DIVERGENCES (Short signals) ===
    if len(recent_highs) >= 2:
        prev_idx, prev_high = recent_highs[-2]
        curr_idx_p, curr_high = recent_highs[-1]
        
        if curr_idx_p - prev_idx >= 5:
            # Regular Bearish: Higher High + Lower RSI
            if curr_high > prev_high and rsis_available[curr_idx_p] < rsis_available[prev_idx]:
                return {'type': 'regular_bearish', 'side': 'short'}
            
            # Hidden Bearish: Lower High + Higher RSI
            if curr_high < prev_high and rsis_available[curr_idx_p] > rsis_available[prev_idx]:
                return {'type': 'hidden_bearish', 'side': 'short'}
    
    # === BULLISH DIVERGENCES (Long signals) ===
    if len(recent_lows) >= 2:
        prev_idx, prev_low = recent_lows[-2]
        curr_idx_p, curr_low = recent_lows[-1]
        
        if curr_idx_p - prev_idx >= 5:
            # Regular Bullish: Lower Low + Higher RSI
            if curr_low < prev_low and rsis_available[curr_idx_p] > rsis_available[prev_idx]:
                return {'type': 'regular_bullish', 'side': 'long'}
            
            # Hidden Bullish: Higher Low + Lower RSI
            if curr_low > prev_low and rsis_available[curr_idx_p] < rsis_available[prev_idx]:
                return {'type': 'hidden_bullish', 'side': 'long'}
    
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

def simulate_trade(opens, highs, lows, closes, entry_idx, side, entry_price, sl_price, sl_distance, timeout_bars, slippage_pct):
    """Simulate trade with pessimistic assumptions"""
    if side == 'short':
        entry_adj = entry_price * (1 - slippage_pct / 100)
    else:
        entry_adj = entry_price * (1 + slippage_pct / 100)
    
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
        
        # SL CHECKED FIRST (pessimistic)
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
            
            # Deduct commission
            exit_r -= (COMMISSION_PCT * 2 / 100) * abs(exit_r + 1)
            return {'result': 'sl' if exit_r < 0 else 'trailed', 'r': exit_r}
        
        # Move to BE at 0.7R (but actually to 0.4R immediately with fix)
        if not sl_at_be and max_r >= BE_THRESHOLD_R:
            # Immediately trail to 0.4R (optimal strategy)
            trail_level = max_r - TRAIL_DISTANCE_R
            if side == 'short':
                current_sl = entry_adj - (trail_level * sl_distance)
            else:
                current_sl = entry_adj + (trail_level * sl_distance)
            sl_at_be = True
            trailing = True
        
        # Continue trailing
        if trailing and max_r >= TRAIL_START_R:
            trail_level = max_r - TRAIL_DISTANCE_R
            if trail_level > 0:
                if side == 'short':
                    new_sl = entry_adj - (trail_level * sl_distance)
                    current_sl = min(current_sl, new_sl)
                else:
                    new_sl = entry_adj + (trail_level * sl_distance)
                    current_sl = max(current_sl, new_sl)
        
        # Full TP
        if candle_max_r >= MAX_PROFIT_R:
            exit_r = MAX_PROFIT_R - (COMMISSION_PCT * 2 / 100) * MAX_PROFIT_R
            return {'result': 'tp', 'r': exit_r}
    
    # Timeout = LOSS (pessimistic)
    return {'result': 'timeout', 'r': -1.0}

# =============================================================================
# BACKTEST
# =============================================================================

def run_backtest(use_stoch_filter=False, filter_name="BASE"):
    """Run backtest with optional Stochastic RSI filter"""
    print(f"\n{'='*60}")
    print(f"📊 BACKTESTING: {filter_name}")
    print(f"{'='*60}")
    print(f"  Symbols: {NUM_SYMBOLS} | Days: {DATA_DAYS} | Interval: 3m")
    print(f"  StochRSI Filter: {'ENABLED' if use_stoch_filter else 'DISABLED'}")
    if use_stoch_filter:
        print(f"  Shorts: K > {OVERBOUGHT} | Longs: K < {OVERSOLD}")
    
    is_trades = []
    oos_trades = []
    filtered_out = 0
    signals_found = 0
    
    symbols = get_top_symbols(NUM_SYMBOLS)
    if len(symbols) < 10:
        print("❌ Not enough symbols!")
        return None
    
    print(f"\n  Processing {len(symbols)} symbols...")
    
    for sym_idx, sym in enumerate(symbols):
        try:
            df = fetch_klines(sym, '3', DATA_DAYS)
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
            
            # Precompute indicators
            vol_ma = np.full(n, np.nan)
            for i in range(20, n):
                vol_ma[i] = np.mean(volumes[i-20:i])
            
            rsis = np.full(n, np.nan)
            for i in range(RSI_PERIOD + 1, n):
                rsis[i] = calculate_rsi_at_bar(closes[:i+1], RSI_PERIOD)
            
            # Precompute StochRSI K values
            stoch_k = np.full(n, np.nan)
            for i in range(RSI_PERIOD + STOCH_RSI_K + 1, n):
                k, d = calculate_stoch_rsi(rsis[:i+1], STOCH_RSI_K, STOCH_RSI_D, STOCH_RSI_SMOOTH)
                stoch_k[i] = k
            
            last_trade_bar = -COOLDOWN_BARS - 1
            
            for i in range(50, n - TIMEOUT_BARS - 1):
                if i - last_trade_bar < COOLDOWN_BARS:
                    continue
                
                # Volume filter
                if VOLUME_FILTER and not np.isnan(vol_ma[i]):
                    if volumes[i] < vol_ma[i] * MIN_VOL_RATIO:
                        continue
                
                atr = calculate_atr_at_bar(highs[:i+1], lows[:i+1], closes[:i+1], RSI_PERIOD)
                if np.isnan(atr) or atr <= 0:
                    continue
                
                # Detect divergence
                signal = detect_all_divergences(closes, highs, lows, rsis, i, LOOKBACK_BARS)
                
                if not signal:
                    continue
                
                signals_found += 1
                side = signal['side']
                
                # === STOCHASTIC RSI FILTER ===
                if use_stoch_filter:
                    k_value = stoch_k[i]
                    if np.isnan(k_value):
                        filtered_out += 1
                        continue
                    
                    # For shorts: require overbought (K > 80)
                    if side == 'short' and k_value <= OVERBOUGHT:
                        filtered_out += 1
                        continue
                    
                    # For longs: require oversold (K < 20)
                    if side == 'long' and k_value >= OVERSOLD:
                        filtered_out += 1
                        continue
                
                entry_idx = i + 1
                if entry_idx >= n:
                    continue
                
                entry_price = opens[entry_idx]
                
                sl, sl_distance = calculate_sl_strict(
                    highs[:entry_idx], lows[:entry_idx], 
                    side, atr, entry_price, SWING_LOOKBACK
                )
                
                if sl_distance <= 0:
                    continue
                
                result = simulate_trade(
                    opens, highs, lows, closes,
                    entry_idx, side, entry_price, sl, sl_distance, 
                    TIMEOUT_BARS, SLIPPAGE_PCT
                )
                
                trade = {
                    'symbol': sym, 
                    'r': result['r'], 
                    'exit': result['result'],
                    'side': side,
                    'type': signal['type']
                }
                
                if i < split_idx:
                    is_trades.append(trade)
                else:
                    oos_trades.append(trade)
                
                last_trade_bar = i
            
            if (sym_idx + 1) % 10 == 0:
                print(f"  [{sym_idx+1}/{len(symbols)}] Signals: {signals_found}, Filtered: {filtered_out}")
            
            time.sleep(0.02)
        except Exception as e:
            continue
    
    # Analyze results
    def analyze(trades, label=""):
        if not trades:
            return None
        n = len(trades)
        wins = len([t for t in trades if t['r'] > 0])
        losses = len([t for t in trades if t['r'] < 0])
        total_valid = wins + losses
        wr = wins / total_valid * 100 if total_valid > 0 else 0
        lb_wr = wilson_lb(wins, total_valid) * 100
        total_r = sum(t['r'] for t in trades)
        avg_r = total_r / n if n > 0 else 0
        
        # Breakdown by side
        long_trades = [t for t in trades if t['side'] == 'long']
        short_trades = [t for t in trades if t['side'] == 'short']
        long_wr = len([t for t in long_trades if t['r'] > 0]) / len(long_trades) * 100 if long_trades else 0
        short_wr = len([t for t in short_trades if t['r'] > 0]) / len(short_trades) * 100 if short_trades else 0
        
        return {
            'n': n, 'wins': wins, 'losses': losses, 
            'wr': wr, 'lb_wr': lb_wr, 'total_r': total_r, 'avg_r': avg_r,
            'long_n': len(long_trades), 'long_wr': long_wr,
            'short_n': len(short_trades), 'short_wr': short_wr
        }
    
    is_stats = analyze(is_trades, "IS")
    oos_stats = analyze(oos_trades, "OOS")
    
    return {
        'name': filter_name,
        'use_stoch': use_stoch_filter,
        'is': is_stats,
        'oos': oos_stats,
        'signals_found': signals_found,
        'filtered_out': filtered_out
    }

# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 80)
    print("🔬 STOCHASTIC RSI FILTER COMPARISON BACKTEST")
    print("=" * 80)
    print("\nTesting RSI Divergence with/without Stochastic RSI Extremes Filter")
    print(f"\n  Symbols: {NUM_SYMBOLS} | Days: {DATA_DAYS}")
    print(f"  StochRSI Filter: Shorts > {OVERBOUGHT}K, Longs < {OVERSOLD}K")
    print("=" * 80)
    
    results = {}
    
    # Test WITHOUT StochRSI filter (baseline)
    results['base'] = run_backtest(use_stoch_filter=False, filter_name="BASE (No Filter)")
    
    # Test WITH StochRSI filter
    results['stoch'] = run_backtest(use_stoch_filter=True, filter_name="STOCH RSI EXTREMES")
    
    # === COMPARISON ===
    print("\n" + "=" * 80)
    print("📊 COMPARISON: OUT-OF-SAMPLE RESULTS")
    print("=" * 80)
    
    print(f"\n| Strategy           | Trades | WR%    | LB WR% | Total R  | Avg R   |")
    print(f"|--------------------|--------|--------|--------|----------|---------|")
    
    for key, data in results.items():
        if data and data['oos']:
            oos = data['oos']
            name = data['name'][:18]
            print(f"| {name:<18} | {oos['n']:<6} | {oos['wr']:<6.1f} | {oos['lb_wr']:<6.1f} | {oos['total_r']:>+8.1f} | {oos['avg_r']:>+7.3f} |")
    
    # Side breakdown
    print(f"\n📊 SIDE BREAKDOWN (OOS)")
    print(f"\n| Strategy           | Long N | Long WR | Short N | Short WR |")
    print(f"|--------------------|--------|---------|---------|----------|")
    
    for key, data in results.items():
        if data and data['oos']:
            oos = data['oos']
            name = data['name'][:18]
            print(f"| {name:<18} | {oos['long_n']:<6} | {oos['long_wr']:<7.1f} | {oos['short_n']:<7} | {oos['short_wr']:<8.1f} |")
    
    # Filter stats
    if results.get('stoch'):
        stoch = results['stoch']
        print(f"\n📏 STOCH RSI FILTER IMPACT:")
        print(f"  • Signals found: {stoch['signals_found']}")
        print(f"  • Filtered out: {stoch['filtered_out']} ({stoch['filtered_out']/max(stoch['signals_found'],1)*100:.1f}%)")
    
    # Recommendation
    print("\n" + "=" * 80)
    print("💡 RECOMMENDATION")
    print("=" * 80)
    
    base_oos = results.get('base', {}).get('oos')
    stoch_oos = results.get('stoch', {}).get('oos')
    
    if base_oos and stoch_oos:
        if stoch_oos['avg_r'] > base_oos['avg_r']:
            improvement = ((stoch_oos['avg_r'] - base_oos['avg_r']) / abs(base_oos['avg_r'])) * 100 if base_oos['avg_r'] != 0 else 0
            print(f"\n🏆 STOCH RSI FILTER IS BETTER:")
            print(f"   • Avg R: {stoch_oos['avg_r']:+.3f} vs {base_oos['avg_r']:+.3f}")
            print(f"   • WR: {stoch_oos['wr']:.1f}% vs {base_oos['wr']:.1f}%")
            print(f"   • Improvement: {improvement:+.0f}% per trade")
            print(f"\n   CONSIDER implementing StochRSI filter in live bot!")
        else:
            print(f"\n❌ STOCH RSI FILTER IS NOT BETTER:")
            print(f"   • Avg R: {stoch_oos['avg_r']:+.3f} vs {base_oos['avg_r']:+.3f}")
            print(f"   • WR: {stoch_oos['wr']:.1f}% vs {base_oos['wr']:.1f}%")
            print(f"\n   Keep current strategy without StochRSI filter.")
    
    return results

if __name__ == "__main__":
    main()
