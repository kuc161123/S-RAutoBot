#!/usr/bin/env python3
"""
1H STOCHASTIC RSI FILTER BACKTEST
=================================
Tests 1H timeframe with/without StochRSI Extremes Filter

FILTER RULES:
- Shorts: StochRSI K > 80 (overbought)
- Longs: StochRSI K < 20 (oversold)

ROBUSTNESS:
1. 50 symbols (for speed)
2. 60 days of data
3. Walk-forward: 70% train / 30% test
4. NO lookahead (pivot right=0)
5. ATR-based SL (1.0x ATR)
6. Commission: 0.055% per leg
7. Pessimistic timeout (assume loss)
8. SL checked FIRST on every candle
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

DATA_DAYS = 60
NUM_SYMBOLS = 50  # Reduced for speed

# === TRAILING STRATEGY (OPTIMAL) ===
BE_THRESHOLD_R = 0.7
TRAIL_START_R = 0.7
TRAIL_DISTANCE_R = 0.3
MAX_PROFIT_R = 3.0

# === COSTS ===
SLIPPAGE_PCT_1H = 0.05
COMMISSION_PCT = 0.055

# === DIVERGENCE ===
RSI_PERIOD = 14
LOOKBACK_BARS = 14
PIVOT_LEFT = 3
PIVOT_RIGHT = 0

# === SL: ATR-BASED (not pivot) ===
ATR_SL_MULTIPLIER = 1.0  # 1.0x ATR

# === STOCHASTIC RSI ===
STOCH_K_PERIOD = 14
OVERBOUGHT = 80
OVERSOLD = 20

# === FILTERS ===
VOLUME_FILTER = True
MIN_VOL_RATIO = 0.5

# === TIMING ===
COOLDOWN_BARS_1H = 10
TIMEOUT_BARS_1H = 50
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

def get_top_symbols(limit=300):
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

def calculate_stoch_rsi(rsis, k_period=14):
    """Calculate Stochastic RSI K value.
    
    Returns K value (0-100) or None if not enough data.
    """
    if len(rsis) < k_period:
        return None
    
    recent_rsi = rsis[-k_period:]
    valid_rsi = [r for r in recent_rsi if not np.isnan(r)]
    
    if len(valid_rsi) < k_period:
        return None
    
    min_rsi = min(valid_rsi)
    max_rsi = max(valid_rsi)
    current_rsi = rsis[-1]
    
    if np.isnan(current_rsi) or max_rsi == min_rsi:
        return None
    
    return ((current_rsi - min_rsi) / (max_rsi - min_rsi)) * 100

def calculate_atr_sl(entry_price, atr, side, multiplier=1.0):
    """Calculate ATR-based SL (instead of pivot-based).
    
    Returns (sl_price, sl_distance).
    """
    sl_distance = multiplier * atr
    
    if side == 'long':
        sl = entry_price - sl_distance
    else:
        sl = entry_price + sl_distance
    
    return sl, sl_distance


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
        
        # SL CHECKED FIRST
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
    
    # Timeout = LOSS (pessimistic)
    return {'result': 'timeout', 'r': -1.0}

# =============================================================================
# BACKTEST ONE TIMEFRAME
# =============================================================================

def backtest_timeframe(symbols, interval, data_days, cooldown_bars, timeout_bars, slippage_pct, label, stoch_filter=False):
    """Run full backtest for one timeframe with optional StochRSI filter"""
    print(f"\n{'='*60}")
    print(f"📊 BACKTESTING: {label}")
    print(f"{'='*60}")
    print(f"  Symbols: {len(symbols)} | Days: {data_days} | Interval: {interval}min")
    print(f"  StochRSI Filter: {'ENABLED (K>80 short, K<20 long)' if stoch_filter else 'DISABLED'}")
    
    is_trades = []
    oos_trades = []
    weekday_oos = []
    weekend_oos = []
    
    start_time = time.time()
    failed = 0
    filtered_signals = 0
    total_signals = 0
    
    for sym_idx, sym in enumerate(symbols):
        try:
            df = fetch_klines(sym, interval, data_days)
            if df.empty or len(df) < 500:
                failed += 1
                continue
            
            df = df.reset_index()
            opens = df['open'].values
            highs = df['high'].values
            lows = df['low'].values
            closes = df['close'].values
            volumes = df['volume'].values
            timestamps = df['start'].values
            
            n = len(df)
            split_idx = int(n * TRAIN_PCT)
            
            vol_ma = np.full(n, np.nan)
            for i in range(20, n):
                vol_ma[i] = np.mean(volumes[i-20:i])
            
            rsis = np.full(n, np.nan)
            for i in range(RSI_PERIOD + 1, n):
                rsis[i] = calculate_rsi_at_bar(closes[:i+1], RSI_PERIOD)
            
            last_trade_bar = -cooldown_bars - 1
            
            for i in range(50, n - timeout_bars - 1):
                if i - last_trade_bar < cooldown_bars:
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
                
                total_signals += 1
                side = signal['side']
                
                # === STOCHASTIC RSI FILTER ===
                if stoch_filter:
                    stoch_k = calculate_stoch_rsi(rsis[:i+1], STOCH_K_PERIOD)
                    if stoch_k is not None:
                        if side == 'short' and stoch_k <= OVERBOUGHT:
                            filtered_signals += 1
                            continue  # Skip - need overbought for shorts
                        if side == 'long' and stoch_k >= OVERSOLD:
                            filtered_signals += 1
                            continue  # Skip - need oversold for longs
                
                entry_idx = i + 1
                if entry_idx >= n:
                    continue
                
                entry_price = opens[entry_idx]
                
                # === ATR-BASED SL (not pivot) ===
                sl, sl_distance = calculate_atr_sl(entry_price, atr, side, ATR_SL_MULTIPLIER)
                
                if sl_distance <= 0:
                    continue
                
                result = simulate_trade(
                    opens, highs, lows, closes,
                    entry_idx, side, entry_price, sl, sl_distance, 
                    timeout_bars, slippage_pct
                )
                
                trade = {'symbol': sym, 'r': result['r'], 'exit': result['result']}
                
                if i < split_idx:
                    is_trades.append(trade)
                else:
                    oos_trades.append(trade)
                    ts = pd.Timestamp(timestamps[i])
                    if ts.dayofweek >= 5:
                        weekend_oos.append(trade)
                    else:
                        weekday_oos.append(trade)
                
                last_trade_bar = i
            
            if (sym_idx + 1) % 30 == 0:
                oos_count = len(oos_trades)
                if oos_count > 0:
                    wins = len([t for t in oos_trades if t['r'] > 0])
                    wr = wins / oos_count * 100
                    print(f"  [{sym_idx+1}/{len(symbols)}] OOS: {oos_count} trades, {wr:.1f}% WR")
            
            time.sleep(0.02)
        except Exception as e:
            failed += 1
            continue
    
    elapsed = time.time() - start_time
    
    # Analyze
    def analyze(trades):
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
        return {'n': n, 'wins': wins, 'losses': losses, 'wr': wr, 'lb_wr': lb_wr, 'total_r': total_r, 'avg_r': avg_r}
    
    is_stats = analyze(is_trades)
    oos_stats = analyze(oos_trades)
    weekday_stats = analyze(weekday_oos)
    weekend_stats = analyze(weekend_oos)
    
    return {
        'label': label,
        'interval': interval,
        'is': is_stats,
        'oos': oos_stats,
        'weekday': weekday_stats,
        'weekend': weekend_stats,
        'elapsed': elapsed,
        'failed': failed
    }

# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 80)
    print("🔬 1H STOCHASTIC RSI FILTER BACKTEST")
    print("=" * 80)
    print("\nCOMPARING:")
    print("  • 1H BASE (no filter)")
    print("  • 1H + StochRSI EXTREMES (K>80 shorts, K<20 longs)")
    print("")
    print("ROBUSTNESS FEATURES:")
    print(f"  ✓ {NUM_SYMBOLS} symbols (top Bybit by volume)")
    print(f"  ✓ {DATA_DAYS} days of data")
    print("  ✓ Walk-forward: 70% IS / 30% OOS")
    print("  ✓ NO lookahead (pivot right=0)")
    print("  ✓ ATR-based SL (1.0x ATR)")
    print("  ✓ Commission: 0.055% per leg")
    print("  ✓ Timeout = LOSS")
    print("=" * 80)
    
    # Get symbols
    print(f"\n📦 Fetching top {NUM_SYMBOLS} symbols by volume...")
    symbols = get_top_symbols(NUM_SYMBOLS)
    print(f"   Got {len(symbols)} symbols")
    
    if len(symbols) < 20:
        print("❌ Not enough symbols. Check API connection.")
        return
    
    results = {}
    
    # Test 1H WITHOUT StochRSI filter
    results['1h_base'] = backtest_timeframe(
        symbols, '60', DATA_DAYS, 
        COOLDOWN_BARS_1H, TIMEOUT_BARS_1H, 
        SLIPPAGE_PCT_1H, "1H BASE (No Filter)",
        stoch_filter=False
    )
    
    # Test 1H WITH StochRSI filter
    results['1h_stoch'] = backtest_timeframe(
        symbols, '60', DATA_DAYS, 
        COOLDOWN_BARS_1H, TIMEOUT_BARS_1H, 
        SLIPPAGE_PCT_1H, "1H + STOCH RSI EXTREMES",
        stoch_filter=True
    )
    
    # === SUMMARY ===
    print("\n" + "=" * 80)
    print("📊 COMPARISON: OUT-OF-SAMPLE RESULTS")
    print("=" * 80)
    
    print("\n| Strategy              | Trades | WR%    | LB WR% | Total R  | Avg R   |")
    print("|----------------------|--------|--------|--------|----------|---------|")
    
    for name, data in results.items():
        if data.get('oos'):
            oos = data['oos']
            label = "1H BASE (No Filter)" if name == '1h_base' else "1H + STOCH RSI"
            print(f"| {label:<20} | {oos['n']:<6} | {oos['wr']:<6.1f} | {oos['lb_wr']:<6.1f} | {oos['total_r']:>+8.1f} | {oos['avg_r']:>+7.3f} |")
    
    # Recommendation
    print("\n" + "=" * 80)
    print("💡 RECOMMENDATION")
    print("=" * 80)
    
    base = results.get('1h_base', {}).get('oos', {})
    stoch = results.get('1h_stoch', {}).get('oos', {})
    
    if base and stoch:
        if stoch['avg_r'] > base['avg_r']:
            diff = ((stoch['avg_r'] - base['avg_r']) / abs(base['avg_r'])) * 100 if base['avg_r'] != 0 else 0
            print(f"\n🏆 STOCH RSI FILTER IS BETTER:")
            print(f"   • Avg R: {stoch['avg_r']:+.3f} vs {base['avg_r']:+.3f}")
            print(f"   • WR: {stoch['wr']:.1f}% vs {base['wr']:.1f}%")
            print(f"   • Improvement: {diff:+.0f}% per trade")
            print(f"\n   CONSIDER implementing StochRSI filter in live bot!")
        else:
            diff = ((base['avg_r'] - stoch['avg_r']) / abs(stoch['avg_r'])) * 100 if stoch['avg_r'] != 0 else 0
            print(f"\n🏆 BASE (NO FILTER) IS BETTER:")
            print(f"   • Avg R: {base['avg_r']:+.3f} vs {stoch['avg_r']:+.3f}")
            print(f"   • {diff:+.0f}% improvement per trade")
    
    return results

if __name__ == "__main__":
    main()
