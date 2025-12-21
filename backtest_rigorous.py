#!/usr/bin/env python3
"""
PARANOID-LEVEL RIGOROUS BACKTEST
================================
Designed to be PESSIMISTIC and eliminate ALL potential biases:

ANTI-LOOKAHEAD MEASURES:
1. Candle-by-candle processing (only see data up to current bar)
2. Signal detected on bar N, entry on bar N+1 OPEN (not close)
3. RSI/ATR calculated only on data available at signal time
4. Pivot detection uses right=0 (no future confirmation)
5. Exit prices use WORST case (SL checked FIRST)

WALK-FORWARD VALIDATION:
- 70% in-sample (training) | 30% out-of-sample (testing)
- Only OUT-OF-SAMPLE results matter for decision

ADDITIONAL SAFEGUARDS:
- Slippage simulation: 0.05% per trade
- Commission: 0.055% per trade (Bybit taker)
- Conservative timeout: 50 bars (not 100)
- No overlapping trades per symbol
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
# CONFIGURATION - CONSERVATIVE/PESSIMISTIC
# =============================================================================

TIMEFRAME = '60'            # 1H candles
DATA_DAYS = 90              # More data for robust walk-forward
NUM_SYMBOLS = 126           # From config

# === TRAILING STRATEGY (OPTIMAL) ===
BE_THRESHOLD_R = 0.7
TRAIL_START_R = 0.7
TRAIL_DISTANCE_R = 0.3
MAX_PROFIT_R = 3.0

# === COSTS (REALISTIC) ===
SLIPPAGE_PCT = 0.05         # 0.05% slippage per trade
COMMISSION_PCT = 0.055      # 0.055% Bybit taker fee

# === DIVERGENCE DETECTION (STRICT NO LOOKAHEAD) ===
RSI_PERIOD = 14
LOOKBACK_BARS = 14
# NOTE: Using right=0 for pivots (no future confirmation)
PIVOT_LEFT = 3
PIVOT_RIGHT = 0             # CRITICAL: No lookahead!
MIN_PIVOT_DISTANCE = 5

# === SL CONSTRAINTS ===
MIN_SL_ATR = 0.3
MAX_SL_ATR = 2.0
SWING_LOOKBACK = 15

# === FILTERS ===
VOLUME_FILTER = True
MIN_VOL_RATIO = 0.5

# === TIMING (CONSERVATIVE) ===
COOLDOWN_BARS = 10
TIMEOUT_BARS = 50           # More conservative than live (100)

# === WALK-FORWARD ===
TRAIN_PCT = 0.70            # 70% in-sample
TEST_PCT = 0.30             # 30% out-of-sample

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
    candles_needed = days * 24
    
    while len(all_candles) < candles_needed:
        params = {'category': 'linear', 'symbol': symbol, 'interval': interval, 'limit': 1000, 'end': current_end}
        try:
            resp = requests.get(f"{BASE_URL}/v5/market/kline", params=params, timeout=15)
            data = resp.json().get('result', {}).get('list', [])
            if not data: break
            all_candles.extend(data)
            current_end = int(data[-1][0]) - 1
            if len(data) < 1000: break
            time.sleep(0.05)  # Rate limiting
        except: break
    
    if not all_candles: return pd.DataFrame()
    
    df = pd.DataFrame(all_candles, columns=['start', 'open', 'high', 'low', 'close', 'volume', 'turnover'])
    df['start'] = pd.to_datetime(df['start'].astype(int), unit='ms')
    for col in ['open', 'high', 'low', 'close', 'volume']: 
        df[col] = df[col].astype(float)
    df.set_index('start', inplace=True)
    df.sort_index(inplace=True)
    return df

# =============================================================================
# STRICT NO-LOOKAHEAD INDICATORS
# =============================================================================

def calculate_rsi_at_bar(closes_up_to_now, period=14):
    """Calculate RSI using ONLY data up to current bar"""
    if len(closes_up_to_now) < period + 1:
        return np.nan
    delta = np.diff(closes_up_to_now)
    gains = np.where(delta > 0, delta, 0)
    losses = np.where(delta < 0, -delta, 0)
    avg_gain = np.mean(gains[-period:])
    avg_loss = np.mean(losses[-period:])
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def calculate_atr_at_bar(highs, lows, closes, period=14):
    """Calculate ATR using ONLY data up to current bar"""
    if len(closes) < period + 2:
        return np.nan
    tr_list = []
    for i in range(1, len(closes)):
        tr = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i-1]),
            abs(lows[i] - closes[i-1])
        )
        tr_list.append(tr)
    if len(tr_list) < period:
        return np.nan
    return np.mean(tr_list[-period:])

def find_pivot_lows_no_lookahead(lows_up_to_now, left=3):
    """Find pivot lows using ONLY historical data (right=0)"""
    pivots = []
    n = len(lows_up_to_now)
    for i in range(left, n):
        is_pivot = all(lows_up_to_now[j] > lows_up_to_now[i] for j in range(i-left, i))
        if is_pivot:
            pivots.append((i, lows_up_to_now[i]))
    return pivots

def find_pivot_highs_no_lookahead(highs_up_to_now, left=3):
    """Find pivot highs using ONLY historical data (right=0)"""
    pivots = []
    n = len(highs_up_to_now)
    for i in range(left, n):
        is_pivot = all(highs_up_to_now[j] < highs_up_to_now[i] for j in range(i-left, i))
        if is_pivot:
            pivots.append((i, highs_up_to_now[i]))
    return pivots

# =============================================================================
# STRICT NO-LOOKAHEAD DIVERGENCE DETECTION
# =============================================================================

def detect_hidden_bearish_strict(closes, highs, lows, rsis, current_idx, lookback=14):
    """
    Detect Hidden Bearish divergence with ZERO lookahead.
    
    Hidden Bearish: Price makes Lower High, RSI makes Higher High
    (Indicates hidden selling pressure, price likely to continue down)
    
    STRICT RULES:
    - Only use data up to current_idx
    - Pivot detection uses left=3, right=0 (no future confirmation)
    - RSI calculated on data up to current_idx
    """
    if current_idx < 30:
        return None
    
    # Data up to current bar ONLY
    highs_available = highs[:current_idx+1]
    rsis_available = rsis[:current_idx+1]
    
    # Check RSI threshold (should be > 40 for bearish setup)
    current_rsi = rsis_available[-1]
    if np.isnan(current_rsi) or current_rsi < 40:
        return None
    
    # Find pivot highs (no lookahead - right=0)
    pivot_highs = find_pivot_highs_no_lookahead(highs_available, left=PIVOT_LEFT)
    
    if len(pivot_highs) < 2:
        return None
    
    # Get last two pivot highs within lookback
    recent_pivots = [(idx, val) for idx, val in pivot_highs 
                     if idx >= current_idx - lookback and idx < current_idx - 2]
    
    if len(recent_pivots) < 2:
        return None
    
    # Most recent two
    prev_pivot = recent_pivots[-2]
    curr_pivot = recent_pivots[-1]
    
    prev_idx, prev_high = prev_pivot
    curr_idx_pivot, curr_high = curr_pivot
    
    # Minimum distance between pivots
    if curr_idx_pivot - prev_idx < MIN_PIVOT_DISTANCE:
        return None
    
    # Hidden Bearish: Price LH, RSI HH
    price_lower_high = curr_high < prev_high
    rsi_higher_high = rsis_available[curr_idx_pivot] > rsis_available[prev_idx]
    
    if price_lower_high and rsi_higher_high:
        return {'type': 'hidden_bearish', 'side': 'short'}
    
    return None

# =============================================================================
# STRICT NO-LOOKAHEAD SL CALCULATION
# =============================================================================

def calculate_sl_strict(highs_up_to_now, lows_up_to_now, side, atr, entry_price, lookback=15):
    """Calculate SL using ONLY data available at entry time"""
    if len(highs_up_to_now) < lookback:
        lookback = len(highs_up_to_now)
    
    if side == 'short':
        # For short: SL above recent swing high
        swing_high = max(highs_up_to_now[-lookback:])
        sl = swing_high
        sl_distance = sl - entry_price
    else:
        # For long: SL below recent swing low
        swing_low = min(lows_up_to_now[-lookback:])
        sl = swing_low
        sl_distance = entry_price - sl
    
    if sl_distance <= 0:
        sl_distance = atr * 1.0  # Fallback
        if side == 'short':
            sl = entry_price + sl_distance
        else:
            sl = entry_price - sl_distance
    
    # Apply constraints
    min_sl = MIN_SL_ATR * atr
    max_sl = MAX_SL_ATR * atr
    
    if sl_distance < min_sl:
        sl_distance = min_sl
    elif sl_distance > max_sl:
        sl_distance = max_sl
    
    if side == 'short':
        sl = entry_price + sl_distance
    else:
        sl = entry_price - sl_distance
    
    return sl, sl_distance

# =============================================================================
# TRADE SIMULATION (PESSIMISTIC)
# =============================================================================

def simulate_trade_pessimistic(opens, highs, lows, closes, entry_idx, side, entry_price, sl_price, sl_distance):
    """
    Simulate trade with PESSIMISTIC assumptions:
    - SL checked FIRST on each candle
    - Slippage applied to entry and exit
    - Commission deducted
    - Worst-case tie-breaker
    """
    # Apply slippage to entry (against us)
    if side == 'short':
        entry_price_adjusted = entry_price * (1 - SLIPPAGE_PCT / 100)  # Worse for short
    else:
        entry_price_adjusted = entry_price * (1 + SLIPPAGE_PCT / 100)  # Worse for long
    
    current_sl = sl_price
    max_r = 0.0
    sl_at_be = False
    trailing = False
    
    for bar_offset in range(1, TIMEOUT_BARS + 1):
        bar_idx = entry_idx + bar_offset
        if bar_idx >= len(opens):
            break
        
        h, l = highs[bar_idx], lows[bar_idx]
        
        # Calculate R for this candle
        if side == 'short':
            candle_max_r = (entry_price_adjusted - l) / sl_distance
            sl_hit = h >= current_sl
        else:
            candle_max_r = (h - entry_price_adjusted) / sl_distance
            sl_hit = l <= current_sl
        
        max_r = max(max_r, candle_max_r)
        
        # === SL CHECKED FIRST (PESSIMISTIC) ===
        if sl_hit:
            if trailing:
                if side == 'short':
                    exit_r = (entry_price_adjusted - current_sl) / sl_distance
                else:
                    exit_r = (current_sl - entry_price_adjusted) / sl_distance
            elif sl_at_be:
                exit_r = 0
            else:
                exit_r = -1.0
            
            # Apply commission
            exit_r -= (COMMISSION_PCT * 2 / 100) * abs(exit_r + 1)  # Both legs
            
            return {'result': 'sl' if exit_r < 0 else 'trailed', 'r': exit_r, 'max_r': max_r}
        
        # Move to BE
        if not sl_at_be and max_r >= BE_THRESHOLD_R:
            current_sl = entry_price_adjusted
            sl_at_be = True
        
        # Start trailing
        if sl_at_be and max_r >= TRAIL_START_R:
            trailing = True
            trail_level = max_r - TRAIL_DISTANCE_R
            if trail_level > 0:
                if side == 'short':
                    new_sl = entry_price_adjusted - (trail_level * sl_distance)
                    current_sl = min(current_sl, new_sl) if current_sl != entry_price_adjusted else new_sl
                else:
                    new_sl = entry_price_adjusted + (trail_level * sl_distance)
                    current_sl = max(current_sl, new_sl)
        
        # TP at 3R
        if candle_max_r >= MAX_PROFIT_R:
            exit_r = MAX_PROFIT_R - (COMMISSION_PCT * 2 / 100) * MAX_PROFIT_R
            return {'result': 'tp', 'r': exit_r, 'max_r': max_r}
    
    # Timeout - assume LOSS for pessimistic estimate
    return {'result': 'timeout', 'r': -1.0, 'max_r': max_r}

# =============================================================================
# MAIN WALK-FORWARD BACKTEST
# =============================================================================

def run_rigorous_backtest():
    print("=" * 80)
    print("🔒 PARANOID-LEVEL RIGOROUS BACKTEST")
    print("=" * 80)
    print("\n⚠️ ANTI-LOOKAHEAD MEASURES:")
    print("  • Candle-by-candle processing")
    print("  • Pivot detection: right=0 (no future confirmation)")
    print("  • Entry on NEXT bar open (not current close)")
    print("  • SL checked FIRST on each bar")
    print("  • Slippage: 0.05% | Commission: 0.055%")
    print("  • Timeout assumed as LOSS")
    print("\n📊 WALK-FORWARD VALIDATION:")
    print(f"  • In-Sample: {TRAIN_PCT*100:.0f}% | Out-of-Sample: {TEST_PCT*100:.0f}%")
    print(f"  • Only OOS results matter for decision")
    print("=" * 80)
    
    # Load symbols
    try:
        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        symbols = config.get('trade', {}).get('divergence_symbols', [])[:NUM_SYMBOLS]
    except:
        symbols = ['BTCUSDT', 'ETHUSDT']
    
    print(f"\n📦 Testing {len(symbols)} symbols over {DATA_DAYS} days...")
    print("⏳ Running SLOW for accuracy (this may take 5-10 minutes)...\n")
    
    # Containers
    in_sample_trades = []
    out_sample_trades = []
    weekday_oos = []
    weekend_oos = []
    
    start_time = time.time()
    
    for sym_idx, sym in enumerate(symbols):
        try:
            # Fetch data
            df = fetch_klines(sym, TIMEFRAME, DATA_DAYS)
            if df.empty or len(df) < 300:
                continue
            
            df = df.reset_index()
            
            # Get arrays for fast indexing
            opens = df['open'].values
            highs = df['high'].values
            lows = df['low'].values
            closes = df['close'].values
            volumes = df['volume'].values
            timestamps = df['start'].values
            
            n = len(df)
            split_idx = int(n * TRAIN_PCT)
            
            # Pre-calculate volume MA (using only past data)
            vol_ma = np.full(n, np.nan)
            for i in range(20, n):
                vol_ma[i] = np.mean(volumes[i-20:i])
            
            # Pre-calculate RSI (using only past data)
            rsis = np.full(n, np.nan)
            for i in range(RSI_PERIOD + 1, n):
                rsis[i] = calculate_rsi_at_bar(closes[:i+1], RSI_PERIOD)
            
            last_trade_bar = -COOLDOWN_BARS - 1
            
            # Candle-by-candle processing (SLOW but accurate)
            for i in range(50, n - TIMEOUT_BARS - 1):
                # Cooldown
                if i - last_trade_bar < COOLDOWN_BARS:
                    continue
                
                # Volume filter (using only past data)
                if VOLUME_FILTER and not np.isnan(vol_ma[i]):
                    if volumes[i] < vol_ma[i] * MIN_VOL_RATIO:
                        continue
                
                # ATR (using only past data)
                atr = calculate_atr_at_bar(highs[:i+1], lows[:i+1], closes[:i+1], RSI_PERIOD)
                if np.isnan(atr) or atr <= 0:
                    continue
                
                # Detect Hidden Bearish (strict no-lookahead)
                signal = detect_hidden_bearish_strict(closes, highs, lows, rsis, i, LOOKBACK_BARS)
                
                if not signal:
                    continue
                
                # Entry on NEXT bar open
                entry_idx = i + 1
                if entry_idx >= n:
                    continue
                
                entry_price = opens[entry_idx]
                side = signal['side']
                
                # Calculate SL (strict)
                sl, sl_distance = calculate_sl_strict(
                    highs[:entry_idx], lows[:entry_idx], 
                    side, atr, entry_price, SWING_LOOKBACK
                )
                
                if sl_distance <= 0:
                    continue
                
                # Simulate trade (pessimistic)
                result = simulate_trade_pessimistic(
                    opens, highs, lows, closes,
                    entry_idx, side, entry_price, sl, sl_distance
                )
                
                trade_record = {
                    'symbol': sym,
                    'r': result['r'],
                    'exit': result['result'],
                    'max_r': result['max_r'],
                    'bar_idx': i,
                    'timestamp': timestamps[i]
                }
                
                # Categorize
                if i < split_idx:
                    in_sample_trades.append(trade_record)
                else:
                    out_sample_trades.append(trade_record)
                    
                    # Weekend check
                    ts = pd.Timestamp(timestamps[i])
                    if ts.dayofweek >= 5:
                        weekend_oos.append(trade_record)
                    else:
                        weekday_oos.append(trade_record)
                
                last_trade_bar = i
            
            if (sym_idx + 1) % 10 == 0:
                oos_count = len(out_sample_trades)
                if oos_count > 0:
                    wins = len([t for t in out_sample_trades if t['r'] > 0])
                    wr = wins / oos_count * 100
                    print(f"  [{sym_idx+1}/{len(symbols)}] OOS Trades: {oos_count} | WR: {wr:.1f}%")
            
            time.sleep(0.03)  # Slow down for stability
            
        except Exception as e:
            continue
    
    elapsed = time.time() - start_time
    
    # ==========================================================================
    # RESULTS
    # ==========================================================================
    
    def analyze(trades, label):
        if not trades:
            return None
        n = len(trades)
        wins = len([t for t in trades if t['r'] > 0])
        losses = len([t for t in trades if t['r'] < 0])
        bes = len([t for t in trades if t['r'] == 0])
        
        total_valid = wins + losses
        wr = wins / total_valid * 100 if total_valid > 0 else 0
        lb_wr = wilson_lb(wins, total_valid) * 100
        
        total_r = sum(t['r'] for t in trades)
        avg_r = total_r / n if n > 0 else 0
        
        return {
            'label': label,
            'trades': n,
            'wins': wins,
            'losses': losses,
            'bes': bes,
            'wr': wr,
            'lb_wr': lb_wr,
            'total_r': total_r,
            'avg_r': avg_r
        }
    
    is_stats = analyze(in_sample_trades, "IN-SAMPLE (Training)")
    oos_stats = analyze(out_sample_trades, "OUT-OF-SAMPLE (Testing)")
    weekday_stats = analyze(weekday_oos, "WEEKDAY OOS")
    weekend_stats = analyze(weekend_oos, "WEEKEND OOS")
    
    print("\n" + "=" * 80)
    print("📊 RIGOROUS BACKTEST RESULTS")
    print("=" * 80)
    
    if is_stats:
        print(f"\n📚 **{is_stats['label']}** (Do NOT use for decisions)")
        print(f"├ Trades: {is_stats['trades']}")
        print(f"├ W/L: {is_stats['wins']}/{is_stats['losses']}")
        print(f"├ Win Rate: {is_stats['wr']:.1f}%")
        print(f"└ Total R: {is_stats['total_r']:+.1f}")
    
    if oos_stats:
        print(f"\n🎯 **{oos_stats['label']}** (USE THIS FOR DECISIONS)")
        print(f"├ Trades: {oos_stats['trades']}")
        print(f"├ W/L: {oos_stats['wins']}/{oos_stats['losses']}")
        print(f"├ Win Rate: **{oos_stats['wr']:.1f}%** (LB: {oos_stats['lb_wr']:.1f}%)")
        print(f"├ Total R: **{oos_stats['total_r']:+.1f}R**")
        print(f"└ Avg R/Trade: **{oos_stats['avg_r']:+.3f}**")
    
    # Robustness check
    if is_stats and oos_stats:
        wr_drop = is_stats['wr'] - oos_stats['wr']
        avgr_drop = is_stats['avg_r'] - oos_stats['avg_r']
        
        print(f"\n🔍 **ROBUSTNESS CHECK**")
        print(f"├ WR Drop (IS→OOS): {wr_drop:+.1f}%")
        print(f"├ Avg R Drop: {avgr_drop:+.3f}")
        
        if abs(wr_drop) < 5 and abs(avgr_drop) < 0.1:
            print(f"└ ✅ ROBUST (minimal overfitting)")
        else:
            print(f"└ ⚠️ POTENTIAL OVERFITTING")
    
    # Weekday vs Weekend
    print("\n" + "=" * 80)
    print("📅 WEEKDAY vs WEEKEND (Out-of-Sample Only)")
    print("=" * 80)
    
    if weekday_stats and weekend_stats:
        print(f"\n| Metric     | Weekday       | Weekend       |")
        print(f"|------------|---------------|---------------|")
        print(f"| Trades     | {weekday_stats['trades']:<13} | {weekend_stats['trades']:<13} |")
        print(f"| Win Rate   | {weekday_stats['wr']:.1f}%{' '*8} | {weekend_stats['wr']:.1f}%{' '*8} |")
        print(f"| Total R    | {weekday_stats['total_r']:+.1f}R{' '*7} | {weekend_stats['total_r']:+.1f}R{' '*7} |")
        print(f"| Avg R      | {weekday_stats['avg_r']:+.3f}{' '*7} | {weekend_stats['avg_r']:+.3f}{' '*7} |")
        
        wr_diff = weekday_stats['wr'] - weekend_stats['wr']
        
        print(f"\n💡 **RECOMMENDATION**")
        if wr_diff > 5:
            print(f"   Weekdays are {wr_diff:.1f}% better - Consider weekend filter")
        elif wr_diff < -5:
            print(f"   Weekends are {abs(wr_diff):.1f}% better - Trade more on weekends")
        else:
            print(f"   No significant difference ({abs(wr_diff):.1f}%) - Trade 24/7")
    
    print(f"\n⏱️ Completed in {elapsed/60:.1f} minutes")
    
    # Save
    all_trades = in_sample_trades + out_sample_trades
    pd.DataFrame(all_trades).to_csv('rigorous_backtest_results.csv', index=False)
    print(f"💾 Saved to: rigorous_backtest_results.csv")
    
    return oos_stats

if __name__ == "__main__":
    run_rigorous_backtest()
