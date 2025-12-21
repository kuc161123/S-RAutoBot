#!/usr/bin/env python3
"""
ULTRA-REALISTIC BACKTEST
========================
Matches live bot EXACTLY:
- Pivot-based divergence detection
- Pivot SL with ATR constraints (0.3-2.0x)
- Volume filter (vol > 0.5 * MA)
- Optimal trailing: BE at 0.7R, Trail 0.3R behind
- 100-bar timeout (excluded from WR)
- 10-bar cooldown per symbol
- Entry on NEXT candle open
- SL checked FIRST if both hit same candle

INCLUDES: Weekday vs Weekend breakdown analysis
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
# CONFIGURATION - MATCHES LIVE BOT EXACTLY
# =============================================================================

TIMEFRAME = '60'            # 1H candles (matches live)
DATA_DAYS = 60              # 60 days
NUM_SYMBOLS = 150           # From config

# === TRAILING STRATEGY (OPTIMAL) ===
BE_THRESHOLD_R = 0.7        # Move SL to BE at +0.7R
TRAIL_START_R = 0.7         # Start trailing at +0.7R
TRAIL_DISTANCE_R = 0.3      # Trail 0.3R behind
MAX_PROFIT_R = 3.0          # Cap at +3R

# === DIVERGENCE DETECTION ===
RSI_PERIOD = 14
RSI_OVERSOLD = 30
RSI_OVERBOUGHT = 70
LOOKBACK_BARS = 14
PIVOT_LEFT = 3
PIVOT_RIGHT = 3
MIN_PIVOT_DISTANCE = 5

# === SL CONSTRAINTS ===
MIN_SL_ATR = 0.3
MAX_SL_ATR = 2.0
SWING_LOOKBACK = 15

# === FILTERS ===
VOLUME_FILTER = True
MIN_VOL_RATIO = 0.5         # vol > 0.5 * vol_ma(20)

# === TIMING ===
COOLDOWN_BARS = 10          # Per symbol
TIMEOUT_BARS = 100          # ~100 hours on 1H

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
            resp = requests.get(f"{BASE_URL}/v5/market/kline", params=params, timeout=10)
            data = resp.json().get('result', {}).get('list', [])
            if not data: break
            all_candles.extend(data)
            current_end = int(data[-1][0]) - 1
            if len(data) < 1000: break
        except: break
    
    if not all_candles: return pd.DataFrame()
    
    df = pd.DataFrame(all_candles, columns=['start', 'open', 'high', 'low', 'close', 'volume', 'turnover'])
    df['start'] = pd.to_datetime(df['start'].astype(int), unit='ms')
    for col in ['open', 'high', 'low', 'close', 'volume']: 
        df[col] = df[col].astype(float)
    df.set_index('start', inplace=True)
    df.sort_index(inplace=True)
    return df

def calculate_rsi(close, period=14):
    delta = close.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    return 100 - (100 / (1 + rs))

def find_pivots(data, left=3, right=3):
    n = len(data)
    pivot_highs = np.full(n, np.nan)
    pivot_lows = np.full(n, np.nan)
    for i in range(left, n - right):
        is_high = all(data[j] < data[i] for j in range(i - left, i + right + 1) if j != i)
        is_low = all(data[j] > data[i] for j in range(i - left, i + right + 1) if j != i)
        if is_high: pivot_highs[i] = data[i]
        if is_low: pivot_lows[i] = data[i]
    return pivot_highs, pivot_lows

# =============================================================================
# DIVERGENCE DETECTION (matches live divergence_detector.py)
# =============================================================================

def detect_divergence_signals(df, hidden_bearish_only=True):
    """Detect divergences - matches live bot exactly"""
    if len(df) < 100: return []
    
    close = df['close'].values
    rsi = df['rsi'].values
    n = len(df)
    
    price_ph, price_pl = find_pivots(close, PIVOT_LEFT, PIVOT_RIGHT)
    signals = []
    
    for i in range(30, n - 5):
        # Find two pivot lows for bullish divergences
        curr_pl = curr_pli = prev_pl = prev_pli = None
        for j in range(i, max(i - LOOKBACK_BARS, 0), -1):
            if not np.isnan(price_pl[j]):
                if curr_pl is None: curr_pl, curr_pli = price_pl[j], j
                elif prev_pl is None and j < curr_pli - MIN_PIVOT_DISTANCE: 
                    prev_pl, prev_pli = price_pl[j], j
                    break
        
        # Find two pivot highs for bearish divergences
        curr_ph = curr_phi = prev_ph = prev_phi = None
        for j in range(i, max(i - LOOKBACK_BARS, 0), -1):
            if not np.isnan(price_ph[j]):
                if curr_ph is None: curr_ph, curr_phi = price_ph[j], j
                elif prev_ph is None and j < curr_phi - MIN_PIVOT_DISTANCE: 
                    prev_ph, prev_phi = price_ph[j], j
                    break
        
        # Hidden Bearish: Price LH, RSI HH (ACTIVE in live)
        if curr_ph and prev_ph and curr_ph < prev_ph and rsi[curr_phi] > rsi[prev_phi] and rsi[i] > RSI_OVERSOLD + 10:
            signals.append({'idx': i, 'side': 'short', 'type': 'hidden_bearish'})
            continue
        
        # Skip other types if hidden_bearish_only (matches live config)
        if hidden_bearish_only:
            continue
        
        # Regular Bullish: Price LL, RSI HL
        if curr_pl and prev_pl and curr_pl < prev_pl and rsi[curr_pli] > rsi[prev_pli] and rsi[i] < RSI_OVERSOLD + 15:
            signals.append({'idx': i, 'side': 'long', 'type': 'regular_bullish'})
            continue
        # Regular Bearish: Price HH, RSI LH
        if curr_ph and prev_ph and curr_ph > prev_ph and rsi[curr_phi] < rsi[prev_phi] and rsi[i] > RSI_OVERBOUGHT - 15:
            signals.append({'idx': i, 'side': 'short', 'type': 'regular_bearish'})
            continue
        # Hidden Bullish: Price HL, RSI LL
        if curr_pl and prev_pl and curr_pl > prev_pl and rsi[curr_pli] < rsi[prev_pli] and rsi[i] < RSI_OVERBOUGHT - 10:
            signals.append({'idx': i, 'side': 'long', 'type': 'hidden_bullish'})
    
    return signals

# =============================================================================
# PIVOT SL CALCULATION (matches live execute_divergence_trade)
# =============================================================================

def calc_pivot_sl(rows, idx, side, atr):
    """Calculate pivot-based SL with ATR constraints (matches live)"""
    entry = rows[idx + 1].open if idx + 1 < len(rows) else rows[idx].close
    
    # Find swing high/low for SL
    start_lookback = max(0, idx - SWING_LOOKBACK)
    
    if side == 'long':
        swing_low = min(rows[j].low for j in range(start_lookback, idx + 1))
        sl = swing_low
        sl_distance = abs(entry - sl)
    else:
        swing_high = max(rows[j].high for j in range(start_lookback, idx + 1))
        sl = swing_high
        sl_distance = abs(sl - entry)
    
    # Apply min/max constraints
    min_sl = MIN_SL_ATR * atr
    max_sl = MAX_SL_ATR * atr
    
    if sl_distance < min_sl:
        sl_distance = min_sl
        if side == 'long':
            sl = entry - sl_distance
        else:
            sl = entry + sl_distance
    elif sl_distance > max_sl:
        sl_distance = max_sl
        if side == 'long':
            sl = entry - sl_distance
        else:
            sl = entry + sl_distance
    
    return entry, sl, sl_distance

# =============================================================================
# TRADE SIMULATION WITH OPTIMAL TRAILING (matches live monitor_trailing_sl)
# =============================================================================

def simulate_trade_with_trailing(rows, signal_idx, side, entry_price, sl_price, sl_distance):
    """
    Simulate trade with OPTIMAL TRAILING strategy:
    - Move to BE at +0.7R
    - Trail from +0.7R with 0.3R distance
    - Exit at +3R TP or trailed SL
    - SL checked FIRST on each candle
    """
    start_idx = signal_idx + 1
    if start_idx >= len(rows) - TIMEOUT_BARS:
        return {'result': 'timeout', 'r': 0, 'bars': 0, 'max_r': 0}
    
    current_sl = sl_price
    max_r = 0.0
    sl_at_be = False
    trailing = False
    
    for bar_idx in range(1, TIMEOUT_BARS + 1):
        if start_idx + bar_idx >= len(rows):
            return {'result': 'timeout', 'r': 0, 'bars': bar_idx, 'max_r': max_r}
        
        row = rows[start_idx + bar_idx]
        h, l = row.high, row.low
        
        # Calculate current candle max R
        if side == 'long':
            candle_max_r = (h - entry_price) / sl_distance
            sl_hit = l <= current_sl
        else:
            candle_max_r = (entry_price - l) / sl_distance
            sl_hit = h >= current_sl
        
        max_r = max(max_r, candle_max_r)
        
        # === CHECK SL FIRST (pessimistic, matches live) ===
        if sl_hit:
            if trailing:
                # Exit at trailed SL level
                if side == 'long':
                    exit_r = (current_sl - entry_price) / sl_distance
                else:
                    exit_r = (entry_price - current_sl) / sl_distance
                return {'result': 'trailed', 'r': exit_r, 'bars': bar_idx, 'max_r': max_r}
            elif sl_at_be:
                return {'result': 'be', 'r': 0, 'bars': bar_idx, 'max_r': max_r}
            else:
                return {'result': 'loss', 'r': -1.0, 'bars': bar_idx, 'max_r': max_r}
        
        # === MOVE TO BE AT +0.7R ===
        if not sl_at_be and max_r >= BE_THRESHOLD_R:
            current_sl = entry_price
            sl_at_be = True
        
        # === START TRAILING AT +0.7R ===
        if sl_at_be and max_r >= TRAIL_START_R:
            trailing = True
            trail_level = max_r - TRAIL_DISTANCE_R
            if trail_level > 0:
                if side == 'long':
                    new_sl = entry_price + (trail_level * sl_distance)
                    current_sl = max(current_sl, new_sl)
                else:
                    new_sl = entry_price - (trail_level * sl_distance)
                    current_sl = min(current_sl, new_sl) if current_sl != entry_price else new_sl
        
        # === TP AT +3R ===
        if candle_max_r >= MAX_PROFIT_R:
            return {'result': 'tp', 'r': MAX_PROFIT_R, 'bars': bar_idx, 'max_r': max_r}
    
    # Timeout - exit at current trailing level if trailing, else BE or 0
    if trailing:
        exit_r = max(0, max_r - TRAIL_DISTANCE_R)
        return {'result': 'timeout_trail', 'r': exit_r, 'bars': TIMEOUT_BARS, 'max_r': max_r}
    elif sl_at_be:
        return {'result': 'timeout_be', 'r': 0, 'bars': TIMEOUT_BARS, 'max_r': max_r}
    else:
        return {'result': 'timeout', 'r': -1, 'bars': TIMEOUT_BARS, 'max_r': max_r}

# =============================================================================
# MAIN
# =============================================================================

def run_backtest():
    print("=" * 80)
    print("🔬 ULTRA-REALISTIC BACKTEST (Matches Live Bot)")
    print("=" * 80)
    print("\nConfiguration:")
    print(f"  • Timeframe: {TIMEFRAME}min (1H)")
    print(f"  • Trailing: BE at +{BE_THRESHOLD_R}R, Trail {TRAIL_DISTANCE_R}R behind from +{TRAIL_START_R}R")
    print(f"  • Max Profit: +{MAX_PROFIT_R}R")
    print(f"  • SL: Pivot ({MIN_SL_ATR}-{MAX_SL_ATR}×ATR)")
    print(f"  • Volume Filter: {VOLUME_FILTER}")
    print(f"  • Timeout: {TIMEOUT_BARS} bars, Cooldown: {COOLDOWN_BARS} bars")
    print("=" * 80)
    
    # Load symbols from config
    try:
        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        symbols = config.get('trade', {}).get('divergence_symbols', [])[:NUM_SYMBOLS]
    except:
        print("Could not load config.yaml, using default symbols")
        symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT']
    
    print(f"\n📦 Testing {len(symbols)} symbols over {DATA_DAYS} days...\n")
    
    # Stats containers
    all_trades = []
    weekday_trades = []  # Mon-Fri
    weekend_trades = []  # Sat-Sun
    
    by_type = defaultdict(list)
    by_exit = defaultdict(int)
    
    start_time = time.time()
    
    for idx, sym in enumerate(symbols):
        try:
            df = fetch_klines(sym, TIMEFRAME, DATA_DAYS)
            if df.empty or len(df) < 200: 
                continue
            
            # Calculate indicators
            df['rsi'] = calculate_rsi(df['close'], RSI_PERIOD)
            hl = df['high'] - df['low']
            hc = abs(df['high'] - df['close'].shift())
            lc = abs(df['low'] - df['close'].shift())
            df['atr'] = pd.concat([hl, hc, lc], axis=1).max(axis=1).rolling(14).mean()
            df['vol_ma'] = df['volume'].rolling(20).mean()
            df['vol_ok'] = df['volume'] > df['vol_ma'] * MIN_VOL_RATIO
            df = df.dropna()
            
            if len(df) < 100: 
                continue
            
            # Get signals (Hidden Bearish only, matches live)
            signals = detect_divergence_signals(df, hidden_bearish_only=True)
            rows = list(df.itertuples())
            last_signal_idx = -COOLDOWN_BARS - 1
            
            for sig in signals:
                i = sig['idx']
                
                # Cooldown check
                if i - last_signal_idx < COOLDOWN_BARS:
                    continue
                
                if i >= len(rows) - TIMEOUT_BARS:
                    continue
                
                row = rows[i]
                if pd.isna(row.atr) or row.atr <= 0:
                    continue
                
                # Volume filter
                if VOLUME_FILTER and not row.vol_ok:
                    continue
                
                # Calculate SL
                entry, sl, sl_distance = calc_pivot_sl(rows, i, sig['side'], row.atr)
                
                if sl_distance <= 0:
                    continue
                
                # Simulate trade
                result = simulate_trade_with_trailing(rows, i, sig['side'], entry, sl, sl_distance)
                
                # Record trade
                trade_record = {
                    'symbol': sym,
                    'type': sig['type'],
                    'side': sig['side'],
                    'r': result['r'],
                    'exit': result['result'],
                    'max_r': result['max_r'],
                    'bars': result['bars'],
                    'timestamp': row.Index
                }
                
                # Categorize by day of week
                day_of_week = row.Index.dayofweek  # 0=Mon, 6=Sun
                is_weekend = day_of_week >= 5
                
                all_trades.append(trade_record)
                if is_weekend:
                    weekend_trades.append(trade_record)
                else:
                    weekday_trades.append(trade_record)
                
                by_type[sig['type']].append(result['r'])
                by_exit[result['result']] += 1
                
                last_signal_idx = i
            
            if (idx + 1) % 20 == 0:
                wins = len([t for t in all_trades if t['r'] > 0])
                total = len([t for t in all_trades if t['exit'] not in ['timeout', 'timeout_be', 'timeout_trail']])
                wr = wins / total * 100 if total > 0 else 0
                print(f"  [{idx+1}/{len(symbols)}] Trades: {len(all_trades)} | WR: {wr:.1f}%")
            
            time.sleep(0.02)
        except Exception as e:
            continue
    
    elapsed = time.time() - start_time
    
    # ==========================================================================
    # RESULTS
    # ==========================================================================
    
    def analyze_trades(trades, label):
        if not trades:
            return None
        
        # Exclude timeouts from WR calculation
        valid = [t for t in trades if t['exit'] not in ['timeout']]
        wins = len([t for t in valid if t['r'] > 0])
        losses = len([t for t in valid if t['r'] < 0])
        bes = len([t for t in valid if t['r'] == 0])
        
        total = wins + losses
        wr = wins / total * 100 if total > 0 else 0
        lb_wr = wilson_lb(wins, total) * 100
        
        total_r = sum(t['r'] for t in trades)
        avg_r = total_r / len(trades) if trades else 0
        
        # By exit type
        exit_counts = defaultdict(int)
        for t in trades:
            exit_counts[t['exit']] += 1
        
        return {
            'label': label,
            'trades': len(trades),
            'wins': wins,
            'losses': losses,
            'bes': bes,
            'wr': wr,
            'lb_wr': lb_wr,
            'total_r': total_r,
            'avg_r': avg_r,
            'exits': dict(exit_counts)
        }
    
    all_stats = analyze_trades(all_trades, "ALL")
    weekday_stats = analyze_trades(weekday_trades, "WEEKDAY (Mon-Fri)")
    weekend_stats = analyze_trades(weekend_trades, "WEEKEND (Sat-Sun)")
    
    print("\n" + "=" * 80)
    print("📊 RESULTS: ULTRA-REALISTIC BACKTEST")
    print("=" * 80)
    
    # Overall
    if all_stats:
        print(f"\n🌍 **OVERALL ({all_stats['trades']} trades)**")
        print(f"├ Wins: {all_stats['wins']} | Losses: {all_stats['losses']} | BE: {all_stats['bes']}")
        print(f"├ Win Rate: {all_stats['wr']:.1f}% (LB: {all_stats['lb_wr']:.1f}%)")
        print(f"├ Total R: {all_stats['total_r']:+.1f}R")
        print(f"├ Avg R/Trade: {all_stats['avg_r']:+.3f}")
        print(f"└ Exits: {dict(all_stats['exits'])}")
    
    # Weekday
    if weekday_stats:
        print(f"\n📅 **{weekday_stats['label']} ({weekday_stats['trades']} trades)**")
        print(f"├ Wins: {weekday_stats['wins']} | Losses: {weekday_stats['losses']} | BE: {weekday_stats['bes']}")
        print(f"├ Win Rate: {weekday_stats['wr']:.1f}% (LB: {weekday_stats['lb_wr']:.1f}%)")
        print(f"├ Total R: {weekday_stats['total_r']:+.1f}R")
        print(f"├ Avg R/Trade: {weekday_stats['avg_r']:+.3f}")
        print(f"└ Exits: {dict(weekday_stats['exits'])}")
    
    # Weekend
    if weekend_stats:
        print(f"\n🌴 **{weekend_stats['label']} ({weekend_stats['trades']} trades)**")
        print(f"├ Wins: {weekend_stats['wins']} | Losses: {weekend_stats['losses']} | BE: {weekend_stats['bes']}")
        print(f"├ Win Rate: {weekend_stats['wr']:.1f}% (LB: {weekend_stats['lb_wr']:.1f}%)")
        print(f"├ Total R: {weekend_stats['total_r']:+.1f}R")
        print(f"├ Avg R/Trade: {weekend_stats['avg_r']:+.3f}")
        print(f"└ Exits: {dict(weekend_stats['exits'])}")
    
    # Comparison
    print("\n" + "=" * 80)
    print("📈 WEEKDAY vs WEEKEND COMPARISON")
    print("=" * 80)
    
    if weekday_stats and weekend_stats:
        wr_diff = weekday_stats['wr'] - weekend_stats['wr']
        avgr_diff = weekday_stats['avg_r'] - weekend_stats['avg_r']
        
        print(f"\n| Metric           | Weekday       | Weekend       | Diff          |")
        print(f"|-----------------|---------------|---------------|---------------|")
        print(f"| Trades          | {weekday_stats['trades']:<13} | {weekend_stats['trades']:<13} | -             |")
        print(f"| Win Rate        | {weekday_stats['wr']:.1f}%{' ' * 8} | {weekend_stats['wr']:.1f}%{' ' * 8} | {wr_diff:+.1f}%{' ' * 7} |")
        print(f"| Total R         | {weekday_stats['total_r']:+.1f}R{' ' * 7} | {weekend_stats['total_r']:+.1f}R{' ' * 7} | -             |")
        print(f"| Avg R/Trade     | {weekday_stats['avg_r']:+.3f}{' ' * 7} | {weekend_stats['avg_r']:+.3f}{' ' * 7} | {avgr_diff:+.3f}{' ' * 6} |")
        
        # Recommendation
        print("\n" + "=" * 80)
        print("💡 RECOMMENDATION")
        print("=" * 80)
        
        if wr_diff > 3 and avgr_diff > 0.05:
            print("\n✅ **WEEKDAYS SIGNIFICANTLY BETTER**")
            print(f"   • WR: {wr_diff:+.1f}% higher on weekdays")
            print(f"   • Avg R: {avgr_diff:+.3f} higher on weekdays")
            print("\n   📋 Consider: Disable trading on weekends (Sat-Sun)")
        elif wr_diff < -3 and avgr_diff < -0.05:
            print("\n✅ **WEEKENDS SIGNIFICANTLY BETTER**")
            print(f"   • WR: {abs(wr_diff):.1f}% higher on weekends")
            print(f"   • Avg R: {abs(avgr_diff):.3f} higher on weekends")
            print("\n   📋 Consider: Trade MORE aggressively on weekends")
        else:
            print("\n⚖️ **NO SIGNIFICANT DIFFERENCE**")
            print(f"   • WR difference: {abs(wr_diff):.1f}% (< 3% threshold)")
            print(f"   • Avg R difference: {abs(avgr_diff):.3f}")
            print("\n   📋 Keep trading 24/7 - No weekend filter needed")
    
    print(f"\n⏱️ Completed in {elapsed/60:.1f} minutes")
    
    # Save results
    results_file = 'ultra_realistic_backtest_results.csv'
    pd.DataFrame(all_trades).to_csv(results_file, index=False)
    print(f"\n💾 Detailed results saved to: {results_file}")
    
    return all_stats, weekday_stats, weekend_stats

if __name__ == "__main__":
    run_backtest()
