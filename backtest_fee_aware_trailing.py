#!/usr/bin/env python3
"""
FEE-AWARE TRAILING SL OPTIMIZATION
===================================

COPIES THE EXACT BOT LOGIC from bot.py and backtest_trailing_detailed.py
Tests multiple trailing configurations to find the most profitable after fees.

Fee structure:
- Round-trip fees: 0.11% (0.055% maker + 0.055% taker)
- Fee impact on 1R risk with 1% SL = ~11% of risk = 0.11R
"""

import requests
import pandas as pd
import numpy as np
from collections import defaultdict
from datetime import datetime
import time
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION - MATCH BOT EXACTLY
# ============================================================================

TIMEFRAME = '3'          # 3-minute candles (bot uses 3m)
DATA_DAYS = 14           # 14 days of data
NUM_SYMBOLS = 50         # Number of symbols to test

# Fees (Bybit standard)
MAKER_FEE = 0.00055      # 0.055%
TAKER_FEE = 0.00055      # 0.055%
ROUND_TRIP_FEE = MAKER_FEE + TAKER_FEE  # 0.11%

# RSI Settings (match bot)
RSI_PERIOD = 14
RSI_OVERSOLD = 30
RSI_OVERBOUGHT = 70
LOOKBACK_BARS = 14
MIN_PIVOT_DISTANCE = 5
COOLDOWN_BARS = 10

# Trade Settings
DEFAULT_RR = 3.0
MIN_SL_ATR = 0.3
MAX_SL_ATR = 2.0

BASE_URL = "https://api.bybit.com"

# ============================================================================
# TRAILING CONFIGURATIONS TO TEST
# ============================================================================

TRAILING_CONFIGS = [
    # Name, BE_Threshold, BE_Lock, Trail_Start, Trail_Distance, Max_TP
    ("Current (Bot)", 0.7, 0.4, 0.7, 0.3, 3.0),
    ("Conservative-1", 0.5, 0.3, 0.5, 0.2, 2.0),
    ("Conservative-2", 0.6, 0.4, 0.6, 0.25, 2.5),
    ("Aggressive-1", 1.0, 0.5, 1.0, 0.4, 4.0),
    ("Aggressive-2", 0.8, 0.5, 0.8, 0.35, 3.5),
    ("Balanced-1", 0.7, 0.5, 0.7, 0.25, 2.5),
    ("Balanced-2", 0.8, 0.4, 0.8, 0.3, 3.0),
    ("Fee-Aware-1", 0.5, 0.35, 0.5, 0.15, 2.0),
    ("Fee-Aware-2", 0.6, 0.45, 0.6, 0.15, 2.5),
    ("Quick-Lock", 0.4, 0.3, 0.4, 0.15, 2.0),
    ("Static-2R", 99.0, 0.0, 99.0, 0.0, 2.0),  # No trailing, fixed 2R
    ("Static-3R", 99.0, 0.0, 99.0, 0.0, 3.0),  # No trailing, fixed 3R
]

# ============================================================================
# DATA FETCHING
# ============================================================================

def get_symbols(limit):
    """Get top symbols by volume"""
    resp = requests.get(f"{BASE_URL}/v5/market/tickers?category=linear", timeout=10)
    tickers = resp.json().get('result', {}).get('list', [])
    usdt = [t for t in tickers if t['symbol'].endswith('USDT')]
    usdt.sort(key=lambda x: float(x.get('turnover24h', 0)), reverse=True)
    return [t['symbol'] for t in usdt[:limit]]


def fetch_klines(symbol, interval, days):
    """Fetch klines from Bybit"""
    end_ts = int(datetime.now().timestamp() * 1000)
    candles_needed = days * 24 * 60 // int(interval)
    all_candles = []
    current_end = end_ts
    
    while len(all_candles) < candles_needed:
        params = {'category': 'linear', 'symbol': symbol, 'interval': interval, 
                  'limit': 1000, 'end': current_end}
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


# ============================================================================
# INDICATORS (MATCH BOT EXACTLY)
# ============================================================================

def calculate_rsi(close, period=14):
    """RSI calculation - same as bot"""
    delta = close.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    return 100 - (100 / (1 + rs))


def find_pivots(data, left=3, right=3):
    """Find pivot highs and lows - same as bot"""
    n = len(data)
    pivot_highs = np.full(n, np.nan)
    pivot_lows = np.full(n, np.nan)
    for i in range(left, n - right):
        is_high = all(data[j] < data[i] for j in range(i - left, i + right + 1) if j != i)
        is_low = all(data[j] > data[i] for j in range(i - left, i + right + 1) if j != i)
        if is_high: pivot_highs[i] = data[i]
        if is_low: pivot_lows[i] = data[i]
    return pivot_highs, pivot_lows


# ============================================================================
# DIVERGENCE DETECTION (MATCH BOT EXACTLY)
# ============================================================================

def detect_all_divergences(df):
    """Detect RSI divergences - same logic as bot"""
    if len(df) < 100: return []
    
    close = df['close'].values
    rsi = df['rsi'].values
    n = len(df)
    
    price_ph, price_pl = find_pivots(close, 3, 3)
    signals = []
    
    for i in range(30, n - 5):
        # Find pivot lows
        curr_pl = curr_pli = prev_pl = prev_pli = None
        for j in range(i, max(i - LOOKBACK_BARS, 0), -1):
            if not np.isnan(price_pl[j]):
                if curr_pl is None: curr_pl, curr_pli = price_pl[j], j
                elif prev_pl is None and j < curr_pli - MIN_PIVOT_DISTANCE:
                    prev_pl, prev_pli = price_pl[j], j
                    break
        
        # Find pivot highs
        curr_ph = curr_phi = prev_ph = prev_phi = None
        for j in range(i, max(i - LOOKBACK_BARS, 0), -1):
            if not np.isnan(price_ph[j]):
                if curr_ph is None: curr_ph, curr_phi = price_ph[j], j
                elif prev_ph is None and j < curr_phi - MIN_PIVOT_DISTANCE:
                    prev_ph, prev_phi = price_ph[j], j
                    break
        
        # Regular Bullish: Lower low + Higher RSI low
        if curr_pl and prev_pl:
            if curr_pl < prev_pl and rsi[curr_pli] > rsi[prev_pli]:
                if rsi[i] < RSI_OVERSOLD + 15:
                    signals.append({'idx': i, 'type': 'regular_bullish', 'side': 'long', 'swing': curr_pl})
                    continue
        
        # Regular Bearish: Higher high + Lower RSI high
        if curr_ph and prev_ph:
            if curr_ph > prev_ph and rsi[curr_phi] < rsi[prev_phi]:
                if rsi[i] > RSI_OVERBOUGHT - 15:
                    signals.append({'idx': i, 'type': 'regular_bearish', 'side': 'short', 'swing': curr_ph})
                    continue
        
        # Hidden Bullish: Higher low + Lower RSI low
        if curr_pl and prev_pl:
            if curr_pl > prev_pl and rsi[curr_pli] < rsi[prev_pli]:
                if rsi[i] < RSI_OVERBOUGHT - 10:
                    signals.append({'idx': i, 'type': 'hidden_bullish', 'side': 'long', 'swing': curr_pl})
                    continue
        
        # Hidden Bearish: Lower high + Higher RSI high
        if curr_ph and prev_ph:
            if curr_ph < prev_ph and rsi[curr_phi] > rsi[prev_phi]:
                if rsi[i] > RSI_OVERSOLD + 10:
                    signals.append({'idx': i, 'type': 'hidden_bearish', 'side': 'short', 'swing': curr_ph})
    
    return signals


# ============================================================================
# TRADE SIMULATION WITH EXACT BOT TRAILING LOGIC
# ============================================================================

def calc_sltp(rows, idx, side, atr, swing_price, max_tp_r):
    """Calculate entry, SL, TP - same as bot"""
    if idx + 1 >= len(rows):
        return None, None, None, None
    
    entry = rows[idx + 1].open
    sl_dist = abs(swing_price - entry)
    
    # Clamp to ATR bounds
    min_dist = MIN_SL_ATR * atr
    max_dist = MAX_SL_ATR * atr
    if sl_dist < min_dist: sl_dist = min_dist
    if sl_dist > max_dist: sl_dist = max_dist
    
    if side == 'long':
        sl = entry - sl_dist
        tp = entry + (sl_dist * max_tp_r)
    else:
        sl = entry + sl_dist
        tp = entry - (sl_dist * max_tp_r)
    
    return entry, sl, sl_dist, tp


def simulate_trade_with_config(rows, signal_idx, side, sl, sl_dist, tp, entry, config):
    """
    Simulate trade with EXACT bot trailing logic.
    
    Config: (name, be_threshold, be_lock, trail_start, trail_distance, max_tp)
    """
    _, be_threshold, be_lock, trail_start, trail_distance, max_tp = config
    
    entry_idx = signal_idx + 1
    if entry_idx >= len(rows) - 1: 
        return 'skip', 0, 0, 0
    
    current_sl = sl
    max_favorable_r = 0
    be_triggered = False
    trailing_active = False
    
    for bar_idx, row in enumerate(rows[entry_idx:entry_idx + 100]):
        if side == 'long':
            # Calculate R-multiple at this bar
            unrealized_r = (row.high - entry) / sl_dist
            if unrealized_r > max_favorable_r:
                max_favorable_r = unrealized_r
                
                # Trail logic (EXACT BOT LOGIC)
                if max_favorable_r >= trail_start:
                    trailing_active = True
                    # Trail: Lock in (max_r - trail_distance) R
                    new_sl = entry + (max_favorable_r - trail_distance) * sl_dist
                    if new_sl > current_sl:
                        current_sl = new_sl
                elif max_favorable_r >= be_threshold and not be_triggered:
                    # Move to BE + lock
                    new_sl = entry + be_lock * sl_dist
                    if new_sl > current_sl:
                        current_sl = new_sl
                    be_triggered = True
            
            # Check SL hit FIRST (conservative)
            if row.low <= current_sl:
                exit_r = (current_sl - entry) / sl_dist
                return 'sl_hit', exit_r, max_favorable_r, bar_idx
            
            # Check TP hit
            if row.high >= tp:
                return 'tp_hit', max_tp, max_favorable_r, bar_idx
                
        else:  # SHORT
            unrealized_r = (entry - row.low) / sl_dist
            if unrealized_r > max_favorable_r:
                max_favorable_r = unrealized_r
                
                if max_favorable_r >= trail_start:
                    trailing_active = True
                    new_sl = entry - (max_favorable_r - trail_distance) * sl_dist
                    if new_sl < current_sl:
                        current_sl = new_sl
                elif max_favorable_r >= be_threshold and not be_triggered:
                    new_sl = entry - be_lock * sl_dist
                    if new_sl < current_sl:
                        current_sl = new_sl
                    be_triggered = True
            
            if row.high >= current_sl:
                exit_r = (entry - current_sl) / sl_dist
                return 'sl_hit', exit_r, max_favorable_r, bar_idx
            
            if row.low <= tp:
                return 'tp_hit', max_tp, max_favorable_r, bar_idx
    
    # Timeout
    exit_r = (current_sl - entry) / sl_dist if side == 'long' else (entry - current_sl) / sl_dist
    return 'timeout', exit_r, max_favorable_r, 100


def calculate_fee_impact_r(entry, sl_dist):
    """Calculate fee impact as R-multiple"""
    sl_pct = sl_dist / entry  # SL as percentage of entry
    fee_impact = ROUND_TRIP_FEE / sl_pct  # Fee as multiple of risk
    return fee_impact


# ============================================================================
# MAIN
# ============================================================================

def run():
    print("=" * 80)
    print("📊 FEE-AWARE TRAILING SL OPTIMIZATION")
    print("=" * 80)
    print(f"\nFees: {ROUND_TRIP_FEE*100:.3f}% round-trip ({MAKER_FEE*100:.3f}% maker + {TAKER_FEE*100:.3f}% taker)")
    print(f"Testing {len(TRAILING_CONFIGS)} configurations on {NUM_SYMBOLS} symbols")
    
    symbols = get_symbols(NUM_SYMBOLS)
    print(f"\n📋 Fetching data for {len(symbols)} symbols...\n")
    
    # Results per config
    results = {cfg[0]: {'trades': [], 'theo_r': [], 'actual_r': [], 'fees_r': []} for cfg in TRAILING_CONFIGS}
    
    for idx, sym in enumerate(symbols):
        try:
            df = fetch_klines(sym, TIMEFRAME, DATA_DAYS)
            if df.empty or len(df) < 200: continue
            
            df['rsi'] = calculate_rsi(df['close'], RSI_PERIOD)
            hl = df['high'] - df['low']
            hc = abs(df['high'] - df['close'].shift())
            lc = abs(df['low'] - df['close'].shift())
            df['atr'] = pd.concat([hl, hc, lc], axis=1).max(axis=1).rolling(14).mean()
            df['vol_ma'] = df['volume'].rolling(20).mean()
            df['vol_ok'] = df['volume'] > df['vol_ma'] * 0.5
            df = df.dropna()
            
            if len(df) > 0:
                df = df.iloc[:-1]  # Skip forming candle
            
            if len(df) < 100: continue
            
            signals = detect_all_divergences(df)
            rows = list(df.itertuples())
            
            last_trade_idx = -COOLDOWN_BARS
            
            for sig in signals:
                i = sig['idx']
                if i - last_trade_idx < COOLDOWN_BARS: continue
                if i >= len(rows) - 50: continue
                
                row = rows[i]
                if not row.vol_ok or row.atr <= 0: continue
                
                # Test each config
                for config in TRAILING_CONFIGS:
                    cfg_name = config[0]
                    max_tp = config[5]
                    
                    entry, sl, sl_dist, tp = calc_sltp(rows, i, sig['side'], row.atr, sig['swing'], max_tp)
                    if entry is None: continue
                    
                    result, exit_r, max_r, bars = simulate_trade_with_config(
                        rows, i, sig['side'], sl, sl_dist, tp, entry, config
                    )
                    
                    if result == 'skip' or (result == 'timeout' and exit_r == 0):
                        continue
                    
                    # Calculate fee impact
                    fee_r = calculate_fee_impact_r(entry, sl_dist)
                    actual_r = exit_r - fee_r
                    
                    results[cfg_name]['trades'].append({
                        'symbol': sym,
                        'side': sig['side'],
                        'type': sig['type'],
                        'theo_r': exit_r,
                        'fee_r': fee_r,
                        'actual_r': actual_r,
                        'max_r': max_r,
                        'result': result
                    })
                    results[cfg_name]['theo_r'].append(exit_r)
                    results[cfg_name]['actual_r'].append(actual_r)
                    results[cfg_name]['fees_r'].append(fee_r)
                
                last_trade_idx = i
                
        except Exception as e:
            continue
        
        if (idx + 1) % 10 == 0:
            total_trades = len(results[TRAILING_CONFIGS[0][0]]['trades'])
            print(f"  [{idx+1}/{NUM_SYMBOLS}] Total trades per config: {total_trades}")
    
    # ============================================================================
    # RESULTS ANALYSIS
    # ============================================================================
    
    print("\n" + "=" * 100)
    print("📊 RESULTS (Sorted by Actual R after fees)")
    print("=" * 100)
    
    summary = []
    for config in TRAILING_CONFIGS:
        cfg_name = config[0]
        trades = results[cfg_name]['trades']
        
        if not trades:
            continue
        
        n = len(trades)
        theo_wins = sum(1 for t in trades if t['theo_r'] > 0)
        actual_wins = sum(1 for t in trades if t['actual_r'] > 0)
        
        theo_wr = theo_wins / n * 100
        actual_wr = actual_wins / n * 100
        
        total_theo = sum(t['theo_r'] for t in trades)
        total_actual = sum(t['actual_r'] for t in trades)
        total_fees = sum(t['fee_r'] for t in trades)
        
        avg_theo = total_theo / n
        avg_actual = total_actual / n
        
        # Profit factor
        gross_profit = sum(t['actual_r'] for t in trades if t['actual_r'] > 0)
        gross_loss = abs(sum(t['actual_r'] for t in trades if t['actual_r'] < 0))
        pf = gross_profit / gross_loss if gross_loss > 0 else 999
        
        summary.append({
            'Config': cfg_name,
            'N': n,
            'WR_Theo': f"{theo_wr:.1f}%",
            'WR_Actual': f"{actual_wr:.1f}%",
            'Total_Theo': f"{total_theo:+.1f}R",
            'Total_Actual': f"{total_actual:+.1f}R",
            'Fees_Paid': f"{total_fees:.1f}R",
            'EV/Trade': f"{avg_actual:+.3f}R",
            'PF': f"{pf:.2f}",
            'Profitable': '✅' if total_actual > 0 else '❌',
            '_sort': total_actual
        })
    
    # Sort by actual R
    summary.sort(key=lambda x: x['_sort'], reverse=True)
    for s in summary:
        del s['_sort']
    
    # Print table
    df_summary = pd.DataFrame(summary)
    print(df_summary.to_string(index=False))
    
    # Save
    df_summary.to_csv('backtest_trailing_optimization_results.csv', index=False)
    print("\n✅ Saved to backtest_trailing_optimization_results.csv")
    
    # Best config
    if summary:
        best = summary[0]
        print("\n" + "=" * 70)
        print("🏆 BEST CONFIGURATION (After Fees)")
        print("=" * 70)
        for cfg in TRAILING_CONFIGS:
            if cfg[0] == best['Config']:
                print(f"Name: {cfg[0]}")
                print(f"BE Threshold: {cfg[1]}R")
                print(f"BE Lock: {cfg[2]}R")
                print(f"Trail Start: {cfg[3]}R")
                print(f"Trail Distance: {cfg[4]}R behind")
                print(f"Max TP: {cfg[5]}R")
                break
        print(f"\nTrades: {best['N']}")
        print(f"Win Rate (After Fees): {best['WR_Actual']}")
        print(f"Total R (After Fees): {best['Total_Actual']}")
        print(f"EV per Trade: {best['EV/Trade']}")
        print(f"Profit Factor: {best['PF']}")
        print(f"Profitable: {best['Profitable']}")


if __name__ == "__main__":
    run()
