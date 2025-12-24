#!/usr/bin/env python3
"""
COMPREHENSIVE 1H STRATEGY OPTIMIZATION
=======================================
Tests MANY variations to find optimal 1H configuration:

1. SL METHODS: ATR-based (1x, 1.5x, 2x) vs Fixed % (2%, 3%)
2. TRAILING: Multiple BE thresholds and trail distances
3. R:R TARGETS: 1:2, 1:3, 1:4
4. SIGNAL FILTERS: All, Hidden-Only, Bearish-Only
5. COOLDOWN: 5, 10, 15 bars

Goal: Find best combination for live trading.
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime
import time
import warnings
from itertools import product
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

TIMEFRAME = '60'         # 1-hour candles
DATA_DAYS = 90           # 90 days
NUM_SYMBOLS = 100        # Top 100 symbols

# Fees
MAKER_FEE = 0.00055
TAKER_FEE = 0.00055
ROUND_TRIP_FEE = MAKER_FEE + TAKER_FEE

# RSI Settings
RSI_PERIOD = 14
RSI_OVERSOLD = 30
RSI_OVERBOUGHT = 70
LOOKBACK_BARS = 14
MIN_PIVOT_DISTANCE = 5

BASE_URL = "https://api.bybit.com"

# ============================================================================
# STRATEGY VARIATIONS TO TEST
# ============================================================================

# SL Methods: (name, type, multiplier_or_pct)
SL_METHODS = [
    ("ATR_1x", "atr", 1.0),
    ("ATR_1.5x", "atr", 1.5),
    ("ATR_2x", "atr", 2.0),
    ("Fixed_2%", "fixed", 2.0),
    ("Fixed_3%", "fixed", 3.0),
]

# Trailing Configs: (name, BE_threshold, trail_distance, max_tp)
TRAILING_CONFIGS = [
    # Trailing with different aggressiveness
    ("Trail_Tight", 0.3, 0.1, 2.0),      # Current config
    ("Trail_Medium", 0.5, 0.2, 2.0),
    ("Trail_Loose", 0.7, 0.3, 2.0),
    ("Trail_Tight_3R", 0.3, 0.1, 3.0),
    ("Trail_Medium_3R", 0.5, 0.2, 3.0),
    
    # No trailing (pure R:R)
    ("Static_1:2", 99.0, 0.0, 2.0),
    ("Static_1:3", 99.0, 0.0, 3.0),
    ("Static_1:4", 99.0, 0.0, 4.0),
]

# Signal Filters: (name, allowed_types)
SIGNAL_FILTERS = [
    ("All_Signals", ["regular_bullish", "regular_bearish", "hidden_bullish", "hidden_bearish"]),
    ("Hidden_Only", ["hidden_bullish", "hidden_bearish"]),
    ("Bearish_Only", ["regular_bearish", "hidden_bearish"]),
    ("Hidden_Bearish", ["hidden_bearish"]),  # Most selective
]

# Cooldown periods (bars)
COOLDOWNS = [5, 10, 15]

# ============================================================================
# DATA FETCHING
# ============================================================================

def get_symbols(limit):
    resp = requests.get(f"{BASE_URL}/v5/market/tickers?category=linear", timeout=10)
    tickers = resp.json().get('result', {}).get('list', [])
    usdt = [t for t in tickers if t['symbol'].endswith('USDT')]
    usdt.sort(key=lambda x: float(x.get('turnover24h', 0)), reverse=True)
    return [t['symbol'] for t in usdt[:limit]]


def fetch_klines(symbol, interval, days):
    end_ts = int(datetime.now().timestamp() * 1000)
    candles_needed = days * 24
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
# INDICATORS
# ============================================================================

def calculate_rsi(close, period=14):
    delta = close.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    return 100 - (100 / (1 + rs))


def calculate_atr(df, period=14):
    hl = df['high'] - df['low']
    hc = abs(df['high'] - df['close'].shift())
    lc = abs(df['low'] - df['close'].shift())
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.rolling(period).mean()


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


# ============================================================================
# DIVERGENCE DETECTION
# ============================================================================

def detect_divergences(df):
    if len(df) < 100: return []
    
    close = df['close'].values
    rsi = df['rsi'].values
    n = len(df)
    
    price_ph, price_pl = find_pivots(close, 3, 3)
    signals = []
    
    for i in range(30, n - 5):
        confirmed_up_to = i - 3
        
        # Find pivot lows
        curr_pl = curr_pli = prev_pl = prev_pli = None
        for j in range(confirmed_up_to, max(confirmed_up_to - LOOKBACK_BARS, 0), -1):
            if not np.isnan(price_pl[j]):
                if curr_pl is None: 
                    curr_pl, curr_pli = price_pl[j], j
                elif prev_pl is None and j < curr_pli - MIN_PIVOT_DISTANCE:
                    prev_pl, prev_pli = price_pl[j], j
                    break
        
        # Find pivot highs
        curr_ph = curr_phi = prev_ph = prev_phi = None
        for j in range(confirmed_up_to, max(confirmed_up_to - LOOKBACK_BARS, 0), -1):
            if not np.isnan(price_ph[j]):
                if curr_ph is None: 
                    curr_ph, curr_phi = price_ph[j], j
                elif prev_ph is None and j < curr_phi - MIN_PIVOT_DISTANCE:
                    prev_ph, prev_phi = price_ph[j], j
                    break
        
        # Regular Bullish
        if curr_pl and prev_pl:
            if curr_pl < prev_pl and rsi[curr_pli] > rsi[prev_pli]:
                if rsi[i] < RSI_OVERSOLD + 15:
                    signals.append({'idx': i, 'type': 'regular_bullish', 'side': 'long', 'swing': curr_pl})
                    continue
        
        # Regular Bearish
        if curr_ph and prev_ph:
            if curr_ph > prev_ph and rsi[curr_phi] < rsi[prev_phi]:
                if rsi[i] > RSI_OVERBOUGHT - 15:
                    signals.append({'idx': i, 'type': 'regular_bearish', 'side': 'short', 'swing': curr_ph})
                    continue
        
        # Hidden Bullish
        if curr_pl and prev_pl:
            if curr_pl > prev_pl and rsi[curr_pli] < rsi[prev_pli]:
                if rsi[i] < RSI_OVERBOUGHT - 10:
                    signals.append({'idx': i, 'type': 'hidden_bullish', 'side': 'long', 'swing': curr_pl})
                    continue
        
        # Hidden Bearish
        if curr_ph and prev_ph:
            if curr_ph < prev_ph and rsi[curr_phi] > rsi[prev_phi]:
                if rsi[i] > RSI_OVERSOLD + 10:
                    signals.append({'idx': i, 'type': 'hidden_bearish', 'side': 'short', 'swing': curr_ph})
    
    return signals


# ============================================================================
# TRADE SIMULATION
# ============================================================================

def simulate_trade(rows, signal_idx, side, swing_price, atr, entry_price,
                   sl_method, trailing_config):
    
    sl_type, sl_param = sl_method[1], sl_method[2]
    _, be_threshold, trail_distance, max_tp = trailing_config
    
    entry_idx = signal_idx + 1
    if entry_idx >= len(rows) - 1:
        return None
    
    entry = entry_price
    
    # Calculate SL distance based on method
    if sl_type == "atr":
        sl_dist = sl_param * atr
    else:  # fixed %
        sl_dist = entry * (sl_param / 100)
    
    # Ensure minimum SL
    min_sl = entry * 0.02  # 2% minimum
    sl_dist = max(sl_dist, min_sl)
    
    if side == 'long':
        initial_sl = entry - sl_dist
        tp = entry + (max_tp * sl_dist)
    else:
        initial_sl = entry + sl_dist
        tp = entry - (max_tp * sl_dist)
    
    current_sl = initial_sl
    max_favorable_r = 0
    
    for bar_offset in range(1, min(100, len(rows) - entry_idx)):
        bar = rows[entry_idx + bar_offset]
        high = float(bar.high)
        low = float(bar.low)
        
        if side == 'long':
            if low <= current_sl:
                exit_r = (current_sl - entry) / sl_dist
                return {'exit_r': exit_r, 'result': 'sl', 'entry': entry, 'sl_dist': sl_dist}
            
            if high >= tp:
                return {'exit_r': max_tp, 'result': 'tp', 'entry': entry, 'sl_dist': sl_dist}
            
            current_r = (high - entry) / sl_dist
            if current_r > max_favorable_r:
                max_favorable_r = current_r
                if max_favorable_r >= be_threshold and trail_distance > 0:
                    new_sl = entry + (max_favorable_r - trail_distance) * sl_dist
                    if new_sl > current_sl:
                        current_sl = new_sl
        else:
            if high >= current_sl:
                exit_r = (entry - current_sl) / sl_dist
                return {'exit_r': exit_r, 'result': 'sl', 'entry': entry, 'sl_dist': sl_dist}
            
            if low <= tp:
                return {'exit_r': max_tp, 'result': 'tp', 'entry': entry, 'sl_dist': sl_dist}
            
            current_r = (entry - low) / sl_dist
            if current_r > max_favorable_r:
                max_favorable_r = current_r
                if max_favorable_r >= be_threshold and trail_distance > 0:
                    new_sl = entry - (max_favorable_r - trail_distance) * sl_dist
                    if new_sl < current_sl:
                        current_sl = new_sl
    
    # Timeout
    last_bar = rows[min(entry_idx + 99, len(rows) - 1)]
    if side == 'long':
        exit_r = (float(last_bar.close) - entry) / sl_dist
    else:
        exit_r = (entry - float(last_bar.close)) / sl_dist
    
    return {'exit_r': exit_r, 'result': 'timeout', 'entry': entry, 'sl_dist': sl_dist}


def calculate_fee_r(entry, sl_dist):
    sl_pct = sl_dist / entry
    return ROUND_TRIP_FEE / sl_pct


# ============================================================================
# MAIN BACKTEST
# ============================================================================

def run():
    print("=" * 100)
    print("🔬 COMPREHENSIVE 1H STRATEGY OPTIMIZATION")
    print("=" * 100)
    print(f"Timeframe: {TIMEFRAME}min (1 hour)")
    print(f"Symbols: {NUM_SYMBOLS} | Days: {DATA_DAYS}")
    print(f"\nTesting:")
    print(f"  - {len(SL_METHODS)} SL methods")
    print(f"  - {len(TRAILING_CONFIGS)} trailing configs")
    print(f"  - {len(SIGNAL_FILTERS)} signal filters")
    print(f"  - {len(COOLDOWNS)} cooldown periods")
    print(f"  = {len(SL_METHODS) * len(TRAILING_CONFIGS) * len(SIGNAL_FILTERS) * len(COOLDOWNS)} total combinations")
    print()
    
    # Fetch symbols
    symbols = get_symbols(NUM_SYMBOLS)
    print(f"📋 Loaded {len(symbols)} symbols\n")
    
    # Preload all data
    print("📥 Preloading data...")
    symbol_data = {}
    for idx, sym in enumerate(symbols):
        try:
            df = fetch_klines(sym, TIMEFRAME, DATA_DAYS)
            if df.empty or len(df) < 200: continue
            
            df['rsi'] = calculate_rsi(df['close'], RSI_PERIOD)
            df['atr'] = calculate_atr(df, 14)
            df['vol_ma'] = df['volume'].rolling(20).mean()
            df['vol_ok'] = df['volume'] > df['vol_ma'] * 0.5
            df = df.dropna()
            
            if len(df) >= 100:
                signals = detect_divergences(df)
                if signals:
                    symbol_data[sym] = {
                        'df': df,
                        'signals': signals,
                        'rows': list(df.itertuples())
                    }
        except:
            continue
        
        if (idx + 1) % 25 == 0:
            print(f"  [{idx+1}/{NUM_SYMBOLS}] Loaded {len(symbol_data)} symbols with signals")
    
    print(f"\n✅ {len(symbol_data)} symbols with valid signals\n")
    
    # Test all combinations
    results = []
    
    total_combos = len(SL_METHODS) * len(TRAILING_CONFIGS) * len(SIGNAL_FILTERS) * len(COOLDOWNS)
    combo_idx = 0
    
    for sl_method in SL_METHODS:
        for trailing_config in TRAILING_CONFIGS:
            for signal_filter in SIGNAL_FILTERS:
                for cooldown in COOLDOWNS:
                    combo_idx += 1
                    
                    config_name = f"{sl_method[0]}|{trailing_config[0]}|{signal_filter[0]}|CD{cooldown}"
                    
                    trades = []
                    
                    for sym, data in symbol_data.items():
                        df = data['df']
                        signals = data['signals']
                        rows = data['rows']
                        
                        last_trade_idx = -cooldown
                        
                        for sig in signals:
                            # Apply signal filter
                            if sig['type'] not in signal_filter[1]:
                                continue
                            
                            i = sig['idx']
                            if i - last_trade_idx < cooldown: continue
                            if i >= len(rows) - 50: continue
                            
                            row = rows[i]
                            if not row.vol_ok or row.atr <= 0: continue
                            
                            entry_price = float(rows[i+1].open) if i+1 < len(rows) else row.close
                            
                            trade = simulate_trade(
                                rows, i, sig['side'], sig['swing'], row.atr, entry_price,
                                sl_method, trailing_config
                            )
                            
                            if trade is None: continue
                            
                            fee_r = calculate_fee_r(trade['entry'], trade['sl_dist'])
                            actual_r = trade['exit_r'] - fee_r
                            
                            trades.append({
                                'symbol': sym,
                                'type': sig['type'],
                                'actual_r': actual_r,
                                'result': trade['result']
                            })
                            
                            last_trade_idx = i
                    
                    # Calculate metrics
                    if trades:
                        n = len(trades)
                        wins = sum(1 for t in trades if t['actual_r'] > 0)
                        wr = wins / n * 100
                        total_r = sum(t['actual_r'] for t in trades)
                        avg_r = total_r / n
                        
                        gross_profit = sum(t['actual_r'] for t in trades if t['actual_r'] > 0)
                        gross_loss = abs(sum(t['actual_r'] for t in trades if t['actual_r'] < 0))
                        pf = gross_profit / gross_loss if gross_loss > 0 else 999
                        
                        results.append({
                            'Config': config_name,
                            'SL': sl_method[0],
                            'Trail': trailing_config[0],
                            'Filter': signal_filter[0],
                            'Cooldown': cooldown,
                            'N': n,
                            'WR': wr,
                            'Total_R': total_r,
                            'Avg_R': avg_r,
                            'PF': pf,
                            'Status': '✅' if total_r > 0 else '❌'
                        })
                    
                    if combo_idx % 50 == 0:
                        print(f"  [{combo_idx}/{total_combos}] Testing configurations...")
    
    # Sort by Total R
    results.sort(key=lambda x: x['Total_R'], reverse=True)
    
    # Display top 20
    print("\n" + "=" * 120)
    print("🏆 TOP 20 CONFIGURATIONS (Sorted by Total R)")
    print("=" * 120)
    
    df_results = pd.DataFrame(results[:20])
    print(df_results[['SL', 'Trail', 'Filter', 'Cooldown', 'N', 'WR', 'Total_R', 'Avg_R', 'PF', 'Status']].to_string(index=False, 
          formatters={'WR': '{:.1f}%'.format, 'Total_R': '{:+.0f}'.format, 'Avg_R': '{:+.3f}'.format, 'PF': '{:.2f}'.format}))
    
    # Save all results
    pd.DataFrame(results).to_csv('1h_optimization_results.csv', index=False)
    print("\n✅ Full results saved to 1h_optimization_results.csv")
    
    # Best config analysis
    if results:
        best = results[0]
        print("\n" + "=" * 80)
        print("🏆 OPTIMAL 1H CONFIGURATION")
        print("=" * 80)
        print(f"SL Method: {best['SL']}")
        print(f"Trailing: {best['Trail']}")
        print(f"Signal Filter: {best['Filter']}")
        print(f"Cooldown: {best['Cooldown']} bars")
        print(f"\nPerformance:")
        print(f"  Trades: {best['N']}")
        print(f"  Win Rate: {best['WR']:.1f}%")
        print(f"  Total R: {best['Total_R']:+.0f}")
        print(f"  Avg R/Trade: {best['Avg_R']:+.3f}")
        print(f"  Profit Factor: {best['PF']:.2f}")
        
        # Compare trailing vs static
        print("\n" + "=" * 80)
        print("📊 TRAILING vs STATIC COMPARISON")
        print("=" * 80)
        
        trailing_results = [r for r in results if not r['Trail'].startswith('Static')]
        static_results = [r for r in results if r['Trail'].startswith('Static')]
        
        if trailing_results and static_results:
            best_trail = max(trailing_results, key=lambda x: x['Total_R'])
            best_static = max(static_results, key=lambda x: x['Total_R'])
            
            print(f"\nBest Trailing: {best_trail['Trail']} → {best_trail['Total_R']:+.0f}R ({best_trail['WR']:.1f}% WR)")
            print(f"Best Static:   {best_static['Trail']} → {best_static['Total_R']:+.0f}R ({best_static['WR']:.1f}% WR)")
            
            if best_trail['Total_R'] > best_static['Total_R']:
                print("\n✅ TRAILING outperforms STATIC by +{:.0f}R".format(best_trail['Total_R'] - best_static['Total_R']))
            else:
                print("\n✅ STATIC outperforms TRAILING by +{:.0f}R".format(best_static['Total_R'] - best_trail['Total_R']))


if __name__ == "__main__":
    run()
