#!/usr/bin/env python3
"""
3M TIMEFRAME OPTIMIZATION
=========================
Test 3M timeframe with various configurations to find profitable setup.

Challenge: 3M has smaller moves, so fees have higher impact.
Solution: Test multiple SL sizes and R:R ratios to find edge.

Tests:
- SL sizes: 0.5%, 1%, 1.5%, 2%
- R:R targets: 2:1, 3:1, 4:1, 5:1
- Trailing vs Static
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

TIMEFRAME = '3'          # 3-minute candles
DATA_DAYS = 30           # 30 days (faster test)
NUM_SYMBOLS = 50         # Top 50 symbols

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
COOLDOWN = 10

BASE_URL = "https://api.bybit.com"

# ============================================================================
# CONFIGURATIONS TO TEST
# ============================================================================

# SL sizes to test (as percentage of entry price)
SL_SIZES = [0.5, 1.0, 1.5, 2.0]

# Exit strategies: (name, BE_threshold, trail_distance, max_RR)
EXIT_STRATEGIES = [
    # Trailing strategies
    ("Trail_Tight_2R", 0.3, 0.1, 2.0),
    ("Trail_Tight_3R", 0.3, 0.1, 3.0),
    ("Trail_Tight_4R", 0.3, 0.1, 4.0),
    ("Trail_Tight_5R", 0.3, 0.1, 5.0),
    
    # Static R:R (no trailing)
    ("Static_2R", 99.0, 0.0, 2.0),
    ("Static_3R", 99.0, 0.0, 3.0),
    ("Static_4R", 99.0, 0.0, 4.0),
    ("Static_5R", 99.0, 0.0, 5.0),
]

# ============================================================================
# DATA FUNCTIONS
# ============================================================================

def get_symbols(limit):
    resp = requests.get(f"{BASE_URL}/v5/market/tickers?category=linear", timeout=10)
    tickers = resp.json().get('result', {}).get('list', [])
    usdt = [t for t in tickers if t['symbol'].endswith('USDT')]
    usdt.sort(key=lambda x: float(x.get('turnover24h', 0)), reverse=True)
    return [t['symbol'] for t in usdt[:limit]]


def fetch_klines(symbol, interval, days):
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


def calculate_rsi(close, period=14):
    delta = close.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period).mean()
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


def detect_divergences(df):
    if len(df) < 100: return []
    
    close = df['close'].values
    rsi = df['rsi'].values
    n = len(df)
    price_ph, price_pl = find_pivots(close, 3, 3)
    signals = []
    
    for i in range(30, n - 5):
        confirmed_up_to = i - 3
        
        curr_pl = curr_pli = prev_pl = prev_pli = None
        for j in range(confirmed_up_to, max(confirmed_up_to - LOOKBACK_BARS, 0), -1):
            if not np.isnan(price_pl[j]):
                if curr_pl is None: curr_pl, curr_pli = price_pl[j], j
                elif prev_pl is None and j < curr_pli - MIN_PIVOT_DISTANCE:
                    prev_pl, prev_pli = price_pl[j], j
                    break
        
        curr_ph = curr_phi = prev_ph = prev_phi = None
        for j in range(confirmed_up_to, max(confirmed_up_to - LOOKBACK_BARS, 0), -1):
            if not np.isnan(price_ph[j]):
                if curr_ph is None: curr_ph, curr_phi = price_ph[j], j
                elif prev_ph is None and j < curr_phi - MIN_PIVOT_DISTANCE:
                    prev_ph, prev_phi = price_ph[j], j
                    break
        
        if curr_pl and prev_pl:
            if curr_pl < prev_pl and rsi[curr_pli] > rsi[prev_pli]:
                if rsi[i] < RSI_OVERSOLD + 15:
                    signals.append({'idx': i, 'side': 'long'})
                    continue
        
        if curr_ph and prev_ph:
            if curr_ph > prev_ph and rsi[curr_phi] < rsi[prev_phi]:
                if rsi[i] > RSI_OVERBOUGHT - 15:
                    signals.append({'idx': i, 'side': 'short'})
                    continue
        
        if curr_pl and prev_pl:
            if curr_pl > prev_pl and rsi[curr_pli] < rsi[prev_pli]:
                if rsi[i] < RSI_OVERBOUGHT - 10:
                    signals.append({'idx': i, 'side': 'long'})
                    continue
        
        if curr_ph and prev_ph:
            if curr_ph < prev_ph and rsi[curr_phi] > rsi[prev_phi]:
                if rsi[i] > RSI_OVERSOLD + 10:
                    signals.append({'idx': i, 'side': 'short'})
    
    return signals


def simulate_trade(rows, signal_idx, side, entry_price, sl_pct, exit_strategy):
    _, be_threshold, trail_distance, max_rr = exit_strategy
    
    entry_idx = signal_idx + 1
    if entry_idx >= len(rows) - 1: return None
    
    entry = entry_price
    sl_dist = entry * (sl_pct / 100)
    
    if side == 'long':
        initial_sl = entry - sl_dist
        tp = entry + (max_rr * sl_dist)
    else:
        initial_sl = entry + sl_dist
        tp = entry - (max_rr * sl_dist)
    
    current_sl = initial_sl
    max_favorable_r = 0
    
    for bar_offset in range(1, min(200, len(rows) - entry_idx)):
        bar = rows[entry_idx + bar_offset]
        high, low = float(bar.high), float(bar.low)
        
        if side == 'long':
            if low <= current_sl:
                return (current_sl - entry) / sl_dist
            if high >= tp:
                return max_rr
            
            current_r = (high - entry) / sl_dist
            if current_r > max_favorable_r:
                max_favorable_r = current_r
                if max_favorable_r >= be_threshold and trail_distance > 0:
                    new_sl = entry + (max_favorable_r - trail_distance) * sl_dist
                    if new_sl > current_sl: current_sl = new_sl
        else:
            if high >= current_sl:
                return (entry - current_sl) / sl_dist
            if low <= tp:
                return max_rr
            
            current_r = (entry - low) / sl_dist
            if current_r > max_favorable_r:
                max_favorable_r = current_r
                if max_favorable_r >= be_threshold and trail_distance > 0:
                    new_sl = entry - (max_favorable_r - trail_distance) * sl_dist
                    if new_sl < current_sl: current_sl = new_sl
    
    # Timeout
    last_bar = rows[min(entry_idx + 199, len(rows) - 1)]
    if side == 'long':
        return (float(last_bar.close) - entry) / sl_dist
    else:
        return (entry - float(last_bar.close)) / sl_dist


def run():
    print("=" * 100)
    print("🔬 3M TIMEFRAME OPTIMIZATION")
    print("=" * 100)
    print(f"Timeframe: 3min | Symbols: {NUM_SYMBOLS} | Days: {DATA_DAYS}")
    print(f"Testing: {len(SL_SIZES)} SL sizes × {len(EXIT_STRATEGIES)} exit strategies")
    print(f"Total configurations: {len(SL_SIZES) * len(EXIT_STRATEGIES)}")
    print()
    
    # Fetch symbols
    symbols = get_symbols(NUM_SYMBOLS)
    print(f"📋 Loaded {len(symbols)} symbols\n")
    
    # Preload data
    print("📥 Loading data...")
    symbol_data = {}
    for idx, sym in enumerate(symbols):
        try:
            df = fetch_klines(sym, TIMEFRAME, DATA_DAYS)
            if df.empty or len(df) < 500: continue
            
            df['rsi'] = calculate_rsi(df['close'], RSI_PERIOD)
            df['vol_ma'] = df['volume'].rolling(20).mean()
            df['vol_ok'] = df['volume'] > df['vol_ma'] * 0.5
            df = df.dropna()
            
            if len(df) >= 200:
                signals = detect_divergences(df)
                if signals:
                    symbol_data[sym] = {'df': df, 'signals': signals, 'rows': list(df.itertuples())}
        except:
            continue
        
        if (idx + 1) % 10 == 0:
            print(f"  [{idx+1}/{NUM_SYMBOLS}] {len(symbol_data)} symbols")
    
    print(f"\n✅ {len(symbol_data)} symbols with signals\n")
    
    # Test all configurations
    print("🔄 Testing configurations...")
    results = []
    
    for sl_pct in SL_SIZES:
        for exit_strategy in EXIT_STRATEGIES:
            exit_name = exit_strategy[0]
            
            trades = []
            for sym, data in symbol_data.items():
                rows = data['rows']
                signals = data['signals']
                
                last_trade_idx = -COOLDOWN
                for sig in signals:
                    i = sig['idx']
                    if i - last_trade_idx < COOLDOWN: continue
                    if i >= len(rows) - 50: continue
                    
                    row = rows[i]
                    if not row.vol_ok: continue
                    
                    entry_price = float(rows[i+1].open) if i+1 < len(rows) else row.close
                    trade_r = simulate_trade(rows, i, sig['side'], entry_price, sl_pct, exit_strategy)
                    
                    if trade_r is not None:
                        # Deduct fees (in R terms)
                        fee_r = ROUND_TRIP_FEE / (sl_pct / 100)
                        actual_r = trade_r - fee_r
                        trades.append(actual_r)
                        last_trade_idx = i
            
            if trades:
                n = len(trades)
                wins = sum(1 for t in trades if t > 0)
                wr = wins / n * 100
                total_r = sum(trades)
                avg_r = total_r / n
                
                results.append({
                    'SL_Pct': f"{sl_pct}%",
                    'Exit': exit_name,
                    'N': n,
                    'WR': wr,
                    'Total_R': total_r,
                    'Avg_R': avg_r,
                    'Fee_Impact': ROUND_TRIP_FEE / (sl_pct / 100),
                    'Status': '✅' if total_r > 0 else '❌'
                })
    
    # Sort by Total R
    results.sort(key=lambda x: x['Total_R'], reverse=True)
    
    # Display results
    print("\n" + "=" * 120)
    print("📊 ALL RESULTS (Sorted by Total R)")
    print("=" * 120)
    
    df = pd.DataFrame(results)
    print(df.to_string(index=False, formatters={
        'WR': '{:.1f}%'.format, 
        'Total_R': '{:+.0f}'.format, 
        'Avg_R': '{:+.4f}'.format,
        'Fee_Impact': '{:.3f}R'.format
    }))
    
    # Save
    df.to_csv('3m_optimization_results.csv', index=False)
    print("\n✅ Saved to 3m_optimization_results.csv")
    
    # Best result
    if results:
        best = results[0]
        print("\n" + "=" * 80)
        print("🏆 BEST 3M CONFIGURATION")
        print("=" * 80)
        print(f"SL: {best['SL_Pct']}")
        print(f"Exit: {best['Exit']}")
        print(f"Trades: {best['N']}")
        print(f"Win Rate: {best['WR']:.1f}%")
        print(f"Total R: {best['Total_R']:+.0f}")
        print(f"Avg R/Trade: {best['Avg_R']:+.4f}")
        print(f"Fee Impact: {best['Fee_Impact']:.3f}R per trade")
        
        if best['Total_R'] > 0:
            print("\n✅ PROFITABLE CONFIGURATION FOUND!")
        else:
            print("\n⚠️ NO PROFITABLE CONFIGURATION - 3M may not be viable after fees")


if __name__ == "__main__":
    run()
