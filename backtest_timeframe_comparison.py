#!/usr/bin/env python3
"""
TIMEFRAME COMPARISON BACKTEST
=============================
Compare 3M, 15M, and 1H timeframes to find optimal balance between:
- Signal frequency (more trades)
- Profitability (after fees)
- Win Rate stability

Uses same logic as backtest_modern_nolookahead.py but tests multiple TFs.
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime
import time
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

TIMEFRAMES = ['3', '15', '60']  # 3min, 15min, 1hour
DATA_DAYS = 60                   # 60 days of data
NUM_SYMBOLS = 100                # Top 100 symbols (faster test)

# Fees (Bybit standard)
MAKER_FEE = 0.00055
TAKER_FEE = 0.00055
ROUND_TRIP_FEE = MAKER_FEE + TAKER_FEE

# SL Settings - ADJUST PER TIMEFRAME
MIN_SL_PCT = {
    '3': 1.0,    # 1% min for 3M (smaller moves)
    '15': 1.5,   # 1.5% min for 15M
    '60': 2.0    # 2% min for 1H (larger moves)
}

# RSI Settings
RSI_PERIOD = 14
RSI_OVERSOLD = 30
RSI_OVERBOUGHT = 70
LOOKBACK_BARS = 14
MIN_PIVOT_DISTANCE = 5
COOLDOWN_BARS = 10

# Trail Config (Tight-Trail)
BE_THRESHOLD = 0.3
TRAIL_DISTANCE = 0.1
MAX_TP = 2.0

BASE_URL = "https://api.bybit.com"

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
        
        curr_pl = curr_pli = prev_pl = prev_pli = None
        for j in range(confirmed_up_to, max(confirmed_up_to - LOOKBACK_BARS, 0), -1):
            if not np.isnan(price_pl[j]):
                if curr_pl is None: 
                    curr_pl, curr_pli = price_pl[j], j
                elif prev_pl is None and j < curr_pli - MIN_PIVOT_DISTANCE:
                    prev_pl, prev_pli = price_pl[j], j
                    break
        
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
# TRADE SIMULATION (Tight-Trail)
# ============================================================================

def simulate_trade(rows, signal_idx, side, swing_price, atr, min_sl_pct):
    entry_idx = signal_idx + 1
    if entry_idx >= len(rows) - 1:
        return None
    
    entry_bar = rows[entry_idx]
    entry = float(entry_bar.open)
    
    # Calculate SL distance with minimum
    sl_dist = abs(swing_price - entry)
    min_sl_dist = entry * (min_sl_pct / 100)
    sl_dist = max(sl_dist, min_sl_dist, atr)
    
    if side == 'long':
        initial_sl = entry - sl_dist
        tp = entry + (MAX_TP * sl_dist)
    else:
        initial_sl = entry + sl_dist
        tp = entry - (MAX_TP * sl_dist)
    
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
                return {'exit_r': MAX_TP, 'result': 'tp', 'entry': entry, 'sl_dist': sl_dist}
            
            current_r = (high - entry) / sl_dist
            if current_r > max_favorable_r:
                max_favorable_r = current_r
                if max_favorable_r >= BE_THRESHOLD:
                    new_sl = entry + (max_favorable_r - TRAIL_DISTANCE) * sl_dist
                    if new_sl > current_sl:
                        current_sl = new_sl
        else:
            if high >= current_sl:
                exit_r = (entry - current_sl) / sl_dist
                return {'exit_r': exit_r, 'result': 'sl', 'entry': entry, 'sl_dist': sl_dist}
            
            if low <= tp:
                return {'exit_r': MAX_TP, 'result': 'tp', 'entry': entry, 'sl_dist': sl_dist}
            
            current_r = (entry - low) / sl_dist
            if current_r > max_favorable_r:
                max_favorable_r = current_r
                if max_favorable_r >= BE_THRESHOLD:
                    new_sl = entry - (max_favorable_r - TRAIL_DISTANCE) * sl_dist
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
# MAIN
# ============================================================================

def run():
    print("=" * 80)
    print("📊 TIMEFRAME COMPARISON BACKTEST")
    print("=" * 80)
    print(f"Testing: {TIMEFRAMES} (3min, 15min, 1hour)")
    print(f"Symbols: {NUM_SYMBOLS} | Days: {DATA_DAYS}")
    print(f"Strategy: Tight-Trail (BE: {BE_THRESHOLD}R, Trail: {TRAIL_DISTANCE}R, Max: {MAX_TP}R)")
    print()
    
    symbols = get_symbols(NUM_SYMBOLS)
    print(f"📋 Loaded {len(symbols)} symbols\n")
    
    results = {tf: [] for tf in TIMEFRAMES}
    
    for tf in TIMEFRAMES:
        print(f"\n{'='*60}")
        print(f"⏱️  Testing {tf}min timeframe...")
        print(f"{'='*60}")
        
        min_sl = MIN_SL_PCT[tf]
        
        for idx, sym in enumerate(symbols):
            try:
                df = fetch_klines(sym, tf, DATA_DAYS)
                if df.empty or len(df) < 200: continue
                
                df['rsi'] = calculate_rsi(df['close'], RSI_PERIOD)
                df['atr'] = calculate_atr(df, 14)
                df['vol_ma'] = df['volume'].rolling(20).mean()
                df['vol_ok'] = df['volume'] > df['vol_ma'] * 0.5
                df = df.dropna()
                
                if len(df) < 100: continue
                
                signals = detect_divergences(df)
                rows = list(df.itertuples())
                
                last_trade_idx = -COOLDOWN_BARS
                
                for sig in signals:
                    i = sig['idx']
                    if i - last_trade_idx < COOLDOWN_BARS: continue
                    if i >= len(rows) - 50: continue
                    
                    row = rows[i]
                    if not row.vol_ok or row.atr <= 0: continue
                    
                    trade = simulate_trade(rows, i, sig['side'], sig['swing'], row.atr, min_sl)
                    
                    if trade is None: continue
                    
                    fee_r = calculate_fee_r(trade['entry'], trade['sl_dist'])
                    actual_r = trade['exit_r'] - fee_r
                    
                    results[tf].append({
                        'symbol': sym,
                        'side': sig['side'],
                        'type': sig['type'],
                        'theo_r': trade['exit_r'],
                        'fee_r': fee_r,
                        'actual_r': actual_r,
                        'result': trade['result']
                    })
                    
                    last_trade_idx = i
                    
            except Exception as e:
                continue
            
            if (idx + 1) % 25 == 0:
                print(f"  [{idx+1}/{NUM_SYMBOLS}] {tf}min trades so far: {len(results[tf])}")
    
    # ============================================================================
    # ANALYSIS
    # ============================================================================
    
    print("\n" + "=" * 100)
    print("📊 RESULTS COMPARISON")
    print("=" * 100)
    
    summary = []
    for tf in TIMEFRAMES:
        trades = results[tf]
        if not trades: continue
        
        n = len(trades)
        wins = sum(1 for t in trades if t['actual_r'] > 0)
        wr = wins / n * 100
        
        total_r = sum(t['actual_r'] for t in trades)
        total_fees = sum(t['fee_r'] for t in trades)
        avg_r = total_r / n
        avg_fee = total_fees / n
        
        gross_profit = sum(t['actual_r'] for t in trades if t['actual_r'] > 0)
        gross_loss = abs(sum(t['actual_r'] for t in trades if t['actual_r'] < 0))
        pf = gross_profit / gross_loss if gross_loss > 0 else 999
        
        trades_per_day = n / DATA_DAYS
        
        summary.append({
            'Timeframe': f"{tf}min",
            'Trades': n,
            'Trades/Day': f"{trades_per_day:.1f}",
            'WinRate': f"{wr:.1f}%",
            'Total_R': f"{total_r:+.0f}",
            'Avg_R': f"{avg_r:+.3f}",
            'Avg_Fee_R': f"{avg_fee:.3f}",
            'PF': f"{pf:.2f}",
            'Status': '✅' if total_r > 0 else '❌'
        })
    
    df_summary = pd.DataFrame(summary)
    print(df_summary.to_string(index=False))
    
    # Save
    df_summary.to_csv('timeframe_comparison.csv', index=False)
    print("\n✅ Saved to timeframe_comparison.csv")
    
    # RECOMMENDATION
    print("\n" + "=" * 80)
    print("💡 RECOMMENDATION")
    print("=" * 80)
    
    if summary:
        best = max(summary, key=lambda x: float(x['Total_R'].replace('+', '')))
        print(f"\n🏆 BEST TIMEFRAME: {best['Timeframe']}")
        print(f"   Total R: {best['Total_R']}")
        print(f"   Win Rate: {best['WinRate']}")
        print(f"   Trades/Day: {best['Trades/Day']}")
        print(f"   Profit Factor: {best['PF']}")


if __name__ == "__main__":
    run()
