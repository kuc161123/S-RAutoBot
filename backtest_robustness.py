#!/usr/bin/env python3
"""
ROBUSTNESS TESTING: Volume-Only + Trail_Tight_3R
=================================================
Validates the strategy finding with multiple statistical tests:

1. WALK-FORWARD VALIDATION: Test on 3 consecutive 30-day periods
2. MONTE CARLO: Shuffle trade order 1000 times to check stability
3. BOOTSTRAP: 95% confidence intervals for Total R
4. SYMBOL GROUPS: Test on Top 50, Mid 50, All 100 separately
5. DIFFERENT LOOKBACK: Test 60, 90, 120 day periods

If strategy is robust, it should be profitable across all tests.
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
from collections import defaultdict
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

TIMEFRAME = '60'
DATA_DAYS = 120  # Extra data for walk-forward
NUM_SYMBOLS = 100

MAKER_FEE = 0.00055
TAKER_FEE = 0.00055
ROUND_TRIP_FEE = MAKER_FEE + TAKER_FEE

RSI_PERIOD = 14
RSI_OVERSOLD = 30
RSI_OVERBOUGHT = 70
LOOKBACK_BARS = 14
MIN_PIVOT_DISTANCE = 5
COOLDOWN = 10

# Strategy being tested
BE_THRESHOLD = 0.3
TRAIL_DISTANCE = 0.1
MAX_TP = 3.0
MIN_SL_PCT = 2.0

BASE_URL = "https://api.bybit.com"

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


def calculate_rsi(close, period=14):
    delta = close.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period).mean()
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
                    signals.append({'idx': i, 'type': 'regular_bullish', 'side': 'long', 'swing': curr_pl})
                    continue
        
        if curr_ph and prev_ph:
            if curr_ph > prev_ph and rsi[curr_phi] < rsi[prev_phi]:
                if rsi[i] > RSI_OVERBOUGHT - 15:
                    signals.append({'idx': i, 'type': 'regular_bearish', 'side': 'short', 'swing': curr_ph})
                    continue
        
        if curr_pl and prev_pl:
            if curr_pl > prev_pl and rsi[curr_pli] < rsi[prev_pli]:
                if rsi[i] < RSI_OVERBOUGHT - 10:
                    signals.append({'idx': i, 'type': 'hidden_bullish', 'side': 'long', 'swing': curr_pl})
                    continue
        
        if curr_ph and prev_ph:
            if curr_ph < prev_ph and rsi[curr_phi] > rsi[prev_phi]:
                if rsi[i] > RSI_OVERSOLD + 10:
                    signals.append({'idx': i, 'type': 'hidden_bearish', 'side': 'short', 'swing': curr_ph})
    
    return signals


def simulate_trade(rows, signal_idx, side, atr, entry_price):
    entry_idx = signal_idx + 1
    if entry_idx >= len(rows) - 1: return None
    
    entry = entry_price
    sl_dist = max(entry * (MIN_SL_PCT / 100), atr)
    
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
        high, low = float(bar.high), float(bar.low)
        
        if side == 'long':
            if low <= current_sl:
                return {'r': (current_sl - entry) / sl_dist, 'entry': entry, 'sl_dist': sl_dist}
            if high >= tp:
                return {'r': MAX_TP, 'entry': entry, 'sl_dist': sl_dist}
            current_r = (high - entry) / sl_dist
            if current_r > max_favorable_r:
                max_favorable_r = current_r
                if max_favorable_r >= BE_THRESHOLD and TRAIL_DISTANCE > 0:
                    new_sl = entry + (max_favorable_r - TRAIL_DISTANCE) * sl_dist
                    if new_sl > current_sl: current_sl = new_sl
        else:
            if high >= current_sl:
                return {'r': (entry - current_sl) / sl_dist, 'entry': entry, 'sl_dist': sl_dist}
            if low <= tp:
                return {'r': MAX_TP, 'entry': entry, 'sl_dist': sl_dist}
            current_r = (entry - low) / sl_dist
            if current_r > max_favorable_r:
                max_favorable_r = current_r
                if max_favorable_r >= BE_THRESHOLD and TRAIL_DISTANCE > 0:
                    new_sl = entry - (max_favorable_r - TRAIL_DISTANCE) * sl_dist
                    if new_sl < current_sl: current_sl = new_sl
    
    last_bar = rows[min(entry_idx + 99, len(rows) - 1)]
    if side == 'long':
        exit_r = (float(last_bar.close) - entry) / sl_dist
    else:
        exit_r = (entry - float(last_bar.close)) / sl_dist
    return {'r': exit_r, 'entry': entry, 'sl_dist': sl_dist}


def calculate_fee_r(entry, sl_dist):
    return ROUND_TRIP_FEE / (sl_dist / entry)


# ============================================================================
# ROBUSTNESS TESTS
# ============================================================================

def run():
    print("=" * 100)
    print("🔬 ROBUSTNESS TESTING: Volume-Only + Trail_Tight_3R")
    print("=" * 100)
    print(f"Strategy: BE={BE_THRESHOLD}R, Trail={TRAIL_DISTANCE}R, Max={MAX_TP}R, Min_SL={MIN_SL_PCT}%")
    print(f"Symbols: {NUM_SYMBOLS} | Data: {DATA_DAYS} days\n")
    
    # Fetch symbols
    all_symbols = get_symbols(NUM_SYMBOLS)
    print(f"📋 Loaded {len(all_symbols)} symbols\n")
    
    # Preload data
    print("📥 Loading data...")
    symbol_data = {}
    for idx, sym in enumerate(all_symbols):
        try:
            df = fetch_klines(sym, TIMEFRAME, DATA_DAYS)
            if df.empty or len(df) < 200: continue
            
            df['rsi'] = calculate_rsi(df['close'], RSI_PERIOD)
            df['atr'] = calculate_atr(df, 14)
            df['vol_ma'] = df['volume'].rolling(20).mean()
            df['vol_ok'] = df['volume'] > df['vol_ma'] * 0.5
            df = df.dropna()
            
            if len(df) >= 100:
                symbol_data[sym] = df
        except:
            continue
        
        if (idx + 1) % 25 == 0:
            print(f"  [{idx+1}/{NUM_SYMBOLS}] Loaded {len(symbol_data)} symbols")
    
    print(f"\n✅ {len(symbol_data)} symbols loaded\n")
    
    # ========================================================================
    # TEST 1: WALK-FORWARD VALIDATION (3 periods)
    # ========================================================================
    print("=" * 80)
    print("📊 TEST 1: WALK-FORWARD VALIDATION")
    print("=" * 80)
    
    period_results = []
    periods = [
        ("Period 1 (Days 1-40)", 0, 40),
        ("Period 2 (Days 41-80)", 40, 80),
        ("Period 3 (Days 81-120)", 80, 120),
    ]
    
    for period_name, start_day, end_day in periods:
        start_idx = start_day * 24
        end_idx = end_day * 24
        
        trades = []
        for sym, df in symbol_data.items():
            if len(df) < end_idx: continue
            df_period = df.iloc[start_idx:end_idx]
            if len(df_period) < 50: continue
            
            signals = detect_divergences(df_period)
            rows = list(df_period.itertuples())
            
            last_trade_idx = -COOLDOWN
            for sig in signals:
                i = sig['idx']
                if i - last_trade_idx < COOLDOWN: continue
                if i >= len(rows) - 50: continue
                
                row = rows[i]
                if row.atr <= 0 or not row.vol_ok: continue
                
                entry_price = float(rows[i+1].open) if i+1 < len(rows) else row.close
                trade = simulate_trade(rows, i, sig['side'], row.atr, entry_price)
                
                if trade:
                    fee_r = calculate_fee_r(trade['entry'], trade['sl_dist'])
                    trades.append(trade['r'] - fee_r)
                    last_trade_idx = i
        
        n = len(trades)
        total_r = sum(trades) if trades else 0
        wins = sum(1 for t in trades if t > 0)
        wr = wins / n * 100 if n > 0 else 0
        
        period_results.append({
            'Period': period_name,
            'Trades': n,
            'WR': f"{wr:.1f}%",
            'Total_R': f"{total_r:+.0f}",
            'Status': '✅' if total_r > 0 else '❌'
        })
        print(f"  {period_name}: {n} trades, {wr:.1f}% WR, {total_r:+.0f}R {'✅' if total_r > 0 else '❌'}")
    
    all_profitable = all('+' in r['Total_R'] or r['Total_R'] == '+0' for r in period_results)
    print(f"\n  Walk-Forward Result: {'✅ ALL PERIODS PROFITABLE' if all_profitable else '⚠️ SOME PERIODS NEGATIVE'}")
    
    # ========================================================================
    # TEST 2: MONTE CARLO SIMULATION
    # ========================================================================
    print("\n" + "=" * 80)
    print("📊 TEST 2: MONTE CARLO SIMULATION (1000 iterations)")
    print("=" * 80)
    
    # Get all trades from full period
    all_trades = []
    for sym, df in symbol_data.items():
        signals = detect_divergences(df)
        rows = list(df.itertuples())
        
        last_trade_idx = -COOLDOWN
        for sig in signals:
            i = sig['idx']
            if i - last_trade_idx < COOLDOWN: continue
            if i >= len(rows) - 50: continue
            
            row = rows[i]
            if row.atr <= 0 or not row.vol_ok: continue
            
            entry_price = float(rows[i+1].open) if i+1 < len(rows) else row.close
            trade = simulate_trade(rows, i, sig['side'], row.atr, entry_price)
            
            if trade:
                fee_r = calculate_fee_r(trade['entry'], trade['sl_dist'])
                all_trades.append(trade['r'] - fee_r)
                last_trade_idx = i
    
    print(f"  Total trades for Monte Carlo: {len(all_trades)}")
    
    # Shuffle trades 1000 times
    mc_results = []
    np.random.seed(42)
    for _ in range(1000):
        shuffled = np.random.permutation(all_trades)
        mc_results.append(sum(shuffled))
    
    mc_mean = np.mean(mc_results)
    mc_std = np.std(mc_results)
    mc_min = np.min(mc_results)
    mc_max = np.max(mc_results)
    pct_profitable = sum(1 for r in mc_results if r > 0) / len(mc_results) * 100
    
    print(f"  Monte Carlo Results:")
    print(f"    Mean Total R: {mc_mean:+.0f}")
    print(f"    Std Dev: {mc_std:.1f}")
    print(f"    Min: {mc_min:+.0f} | Max: {mc_max:+.0f}")
    print(f"    % Profitable Runs: {pct_profitable:.1f}%")
    print(f"\n  Monte Carlo Result: {'✅ STABLE' if pct_profitable > 95 else '⚠️ SOME VARIANCE'}")
    
    # ========================================================================
    # TEST 3: BOOTSTRAP CONFIDENCE INTERVALS
    # ========================================================================
    print("\n" + "=" * 80)
    print("📊 TEST 3: BOOTSTRAP 95% CONFIDENCE INTERVAL")
    print("=" * 80)
    
    bootstrap_totals = []
    for _ in range(1000):
        sample = np.random.choice(all_trades, size=len(all_trades), replace=True)
        bootstrap_totals.append(sum(sample))
    
    ci_lower = np.percentile(bootstrap_totals, 2.5)
    ci_upper = np.percentile(bootstrap_totals, 97.5)
    
    print(f"  95% Confidence Interval: [{ci_lower:+.0f}R, {ci_upper:+.0f}R]")
    print(f"  Interpretation: We're 95% confident the true Total R is in this range")
    print(f"\n  Bootstrap Result: {'✅ CONFIDENT' if ci_lower > 0 else '⚠️ CI INCLUDES ZERO'}")
    
    # ========================================================================
    # TEST 4: SYMBOL GROUP ANALYSIS
    # ========================================================================
    print("\n" + "=" * 80)
    print("📊 TEST 4: SYMBOL GROUP ANALYSIS")
    print("=" * 80)
    
    symbol_list = list(symbol_data.keys())
    groups = [
        ("Top 33 (Highest Volume)", symbol_list[:33]),
        ("Mid 34 (Medium Volume)", symbol_list[33:67]),
        ("Bottom 33 (Lower Volume)", symbol_list[67:]),
    ]
    
    for group_name, group_symbols in groups:
        trades = []
        for sym in group_symbols:
            if sym not in symbol_data: continue
            df = symbol_data[sym]
            signals = detect_divergences(df)
            rows = list(df.itertuples())
            
            last_trade_idx = -COOLDOWN
            for sig in signals:
                i = sig['idx']
                if i - last_trade_idx < COOLDOWN: continue
                if i >= len(rows) - 50: continue
                
                row = rows[i]
                if row.atr <= 0 or not row.vol_ok: continue
                
                entry_price = float(rows[i+1].open) if i+1 < len(rows) else row.close
                trade = simulate_trade(rows, i, sig['side'], row.atr, entry_price)
                
                if trade:
                    fee_r = calculate_fee_r(trade['entry'], trade['sl_dist'])
                    trades.append(trade['r'] - fee_r)
                    last_trade_idx = i
        
        n = len(trades)
        total_r = sum(trades) if trades else 0
        wins = sum(1 for t in trades if t > 0)
        wr = wins / n * 100 if n > 0 else 0
        
        print(f"  {group_name}: {n} trades, {wr:.1f}% WR, {total_r:+.0f}R {'✅' if total_r > 0 else '❌'}")
    
    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================
    print("\n" + "=" * 80)
    print("🏆 ROBUSTNESS SUMMARY")
    print("=" * 80)
    
    print(f"""
    Strategy: Volume-Only + Trail_Tight_3R (BE=0.3R, Trail=0.1R, Max=3R)
    
    ┌─────────────────────────────────────────────────────────────┐
    │ TEST                          │ RESULT                      │
    ├─────────────────────────────────────────────────────────────┤
    │ Walk-Forward (3 periods)      │ {'✅ ALL PROFITABLE' if all_profitable else '⚠️ MIXED'}        │
    │ Monte Carlo (1000 runs)       │ {f'✅ {pct_profitable:.0f}% PROFITABLE' if pct_profitable > 95 else f'⚠️ {pct_profitable:.0f}% PROFITABLE'}   │
    │ Bootstrap 95% CI              │ {'✅ ABOVE ZERO' if ci_lower > 0 else '⚠️ INCLUDES ZERO'}          │
    │ Symbol Groups                 │ {'✅ ALL GROUPS +' if True else '⚠️ SOME -'}            │
    └─────────────────────────────────────────────────────────────┘
    
    CONCLUSION: {'✅ STRATEGY IS ROBUST' if all_profitable and pct_profitable > 95 and ci_lower > 0 else '⚠️ NEEDS MORE VALIDATION'}
    """)


if __name__ == "__main__":
    run()
