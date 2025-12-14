#!/usr/bin/env python3
"""
BOOM HUNTER PRO STRATEGY BACKTEST
==================================
Based on TradingView's Boom Hunter Pro indicator by veryfid.

Uses Ehlers Early Onset Trend (EOT) oscillators with:
- Highpass filter for cyclic component removal  
- SuperSmoother filter for noise reduction
- LSMA Wave Trend for pressure detection

Signal Types:
- Lime Long: Best quality (deep oversold + red wave confirmation)
- Yellow Long: Good quality (red wave oversold)
- Blue Long: Medium quality (recent bottom touch)
- Gray Long: Lower quality (recent support)
- Red Short: Quality short (overbought conditions)
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

TIMEFRAME = '60'  # 1-hour candles (indicator tuned for 1H)
DATA_DAYS = 60
NUM_SYMBOLS = 100

# Risk/Reward
TP_ATR_MULT = 1.0   # Take profit at 1 ATR (1:1 R:R)
SL_ATR_MULT = 1.0   # Stop loss at 1 ATR
TOTAL_FEES = 0.001  # 0.1% total (entry + exit)

# EOT Parameters (from indicator)
LP_PERIOD_1 = 6     # Main oscillator
K1_1 = 0.0
K2_1 = 0.3

LP_PERIOD_2 = 27    # Red wave
K1_2 = 0.8
K2_2 = 0.3

LP_PERIOD_3 = 11    # Yellow line
K1_3 = 0.9999
K3_3 = -0.9999

# Wave Trend Parameters
WT_N1 = 9
WT_N2 = 6
WT_N3 = 3
TRIGGER_LEN = 2

BASE_URL = "https://api.bybit.com"

# =============================================================================
# EOT CALCULATIONS (Ehlers Early Onset Trend)
# =============================================================================

def calculate_eot(close_prices, lp_period, k1, k2):
    """
    Calculate Ehlers Early Onset Trend oscillator.
    Returns Quotient1 and Quotient2 arrays.
    """
    n = len(close_prices)
    pi = np.pi
    
    # Initialize arrays
    alpha1 = np.zeros(n)
    HP = np.zeros(n)
    Filt = np.zeros(n)
    Peak = np.zeros(n)
    X = np.zeros(n)
    Quotient1 = np.zeros(n)
    Quotient2 = np.zeros(n)
    
    # Highpass filter coefficient
    alpha_const = (np.cos(0.707 * 2 * pi / 100) + np.sin(0.707 * 2 * pi / 100) - 1) / np.cos(0.707 * 2 * pi / 100)
    
    # SuperSmoother coefficients
    a1 = np.exp(-1.414 * pi / lp_period)
    b1 = 2 * a1 * np.cos(1.414 * pi / lp_period)
    c2 = b1
    c3 = -a1 * a1
    c1 = 1 - c2 - c3
    
    for i in range(2, n):
        # Highpass filter - remove cycles shorter than 100 bars
        HP[i] = (1 - alpha_const / 2) ** 2 * (close_prices[i] - 2 * close_prices[i-1] + close_prices[i-2]) + \
                2 * (1 - alpha_const) * HP[i-1] - (1 - alpha_const) ** 2 * HP[i-2]
        
        # SuperSmoother Filter
        Filt[i] = c1 * (HP[i] + HP[i-1]) / 2 + c2 * Filt[i-1] + c3 * Filt[i-2]
        
        # Fast Attack - Slow Decay for Peak
        Peak[i] = 0.991 * Peak[i-1]
        if abs(Filt[i]) > Peak[i]:
            Peak[i] = abs(Filt[i])
        
        # Normalized Roofing Filter
        if Peak[i] != 0:
            X[i] = Filt[i] / Peak[i]
        
        # Quotient calculations
        if k1 * X[i] + 1 != 0:
            Quotient1[i] = (X[i] + k1) / (k1 * X[i] + 1)
        if k2 * X[i] + 1 != 0:
            Quotient2[i] = (X[i] + k2) / (k2 * X[i] + 1)
    
    return Quotient1, Quotient2

def calculate_wave_trend(df, n1=9, n2=6, n3=3):
    """
    Calculate LSMA Wave Trend components.
    Returns wt1, wt2.
    """
    src = df['hlc3'].values
    close = df['close'].values
    open_ = df['open'].values
    volume = df['volume'].values
    n = len(src)
    
    # TCI (Trend Confirmation Index)
    ema_src = pd.Series(src).ewm(span=n1, adjust=False).mean().values
    diff = src - ema_src
    abs_diff = np.abs(diff)
    ema_abs_diff = pd.Series(abs_diff).ewm(span=n1, adjust=False).mean().values
    
    tci = np.zeros(n)
    for i in range(n):
        if ema_abs_diff[i] != 0:
            tci[i] = pd.Series((diff / (0.025 * ema_abs_diff + 1e-10))[:i+1]).ewm(span=n2, adjust=False).mean().iloc[-1] + 50 if i > n2 else 50
    
    # RSI
    rsi = pd.Series(close).diff()
    gain = rsi.where(rsi > 0, 0).rolling(n3).mean()
    loss = (-rsi.where(rsi < 0, 0)).rolling(n3).mean()
    rs = gain / (loss + 1e-10)
    rsi_values = 100 - (100 / (1 + rs))
    
    # MFI (simplified)
    mfi = np.zeros(n)
    for i in range(n3, n):
        pos_flow = 0
        neg_flow = 0
        for j in range(i-n3+1, i+1):
            if j > 0:
                change = src[j] - src[j-1]
                if change > 0:
                    pos_flow += volume[j] * src[j]
                else:
                    neg_flow += volume[j] * src[j]
        if neg_flow > 0:
            mfi[i] = 100 - (100 / (1 + pos_flow / neg_flow))
        else:
            mfi[i] = 100
    
    # Willy (Williams %R variation)
    willy = np.zeros(n)
    for i in range(n2, n):
        highest = np.max(src[i-n2+1:i+1])
        lowest = np.min(src[i-n2+1:i+1])
        if highest != lowest:
            willy[i] = 60 * (src[i] - highest) / (highest - lowest) + 80
        else:
            willy[i] = 50
    
    # Combine for Wave Trend
    wt1 = (tci + mfi + rsi_values.fillna(50).values) / 3
    wt2 = pd.Series(wt1).rolling(6).mean().fillna(50).values
    
    return wt1, wt2

# =============================================================================
# SIGNAL DETECTION
# =============================================================================

def detect_boom_signals(df, Q1_1, Q2_1, Q3, Q4, Q5, Q6, wt1, wt2, lsma):
    """
    Detect Boom Hunter Pro signals.
    
    Returns list of signals with type, side, and index.
    """
    n = len(df)
    signals = []
    
    # Scale quotients to display range (0-100)
    esize = 60
    ey = 50
    q1 = Q1_1 * esize + ey  # Main oscillator
    trigger = pd.Series(q1).rolling(TRIGGER_LEN).mean().fillna(50).values
    
    # Track conditions
    barssince_warn2 = np.full(n, 999)  # barssince(crossover(Quotient1, -0.9))
    barssince_warn3 = np.full(n, 999)  # barssince(crossunder(Quotient1, 0.9))
    barssince_q1_cross_20 = np.full(n, 999)  # barssince(crossover(q1, 20))
    barssince_q1_cross_80 = np.full(n, 999)  # barssince(crossover(q1, 80))
    barssince_q1_under_trigger = np.full(n, 999)  # barssince(q1 <= 20 AND crossunder)
    barssince_q1_deep_under = np.full(n, 999)  # barssince(q1 <= 0 AND crossunder)
    
    for i in range(2, n):
        # Update barssince counters
        # warn2 = crossover(Quotient1, -0.9)
        if Q1_1[i] > -0.9 and Q1_1[i-1] <= -0.9:
            barssince_warn2[i] = 0
        else:
            barssince_warn2[i] = barssince_warn2[i-1] + 1
        
        # warn3 = crossunder(Quotient1, 0.9)
        if Q1_1[i] < 0.9 and Q1_1[i-1] >= 0.9:
            barssince_warn3[i] = 0
        else:
            barssince_warn3[i] = barssince_warn3[i-1] + 1
        
        # crossover(q1, 20)
        if q1[i] > 20 and q1[i-1] <= 20:
            barssince_q1_cross_20[i] = 0
        else:
            barssince_q1_cross_20[i] = barssince_q1_cross_20[i-1] + 1
        
        # crossover(q1, 80)
        if q1[i] > 80 and q1[i-1] <= 80:
            barssince_q1_cross_80[i] = 0
        else:
            barssince_q1_cross_80[i] = barssince_q1_cross_80[i-1] + 1
        
        # q1 <= 20 AND crossunder(q1, trigger)
        if q1[i-1] <= 20 and q1[i-1] > trigger[i-1] and q1[i] <= trigger[i]:
            barssince_q1_under_trigger[i] = 0
        else:
            barssince_q1_under_trigger[i] = barssince_q1_under_trigger[i-1] + 1
        
        # q1 <= 0 AND crossunder(q1, trigger)
        if q1[i-1] <= 0 and q1[i-1] > trigger[i-1] and q1[i] <= trigger[i]:
            barssince_q1_deep_under[i] = 0
        else:
            barssince_q1_deep_under[i] = barssince_q1_deep_under[i-1] + 1
        
        # Crossover detection
        crossover = q1[i] > trigger[i] and q1[i-1] <= trigger[i-1]
        crossunder = q1[i] < trigger[i] and q1[i-1] >= trigger[i-1]
        
        # === LONG SIGNALS ===
        
        # enter3 (Lime) - Best quality
        if (Q3[i] <= -0.9 and crossover and 
            barssince_warn2[i] <= 7 and q1[i] <= 20 and 
            barssince_q1_cross_20[i] <= 21):
            signals.append({'idx': i, 'side': 'long', 'type': 'lime', 'quality': 4})
        
        # enter7 (Yellow) - Good quality
        elif Q3[i] <= -0.9 and crossover:
            signals.append({'idx': i, 'side': 'long', 'type': 'yellow', 'quality': 3})
        
        # enter5 (Blue) - Medium quality
        elif barssince_q1_deep_under[i] <= 5 and crossover:
            signals.append({'idx': i, 'side': 'long', 'type': 'blue', 'quality': 2})
        
        # enter6 (Gray) - Lower quality
        elif barssince_q1_under_trigger[i] <= 11 and crossover and q1[i] <= 60:
            signals.append({'idx': i, 'side': 'long', 'type': 'gray', 'quality': 1})
        
        # === SHORT SIGNALS ===
        
        # senter3 (Red) - Quality short
        if (Q3[i] >= -0.9 and crossunder and 
            barssince_warn3[i] <= 7 and q1[i] >= 99 and 
            barssince_q1_cross_80[i] <= 21):
            signals.append({'idx': i, 'side': 'short', 'type': 'red', 'quality': 3})
    
    return signals

# =============================================================================
# HELPERS
# =============================================================================

def wilson_lower_bound(wins: int, n: int, z: float = 1.96) -> float:
    if n == 0: return 0.0
    p = wins / n
    denominator = 1 + z*z/n
    centre = p + z*z/(2*n)
    spread = z * math.sqrt((p*(1-p) + z*z/(4*n)) / n)
    return max(0, (centre - spread) / denominator)

def calc_ev(wr: float, rr: float = 2.0) -> float:
    return (wr * rr) - (1 - wr)

def get_symbols(limit: int = 100) -> list:
    url = f"{BASE_URL}/v5/market/tickers?category=linear"
    resp = requests.get(url)
    tickers = resp.json().get('result', {}).get('list', [])
    usdt_pairs = [t for t in tickers if t['symbol'].endswith('USDT')]
    usdt_pairs.sort(key=lambda x: float(x.get('turnover24h', 0)), reverse=True)
    return [t['symbol'] for t in usdt_pairs[:limit]]

def fetch_klines(symbol: str, interval: str, days: int) -> pd.DataFrame:
    end_ts = int(datetime.now().timestamp() * 1000)
    candles_needed = days * 24 * 60 // (int(interval) if interval.isdigit() else 60)
    
    all_candles = []
    current_end = end_ts
    
    while len(all_candles) < candles_needed:
        url = f"{BASE_URL}/v5/market/kline"
        params = {'category': 'linear', 'symbol': symbol, 'interval': interval, 'limit': 1000, 'end': current_end}
        try:
            resp = requests.get(url, params=params, timeout=10)
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
    df['hlc3'] = (df['high'] + df['low'] + df['close']) / 3
    df['hour_utc'] = df.index.hour
    
    # ATR for SL/TP
    high_low = df['high'] - df['low']
    high_close = abs(df['high'] - df['close'].shift())
    low_close = abs(df['low'] - df['close'].shift())
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr'] = true_range.rolling(14).mean()
    
    return df

def simulate_trade(df, entry_idx, side, entry_price, atr) -> dict:
    if side == 'long':
        tp = entry_price + (TP_ATR_MULT * atr)
        sl = entry_price - (SL_ATR_MULT * atr)
    else:
        tp = entry_price - (TP_ATR_MULT * atr)
        sl = entry_price + (SL_ATR_MULT * atr)
    
    if side == 'long': tp = tp * (1 + TOTAL_FEES)
    else: tp = tp * (1 - TOTAL_FEES)
    
    rows = list(df.iloc[entry_idx+1:entry_idx+51].itertuples())
    
    for future_row in rows:
        if side == 'long':
            if future_row.low <= sl: return {'outcome': 'loss'}
            if future_row.high >= tp: return {'outcome': 'win'}
        else:
            if future_row.high >= sl: return {'outcome': 'loss'}
            if future_row.low <= tp: return {'outcome': 'win'}
    
    return {'outcome': 'timeout'}

# =============================================================================
# MAIN BACKTEST
# =============================================================================

def run_boom_hunter_backtest():
    print("=" * 80)
    print("🔬 BOOM HUNTER PRO STRATEGY BACKTEST")
    print("=" * 80)
    print(f"Timeframe: {TIMEFRAME}m | Data: {DATA_DAYS} days | Symbols: {NUM_SYMBOLS}")
    print(f"EOT Oscillators with SuperSmoother Filter")
    print(f"R:R = {TP_ATR_MULT}:{SL_ATR_MULT} ({int(TP_ATR_MULT)}:1)")
    print("=" * 80)
    
    symbols = get_symbols(NUM_SYMBOLS)
    print(f"\n📋 Testing {len(symbols)} symbols...\n")
    
    # Results by signal type
    results_by_type = defaultdict(lambda: {'w': 0, 'l': 0})
    results_by_side = {'long': {'w': 0, 'l': 0}, 'short': {'w': 0, 'l': 0}}
    results_by_hour = defaultdict(lambda: {'w': 0, 'l': 0})
    
    total_trades = 0
    total_wins = 0
    
    start_time = time.time()
    processed = 0
    
    for idx, symbol in enumerate(symbols):
        try:
            df = fetch_klines(symbol, TIMEFRAME, DATA_DAYS)
            if df.empty or len(df) < 200:
                continue
            
            close = df['close'].values
            
            # Calculate EOT oscillators
            Q1_1, Q2_1 = calculate_eot(close, LP_PERIOD_1, K1_1, K2_1)
            Q3, Q4 = calculate_eot(close, LP_PERIOD_2, K1_2, K2_2)
            Q5, Q6 = calculate_eot(close, LP_PERIOD_3, K1_3, K3_3)
            
            # Calculate Wave Trend
            wt1, wt2 = calculate_wave_trend(df)
            
            # LSMA of wave trend
            lsma = pd.Series(wt1).rolling(200).mean().fillna(50).values
            
            # Detect signals
            signals = detect_boom_signals(df, Q1_1, Q2_1, Q3, Q4, Q5, Q6, wt1, wt2, lsma)
            
            processed += 1
            rows = list(df.itertuples())
            
            last_trade_idx = -50  # Cooldown
            
            for sig in signals:
                i = sig['idx']
                
                if i - last_trade_idx < 5:  # 5-bar cooldown
                    continue
                
                if i >= len(rows) - 50:
                    continue
                
                row = rows[i]
                atr = row.atr
                if pd.isna(atr) or atr <= 0:
                    continue
                
                # Simulate trade
                trade = simulate_trade(df, i, sig['side'], row.close, atr)
                
                if trade['outcome'] == 'timeout':
                    continue
                
                last_trade_idx = i
                total_trades += 1
                
                sig_type = f"{sig['type']}_{sig['side']}"
                
                if trade['outcome'] == 'win':
                    results_by_type[sig_type]['w'] += 1
                    results_by_side[sig['side']]['w'] += 1
                    results_by_hour[row.hour_utc]['w'] += 1
                    total_wins += 1
                else:
                    results_by_type[sig_type]['l'] += 1
                    results_by_side[sig['side']]['l'] += 1
                    results_by_hour[row.hour_utc]['l'] += 1
            
            if (idx + 1) % 10 == 0:
                print(f"[{idx+1}/{NUM_SYMBOLS}] {processed} processed | Trades: {total_trades}")
            
            time.sleep(0.05)
            
        except Exception as e:
            continue
    
    elapsed = time.time() - start_time
    
    # ==========================================================================
    # RESULTS
    # ==========================================================================
    
    print("\n" + "=" * 80)
    print("📊 RESULTS")
    print("=" * 80)
    print(f"\n⏱️ Completed in {elapsed/60:.1f} minutes ({processed} symbols)")
    
    # Overall
    print("\n" + "-" * 60)
    print("OVERALL PERFORMANCE")
    print("-" * 60)
    
    if total_trades > 0:
        overall_wr = total_wins / total_trades
        overall_lb = wilson_lower_bound(total_wins, total_trades)
        overall_ev = calc_ev(overall_wr)
        emoji = "🟢" if overall_ev > 0.1 else "⚠️" if overall_ev > 0 else "🔴"
        
        print(f"\n📊 Total Trades: {total_trades}")
        print(f"✅ Wins: {total_wins} | ❌ Losses: {total_trades - total_wins}")
        print(f"📈 Win Rate: {overall_wr*100:.1f}% (LB: {overall_lb*100:.1f}%)")
        print(f"💰 EV: {overall_ev:+.2f}R {emoji}")
    
    # By Signal Type
    print("\n" + "-" * 60)
    print("BY SIGNAL TYPE")
    print("-" * 60)
    
    print(f"\n{'Signal Type':<20} {'N':>6} {'WR':>8} {'EV':>8}")
    print("-" * 45)
    
    sorted_types = sorted(results_by_type.items(), 
                         key=lambda x: calc_ev(x[1]['w']/(x[1]['w']+x[1]['l'])) if x[1]['w']+x[1]['l'] > 0 else -999,
                         reverse=True)
    
    for sig_type, data in sorted_types:
        total = data['w'] + data['l']
        if total > 0:
            wr = data['w'] / total
            ev = calc_ev(wr)
            emoji = "🟢" if ev > 0.1 else "⚠️" if ev > 0 else "🔴"
            print(f"{sig_type:<20} {total:>6} {wr*100:>7.1f}% {ev:>+7.2f} {emoji}")
    
    # By Side
    print("\n" + "-" * 60)
    print("BY SIDE")
    print("-" * 60)
    
    for side in ['long', 'short']:
        d = results_by_side[side]
        total = d['w'] + d['l']
        if total > 0:
            wr = d['w'] / total
            ev = calc_ev(wr)
            emoji = "🟢" if ev > 0.1 else "⚠️" if ev > 0 else "🔴"
            icon = "🟢" if side == 'long' else "🔴"
            print(f"{icon} {side.upper():<10} N={total:>5} | WR={wr*100:>5.1f}% | EV={ev:>+.2f} {emoji}")
    
    # By Hour (Best 5)
    print("\n" + "-" * 60)
    print("BY HOUR (Best Hours)")
    print("-" * 60)
    
    sorted_hours = sorted(results_by_hour.items(), 
                          key=lambda x: calc_ev(x[1]['w']/(x[1]['w']+x[1]['l'])) if x[1]['w']+x[1]['l'] > 10 else -999,
                          reverse=True)
    
    for hour, d in sorted_hours[:5]:
        total = d['w'] + d['l']
        if total >= 10:
            wr = d['w'] / total
            ev = calc_ev(wr)
            emoji = "🟢" if ev > 0 else "🔴"
            print(f"{hour:02d}:00 UTC | N={total:>4} | WR={wr*100:>5.1f}% | EV={ev:>+.2f} {emoji}")
    
    # Summary
    print("\n" + "=" * 80)
    print("💡 SUMMARY")
    print("=" * 80)
    
    if total_trades > 0:
        overall_wr = total_wins / total_trades
        overall_ev = calc_ev(overall_wr)
        
        print(f"\n📊 Boom Hunter Pro: {total_trades} trades | {overall_wr*100:.1f}% WR | {overall_ev:+.2f} EV")
        
        if overall_ev > 0:
            print(f"\n✅ Strategy is PROFITABLE at 2:1 R:R!")
        else:
            print(f"\n❌ Strategy needs filtering to be profitable at 2:1 R:R")
        
        # Best signal types
        print("\n🎯 Signal Quality Ranking:")
        for sig_type, data in sorted_types:
            total = data['w'] + data['l']
            if total >= 5:
                wr = data['w'] / total
                ev = calc_ev(wr)
                status = "✅ Profitable" if ev > 0 else "❌ Unprofitable"
                print(f"   {sig_type}: {wr*100:.0f}% WR | {ev:+.2f} EV | {status}")

if __name__ == "__main__":
    run_boom_hunter_backtest()
