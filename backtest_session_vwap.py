#!/usr/bin/env python3
"""
SESSION VWAP vs ROLLING VWAP BACKTEST
=====================================
Comparing:
1. Session VWAP: Resets at 00:00 UTC daily (institutional style)
2. Rolling VWAP: 24-hour rolling window (current bot)

Session VWAP is how institutions calculate VWAP - fresh start each day.
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

TIMEFRAME = '3'
DATA_DAYS = 60
NUM_SYMBOLS = 50

TP_ATR_MULT = 2.0
SL_ATR_MULT = 1.0
TOTAL_FEES = 0.001

BASE_URL = "https://api.bybit.com"

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

def get_symbols(limit: int = 50) -> list:
    url = f"{BASE_URL}/v5/market/tickers?category=linear"
    resp = requests.get(url)
    tickers = resp.json().get('result', {}).get('list', [])
    usdt_pairs = [t for t in tickers if t['symbol'].endswith('USDT')]
    usdt_pairs.sort(key=lambda x: float(x.get('turnover24h', 0)), reverse=True)
    return [t['symbol'] for t in usdt_pairs[:limit]]

def fetch_klines(symbol: str, interval: str, days: int) -> pd.DataFrame:
    end_ts = int(datetime.now().timestamp() * 1000)
    candles_needed = days * 24 * 60 // int(interval)
    
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
    df['hour_utc'] = df.index.hour
    df['date'] = df.index.date
    return df

def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if len(df) < 50: return pd.DataFrame()
    
    # RSI
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # MACD
    ema12 = df['close'].ewm(span=12).mean()
    ema26 = df['close'].ewm(span=26).mean()
    df['macd'] = ema12 - ema26
    df['macd_signal'] = df['macd'].ewm(span=9).mean()
    
    # ATR
    high_low = df['high'] - df['low']
    high_close = abs(df['high'] - df['close'].shift())
    low_close = abs(df['low'] - df['close'].shift())
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr'] = true_range.rolling(14).mean()
    
    # Fib levels
    df['roll_high'] = df['high'].rolling(50).max()
    df['roll_low'] = df['low'].rolling(50).min()
    
    # =====================================================
    # ROLLING VWAP (current bot - 24h rolling)
    # =====================================================
    tp = (df['high'] + df['low'] + df['close']) / 3
    df['vwap_rolling'] = (tp * df['volume']).rolling(480).sum() / df['volume'].rolling(480).sum()
    
    # =====================================================
    # SESSION VWAP (resets at 00:00 UTC each day)
    # =====================================================
    df['tp'] = tp
    df['tp_vol'] = tp * df['volume']
    
    # Group by date and calculate cumulative VWAP within each day
    df['session_cum_tpvol'] = df.groupby('date')['tp_vol'].cumsum()
    df['session_cum_vol'] = df.groupby('date')['volume'].cumsum()
    df['vwap_session'] = df['session_cum_tpvol'] / df['session_cum_vol']
    
    # Clean up temp columns
    df.drop(['tp', 'tp_vol', 'session_cum_tpvol', 'session_cum_vol'], axis=1, inplace=True)
    
    return df.dropna()

def check_vwap_signal(close, low, high, vwap) -> str:
    """Cross strategy - price crosses through VWAP"""
    if low <= vwap and close > vwap:
        return 'long'
    if high >= vwap and close < vwap:
        return 'short'
    return None

def get_combo(row) -> str:
    rsi = row['rsi']
    if rsi < 40: r_bin = 'oversold'
    elif rsi > 60: r_bin = 'overbought'
    else: r_bin = 'neutral'
    
    m_bin = 'bull' if row['macd'] > row['macd_signal'] else 'bear'
    
    high, low, close = row['roll_high'], row['roll_low'], row['close']
    if high == low: f_bin = 'low'
    else:
        fib = (high - close) / (high - low) * 100
        if fib < 38: f_bin = 'low'
        elif fib < 62: f_bin = 'mid'
        else: f_bin = 'high'
    
    return f"RSI:{r_bin} MACD:{m_bin} Fib:{f_bin}"

def simulate_trade(df, entry_idx, side, entry_price, atr) -> dict:
    if side == 'long':
        tp = entry_price + (TP_ATR_MULT * atr)
        sl = entry_price - (SL_ATR_MULT * atr)
    else:
        tp = entry_price - (TP_ATR_MULT * atr)
        sl = entry_price + (SL_ATR_MULT * atr)
    
    if side == 'long': tp = tp * (1 + TOTAL_FEES)
    else: tp = tp * (1 - TOTAL_FEES)
    
    rows = list(df.iloc[entry_idx+1:entry_idx+101].itertuples())
    
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

def run_session_vwap_backtest():
    print("=" * 80)
    print("🔬 SESSION VWAP vs ROLLING VWAP BACKTEST")
    print("=" * 80)
    print(f"Timeframe: {TIMEFRAME}m | Data: {DATA_DAYS} days | Symbols: {NUM_SYMBOLS}")
    print(f"Session VWAP: Resets at 00:00 UTC daily")
    print(f"Rolling VWAP: 24-hour rolling window (480 × 3m candles)")
    print("=" * 80)
    
    symbols = get_symbols(NUM_SYMBOLS)
    print(f"\n📋 Testing {len(symbols)} symbols...\n")
    
    # Results
    results = {
        'rolling': {'all': {'w': 0, 'n': 0}, 'long': {'w': 0, 'n': 0}, 'short': {'w': 0, 'n': 0}},
        'session': {'all': {'w': 0, 'n': 0}, 'long': {'w': 0, 'n': 0}, 'short': {'w': 0, 'n': 0}}
    }
    
    rolling_by_hour = defaultdict(lambda: {'w': 0, 'n': 0})
    session_by_hour = defaultdict(lambda: {'w': 0, 'n': 0})
    
    rolling_by_combo = defaultdict(lambda: {'w': 0, 'n': 0})
    session_by_combo = defaultdict(lambda: {'w': 0, 'n': 0})
    
    # Track session hours (time since 00:00 UTC)
    session_hour_stats = defaultdict(lambda: {'w': 0, 'n': 0})
    
    start_time = time.time()
    processed = 0
    
    for idx, symbol in enumerate(symbols):
        try:
            df = fetch_klines(symbol, TIMEFRAME, DATA_DAYS)
            if df.empty or len(df) < 1000:
                continue
            
            df = calculate_indicators(df)
            if df.empty:
                continue
            
            processed += 1
            rows = list(df.itertuples())
            
            for i in range(1, len(rows) - 100):
                row = rows[i]
                
                atr = row.atr
                if pd.isna(atr) or atr <= 0:
                    continue
                
                hour = row.hour_utc
                combo = get_combo({
                    'rsi': row.rsi, 'macd': row.macd, 'macd_signal': row.macd_signal,
                    'roll_high': row.roll_high, 'roll_low': row.roll_low, 'close': row.close
                })
                
                # Test ROLLING VWAP
                rolling_side = check_vwap_signal(row.close, row.low, row.high, row.vwap_rolling)
                if rolling_side:
                    trade = simulate_trade(df, i, rolling_side, row.close, atr)
                    if trade['outcome'] != 'timeout':
                        win = 1 if trade['outcome'] == 'win' else 0
                        results['rolling']['all']['n'] += 1
                        results['rolling']['all']['w'] += win
                        results['rolling'][rolling_side]['n'] += 1
                        results['rolling'][rolling_side]['w'] += win
                        rolling_by_hour[hour]['n'] += 1
                        rolling_by_hour[hour]['w'] += win
                        rolling_by_combo[combo]['n'] += 1
                        rolling_by_combo[combo]['w'] += win
                
                # Test SESSION VWAP
                session_side = check_vwap_signal(row.close, row.low, row.high, row.vwap_session)
                if session_side:
                    trade = simulate_trade(df, i, session_side, row.close, atr)
                    if trade['outcome'] != 'timeout':
                        win = 1 if trade['outcome'] == 'win' else 0
                        results['session']['all']['n'] += 1
                        results['session']['all']['w'] += win
                        results['session'][session_side]['n'] += 1
                        results['session'][session_side]['w'] += win
                        session_by_hour[hour]['n'] += 1
                        session_by_hour[hour]['w'] += win
                        session_by_combo[combo]['n'] += 1
                        session_by_combo[combo]['w'] += win
                        
                        # Track hours since session start (00:00 UTC)
                        session_hour_stats[hour]['n'] += 1
                        session_hour_stats[hour]['w'] += win
            
            if (idx + 1) % 10 == 0:
                print(f"[{idx+1}/{NUM_SYMBOLS}] {processed} processed | Rolling: {results['rolling']['all']['n']} | Session: {results['session']['all']['n']}")
            
            time.sleep(0.03)
            
        except Exception as e:
            continue
    
    elapsed = time.time() - start_time
    
    # ==========================================================================
    # RESULTS
    # ==========================================================================
    
    print("\n" + "=" * 80)
    print("📊 COMPARISON RESULTS")
    print("=" * 80)
    print(f"\n⏱️ Completed in {elapsed/60:.1f} minutes ({processed} symbols)")
    
    print("\n" + "-" * 60)
    print("OVERALL COMPARISON")
    print("-" * 60)
    
    print(f"\n{'Strategy':<25} {'Trades':>8} {'WR':>8} {'LB WR':>8} {'EV':>8}")
    print("-" * 60)
    
    for strat in ['rolling', 'session']:
        d = results[strat]['all']
        if d['n'] > 0:
            wr = d['w'] / d['n']
            lb = wilson_lower_bound(d['w'], d['n'])
            ev = calc_ev(wr)
            emoji = "🟢" if ev > 0.1 else "⚠️" if ev > 0 else "🔴"
            name = "Rolling VWAP (current)" if strat == 'rolling' else "Session VWAP (00:00 reset)"
            print(f"{name:<25} {d['n']:>8} {wr*100:>7.1f}% {lb*100:>7.1f}% {ev:>+7.2f} {emoji}")
    
    print("\n" + "-" * 60)
    print("BY SIDE")
    print("-" * 60)
    
    for strat in ['rolling', 'session']:
        name = "ROLLING" if strat == 'rolling' else "SESSION"
        print(f"\n{name}:")
        for side in ['long', 'short']:
            d = results[strat][side]
            if d['n'] > 0:
                wr = d['w'] / d['n']
                ev = calc_ev(wr)
                emoji = "🟢" if ev > 0.1 else "⚠️" if ev > 0 else "🔴"
                print(f"   {side.upper():<10} N={d['n']:>5} | WR={wr*100:>5.1f}% | EV={ev:>+.2f} {emoji}")
    
    # Session VWAP: Performance by hour since session start
    print("\n" + "-" * 60)
    print("SESSION VWAP: PERFORMANCE BY HOUR SINCE 00:00 UTC")
    print("-" * 60)
    print("(Shows if early/late session is better)")
    
    sorted_session_hours = sorted(session_hour_stats.items(), key=lambda x: x[0])
    print(f"\n{'Hour':>8} {'Trades':>8} {'WR':>8} {'EV':>8}")
    print("-" * 40)
    for hour, d in sorted_session_hours:
        if d['n'] >= 50:
            wr = d['w'] / d['n']
            ev = calc_ev(wr)
            emoji = "🟢" if ev > 0 else "🔴"
            print(f"{hour:02d}:00   {d['n']:>8} {wr*100:>7.1f}% {ev:>+7.2f} {emoji}")
    
    # Top combos
    print("\n" + "-" * 60)
    print("TOP 5 COMBOS BY STRATEGY")
    print("-" * 60)
    
    for strat, combo_dict in [('ROLLING', rolling_by_combo), ('SESSION', session_by_combo)]:
        print(f"\n{strat}:")
        sorted_combos = sorted(combo_dict.items(), key=lambda x: calc_ev(x[1]['w']/x[1]['n']) if x[1]['n'] > 50 else -999, reverse=True)
        for combo, d in sorted_combos[:5]:
            if d['n'] >= 50:
                wr = d['w'] / d['n']
                ev = calc_ev(wr)
                print(f"   {combo:<40} | N={d['n']:>4} | WR={wr*100:>5.1f}% | EV={ev:>+.2f}")
    
    # Summary
    print("\n" + "=" * 80)
    print("💡 SUMMARY")
    print("=" * 80)
    
    rolling_all = results['rolling']['all']
    session_all = results['session']['all']
    
    if rolling_all['n'] > 0 and session_all['n'] > 0:
        rolling_wr = rolling_all['w'] / rolling_all['n']
        session_wr = session_all['w'] / session_all['n']
        rolling_ev = calc_ev(rolling_wr)
        session_ev = calc_ev(session_wr)
        
        print(f"\n📊 Rolling VWAP:  {rolling_all['n']:>5} trades | {rolling_wr*100:.1f}% WR | {rolling_ev:+.2f} EV")
        print(f"📊 Session VWAP:  {session_all['n']:>5} trades | {session_wr*100:.1f}% WR | {session_ev:+.2f} EV")
        
        if rolling_ev > session_ev:
            diff = rolling_ev - session_ev
            print(f"\n✅ ROLLING VWAP (current) is better by {diff:.2f} EV")
        elif session_ev > rolling_ev:
            diff = session_ev - rolling_ev
            print(f"\n✅ SESSION VWAP is better by {diff:.2f} EV")
        else:
            print(f"\n⚖️ Both strategies perform similarly")
        
        # Trade frequency
        print(f"\n📈 Trade Frequency:")
        print(f"   Rolling: {rolling_all['n']} trades ({rolling_all['n']/processed:.0f} per symbol)")
        print(f"   Session: {session_all['n']} trades ({session_all['n']/processed:.0f} per symbol)")
    
    return results

if __name__ == "__main__":
    run_session_vwap_backtest()
