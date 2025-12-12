#!/usr/bin/env python3
"""
COMPREHENSIVE 400-SYMBOL BACKTEST
=================================
Master Trader Analysis - Full Universe

Building on V2 insights, now testing all 400 symbols to find:
1. Additional winning symbols
2. Best combo+hour combinations with higher sample sizes
3. Symbol-specific edges

Based on discoveries:
- Shorts outperform longs significantly  
- Best hours: 03:00, 05:00, 08:00, 22:00 UTC
- Best combos: RSI:oversold with bullish MACD
- Stacked filters: 42.6% WR → +0.28R EV
"""

import requests
import pandas as pd
import numpy as np
import yaml
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
DATA_DAYS = 60  # 60 days for robust testing

TP_ATR_MULT = 2.0
SL_ATR_MULT = 1.0

# From V2 discoveries
BEST_HOURS_T1 = [3, 5, 8]      # Tier 1 hours (50%+ WR)
BEST_HOURS_T2 = [22, 9, 11]   # Tier 2 hours (45-50% WR)
ALL_BEST_HOURS = BEST_HOURS_T1 + BEST_HOURS_T2

# Best combos from V2 (in order of performance)
BEST_COMBOS_T1 = [
    'RSI:oversold MACD:bull Fib:high',     # 61% WR @ 05:00
    'RSI:oversold MACD:bear Fib:low',      # 60% WR @ 03:00
]
BEST_COMBOS_T2 = [
    'RSI:overbought MACD:bear Fib:low',    # 57% WR @ 13:00
    'RSI:neutral MACD:bull Fib:low',       # 56% WR @ 22:00
    'RSI:overbought MACD:bull Fib:mid',    # 54% WR @ 22:00
    'RSI:neutral MACD:bull Fib:mid',       # 52% WR @ 22:00
    'RSI:neutral MACD:bear Fib:low',       # 53% WR @ 09:00
]
ALL_BEST_COMBOS = BEST_COMBOS_T1 + BEST_COMBOS_T2

# Thresholds
MIN_TRADES_SYMBOL = 10      # Min trades for symbol-level analysis
MIN_TRADES_COMBO = 15       # Min trades for combo-level analysis
TARGET_LB_WR = 38.0         # Target lower bound WR (breakeven at 2:1 is 33%)
TARGET_RAW_WR = 42.0        # Target raw WR

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

def get_symbols(limit: int = 400) -> list:
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
    return df

def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if len(df) < 50: return pd.DataFrame()
    
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    ema12 = df['close'].ewm(span=12).mean()
    ema26 = df['close'].ewm(span=26).mean()
    df['macd'] = ema12 - ema26
    df['macd_signal'] = df['macd'].ewm(span=9).mean()
    
    high_low = df['high'] - df['low']
    high_close = abs(df['high'] - df['close'].shift())
    low_close = abs(df['low'] - df['close'].shift())
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr'] = true_range.rolling(14).mean()
    
    df['roll_high'] = df['high'].rolling(50).max()
    df['roll_low'] = df['low'].rolling(50).min()
    
    tp = (df['high'] + df['low'] + df['close']) / 3
    df['vwap'] = (tp * df['volume']).cumsum() / df['volume'].cumsum()
    
    return df.dropna()

def check_vwap_signal(row, prev_row) -> str:
    if prev_row is None: return None
    if row['low'] <= row['vwap'] and row['close'] > row['vwap']:
        return 'long'
    if row['high'] >= row['vwap'] and row['close'] < row['vwap']:
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

def run_comprehensive_backtest():
    print("=" * 80)
    print("🔬 COMPREHENSIVE 400-SYMBOL BACKTEST")
    print("=" * 80)
    print(f"Timeframe: {TIMEFRAME}m | Data: {DATA_DAYS} days")
    print(f"Stacked filters from V2 analysis:")
    print(f"  • SHORTS ONLY")
    print(f"  • Tier 1 Hours: {BEST_HOURS_T1}")
    print(f"  • Tier 2 Hours: {BEST_HOURS_T2}")
    print(f"  • Tier 1 Combos: {len(BEST_COMBOS_T1)}")
    print(f"  • Tier 2 Combos: {len(BEST_COMBOS_T2)}")
    print("=" * 80)
    
    symbols = get_symbols(400)
    print(f"\n📋 Testing {len(symbols)} symbols...\n")
    
    # Aggregate tracking
    aggregate = {
        'all_shorts': {'w': 0, 'n': 0},
        't1_stacked': {'w': 0, 'n': 0},  # T1 hour + T1 combo
        't2_stacked': {'w': 0, 'n': 0},  # Any best hour + any best combo
        'best_hour_only': {'w': 0, 'n': 0},
        'best_combo_only': {'w': 0, 'n': 0},
    }
    
    # Per-symbol tracking
    symbol_results = defaultdict(lambda: {
        'all': {'w': 0, 'n': 0},
        'stacked': {'w': 0, 'n': 0}
    })
    
    # Combo+Hour tracking
    combo_hour_stats = defaultdict(lambda: {'w': 0, 'n': 0})
    
    # Hour-only tracking
    hour_stats = defaultdict(lambda: {'w': 0, 'n': 0})
    
    # Combo-only tracking
    combo_stats = defaultdict(lambda: {'w': 0, 'n': 0})
    
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
                prev_row = rows[i-1]
                
                side = check_vwap_signal(
                    {'low': row.low, 'high': row.high, 'close': row.close, 'vwap': row.vwap},
                    {'vwap': prev_row.vwap}
                )
                
                if side != 'short':
                    continue
                
                hour = row.hour_utc
                combo = get_combo({
                    'rsi': row.rsi, 'macd': row.macd, 'macd_signal': row.macd_signal,
                    'roll_high': row.roll_high, 'roll_low': row.roll_low, 'close': row.close
                })
                
                atr = row.atr
                if pd.isna(atr) or atr <= 0:
                    continue
                
                trade = simulate_trade(df, i, 'short', row.close, atr)
                win = 1 if trade['outcome'] == 'win' else 0
                
                # All shorts
                aggregate['all_shorts']['n'] += 1
                aggregate['all_shorts']['w'] += win
                
                # Hour stats
                hour_stats[hour]['n'] += 1
                hour_stats[hour]['w'] += win
                
                # Combo stats
                combo_stats[combo]['n'] += 1
                combo_stats[combo]['w'] += win
                
                # Combo+Hour
                key = f"{combo}@{hour:02d}:00"
                combo_hour_stats[key]['n'] += 1
                combo_hour_stats[key]['w'] += win
                
                # Symbol tracking (all shorts)
                symbol_results[symbol]['all']['n'] += 1
                symbol_results[symbol]['all']['w'] += win
                
                # Check filters
                in_best_hour = hour in ALL_BEST_HOURS
                in_best_combo = combo in ALL_BEST_COMBOS
                in_t1_hour = hour in BEST_HOURS_T1
                in_t1_combo = combo in BEST_COMBOS_T1
                
                if in_best_hour:
                    aggregate['best_hour_only']['n'] += 1
                    aggregate['best_hour_only']['w'] += win
                
                if in_best_combo:
                    aggregate['best_combo_only']['n'] += 1
                    aggregate['best_combo_only']['w'] += win
                
                # T1 Stacked (premium)
                if in_t1_hour and in_t1_combo:
                    aggregate['t1_stacked']['n'] += 1
                    aggregate['t1_stacked']['w'] += win
                
                # T2 Stacked (any best hour + any best combo)
                if in_best_hour and in_best_combo:
                    aggregate['t2_stacked']['n'] += 1
                    aggregate['t2_stacked']['w'] += win
                    symbol_results[symbol]['stacked']['n'] += 1
                    symbol_results[symbol]['stacked']['w'] += win
            
            # Progress
            if (idx + 1) % 50 == 0:
                elapsed = time.time() - start_time
                rate = processed / elapsed * 60
                stacked_n = aggregate['t2_stacked']['n']
                stacked_wr = aggregate['t2_stacked']['w'] / stacked_n * 100 if stacked_n > 0 else 0
                print(f"[{idx+1}/400] {processed} processed | Stacked: {stacked_n} trades, {stacked_wr:.1f}% WR | {rate:.0f} sym/min")
            
            time.sleep(0.03)
            
        except Exception as e:
            continue
    
    elapsed = time.time() - start_time
    
    # ==========================================================================
    # RESULTS
    # ==========================================================================
    
    print("\n" + "=" * 80)
    print("📊 AGGREGATE RESULTS")
    print("=" * 80)
    
    print(f"\n⏱️ Completed in {elapsed/60:.1f} minutes ({processed} symbols)")
    
    print(f"\n{'Filter':<35} {'N':>8} {'WR':>8} {'LB WR':>8} {'EV':>8}")
    print("-" * 70)
    
    for name, data in [
        ('All Shorts (baseline)', aggregate['all_shorts']),
        ('Best Hour Only', aggregate['best_hour_only']),
        ('Best Combo Only', aggregate['best_combo_only']),
        ('STACKED (Hour+Combo)', aggregate['t2_stacked']),
        ('T1 PREMIUM (Best of best)', aggregate['t1_stacked']),
    ]:
        if data['n'] > 0:
            wr = data['w'] / data['n']
            lb = wilson_lower_bound(data['w'], data['n'])
            ev = calc_ev(wr)
            emoji = "🟢" if ev > 0.1 else "⚠️" if ev > 0 else "🔴"
            print(f"{name:<35} {data['n']:>8} {wr*100:>7.1f}% {lb*100:>7.1f}% {ev:>+7.2f} {emoji}")
    
    # Best hours
    print("\n" + "=" * 80)
    print("⏰ PERFORMANCE BY HOUR (All 400 Symbols - Shorts)")
    print("=" * 80)
    
    hour_results = []
    for hour, data in hour_stats.items():
        if data['n'] >= 100:
            wr = data['w'] / data['n']
            lb = wilson_lower_bound(data['w'], data['n'])
            ev = calc_ev(wr)
            hour_results.append((hour, data['n'], wr, lb, ev))
    
    hour_results.sort(key=lambda x: x[4], reverse=True)
    
    print(f"\n{'Hour':<8} {'N':>8} {'WR':>8} {'LB':>8} {'EV':>8}")
    print("-" * 45)
    for hour, n, wr, lb, ev in hour_results[:12]:
        emoji = "🟢" if ev > 0.1 else "⚠️" if ev > 0 else "🔴"
        print(f"{hour:02d}:00   {n:>8} {wr*100:>7.1f}% {lb*100:>7.1f}% {ev:>+7.2f} {emoji}")
    
    # Best combos
    print("\n" + "=" * 80)
    print("🧩 PERFORMANCE BY COMBO (All 400 Symbols - Shorts)")
    print("=" * 80)
    
    combo_results = []
    for combo, data in combo_stats.items():
        if data['n'] >= 200:
            wr = data['w'] / data['n']
            lb = wilson_lower_bound(data['w'], data['n'])
            ev = calc_ev(wr)
            combo_results.append((combo, data['n'], wr, lb, ev))
    
    combo_results.sort(key=lambda x: x[4], reverse=True)
    
    print(f"\n{'Combo':<45} {'N':>6} {'WR':>7} {'EV':>7}")
    print("-" * 70)
    for combo, n, wr, lb, ev in combo_results:
        emoji = "✅" if ev > 0.15 else ""
        print(f"{combo:<45} {n:>6} {wr*100:>6.1f}% {ev:>+6.2f} {emoji}")
    
    # Best combo+hour combinations (NEW: top 20)
    print("\n" + "=" * 80)
    print("🏆 TOP 20 COMBO+HOUR COMBINATIONS (Sorted by EV)")
    print("=" * 80)
    
    ch_results = []
    for key, data in combo_hour_stats.items():
        if data['n'] >= 20:
            wr = data['w'] / data['n']
            lb = wilson_lower_bound(data['w'], data['n'])
            ev = calc_ev(wr)
            if wr >= 0.40:  # 40%+ WR
                ch_results.append((key, data['n'], wr, lb, ev))
    
    ch_results.sort(key=lambda x: x[4], reverse=True)
    
    print(f"\n{'Combo @ Hour':<55} {'N':>5} {'WR':>7} {'EV':>7}")
    print("-" * 80)
    for key, n, wr, lb, ev in ch_results[:20]:
        emoji = "🔥" if ev > 0.4 else "✅" if ev > 0.2 else ""
        display = key[:52] + "..." if len(key) > 55 else key
        print(f"{display:<55} {n:>5} {wr*100:>6.1f}% {ev:>+6.2f} {emoji}")
    
    # Winning symbols
    print("\n" + "=" * 80)
    print("🎯 WINNING SYMBOLS (Stacked Filter, LB WR >= 38%)")
    print("=" * 80)
    
    winners = []
    for sym, data in symbol_results.items():
        stacked = data['stacked']
        if stacked['n'] >= MIN_TRADES_SYMBOL:
            wr = stacked['w'] / stacked['n']
            lb = wilson_lower_bound(stacked['w'], stacked['n'])
            ev = calc_ev(wr)
            if lb >= TARGET_LB_WR / 100:
                winners.append({'symbol': sym, 'n': stacked['n'], 'wr': wr*100, 'lb_wr': lb*100, 'ev': ev})
    
    winners.sort(key=lambda x: x['lb_wr'], reverse=True)
    
    if winners:
        print(f"\n{len(winners)} symbols found:\n")
        for w in winners[:30]:
            print(f"   {w['symbol']:<15} LB WR={w['lb_wr']:.1f}% | N={w['n']:>3} | EV={w['ev']:+.2f}")
    else:
        print("\n   None found at 38% LB WR threshold")
    
    # ==========================================================================
    # GENERATE UPDATED CONFIG
    # ==========================================================================
    
    print("\n" + "=" * 80)
    print("📁 GENERATING UPDATED CONFIG")
    print("=" * 80)
    
    # Build premium setups from top combo+hour combos
    premium_setups = []
    for key, n, wr, lb, ev in ch_results[:15]:  # Top 15
        parts = key.split('@')
        combo = parts[0]
        hour = int(parts[1].replace(':00', ''))
        
        premium_setups.append({
            'combo': combo,
            'hours': [hour],
            'expected_wr': round(wr * 100),
            'sample': n,
            'ev': round(ev, 2)
        })
    
    # Find best hours (from aggregate)
    best_hours = [h for h, n, wr, lb, ev in hour_results[:6] if ev > 0.05]
    
    config = {
        'global_rules': {
            'side': 'short_only',
            'enabled': True
        },
        'time_filter': {
            'best_hours': sorted(best_hours),
            'note': 'From 400-symbol backtest'
        },
        'premium_setups': premium_setups,
        'verified_symbols': {w['symbol']: {'lb_wr': round(w['lb_wr'], 1), 'ev': round(w['ev'], 2)} for w in winners[:20]}
    }
    
    with open('backtest_golden_combos_v3.yaml', 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"\n✅ Config saved: backtest_golden_combos_v3.yaml")
    print(f"   • {len(premium_setups)} premium setups")
    print(f"   • Best hours: {sorted(best_hours)}")
    print(f"   • {len(winners)} verified symbols")
    
    # Summary
    print("\n" + "=" * 80)
    print("💡 KEY INSIGHTS FROM 400-SYMBOL ANALYSIS")
    print("=" * 80)
    
    if aggregate['t2_stacked']['n'] > 0:
        base_wr = aggregate['all_shorts']['w'] / aggregate['all_shorts']['n']
        stacked_wr = aggregate['t2_stacked']['w'] / aggregate['t2_stacked']['n']
        improvement = (stacked_wr - base_wr) * 100
        stacked_ev = calc_ev(stacked_wr)
        
        print(f"\n📈 Stacked filters improve WR by: {improvement:+.1f}%")
        print(f"   Baseline: {base_wr*100:.1f}% → Stacked: {stacked_wr*100:.1f}%")
        print(f"   EV: {stacked_ev:+.2f}R at 2:1 R:R")
        
        if stacked_wr > 0.33:
            print("\n✅ STRATEGY IS PROFITABLE at 2:1 R:R")
        else:
            print("\n⚠️ Strategy below breakeven - needs adjustment")
    
    return winners, ch_results

if __name__ == "__main__":
    winners, combos = run_comprehensive_backtest()
