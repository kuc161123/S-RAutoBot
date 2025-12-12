#!/usr/bin/env python3
"""
PRECISION EDGE BACKTEST V2
==========================
Based on V1 discoveries:
- Best hours: 03:00, 22:00 UTC (42%+ WR)
- Best combo: RSI:oversold MACD:bull Fib:high (42% WR)
- Shorts only

This version adds:
1. COMBO + HOUR stacking (both filters must pass)
2. More symbols
3. Lower sample threshold for discovery
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
# CONFIGURATION - PRECISION TUNED
# =============================================================================

TIMEFRAME = '3'
DATA_DAYS = 60

TP_ATR_MULT = 2.0
SL_ATR_MULT = 1.0

# DISCOVERED EDGES (from V1)
BEST_HOURS = [3, 22, 9]  # V1 winners: 03:00 (42.7%), 22:00 (42.4%), 09:00 (36.9%)
BEST_COMBOS = [
    'RSI:oversold MACD:bull Fib:high',     # 42.2% WR, +0.27 EV
    'RSI:overbought MACD:bear Fib:mid',    # 40.0% WR, +0.20 EV  
    'RSI:overbought MACD:bull Fib:mid',    # 36.8% WR, +0.11 EV
]

# Relaxed thresholds for discovery
MIN_TRADES = 10
TARGET_LB_WR = 35.0  # Breakeven at 2:1 RR is 33%

TOTAL_FEES = 0.001
BASE_URL = "https://api.bybit.com"

# =============================================================================
# HELPERS (same as before)
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

def get_symbols(limit: int = 200) -> list:
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
# BACKTEST
# =============================================================================

def run_precision_backtest():
    print("=" * 70)
    print("🎯 PRECISION EDGE BACKTEST V2")
    print("=" * 70)
    print("Testing stacked edges:")
    print(f"  • SHORTS ONLY")
    print(f"  • BEST HOURS: {BEST_HOURS}")
    print(f"  • BEST COMBOS: {len(BEST_COMBOS)} combos")
    print("=" * 70)
    
    symbols = get_symbols(200)
    print(f"\n📋 Testing {len(symbols)} symbols...")
    
    # Results by filter type
    results = {
        'all_shorts': {'w': 0, 'n': 0},
        'best_hour': {'w': 0, 'n': 0},
        'best_combo': {'w': 0, 'n': 0},
        'stacked': {'w': 0, 'n': 0},  # Both hour AND combo filter
    }
    
    # Per-symbol tracking for stacked filter
    symbol_results = defaultdict(lambda: {'w': 0, 'n': 0})
    
    # Track combo+hour combinations
    combo_hour = defaultdict(lambda: {'w': 0, 'n': 0})
    
    for idx, symbol in enumerate(symbols):
        try:
            df = fetch_klines(symbol, TIMEFRAME, DATA_DAYS)
            if df.empty or len(df) < 1000: continue
            df = calculate_indicators(df)
            if df.empty: continue
            
            rows = list(df.itertuples())
            
            for i in range(1, len(rows) - 100):
                row = rows[i]
                prev_row = rows[i-1]
                
                side = check_vwap_signal(
                    {'low': row.low, 'high': row.high, 'close': row.close, 'vwap': row.vwap},
                    {'vwap': prev_row.vwap}
                )
                
                if side != 'short': continue  # Shorts only
                
                hour = row.hour_utc
                combo = get_combo({
                    'rsi': row.rsi, 'macd': row.macd, 'macd_signal': row.macd_signal,
                    'roll_high': row.roll_high, 'roll_low': row.roll_low, 'close': row.close
                })
                
                atr = row.atr
                if pd.isna(atr) or atr <= 0: continue
                
                trade = simulate_trade(df, i, 'short', row.close, atr)
                win = 1 if trade['outcome'] == 'win' else 0
                
                # All shorts
                results['all_shorts']['n'] += 1
                results['all_shorts']['w'] += win
                
                # Best hour filter
                if hour in BEST_HOURS:
                    results['best_hour']['n'] += 1
                    results['best_hour']['w'] += win
                
                # Best combo filter
                if combo in BEST_COMBOS:
                    results['best_combo']['n'] += 1
                    results['best_combo']['w'] += win
                
                # STACKED: Both hour AND combo
                if hour in BEST_HOURS and combo in BEST_COMBOS:
                    results['stacked']['n'] += 1
                    results['stacked']['w'] += win
                    symbol_results[symbol]['n'] += 1
                    symbol_results[symbol]['w'] += win
                
                # Track combo+hour
                key = f"{combo}@{hour:02d}:00"
                combo_hour[key]['n'] += 1
                combo_hour[key]['w'] += win
            
            if (idx + 1) % 40 == 0:
                print(f"[{idx+1}/{len(symbols)}] Processed...")
            
            time.sleep(0.05)
            
        except: continue
    
    # ==========================================================================
    # RESULTS
    # ==========================================================================
    
    print("\n" + "=" * 70)
    print("📊 FILTER COMPARISON")
    print("=" * 70)
    
    filters = [
        ('All Shorts (no filter)', results['all_shorts']),
        ('Best Hour Only', results['best_hour']),
        ('Best Combo Only', results['best_combo']),
        ('STACKED (Hour + Combo)', results['stacked']),
    ]
    
    print(f"\n{'Filter':<30} {'N':>8} {'WR':>8} {'LB WR':>8} {'EV':>8}")
    print("-" * 65)
    
    for name, data in filters:
        if data['n'] > 0:
            wr = data['w'] / data['n']
            lb = wilson_lower_bound(data['w'], data['n'])
            ev = calc_ev(wr)
            emoji = "🟢" if ev > 0.1 else "⚠️" if ev > 0 else "🔴"
            print(f"{name:<30} {data['n']:>8} {wr*100:>7.1f}% {lb*100:>7.1f}% {ev:>+7.2f} {emoji}")
        else:
            print(f"{name:<30} {'N/A':>8}")
    
    # Top combo+hour combinations
    print("\n" + "=" * 70)
    print("🏆 BEST COMBO+HOUR COMBINATIONS")
    print("=" * 70)
    
    ch_results = []
    for key, data in combo_hour.items():
        if data['n'] >= 10:
            wr = data['w'] / data['n']
            lb = wilson_lower_bound(data['w'], data['n'])
            ev = calc_ev(wr)
            if wr >= 0.4:  # 40%+ WR only
                ch_results.append((key, data['n'], wr, lb, ev))
    
    ch_results.sort(key=lambda x: x[4], reverse=True)
    
    print(f"\n{'Combo @ Hour':<55} {'N':>5} {'WR':>7} {'EV':>7}")
    print("-" * 75)
    for key, n, wr, lb, ev in ch_results[:15]:
        emoji = "🔥" if ev > 0.3 else "✅" if ev > 0.15 else ""
        display = key[:52] + "..." if len(key) > 55 else key
        print(f"{display:<55} {n:>5} {wr*100:>6.1f}% {ev:>+6.2f} {emoji}")
    
    # Per-symbol winners (stacked filter)
    winners = []
    for sym, data in symbol_results.items():
        if data['n'] >= MIN_TRADES:
            wr = data['w'] / data['n']
            lb = wilson_lower_bound(data['w'], data['n'])
            ev = calc_ev(wr)
            if lb >= TARGET_LB_WR / 100:
                winners.append({'symbol': sym, 'n': data['n'], 'wr': wr*100, 'lb_wr': lb*100, 'ev': ev})
    
    winners.sort(key=lambda x: x['lb_wr'], reverse=True)
    
    print(f"\n🏆 WINNING SYMBOLS (Stacked Filter, LB >= {TARGET_LB_WR}%):")
    if winners:
        for w in winners[:20]:
            print(f"   {w['symbol']}: LB WR={w['lb_wr']:.1f}% | N={w['n']} | EV={w['ev']:+.2f}")
        
        # Generate config
        config = {}
        for w in winners:
            config[w['symbol']] = {
                'allowed_combos_short': BEST_COMBOS,
                'allowed_hours': BEST_HOURS,
                'lb_wr': round(w['lb_wr'], 1),
                'ev': round(w['ev'], 2)
            }
        
        with open('precision_edge_config.yaml', 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        print(f"\n✅ Config saved: precision_edge_config.yaml ({len(winners)} symbols)")
    else:
        print("   None found at current threshold")
    
    # Key insights
    print("\n" + "=" * 70)
    print("💡 KEY INSIGHTS")
    print("=" * 70)
    
    if results['stacked']['n'] > 0:
        stacked_wr = results['stacked']['w'] / results['stacked']['n']
        all_wr = results['all_shorts']['w'] / results['all_shorts']['n'] if results['all_shorts']['n'] > 0 else 0
        
        improvement = (stacked_wr - all_wr) * 100
        print(f"\n📈 Stacking filters improves WR by: {improvement:+.1f}%")
        print(f"   From {all_wr*100:.1f}% → {stacked_wr*100:.1f}%")
        
        if stacked_wr > 0.33:
            print("\n✅ STACKED FILTER IS PROFITABLE at 2:1 R:R")
        else:
            print("\n⚠️ Still below breakeven - need more refinement")
    
    print("\n🎯 RECOMMENDED TRADING RULES:")
    print("   1. SHORTS ONLY")
    print(f"   2. Trade during: {BEST_HOURS} UTC")
    print(f"   3. Use combos: {', '.join(BEST_COMBOS[:2])}")
    print("   4. Focus on winning symbols only")

if __name__ == "__main__":
    run_precision_backtest()
