#!/usr/bin/env python3
"""
EDGE-OPTIMIZED VWAP BACKTEST
============================
Based on Live Data Insights:
1. SHORTS ONLY - Shorts outperform longs significantly
2. ASIAN SESSION - Best performance during Asian hours
3. TIME FILTERS - 00:00-02:00 UTC is the "golden hour"
4. BTC FILTER - Trade with BTC trend

This backtest incorporates the discovered edges to validate
if they hold in historical data.
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
# CONFIGURATION - OPTIMIZED BASED ON LIVE DATA
# =============================================================================

TIMEFRAME = '3'
DATA_DAYS = 60  # Shorter period for faster iteration

# R:R
TP_ATR_MULT = 2.0
SL_ATR_MULT = 1.0

# Thresholds (relaxed to find more opportunities)
MIN_TRADES = 15
TARGET_LB_WR = 38.0  # Relaxed from 45% - at 2:1 R:R, 33% is breakeven
TARGET_RAW_WR = 42.0  # Raw WR target

# Fee buffer
TOTAL_FEES = 0.001

# EDGE FILTERS (from live data analysis)
SHORTS_ONLY = True  # Live data shows shorts outperform
BEST_HOURS = [0, 1, 2, 21, 22, 23]  # Asian session + golden hours
WORST_HOURS = [14, 15, 16, 17, 18, 19]  # Avoid these hours

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

def get_symbols(limit: int = 100) -> list:
    """Fetch top symbols."""
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
        params = {
            'category': 'linear',
            'symbol': symbol,
            'interval': interval,
            'limit': 1000,
            'end': current_end
        }
        
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
    
    # Extract hour for filtering
    df['hour_utc'] = df.index.hour
    
    return df

# =============================================================================
# INDICATORS
# =============================================================================

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
    
    # Fib
    df['roll_high'] = df['high'].rolling(50).max()
    df['roll_low'] = df['low'].rolling(50).min()
    
    # VWAP
    tp = (df['high'] + df['low'] + df['close']) / 3
    df['vwap'] = (tp * df['volume']).cumsum() / df['volume'].cumsum()
    
    return df.dropna()

# =============================================================================
# SIGNAL & COMBO
# =============================================================================

def check_vwap_signal(row, prev_row) -> str:
    if prev_row is None: return None
    
    # Long signal
    if row['low'] <= row['vwap'] and row['close'] > row['vwap']:
        return 'long'
    
    # Short signal
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

# =============================================================================
# TRADE SIMULATION
# =============================================================================

def simulate_trade(df: pd.DataFrame, entry_idx: int, side: str, 
                   entry_price: float, atr: float) -> dict:
    if side == 'long':
        tp = entry_price + (TP_ATR_MULT * atr)
        sl = entry_price - (SL_ATR_MULT * atr)
    else:
        tp = entry_price - (TP_ATR_MULT * atr)
        sl = entry_price + (SL_ATR_MULT * atr)
    
    # Fee buffer
    if side == 'long':
        tp = tp * (1 + TOTAL_FEES)
    else:
        tp = tp * (1 - TOTAL_FEES)
    
    max_bars = 100
    rows = list(df.iloc[entry_idx+1:entry_idx+1+max_bars].itertuples())
    
    for future_row in rows:
        if side == 'long':
            if future_row.low <= sl:
                return {'outcome': 'loss', 'pnl': -1.0}
            if future_row.high >= tp:
                return {'outcome': 'win', 'pnl': TP_ATR_MULT}
        else:
            if future_row.high >= sl:
                return {'outcome': 'loss', 'pnl': -1.0}
            if future_row.low <= tp:
                return {'outcome': 'win', 'pnl': TP_ATR_MULT}
    
    return {'outcome': 'timeout', 'pnl': -0.5}

# =============================================================================
# BACKTEST
# =============================================================================

def run_backtest(symbol: str, df: pd.DataFrame) -> dict:
    """Run backtest with edge filters."""
    results = {
        'symbol': symbol,
        'all': {'w': 0, 'n': 0},        # All trades
        'filtered': {'w': 0, 'n': 0},    # With edge filters
        'by_hour': defaultdict(lambda: {'w': 0, 'n': 0}),
        'by_combo': defaultdict(lambda: {'w': 0, 'n': 0})
    }
    
    rows = list(df.itertuples())
    
    for i in range(1, len(rows) - 100):
        row = rows[i]
        prev_row = rows[i-1]
        
        side = check_vwap_signal(
            {'low': row.low, 'high': row.high, 'close': row.close, 'vwap': row.vwap},
            {'vwap': prev_row.vwap}
        )
        
        if not side: continue
        
        # EDGE FILTER 1: Shorts only
        if SHORTS_ONLY and side == 'long':
            continue
        
        entry_price = row.close
        atr = row.atr
        hour = row.hour_utc
        combo = get_combo({
            'rsi': row.rsi,
            'macd': row.macd,
            'macd_signal': row.macd_signal,
            'roll_high': row.roll_high,
            'roll_low': row.roll_low,
            'close': row.close
        })
        
        if pd.isna(atr) or atr <= 0: continue
        
        # Simulate trade
        trade = simulate_trade(df, i, side, entry_price, atr)
        win = 1 if trade['outcome'] == 'win' else 0
        
        # Record ALL trades (for comparison)
        results['all']['n'] += 1
        results['all']['w'] += win
        
        # By hour
        results['by_hour'][hour]['n'] += 1
        results['by_hour'][hour]['w'] += win
        
        # By combo
        results['by_combo'][combo]['n'] += 1
        results['by_combo'][combo]['w'] += win
        
        # EDGE FILTER 2: Best hours only
        if hour in BEST_HOURS:
            results['filtered']['n'] += 1
            results['filtered']['w'] += win
    
    return results

# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("🎯 EDGE-OPTIMIZED VWAP BACKTEST")
    print("=" * 70)
    print("Testing discovered edges from live data analysis:")
    print(f"  • SHORTS ONLY: {SHORTS_ONLY}")
    print(f"  • BEST HOURS: {BEST_HOURS}")
    print(f"  • MIN_TRADES: {MIN_TRADES}")
    print(f"  • TARGET: {TARGET_LB_WR}% LB WR / {TARGET_RAW_WR}% Raw WR")
    print("=" * 70)
    
    symbols = get_symbols(100)  # Start with 100 symbols
    print(f"\n📋 Testing {len(symbols)} symbols...")
    
    # Aggregate results
    all_results = []
    total_all = {'w': 0, 'n': 0}
    total_filtered = {'w': 0, 'n': 0}
    hour_aggregate = defaultdict(lambda: {'w': 0, 'n': 0})
    combo_aggregate = defaultdict(lambda: {'w': 0, 'n': 0})
    
    winning_symbols = []
    
    for idx, symbol in enumerate(symbols):
        try:
            df = fetch_klines(symbol, TIMEFRAME, DATA_DAYS)
            if df.empty or len(df) < 1000:
                continue
            
            df = calculate_indicators(df)
            if df.empty: continue
            
            results = run_backtest(symbol, df)
            all_results.append(results)
            
            # Aggregate
            total_all['n'] += results['all']['n']
            total_all['w'] += results['all']['w']
            total_filtered['n'] += results['filtered']['n']
            total_filtered['w'] += results['filtered']['w']
            
            for hour, data in results['by_hour'].items():
                hour_aggregate[hour]['n'] += data['n']
                hour_aggregate[hour]['w'] += data['w']
            
            for combo, data in results['by_combo'].items():
                combo_aggregate[combo]['n'] += data['n']
                combo_aggregate[combo]['w'] += data['w']
            
            # Check if symbol+filtered is a winner
            f = results['filtered']
            if f['n'] >= MIN_TRADES:
                wr = f['w'] / f['n']
                lb_wr = wilson_lower_bound(f['w'], f['n'])
                ev = calc_ev(wr)
                if lb_wr >= TARGET_LB_WR / 100:
                    winning_symbols.append({
                        'symbol': symbol,
                        'n': f['n'],
                        'wr': wr * 100,
                        'lb_wr': lb_wr * 100,
                        'ev': ev
                    })
            
            # Progress
            if (idx + 1) % 20 == 0:
                print(f"[{idx+1}/{len(symbols)}] Processed... {len(winning_symbols)} winners so far")
            
            time.sleep(0.1)
            
        except Exception as e:
            continue
    
    # ==========================================================================
    # RESULTS
    # ==========================================================================
    
    print("\n" + "=" * 70)
    print("📊 AGGREGATE RESULTS")
    print("=" * 70)
    
    # All trades
    if total_all['n'] > 0:
        all_wr = total_all['w'] / total_all['n']
        all_lb = wilson_lower_bound(total_all['w'], total_all['n'])
        all_ev = calc_ev(all_wr)
        print(f"\n📈 ALL TRADES (shorts during any hour):")
        print(f"   N: {total_all['n']} | WR: {all_wr*100:.1f}% | LB: {all_lb*100:.1f}% | EV: {all_ev:+.2f}")
    
    # Filtered trades
    if total_filtered['n'] > 0:
        filt_wr = total_filtered['w'] / total_filtered['n']
        filt_lb = wilson_lower_bound(total_filtered['w'], total_filtered['n'])
        filt_ev = calc_ev(filt_wr)
        print(f"\n🎯 FILTERED (shorts during BEST hours only):")
        print(f"   N: {total_filtered['n']} | WR: {filt_wr*100:.1f}% | LB: {filt_lb*100:.1f}% | EV: {filt_ev:+.2f}")
        
        improvement = (filt_wr - all_wr) * 100
        print(f"   📈 Improvement: {improvement:+.1f}% WR")
    
    # By hour
    print(f"\n⏰ PERFORMANCE BY HOUR (Shorts Only):")
    print(f"   {'Hour':<8} {'N':>6} {'WR':>8} {'LB WR':>8} {'EV':>8}")
    print("   " + "-" * 40)
    
    hour_results = []
    for hour, data in sorted(hour_aggregate.items()):
        if data['n'] >= 10:
            wr = data['w'] / data['n']
            lb = wilson_lower_bound(data['w'], data['n'])
            ev = calc_ev(wr)
            hour_results.append((hour, data['n'], wr, lb, ev))
    
    hour_results.sort(key=lambda x: x[4], reverse=True)
    for hour, n, wr, lb, ev in hour_results:
        emoji = "🟢" if ev > 0.1 else "🔴" if ev < 0 else "⚪"
        print(f"   {hour:02d}:00   {n:>6} {wr*100:>7.1f}% {lb*100:>7.1f}% {ev:>+7.2f} {emoji}")
    
    # By combo
    print(f"\n🧩 PERFORMANCE BY COMBO (Shorts Only):")
    print(f"   {'Combo':<40} {'N':>5} {'WR':>7} {'EV':>7}")
    print("   " + "-" * 60)
    
    combo_results = []
    for combo, data in combo_aggregate.items():
        if data['n'] >= 20:
            wr = data['w'] / data['n']
            ev = calc_ev(wr)
            combo_results.append((combo, data['n'], wr, ev))
    
    combo_results.sort(key=lambda x: x[3], reverse=True)
    for combo, n, wr, ev in combo_results[:10]:
        emoji = "✅" if ev > 0.15 else ""
        print(f"   {combo:<40} {n:>5} {wr*100:>6.1f}% {ev:>+6.2f} {emoji}")
    
    # Winning symbols
    print(f"\n🏆 WINNING SYMBOLS ({len(winning_symbols)}):")
    if winning_symbols:
        winning_symbols.sort(key=lambda x: x['lb_wr'], reverse=True)
        for w in winning_symbols[:20]:
            print(f"   {w['symbol']}: LB WR={w['lb_wr']:.1f}% | N={w['n']} | EV={w['ev']:+.2f}")
        
        # Save config
        config = {}
        for w in winning_symbols:
            config[w['symbol']] = {
                'allowed_combos_short': ['ALL'],  # Allow all combos for shorts
                'filter': {
                    'hours': BEST_HOURS,
                    'side': 'short'
                }
            }
        
        with open('optimized_edge_config.yaml', 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        print(f"\n✅ Config saved to: optimized_edge_config.yaml")
    else:
        print("   None found - adjusting thresholds...")
    
    # Summary
    print("\n" + "=" * 70)
    print("💡 CONCLUSIONS")
    print("=" * 70)
    
    if total_filtered['n'] > 0 and total_all['n'] > 0:
        if filt_wr > all_wr:
            print("✅ TIME FILTER VALIDATES - Best hours outperform all hours")
        else:
            print("❌ TIME FILTER NOT VALIDATED - No improvement from hour filter")
    
    print("\nRecommended strategy:")
    print("  1. SHORTS ONLY")
    print("  2. Trade during: 00:00-02:00, 21:00-23:00 UTC")
    print("  3. Avoid: 14:00-19:00 UTC")
    print("  4. Focus on best-performing combos")

if __name__ == "__main__":
    main()
