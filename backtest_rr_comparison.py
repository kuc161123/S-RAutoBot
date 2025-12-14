#!/usr/bin/env python3
"""
R:R COMPARISON BACKTEST
========================
Tests multiple Risk:Reward ratios to find optimal setting.
Compares: 1:1, 1.5:1, 2:1, 2.5:1, 3:1
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

TIMEFRAME = '15'
DATA_DAYS = 60
NUM_SYMBOLS = 200

# R:R ratios to test
RR_RATIOS = [1.0, 1.5, 2.0, 2.5, 3.0]

# Costs
SLIPPAGE_PCT = 0.0005
FEE_PCT = 0.0004
TOTAL_COST = (SLIPPAGE_PCT + FEE_PCT) * 2

# Divergence settings
RSI_PERIOD = 14
RSI_OVERSOLD = 30
RSI_OVERBOUGHT = 70
LOOKBACK_BARS = 14
MIN_PIVOT_DISTANCE = 5

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

def calc_ev(wr: float, rr: float) -> float:
    """Calculate Expected Value given win rate and R:R"""
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
    return df

def calculate_rsi(close, period=14):
    delta = close.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
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

def detect_divergence_signals(df):
    if len(df) < 100:
        return []
    
    close = df['close'].values
    rsi = df['rsi'].values
    n = len(df)
    
    price_pivot_highs, price_pivot_lows = find_pivots(close, left=3, right=3)
    signals = []
    
    for i in range(30, n - 5):
        curr_price_low = curr_price_low_idx = None
        prev_price_low = prev_price_low_idx = None
        
        for j in range(i, max(i - LOOKBACK_BARS, 0), -1):
            if not np.isnan(price_pivot_lows[j]):
                if curr_price_low is None:
                    curr_price_low = price_pivot_lows[j]
                    curr_price_low_idx = j
                elif prev_price_low is None and j < curr_price_low_idx - MIN_PIVOT_DISTANCE:
                    prev_price_low = price_pivot_lows[j]
                    prev_price_low_idx = j
                    break
        
        curr_price_high = curr_price_high_idx = None
        prev_price_high = prev_price_high_idx = None
        
        for j in range(i, max(i - LOOKBACK_BARS, 0), -1):
            if not np.isnan(price_pivot_highs[j]):
                if curr_price_high is None:
                    curr_price_high = price_pivot_highs[j]
                    curr_price_high_idx = j
                elif prev_price_high is None and j < curr_price_high_idx - MIN_PIVOT_DISTANCE:
                    prev_price_high = price_pivot_highs[j]
                    prev_price_high_idx = j
                    break
        
        # Regular Bullish
        if curr_price_low is not None and prev_price_low is not None:
            if curr_price_low < prev_price_low:
                curr_rsi = rsi[curr_price_low_idx]
                prev_rsi = rsi[prev_price_low_idx]
                if curr_rsi > prev_rsi and rsi[i] < RSI_OVERSOLD + 15:
                    signals.append({'idx': i, 'side': 'long', 'type': 'regular_bullish'})
                    continue
        
        # Regular Bearish
        if curr_price_high is not None and prev_price_high is not None:
            if curr_price_high > prev_price_high:
                curr_rsi = rsi[curr_price_high_idx]
                prev_rsi = rsi[prev_price_high_idx]
                if curr_rsi < prev_rsi and rsi[i] > RSI_OVERBOUGHT - 15:
                    signals.append({'idx': i, 'side': 'short', 'type': 'regular_bearish'})
                    continue
        
        # Hidden Bullish
        if curr_price_low is not None and prev_price_low is not None:
            if curr_price_low > prev_price_low:
                curr_rsi = rsi[curr_price_low_idx]
                prev_rsi = rsi[prev_price_low_idx]
                if curr_rsi < prev_rsi and rsi[i] < RSI_OVERBOUGHT - 10:
                    signals.append({'idx': i, 'side': 'long', 'type': 'hidden_bullish'})
                    continue
        
        # Hidden Bearish
        if curr_price_high is not None and prev_price_high is not None:
            if curr_price_high < prev_price_high:
                curr_rsi = rsi[curr_price_high_idx]
                prev_rsi = rsi[prev_price_high_idx]
                if curr_rsi > prev_rsi and rsi[i] > RSI_OVERSOLD + 10:
                    signals.append({'idx': i, 'side': 'short', 'type': 'hidden_bearish'})
    
    return signals

def simulate_trade(df, signal_idx, side, atr, tp_mult, sl_mult=1.0):
    """Simulate trade with given TP multiplier"""
    rows = list(df.itertuples())
    
    entry_idx = signal_idx + 1
    if entry_idx >= len(rows) - 50:
        return {'outcome': 'timeout', 'bars': 0}
    
    entry_row = rows[entry_idx]
    base_entry = entry_row.open
    
    if side == 'long':
        entry_price = base_entry * (1 + SLIPPAGE_PCT)
        tp = entry_price + (tp_mult * atr)
        sl = entry_price - (sl_mult * atr)
        tp = tp * (1 - TOTAL_COST)
    else:
        entry_price = base_entry * (1 - SLIPPAGE_PCT)
        tp = entry_price - (tp_mult * atr)
        sl = entry_price + (sl_mult * atr)
        tp = tp * (1 + TOTAL_COST)
    
    for bar_offset, future_row in enumerate(rows[entry_idx+1:entry_idx+150]):
        if side == 'long':
            if future_row.low <= sl:
                return {'outcome': 'loss', 'bars': bar_offset + 1}
            if future_row.high >= tp:
                return {'outcome': 'win', 'bars': bar_offset + 1}
        else:
            if future_row.high >= sl:
                return {'outcome': 'loss', 'bars': bar_offset + 1}
            if future_row.low <= tp:
                return {'outcome': 'win', 'bars': bar_offset + 1}
    
    return {'outcome': 'timeout', 'bars': 150}

# =============================================================================
# MAIN
# =============================================================================

def run_rr_comparison():
    print("=" * 80)
    print("🔬 R:R COMPARISON BACKTEST")
    print("=" * 80)
    print(f"Timeframe: {TIMEFRAME}m | Data: {DATA_DAYS} days | Symbols: {NUM_SYMBOLS}")
    print(f"Testing R:R ratios: {RR_RATIOS}")
    print("=" * 80)
    
    symbols = get_symbols(NUM_SYMBOLS)
    print(f"\n📋 Fetching data for {len(symbols)} symbols...\n")
    
    # Prepare data storage
    all_signals = []  # Store all detected signals for replay
    
    # First pass: collect all signals and data
    start_time = time.time()
    processed = 0
    
    for idx, symbol in enumerate(symbols):
        try:
            df = fetch_klines(symbol, TIMEFRAME, DATA_DAYS)
            if df.empty or len(df) < 500:
                continue
            
            df['rsi'] = calculate_rsi(df['close'], RSI_PERIOD)
            
            high_low = df['high'] - df['low']
            high_close = abs(df['high'] - df['close'].shift())
            low_close = abs(df['low'] - df['close'].shift())
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            df['atr'] = true_range.rolling(14).mean()
            
            df['vol_ma'] = df['volume'].rolling(20).mean()
            df['vol_ok'] = df['volume'] > df['vol_ma'] * 0.5
            
            df = df.dropna()
            if len(df) < 200:
                continue
            
            processed += 1
            
            signals = detect_divergence_signals(df)
            rows = list(df.itertuples())
            last_trade_idx = -20
            
            for sig in signals:
                i = sig['idx']
                
                if i - last_trade_idx < 10:
                    continue
                if i >= len(rows) - 100:
                    continue
                
                row = rows[i]
                atr = row.atr
                if pd.isna(atr) or atr <= 0:
                    continue
                if not row.vol_ok:
                    continue
                
                # Store signal for multi-RR testing
                all_signals.append({
                    'df': df,
                    'signal_idx': i,
                    'side': sig['side'],
                    'atr': atr
                })
                last_trade_idx = i
            
            if (idx + 1) % 20 == 0:
                print(f"[{idx+1}/{NUM_SYMBOLS}] {processed} processed | Signals: {len(all_signals)}")
            
            time.sleep(0.03)
            
        except Exception as e:
            continue
    
    print(f"\n📊 Collected {len(all_signals)} signals from {processed} symbols")
    print(f"⏱️ Data fetch completed in {(time.time() - start_time)/60:.1f} minutes")
    
    # Second pass: simulate each R:R
    print("\n" + "=" * 80)
    print("📊 SIMULATING DIFFERENT R:R RATIOS")
    print("=" * 80)
    
    results = {}
    
    for rr in RR_RATIOS:
        wins = 0
        losses = 0
        bars_win = []
        bars_loss = []
        
        for sig in all_signals:
            trade = simulate_trade(sig['df'], sig['signal_idx'], sig['side'], sig['atr'], tp_mult=rr)
            
            if trade['outcome'] == 'win':
                wins += 1
                bars_win.append(trade['bars'])
            elif trade['outcome'] == 'loss':
                losses += 1
                bars_loss.append(trade['bars'])
        
        total = wins + losses
        wr = wins / total if total > 0 else 0
        ev = calc_ev(wr, rr)
        lb = wilson_lower_bound(wins, total)
        
        results[rr] = {
            'wins': wins,
            'losses': losses,
            'total': total,
            'wr': wr,
            'lb': lb,
            'ev': ev,
            'avg_bars_win': np.mean(bars_win) if bars_win else 0,
            'avg_bars_loss': np.mean(bars_loss) if bars_loss else 0
        }
    
    # Print comparison table
    print("\n" + "-" * 80)
    print(f"{'R:R':<8} {'Trades':<10} {'Wins':<8} {'Losses':<8} {'WR%':<10} {'LB%':<10} {'EV':<10} {'Bars Win':<10} {'Bars Loss':<10}")
    print("-" * 80)
    
    best_ev = -999
    best_rr = None
    
    for rr in RR_RATIOS:
        r = results[rr]
        emoji = "✅" if r['ev'] > 0.5 else "⚠️" if r['ev'] > 0 else "❌"
        
        print(f"{rr}:1{'':<4} {r['total']:<10} {r['wins']:<8} {r['losses']:<8} {r['wr']*100:.1f}%{'':<4} {r['lb']*100:.1f}%{'':<4} {r['ev']:+.2f}{'':<4} {r['avg_bars_win']:.1f}{'':<6} {r['avg_bars_loss']:.1f}{'':<6} {emoji}")
        
        if r['ev'] > best_ev:
            best_ev = r['ev']
            best_rr = rr
    
    # Summary
    print("\n" + "=" * 80)
    print("💡 SUMMARY")
    print("=" * 80)
    
    print(f"\n🏆 Best R:R by EV: **{best_rr}:1** with EV = {best_ev:+.2f}")
    
    # Trade-off analysis
    print("\n📊 Trade-off Analysis:")
    for rr in RR_RATIOS:
        r = results[rr]
        print(f"  {rr}:1 → WR={r['wr']*100:.1f}%, EV={r['ev']:+.2f}, Avg Win Time={r['avg_bars_win']:.0f} bars")
    
    print("\n💡 Note:")
    print("  - Higher R:R = Lower WR but bigger wins")
    print("  - Lower R:R = Higher WR but smaller wins")
    print("  - EV accounts for both WR and R:R")

if __name__ == "__main__":
    run_rr_comparison()
