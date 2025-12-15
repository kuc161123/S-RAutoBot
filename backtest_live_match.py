#!/usr/bin/env python3
"""
LIVE-MATCHING BACKTEST
======================
This backtest is designed to EXACTLY match the live bot's behavior:
- 25-hour timeout (100 bars), excluded from WR
- 10-bar cooldown per symbol
- Pivot-based SL with 0.3-2.0x ATR constraints
- 3:1 R:R
- Volume filter: vol > 0.5 * vol_ma(20)
- Entry on NEXT candle open
- SL checked FIRST if both hit same candle

Run this to get the TRUE expected performance to compare with live.
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
# CONFIGURATION - MATCHES LIVE BOT EXACTLY
# =============================================================================

TIMEFRAME = '15'
DATA_DAYS = 60
NUM_SYMBOLS = 200  # Top 200 by volume (same as live)

# NO slippage/fee adjustment - real Bybit handles this
APPLY_SLIPPAGE = False

RSI_PERIOD = 14
RSI_OVERSOLD = 30
RSI_OVERBOUGHT = 70
LOOKBACK_BARS = 14
MIN_PIVOT_DISTANCE = 5
PIVOT_LEFT = 3
PIVOT_RIGHT = 3

# SL constraints (match live)
MIN_SL_ATR = 0.3
MAX_SL_ATR = 2.0

# R:R (match live)
RR_RATIO = 3.0

# Cooldown (match live)
COOLDOWN_BARS = 10  # Per symbol (not per combo)

# Timeout (match live)
TIMEOUT_BARS = 100  # ~25 hours on 15min candles

# Swing lookback for pivot SL
SWING_LOOKBACK = 15

BASE_URL = "https://api.bybit.com"

# =============================================================================
# HELPERS
# =============================================================================

def wilson_lb(wins, n, z=1.96):
    if n == 0: return 0.0
    p = wins / n
    denom = 1 + z*z/n
    centre = p + z*z/(2*n)
    spread = z * math.sqrt((p*(1-p) + z*z/(4*n)) / n)
    return max(0, (centre - spread) / denom)

def get_symbols(limit=200):
    resp = requests.get(f"{BASE_URL}/v5/market/tickers?category=linear")
    tickers = resp.json().get('result', {}).get('list', [])
    usdt = [t for t in tickers if t['symbol'].endswith('USDT')]
    usdt.sort(key=lambda x: float(x.get('turnover24h', 0)), reverse=True)
    return [t['symbol'] for t in usdt[:limit]]

def fetch_klines(symbol, interval, days):
    end_ts = int(datetime.now().timestamp() * 1000)
    all_candles = []
    current_end = end_ts
    candles_needed = days * 24 * 60 // int(interval)
    
    while len(all_candles) < candles_needed:
        params = {'category': 'linear', 'symbol': symbol, 'interval': interval, 'limit': 1000, 'end': current_end}
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

# =============================================================================
# DIVERGENCE DETECTION (matches live divergence_detector.py)
# =============================================================================

def detect_divergence_signals(df):
    if len(df) < 100: return []
    
    close = df['close'].values
    rsi = df['rsi'].values
    n = len(df)
    
    price_ph, price_pl = find_pivots(close, PIVOT_LEFT, PIVOT_RIGHT)
    signals = []
    
    for i in range(30, n - 5):
        curr_pl = curr_pli = prev_pl = prev_pli = None
        for j in range(i, max(i - LOOKBACK_BARS, 0), -1):
            if not np.isnan(price_pl[j]):
                if curr_pl is None: curr_pl, curr_pli = price_pl[j], j
                elif prev_pl is None and j < curr_pli - MIN_PIVOT_DISTANCE: 
                    prev_pl, prev_pli = price_pl[j], j
                    break
        
        curr_ph = curr_phi = prev_ph = prev_phi = None
        for j in range(i, max(i - LOOKBACK_BARS, 0), -1):
            if not np.isnan(price_ph[j]):
                if curr_ph is None: curr_ph, curr_phi = price_ph[j], j
                elif prev_ph is None and j < curr_phi - MIN_PIVOT_DISTANCE: 
                    prev_ph, prev_phi = price_ph[j], j
                    break
        
        # Regular Bullish: Price LL, RSI HL
        if curr_pl and prev_pl and curr_pl < prev_pl and rsi[curr_pli] > rsi[prev_pli] and rsi[i] < RSI_OVERSOLD + 15:
            signals.append({'idx': i, 'side': 'long', 'type': 'regular_bullish'})
            continue
        # Regular Bearish: Price HH, RSI LH
        if curr_ph and prev_ph and curr_ph > prev_ph and rsi[curr_phi] < rsi[prev_phi] and rsi[i] > RSI_OVERBOUGHT - 15:
            signals.append({'idx': i, 'side': 'short', 'type': 'regular_bearish'})
            continue
        # Hidden Bullish: Price HL, RSI LL
        if curr_pl and prev_pl and curr_pl > prev_pl and rsi[curr_pli] < rsi[prev_pli] and rsi[i] < RSI_OVERBOUGHT - 10:
            signals.append({'idx': i, 'side': 'long', 'type': 'hidden_bullish'})
            continue
        # Hidden Bearish: Price LH, RSI HH
        if curr_ph and prev_ph and curr_ph < prev_ph and rsi[curr_phi] > rsi[prev_phi] and rsi[i] > RSI_OVERSOLD + 10:
            signals.append({'idx': i, 'side': 'short', 'type': 'hidden_bearish'})
    
    return signals

# =============================================================================
# PIVOT SL/TP CALCULATION (matches live execute_divergence_trade)
# =============================================================================

def calc_pivot_sltp(rows, idx, side, atr):
    """Calculate pivot-based SL and 3:1 TP (matches live bot exactly)"""
    # Entry on NEXT candle open
    entry = rows[idx + 1].open if idx + 1 < len(rows) else rows[idx].close
    
    # Find swing high/low for SL
    start_lookback = max(0, idx - SWING_LOOKBACK)
    
    if side == 'long':
        swing_low = min(rows[j].low for j in range(start_lookback, idx + 1))
        sl = swing_low
        sl_distance = abs(entry - sl)
    else:
        swing_high = max(rows[j].high for j in range(start_lookback, idx + 1))
        sl = swing_high
        sl_distance = abs(sl - entry)
    
    # Apply min/max constraints
    min_sl = MIN_SL_ATR * atr
    max_sl = MAX_SL_ATR * atr
    
    if sl_distance < min_sl:
        sl_distance = min_sl
        if side == 'long':
            sl = entry - sl_distance
        else:
            sl = entry + sl_distance
    elif sl_distance > max_sl:
        sl_distance = max_sl
        if side == 'long':
            sl = entry - sl_distance
        else:
            sl = entry + sl_distance
    
    # Calculate TP with R:R ratio
    if side == 'long':
        tp = entry + (RR_RATIO * sl_distance)
    else:
        tp = entry - (RR_RATIO * sl_distance)
    
    return entry, sl, tp, sl_distance / atr

# =============================================================================
# TRADE SIMULATION (matches live behavior)
# =============================================================================

def simulate_trade(rows, signal_idx, side, sl, tp, entry):
    """Simulate trade with SL checked FIRST on same candle"""
    start_idx = signal_idx + 1
    if start_idx >= len(rows) - TIMEOUT_BARS:
        return 'timeout', 0
    
    for bar_idx in range(1, TIMEOUT_BARS + 1):
        if start_idx + bar_idx >= len(rows):
            return 'timeout', bar_idx
        
        row = rows[start_idx + bar_idx]
        
        # SL checked FIRST (matches live)
        if side == 'long':
            if row.low <= sl:
                return 'loss', bar_idx
            if row.high >= tp:
                return 'win', bar_idx
        else:
            if row.high >= sl:
                return 'loss', bar_idx
            if row.low <= tp:
                return 'win', bar_idx
    
    return 'timeout', TIMEOUT_BARS

# =============================================================================
# MAIN
# =============================================================================

def run_backtest():
    print("=" * 80)
    print("🔬 LIVE-MATCHING BACKTEST")
    print("=" * 80)
    print("\nConfiguration (MATCHES LIVE BOT):")
    print(f"  • Timeout: {TIMEOUT_BARS} bars (~{TIMEOUT_BARS * 15 / 60:.0f}h), EXCLUDED from WR")
    print(f"  • Cooldown: {COOLDOWN_BARS} bars per symbol (~{COOLDOWN_BARS * 15 / 60:.1f}h)")
    print(f"  • SL: Pivot-based ({MIN_SL_ATR}-{MAX_SL_ATR}×ATR)")
    print(f"  • R:R: {RR_RATIO}:1")
    print(f"  • Symbols: {NUM_SYMBOLS} (by volume)")
    print(f"  • Days: {DATA_DAYS}")
    print("=" * 80)
    
    symbols = get_symbols(NUM_SYMBOLS)
    print(f"\n📦 Fetching data for {len(symbols)} symbols...\n")
    
    # Stats
    wins = 0
    losses = 0
    timeouts = 0
    skipped_cooldown = 0
    skipped_volume = 0
    
    by_type = defaultdict(lambda: {'w': 0, 'l': 0})
    
    start = time.time()
    
    for idx, sym in enumerate(symbols):
        try:
            df = fetch_klines(sym, TIMEFRAME, DATA_DAYS)
            if df.empty or len(df) < 400: 
                continue
            
            # Calculate indicators (same as live)
            df['rsi'] = calculate_rsi(df['close'], RSI_PERIOD)
            hl = df['high'] - df['low']
            hc = abs(df['high'] - df['close'].shift())
            lc = abs(df['low'] - df['close'].shift())
            df['atr'] = pd.concat([hl, hc, lc], axis=1).max(axis=1).rolling(14).mean()
            df['vol_ma'] = df['volume'].rolling(20).mean()
            df['vol_ok'] = df['volume'] > df['vol_ma'] * 0.5
            df = df.dropna()
            
            if len(df) < 150: 
                continue
            
            signals = detect_divergence_signals(df)
            rows = list(df.itertuples())
            last_signal_idx = -COOLDOWN_BARS - 1  # Per symbol cooldown
            
            for sig in signals:
                i = sig['idx']
                
                # COOLDOWN: 10 bars per symbol (matches live)
                if i - last_signal_idx < COOLDOWN_BARS:
                    skipped_cooldown += 1
                    continue
                
                if i >= len(rows) - TIMEOUT_BARS:
                    continue
                
                row = rows[i]
                if pd.isna(row.atr) or row.atr <= 0:
                    continue
                
                # VOLUME FILTER (matches live)
                if not row.vol_ok:
                    skipped_volume += 1
                    continue
                
                # Calculate SL/TP
                entry, sl, tp, sl_atr = calc_pivot_sltp(rows, i, sig['side'], row.atr)
                
                # Simulate trade
                result, bars = simulate_trade(rows, i, sig['side'], sl, tp, entry)
                
                if result == 'win':
                    wins += 1
                    by_type[sig['type']]['w'] += 1
                elif result == 'loss':
                    losses += 1
                    by_type[sig['type']]['l'] += 1
                else:
                    timeouts += 1  # EXCLUDED from WR (matches live)
                
                last_signal_idx = i
            
            if (idx + 1) % 25 == 0:
                total = wins + losses
                wr = wins / total * 100 if total > 0 else 0
                print(f"  [{idx+1}/{len(symbols)}] W/L: {wins}/{losses} | WR: {wr:.1f}% | Timeouts: {timeouts}")
            
            time.sleep(0.02)
        except Exception as e:
            continue
    
    elapsed = time.time() - start
    
    # Results
    print("\n" + "=" * 80)
    print("📊 RESULTS (Live-Matching Parameters)")
    print("=" * 80)
    
    total = wins + losses
    wr = wins / total * 100 if total > 0 else 0
    ev = (wr/100 * RR_RATIO) - ((100-wr)/100 * 1) if total > 0 else 0
    lb_wr = wilson_lb(wins, total) * 100
    total_r = (wins * RR_RATIO) - losses
    
    print(f"\n📈 OVERALL PERFORMANCE")
    print(f"├ Trades: {total} (excl. {timeouts} timeouts)")
    print(f"├ Wins: {wins}")
    print(f"├ Losses: {losses}")
    print(f"├ Win Rate: {wr:.1f}% (LB: {lb_wr:.1f}%)")
    print(f"├ EV: {ev:+.2f}R per trade")
    print(f"├ Total P&L: {total_r:+,.0f}R")
    print(f"└ R:R: {RR_RATIO}:1")
    
    print(f"\n📊 BY DIVERGENCE TYPE")
    for sig_type, stats in sorted(by_type.items()):
        t = stats['w'] + stats['l']
        if t > 0:
            type_wr = stats['w'] / t * 100
            print(f"├ {sig_type}: {stats['w']}W/{stats['l']}L = {type_wr:.1f}% WR")
    
    print(f"\n📉 FILTERED SIGNALS")
    print(f"├ Skipped (cooldown): {skipped_cooldown}")
    print(f"├ Skipped (volume): {skipped_volume}")
    print(f"└ Timeouts (excluded): {timeouts}")
    
    print(f"\n⏱️ Completed in {elapsed/60:.1f} minutes")
    
    # Summary for comparison
    print("\n" + "=" * 80)
    print("🎯 EXPECTED LIVE PERFORMANCE")
    print("=" * 80)
    print(f"\nIf live bot is working correctly, expect:")
    print(f"  • Win Rate: ~{wr:.0f}% (±5%)")
    print(f"  • EV: ~{ev:+.2f}R per trade")
    print(f"  • Trades/day: ~{total / DATA_DAYS:.0f}")
    
    return {
        'wins': wins,
        'losses': losses,
        'timeouts': timeouts,
        'wr': wr,
        'ev': ev,
        'total_r': total_r
    }

if __name__ == "__main__":
    run_backtest()
