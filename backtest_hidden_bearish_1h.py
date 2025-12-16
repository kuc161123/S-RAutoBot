#!/usr/bin/env python3
"""
HIDDEN BEARISH 1H BACKTEST (Pivot SL + 3:1 R:R)
===============================================
Exact simulation of Live Bot logic:
- Timeframe: 60m (1H)
- Strategy: Hidden Bearish Divergence ONLY
- Stop Loss: Pivot Swing High (Min 0.3 ATR, Max 2.0 ATR)
- Take Profit: 3.0x Risk
- Entry: Next Candle Open
- Filter: Volume > 50% MA
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

TIMEFRAME = '60'       # 1H Timeframe
DATA_DAYS = 60         # 60 Days of data
NUM_SYMBOLS = 150      # Top 150 symbols by volume
TARGET_STRATEGY = 'hidden_bearish'  # ONLY this strategy

# Realistic Costs
SLIPPAGE_PCT = 0.0005  # 0.05%
FEE_PCT = 0.0004       # 0.04%
TOTAL_COST = (SLIPPAGE_PCT + FEE_PCT) * 2

# Strategy Settings (Live Bot Parity)
RSI_PERIOD = 14
RSI_OVERSOLD = 30
RSI_OVERBOUGHT = 70
LOOKBACK_BARS = 14
MIN_PIVOT_DISTANCE = 5

RISK_REWARD = 3.0      # 3:1 R:R
MIN_SL_ATR = 0.3
MAX_SL_ATR = 2.0

BASE_URL = "https://api.bybit.com"

# =============================================================================
# HELPERS
# =============================================================================

def calc_ev(wr: float, rr: float) -> float:
    return (wr * rr) - (1 - wr)

def get_symbols(limit: int = 150) -> list:
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

# =============================================================================
# LOGIC MATCHING LIVE BOT
# =============================================================================

def detect_divergence_signals(df):
    """Detect ONLY Hidden Bearish signals to match Live Bot"""
    if len(df) < 100: return []
    
    close = df['close'].values
    rsi = df['rsi'].values
    n = len(df)
    
    price_ph, price_pl = find_pivots(close, 3, 3)
    signals = []
    
    for i in range(30, n - 5):
        # Look for Hidden Bearish: Price LH, RSI HH
        # 1. Find Current Pivot High
        curr_ph = curr_phi = prev_ph = prev_phi = None
        
        # Scan backwards for pivots
        for j in range(i, max(i - LOOKBACK_BARS, 0), -1):
            if not np.isnan(price_ph[j]):
                if curr_ph is None: 
                    curr_ph, curr_phi = price_ph[j], j
                elif prev_ph is None and j < curr_phi - MIN_PIVOT_DISTANCE: 
                    prev_ph, prev_phi = price_ph[j], j
                    break
        
        # Check condition
        if curr_ph and prev_ph:
            # Price Lower High (Trend Continuation check?)
            # Standard Hidden Bearish: Price Lower High, RSI Higher High
            if curr_ph < prev_ph:  # Lower High
                if rsi[curr_phi] > rsi[prev_phi]: # Higher High in RSI
                   if rsi[i] > RSI_OVERSOLD + 10: # Sanity check (not too oversold)
                       signals.append({'idx': i, 'side': 'short', 'type': 'hidden_bearish', 'swing': curr_ph})

    return signals

def calc_pivot_sltp(rows, idx, side, atr, swing_price):
    """
    Calculate SL/TP using Pivot Logic (Parity with bot.py)
    """
    entry = rows[idx + 1].open if idx + 1 < len(rows) else rows[idx].close
    
    # 1. Base SL at Swing
    sl = swing_price
    
    # 2. Calculate Distance
    sl_dist = abs(entry - sl)
    
    # 3. Apply Min/Max ATR Clamps
    min_dist = MIN_SL_ATR * atr
    max_dist = MAX_SL_ATR * atr
    
    if sl_dist < min_dist: sl_dist = min_dist
    if sl_dist > max_dist: sl_dist = max_dist
    
    # 4. Final SL Price
    if side == 'long':
        sl = entry - sl_dist
    else: # short
        sl = entry + sl_dist
        
    # 5. TP at Reward Ratio
    processed_sl_dist = abs(entry - sl)
    if side == 'long':
        tp = entry + (processed_sl_dist * RISK_REWARD)
    else:
        tp = entry - (processed_sl_dist * RISK_REWARD)
        
    return entry, sl, tp

def simulate_trade(rows, signal_idx, side, sl, tp, entry):
    """
    Simulate trade candle-by-candle including the entry candle for immediate risk
    """
    entry_idx = signal_idx + 1
    if entry_idx >= len(rows) - 1:
        return 'timeout', 0
        
    # Apply entry slippage
    if side == 'long':
        entry = entry * (1 + SLIPPAGE_PCT)
        tp = tp * (1 - TOTAL_COST) # Net TP
    else:
        entry = entry * (1 - SLIPPAGE_PCT)
        tp = tp * (1 + TOTAL_COST) # Net TP
        
    # Check candles starting from entry candle
    # Note: On entry candle, we only check risk if Price moves against us AFTER open?
    # Conservative: Check High/Low of entry candle. If Low < SL, assuming we stopped out.
    # If High > TP, assume we won?
    # To be "Safer" (Conservative):
    # - If Low <= SL: LOSS (Hit stop)
    # - If High >= TP: WIN (Hit target)
    # - If BOTH: LOSS (Assume SL hit first or volatility kill)
    
    for bar_idx, row in enumerate(rows[entry_idx:entry_idx + 100]):
        outcome = None
        
        if side == 'long':
            hit_sl = row.low <= sl
            hit_tp = row.high >= tp
        else: # short
            hit_sl = row.high >= sl
            hit_tp = row.low <= tp
            
        if hit_sl and hit_tp:
            return 'loss', bar_idx # Assume worst case (stopped out)
        elif hit_sl:
            return 'loss', bar_idx
        elif hit_tp:
            return 'win', bar_idx
            
    return 'timeout', 100

# =============================================================================
# MAIN
# =============================================================================

def run():
    print("=" * 60)
    print("🐻 HIDDEN BEARISH 1H BACKTEST (Precision Mode)")
    print("=" * 60)
    print(f"Timeframe: {TIMEFRAME}m | R:R: {RISK_REWARD}:1 | SL: Pivot (ATR Buffered)")
    print(f"Slippage: {SLIPPAGE_PCT*100:.2f}% | Fees: {FEE_PCT*100:.2f}%")
    print("-" * 60)
    
    symbols = get_symbols(NUM_SYMBOLS)
    print(f"Testing {len(symbols)} symbols over {DATA_DAYS} days...")
    
    wins = 0
    losses = 0
    total_r = 0.0
    
    start_time = time.time()
    
    symbol_stats = []

    for idx, sym in enumerate(symbols):
        try:
            df = fetch_klines(sym, TIMEFRAME, DATA_DAYS)
            if df.empty or len(df) < 200: continue
            
            # Indicators
            df['rsi'] = calculate_rsi(df['close'], RSI_PERIOD)
            
            hl = df['high'] - df['low']
            hc = abs(df['high'] - df['close'].shift())
            lc = abs(df['low'] - df['close'].shift())
            df['atr'] = pd.concat([hl, hc, lc], axis=1).max(axis=1).rolling(14).mean()
            
            df['vol_ma'] = df['volume'].rolling(20).mean()
            df['vol_ok'] = df['volume'] > df['vol_ma'] * 0.5
            
            df = df.dropna()
            
            signals = detect_divergence_signals(df)
            rows = list(df.itertuples())
            last_trade_idx = -20
            
            sym_wins = 0
            sym_losses = 0
            sym_r = 0.0
            
            for sig in signals:
                i = sig['idx']
                if i - last_trade_idx < 10: continue # Cooldown
                if i >= len(rows) - 50: continue
                
                row = rows[i]
                if not row.vol_ok or row.atr <= 0: continue
                
                # Execute
                entry, sl, tp = calc_pivot_sltp(rows, i, 'short', row.atr, sig['swing'])
                
                res, bars = simulate_trade(rows, i, 'short', sl, tp, entry)
                
                if res == 'win':
                    wins += 1
                    sym_wins += 1
                    total_r += RISK_REWARD
                    sym_r += RISK_REWARD
                    last_trade_idx = i
                elif res == 'loss':
                    losses += 1
                    sym_losses += 1
                    total_r -= 1.0 # Risk is 1R
                    sym_r -= 1.0
                    last_trade_idx = i
            
            # Track symbol stats
            sym_total = sym_wins + sym_losses
            if sym_total > 0:
                symbol_stats.append({
                    'symbol': sym,
                    'wins': sym_wins,
                    'losses': sym_losses,
                    'total': sym_total,
                    'wr': sym_wins / sym_total,
                    'r_total': sym_r
                })
                    
        except Exception as e:
            continue
            
        if (idx+1) % 50 == 0:
            print(f"Processed {idx+1}...")
            
    total = wins + losses
    if total > 0:
        wr = wins / total
        ev = calc_ev(wr, RISK_REWARD)
        print("\n" + "=" * 60)
        print("📊 FINAL RESULTS")
        print("=" * 60)
        print(f"Total Trades: {total}")
        print(f"✅ Wins: {wins}")
        print(f"❌ Losses: {losses}")
        print(f"📈 Win Rate: {wr*100:.1f}%")
        print(f"💰 Total R: {total_r:+.1f}R")
        print(f"📊 EV: {ev:+.2f}R")
        print(f"⏱️ Trades/Day: {total/DATA_DAYS:.1f}")
        print("=" * 60)
        
        # Filter and Print Golden List
        print("\n🏆 GOLDEN CONFIGURATION (YAML)")
        print("Copy below into config.yaml -> divergence_symbols:")
        print("-" * 60)
        print("  divergence_symbols:")
        
        # Sort by WR desc
        symbol_stats.sort(key=lambda x: x['wr'], reverse=True)
        
        valid_count = 0
        for s in symbol_stats:
            if s['wr'] >= 0.40 and s['total'] >= 5: # Min 40% WR, Min 5 trades
                valid_count += 1
                comment = f"# {s['wr']*100:.1f}% WR, {s['total']} trades, {s['r_total']:+.1f}R"
                print(f"    - {s['symbol']:<12} {comment}")
        
        print("-" * 60)
        print(f"Total Valid Symbols: {valid_count}")
        
    else:
        print("No trades found.")

if __name__ == "__main__":
    run()
