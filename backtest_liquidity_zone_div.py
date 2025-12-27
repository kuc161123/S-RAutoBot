#!/usr/bin/env python3
"""
LIQUIDITY ZONE DIVERGENCE + BOS - 5M TIMEFRAME
==============================================
Hypothesis: Divergences are more reliable when they occur at significant 
liquidity zones (prior swing highs/lows, equal highs/lows).

Entry Logic:
1. Identify liquidity zones (clusters of prior swing highs/lows)
2. Divergence must occur AT or NEAR a liquidity zone
3. BOS confirms the reversal
4. Enter next candle after BOS

Liquidity Zone Detection:
- Prior swing highs within X ATR become resistance zones
- Prior swing lows within X ATR become support zones
- Equal highs/lows (within 0.1% tolerance) are stronger zones
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Config
TIMEFRAME = '5'
DATA_DAYS = 60
NUM_SYMBOLS = 40
MAX_WAIT_CANDLES = 10
SL_ATR_MULT = 0.8

# Test R:R
RR_RATIOS = [1.0, 1.5, 2.0, 2.5]

# Liquidity Zone Settings
ZONE_ATR_THRESHOLD = 1.5  # Zone must be within X ATR of current price
ZONE_LOOKBACK = 100  # Bars to look back for zones
ZONE_TOUCH_THRESHOLD = 0.002  # 0.2% - how close price must be to zone

# Costs
SLIPPAGE_PCT = 0.0003
FEE_PCT = 0.0006

RSI_PERIOD = 14
LOOKBACK_BARS = 10
MIN_PIVOT_DISTANCE = 3

BASE_URL = "https://api.bybit.com"

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

def add_indicators(df):
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(RSI_PERIOD).mean()
    avg_loss = loss.rolling(RSI_PERIOD).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    df['rsi'] = 100 - (100 / (1 + rs))
    
    hl = df['high'] - df['low']
    hc = abs(df['high'] - df['close'].shift())
    lc = abs(df['low'] - df['close'].shift())
    df['atr'] = pd.concat([hl, hc, lc], axis=1).max(axis=1).rolling(14).mean()
    
    df['ema200'] = df['close'].ewm(span=200, adjust=False).mean()
    df['vol_ma'] = df['volume'].rolling(20).mean()
    
    return df.dropna()

def find_pivots(data, left=3, right=3):
    n = len(data)
    pivot_highs = np.full(n, np.nan)
    pivot_lows = np.full(n, np.nan)
    pivot_high_vals = []
    pivot_low_vals = []
    
    for i in range(left, n - right):
        is_high = all(data[j] < data[i] for j in range(i - left, i + right + 1) if j != i)
        is_low = all(data[j] > data[i] for j in range(i - left, i + right + 1) if j != i)
        if is_high: 
            pivot_highs[i] = data[i]
            pivot_high_vals.append((i, data[i]))
        if is_low: 
            pivot_lows[i] = data[i]
            pivot_low_vals.append((i, data[i]))
    
    return pivot_highs, pivot_lows, pivot_high_vals, pivot_low_vals

def find_liquidity_zones(pivot_high_vals, pivot_low_vals, lookback_idx, atr):
    """Find liquidity zones from prior swing highs/lows."""
    zones = {'resistance': [], 'support': []}
    
    # Get recent pivots
    recent_highs = [(i, v) for i, v in pivot_high_vals if i < lookback_idx and i > lookback_idx - ZONE_LOOKBACK]
    recent_lows = [(i, v) for i, v in pivot_low_vals if i < lookback_idx and i > lookback_idx - ZONE_LOOKBACK]
    
    # Cluster nearby pivots into zones
    for _, val in recent_highs:
        zones['resistance'].append(val)
    
    for _, val in recent_lows:
        zones['support'].append(val)
    
    return zones

def is_at_liquidity_zone(price, zones, atr, zone_type='support'):
    """Check if price is at a liquidity zone."""
    zone_list = zones.get(zone_type, [])
    
    for zone_price in zone_list:
        # Price must be within threshold of zone
        dist = abs(price - zone_price) / price
        if dist < ZONE_TOUCH_THRESHOLD:
            return True
        # Or within ATR range
        if abs(price - zone_price) < atr * ZONE_ATR_THRESHOLD:
            return True
    
    return False

def detect_divergences_at_liquidity(df, pivot_high_vals, pivot_low_vals):
    if len(df) < 100: return []
    
    close = df['close'].values
    high = df['high'].values
    low = df['low'].values
    rsi = df['rsi'].values
    atr = df['atr'].values
    n = len(df)
    
    price_ph, price_pl = find_pivots(close, 3, 3)[:2]
    signals = []
    
    for i in range(ZONE_LOOKBACK, n - 15):
        current_atr = atr[i]
        zones = find_liquidity_zones(pivot_high_vals, pivot_low_vals, i, current_atr)
        
        # Find pivot lows
        curr_pl = curr_pli = prev_pl = prev_pli = None
        for j in range(i, max(i - LOOKBACK_BARS, 0), -1):
            if not np.isnan(price_pl[j]):
                if curr_pl is None: curr_pl, curr_pli = price_pl[j], j
                elif prev_pl is None and j < curr_pli - MIN_PIVOT_DISTANCE:
                    prev_pl, prev_pli = price_pl[j], j
                    break
        
        # Find pivot highs
        curr_ph = curr_phi = prev_ph = prev_phi = None
        for j in range(i, max(i - LOOKBACK_BARS, 0), -1):
            if not np.isnan(price_ph[j]):
                if curr_ph is None: curr_ph, curr_phi = price_ph[j], j
                elif prev_ph is None and j < curr_phi - MIN_PIVOT_DISTANCE:
                    prev_ph, prev_phi = price_ph[j], j
                    break
        
        # Bullish divergence AT SUPPORT zone
        if curr_pl and prev_pl:
            if curr_pl < prev_pl and rsi[curr_pli] > rsi[prev_pli]:
                if rsi[i] < 40:
                    # Check if at support zone
                    at_support = is_at_liquidity_zone(low[i], zones, current_atr, 'support')
                    if at_support:
                        swing_high = max(high[max(0, i-LOOKBACK_BARS):i+1])
                        signals.append({'idx': i, 'side': 'long', 'swing': swing_high, 'at_zone': True})
                    else:
                        swing_high = max(high[max(0, i-LOOKBACK_BARS):i+1])
                        signals.append({'idx': i, 'side': 'long', 'swing': swing_high, 'at_zone': False})
                    continue
        
        # Bearish divergence AT RESISTANCE zone
        if curr_ph and prev_ph:
            if curr_ph > prev_ph and rsi[curr_phi] < rsi[prev_phi]:
                if rsi[i] > 60:
                    # Check if at resistance zone
                    at_resistance = is_at_liquidity_zone(high[i], zones, current_atr, 'resistance')
                    if at_resistance:
                        swing_low = min(low[max(0, i-LOOKBACK_BARS):i+1])
                        signals.append({'idx': i, 'side': 'short', 'swing': swing_low, 'at_zone': True})
                    else:
                        swing_low = min(low[max(0, i-LOOKBACK_BARS):i+1])
                        signals.append({'idx': i, 'side': 'short', 'swing': swing_low, 'at_zone': False})
    
    return signals

def run_backtest(df, signals, rr, only_at_zone=False):
    rows = list(df.itertuples())
    atr = df['atr'].values
    
    trades = []
    
    for sig in signals:
        # Filter by zone if required
        if only_at_zone and not sig.get('at_zone', False):
            continue
        
        div_idx = sig['idx']
        side = sig['side']
        
        bos_idx = None
        for j in range(div_idx + 1, min(div_idx + 1 + MAX_WAIT_CANDLES, len(rows) - 10)):
            if side == 'long':
                if rows[j].close > sig['swing']:
                    bos_idx = j
                    break
            else:
                if rows[j].close < sig['swing']:
                    bos_idx = j
                    break
        
        if bos_idx is None:
            continue
        
        entry_idx = bos_idx + 1
        if entry_idx >= len(rows) - 50:
            continue
        
        entry_price = rows[entry_idx].open
        if side == 'long':
            entry_price *= (1 + SLIPPAGE_PCT)
        else:
            entry_price *= (1 - SLIPPAGE_PCT)
        
        entry_atr = atr[entry_idx - 1]
        sl_dist = entry_atr * SL_ATR_MULT
        
        if side == 'long':
            sl = entry_price - sl_dist
            tp = entry_price + (sl_dist * rr)
        else:
            sl = entry_price + sl_dist
            tp = entry_price - (sl_dist * rr)
        
        result = None
        for k in range(entry_idx, min(entry_idx + 200, len(rows))):
            row = rows[k]
            if side == 'long':
                if row.low <= sl: result = 'loss'; break
                if row.high >= tp: result = 'win'; break
            else:
                if row.high >= sl: result = 'loss'; break
                if row.low <= tp: result = 'win'; break
        
        if result:
            risk_pct = sl_dist / entry_price
            fee_cost = (FEE_PCT * 2 + SLIPPAGE_PCT) / risk_pct
            r = (rr - fee_cost) if result == 'win' else (-1.0 - fee_cost)
            trades.append({'result': result, 'r': r, 'at_zone': sig.get('at_zone', False)})
    
    return trades

def main():
    print("="*80)
    print("LIQUIDITY ZONE DIVERGENCE + BOS - 5M TIMEFRAME")
    print("="*80)
    print(f"Hypothesis: Divergences at liquidity zones are more reliable")
    print(f"Data: {DATA_DAYS} days, {NUM_SYMBOLS} symbols")
    print("="*80)
    
    symbols = get_symbols(NUM_SYMBOLS)
    print(f"Fetching data for {len(symbols)} symbols...")
    
    symbol_data = {}
    for idx, sym in enumerate(symbols):
        df = fetch_klines(sym, TIMEFRAME, DATA_DAYS)
        if df.empty or len(df) < 500:
            continue
        df = add_indicators(df)
        if len(df) < ZONE_LOOKBACK + 50:
            continue
        
        # Get pivots for zone detection
        close = df['close'].values
        _, _, ph_vals, pl_vals = find_pivots(close, 3, 3)
        
        signals = detect_divergences_at_liquidity(df, ph_vals, pl_vals)
        if signals:
            symbol_data[sym] = (df, signals)
        
        if (idx + 1) % 10 == 0:
            print(f"  {idx+1}/{len(symbols)} symbols...")
    
    print(f"\nSymbols with signals: {len(symbol_data)}")
    total_signals = sum(len(sig) for df, sig in symbol_data.values())
    at_zone_signals = sum(sum(1 for s in sig if s.get('at_zone')) for df, sig in symbol_data.values())
    print(f"Total signals: {total_signals}")
    print(f"Signals AT liquidity zones: {at_zone_signals} ({at_zone_signals/total_signals*100:.1f}%)")
    
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    print(f"{'Filter':<25} | {'RR':<4} | {'N':<6} | {'WR':<7} | {'Avg R':<10} | {'Status'}")
    print("-"*80)
    
    results = []
    
    for rr in RR_RATIOS:
        # All divergences
        all_trades = []
        for sym, (df, signals) in symbol_data.items():
            trades = run_backtest(df, signals, rr, only_at_zone=False)
            all_trades.extend(trades)
        
        if all_trades:
            wins = sum(1 for t in all_trades if t['result'] == 'win')
            n = len(all_trades)
            wr = wins / n * 100
            total_r = sum(t['r'] for t in all_trades)
            avg_r = total_r / n
            status = "✅" if avg_r > 0 else "❌"
            print(f"{'All Divergences':<25} | {rr:<4.1f} | {n:<6} | {wr:>5.1f}% | {avg_r:>+8.4f}R | {status}")
            results.append({'name': 'All', 'rr': rr, 'n': n, 'wr': wr, 'avg_r': avg_r})
        
        # Only at liquidity zones
        zone_trades = []
        for sym, (df, signals) in symbol_data.items():
            trades = run_backtest(df, signals, rr, only_at_zone=True)
            zone_trades.extend(trades)
        
        if zone_trades:
            wins = sum(1 for t in zone_trades if t['result'] == 'win')
            n = len(zone_trades)
            wr = wins / n * 100
            total_r = sum(t['r'] for t in zone_trades)
            avg_r = total_r / n
            status = "✅ PROFIT" if avg_r > 0 else "❌"
            print(f"{'AT LIQUIDITY ZONE':<25} | {rr:<4.1f} | {n:<6} | {wr:>5.1f}% | {avg_r:>+8.4f}R | {status}")
            results.append({'name': 'At Zone', 'rr': rr, 'n': n, 'wr': wr, 'avg_r': avg_r})
        
        print("-"*80)
    
    print("="*80)
    
    # Summary
    profitable = [r for r in results if r['avg_r'] > 0]
    if profitable:
        print(f"\n🏆 FOUND {len(profitable)} PROFITABLE CONFIGURATIONS!")
        for r in profitable:
            print(f"  ✅ {r['name']} | RR={r['rr']} | {r['wr']:.1f}% WR | {r['avg_r']:+.4f}R avg ({r['n']} trades)")
    else:
        print("\n❌ NO PROFITABLE CONFIGURATIONS")
        # Show comparison
        zone_results = [r for r in results if r['name'] == 'At Zone']
        all_results = [r for r in results if r['name'] == 'All']
        if zone_results and all_results:
            print("\nZone filter improvement:")
            for z, a in zip(zone_results, all_results):
                if z['rr'] == a['rr']:
                    wr_diff = z['wr'] - a['wr']
                    print(f"  RR {z['rr']}: Zone {z['wr']:.1f}% vs All {a['wr']:.1f}% ({wr_diff:+.1f}% WR diff)")

if __name__ == "__main__":
    main()
