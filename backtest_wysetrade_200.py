#!/usr/bin/env python3
"""
WYSETRADE 200 SCALE TEST (5M NY Session)
========================================

Goal: Validate the "5M NY Session" Edge across the Top 200 Crypto Assets.
Output: A filtered list of symbols that are profitable with this strategy.

Strategy:
- Timeframe: 5M
- Hours: 13:00 - 21:00 UTC
- Logic: RSI Div + Key Level + Structure Break
- Risk: 2:1 R:R, Trailing SL

Author: AutoBot Architect
"""

import pandas as pd
import numpy as np
import requests
import time
import sys
import yaml

# ============================================
# CONFIGURATION
# ============================================

DAYS = 60 # 2 months
TIMEFRAME = 5
WIN_COST = 0.0006
LOSS_COST = 0.00125

# ============================================
# DATA ENGINE
# ============================================

def get_top_200_symbols():
    try:
        url = "https://api.bybit.com/v5/market/tickers"
        params = {'category': 'linear'}
        r = requests.get(url, params=params).json()
        if r['retCode'] != 0: return []
        
        tickers = r['result']['list']
        # Filter USDT perps only
        usdt_pairs = [t for t in tickers if t['symbol'].endswith('USDT') and '24' not in t['symbol']]
        
        # Sort by turnover
        usdt_pairs.sort(key=lambda x: float(x.get('turnover24h', 0)), reverse=True)
        
        # Return top 200 symbols
        return [t['symbol'] for t in usdt_pairs[:200]]
    except Exception as e:
        print(f"Error fetching symbols: {e}")
        return []

def fetch_data(symbol):
    try:
        url = "https://api.bybit.com/v5/market/kline"
        all_kline = []
        end_ts = int(time.time() * 1000)
        start_ts = int((time.time() - DAYS * 24 * 3600) * 1000)
        
        # Optimization: Fetch only necessary hours? Difficult with API. Fetch all.
        while end_ts > start_ts:
            params = {'category': 'linear', 'symbol': symbol, 'interval': str(TIMEFRAME), 'limit': 1000, 'end': end_ts}
            r = requests.get(url, params=params).json()
            if r['retCode'] != 0 or not r['result']['list']: break
            klines = r['result']['list']
            all_kline.extend(klines)
            end_ts = int(klines[-1][0]) - 1
            time.sleep(0.05) # Rate limit
        
        if not all_kline: return pd.DataFrame()
        
        df = pd.DataFrame(all_kline, columns=['ts', 'open', 'high', 'low', 'close', 'vol', 'to'])
        df = df.iloc[::-1].reset_index(drop=True)
        for c in ['open', 'high', 'low', 'close', 'vol']: df[c] = df[c].astype(float)
        
        # Datetime
        df['datetime'] = pd.to_datetime(df['ts'].astype(float), unit='ms')
        df['hour'] = df['datetime'].dt.hour
        
        return df
    except: return pd.DataFrame()

def prepare_indicators(df):
    df = df.copy()
    close = df['close']
    
    # RSI
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 1e-9)
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # Key Levels
    df['key_sup'] = df['low'].rolling(200).min()
    df['key_res'] = df['high'].rolling(200).max()
    
    # ADX
    h, l = df['high'], df['low']
    c_prev = close.shift(1)
    tr = pd.concat([h-l, (h-c_prev).abs(), (l-c_prev).abs()], axis=1).max(axis=1)
    plus_dm = h.diff()
    minus_dm = l.diff()
    plus_dm = np.where((plus_dm > minus_dm) & (plus_dm > 0), plus_dm, 0.0)
    minus_dm = np.where((minus_dm > plus_dm) & (minus_dm > 0), minus_dm, 0.0)
    tr_s = tr.rolling(14).sum()
    plus_di = 100 * (pd.Series(plus_dm).rolling(14).sum() / tr_s)
    minus_di = 100 * (pd.Series(minus_dm).rolling(14).sum() / tr_s)
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di).abs()) * 100
    df['adx'] = dx.rolling(14).mean()
    
    # Swing Points for Confirmation
    df['swing_high'] = df['high'].rolling(15).max()
    df['swing_low'] = df['low'].rolling(15).min()
    
    return df

# ============================================
# STRATEGY: NY SESSION WYSETRADE
# ============================================

def run_strategy(df):
    trades = []
    
    # NY SESSION: 13 to 21 UTC
    # ADX > 20
    
    potential_long = None
    potential_short = None
    rr = 2.0
    
    for i in range(200, len(df)-1):
        row = df.iloc[i]
        
        # TIME FILTER
        if not (13 <= row['hour'] < 21): 
            continue
            
        # ADX FILTER
        if row['adx'] < 20: continue
        
        # Setup
        # Strict Key Level Check (<1%)
        bull_div = (row['low'] <= df['low'].iloc[i-10:i].min() and 
                    row['rsi'] > df['rsi'].iloc[i-10:i].min() and 
                    row['rsi'] < 40 and
                    abs(row['low'] - row['key_sup'])/row['key_sup'] < 0.01)
                    
        bear_div = (row['high'] >= df['high'].iloc[i-10:i].max() and 
                    row['rsi'] < df['rsi'].iloc[i-10:i].max() and 
                    row['rsi'] > 60 and
                    abs(row['high'] - row['key_res'])/row['key_res'] < 0.01)
        
        if bull_div: potential_long = {'idx': i, 'trigger': df['swing_high'].iloc[i], 'sl': row['low']}
        if bear_div: potential_short = {'idx': i, 'trigger': df['swing_low'].iloc[i], 'sl': row['high']}
        
        # Trigger
        entry = 0; sl = 0; side = None
        if potential_long:
            if i - potential_long['idx'] > 20: potential_long = None
            elif row['close'] > potential_long['trigger']:
                side = 'long'; entry = df.iloc[i+1]['open']; sl = potential_long['sl']; potential_long = None
        elif potential_short:
            if i - potential_short['idx'] > 20: potential_short = None
            elif row['close'] < potential_short['trigger']:
                side = 'short'; entry = df.iloc[i+1]['open']; sl = potential_short['sl']; potential_short = None
                
        if side:
            risk = abs(entry-sl)
            if risk == 0 or risk/entry > 0.04: continue
            
            risk_pct = risk/entry
            tp_dist = risk*rr
            tp_price = entry+tp_dist if side=='long' else entry-tp_dist
            
            outcome = 'loss'
            curr_sl = sl
            be_hit = False
            
            # 500 bar hold (~2 days) max
            for j in range(i+1, min(i+500, len(df))):
                c = df.iloc[j]
                if side == 'long':
                    if c['low'] <= curr_sl: outcome = 'loss' if not be_hit else 'be'; break
                    if c['high'] >= entry+risk: curr_sl = entry; be_hit = True # BE
                    if c['high'] >= tp_price: outcome = 'win'; break
                else:
                    if c['high'] >= curr_sl: outcome = 'loss' if not be_hit else 'be'; break
                    if c['low'] <= entry-risk: curr_sl = entry; be_hit = True
                    if c['low'] <= tp_price: outcome = 'win'; break
            
            res_r = 0
            if outcome == 'win': res_r = rr - (WIN_COST / risk_pct)
            elif outcome == 'loss': res_r = -1.0 - (LOSS_COST / risk_pct)
            elif outcome == 'be': res_r = 0.0 - (WIN_COST / risk_pct)
            
            trades.append(res_r)
            
    return trades

# ============================================
# MAIN
# ============================================

def main():
    print("🚀 WYSETRADE 200 (5M NY Session)")
    print("Fetching top 200 symbols...")
    symbols = get_top_200_symbols()
    print(f"Found {len(symbols)} symbols.")
    print("-" * 65)
    print(f"{'Symbol':<12} | {'Trades':<8} {'Net R':<8} {'Avg R':<8}")
    print("-" * 65)
    
    results = []
    
    for i, sym in enumerate(symbols):
        sys.stdout.write(f"\rProcessing {i+1}/{len(symbols)}: {sym:<10}")
        sys.stdout.flush()
        
        df = fetch_data(sym)
        if len(df) < 1000: continue
        
        df = prepare_indicators(df)
        trades = run_strategy(df)
        
        if not trades: continue
        
        net_r = sum(trades)
        count = len(trades)
        avg_r = net_r / count
        
        results.append({
            'symbol': sym,
            'trades': count,
            'net_r': net_r,
            'avg_r': avg_r
        })
        
        # Live log profitable ones
        if net_r > 0:
            print(f"\r{sym:<12} | {count:<8} {net_r:<8.1f} {avg_r:<8.3f} ✅")
            
    print("\n" + "="*65)
    print("🏆 FINAL RESULTS")
    print("="*65)
    
    profitable = [r for r in results if r['net_r'] > 0]
    total_trades = sum(r['trades'] for r in results)
    total_r = sum(r['net_r'] for r in results)
    
    print(f"Total Symbols Tested: {len(results)}")
    print(f"Profitable Symbols: {len(profitable)}")
    print(f"Total Trades: {total_trades}")
    print(f"Total Net R: {total_r:.1f}R")
    print(f"Overall Expectancy: {total_r/total_trades:.3f}R")
    
    # Save Profitable List
    if profitable:
        profitable.sort(key=lambda x: x['net_r'], reverse=True)
        top_symbols = [r['symbol'] for r in profitable]
        
        with open('ny_session_golden_list.yaml', 'w') as f:
            yaml.dump({'symbols': top_symbols}, f)
        print(f"\n📁 Validated symbol list saved to 'ny_session_golden_list.yaml'.")

if __name__ == "__main__":
    main()
