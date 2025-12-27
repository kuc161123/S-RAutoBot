#!/usr/bin/env python3
"""
WYSETRADE SCALE TEST (30M Strict)
=================================

Goal: Verify if the Profitable Strict Wysetrade Logic holds up on a wider basket of Alts.
Hypothesis: We can achieve high trade frequency by trading 30-50 pairs instead of relaxing filters.

Universe: Top 30 Liquid Pairs
Timeframe: 30M
Strategy: Strict Key Level + Structure Break.
Fee Model: Precise Bybit.

Author: AutoBot Architect
"""

import pandas as pd
import numpy as np
import requests
import time
import sys

# ============================================
# CONFIGURATION
# ============================================

# Top 30 Liquid Pairs
SYMBOLS = [
    'BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'BNBUSDT', 'XRPUSDT', 
    'ADAUSDT', 'DOGEUSDT', 'AVAXUSDT', 'LINKUSDT', 'DOTUSDT',
    'LTCUSDT', 'MATICUSDT', 'UNIUSDT', 'ATOMUSDT', 'IMXUSDT',
    'NEARUSDT', 'ETCUSDT', 'FILUSDT', 'HBARUSDT', 'APTUSDT',
    'ARB', 'OP', 'SUI', 'INJ', 'RNDR', # Need to append USDT
    'FET', 'AGIX', 'PEPE', 'WLD', 'TIA'
]
SYMBOLS = [s if 'USDT' in s else s+'USDT' for s in SYMBOLS]

TIMEFRAME = 30
DAYS = 90

WIN_COST = 0.0006
LOSS_COST = 0.00125

# ============================================
# LOGIC
# ============================================

def fetch_data(symbol):
    try:
        url = "https://api.bybit.com/v5/market/kline"
        all_kline = []
        end_ts = int(time.time() * 1000)
        start_ts = int((time.time() - DAYS * 24 * 3600) * 1000)
        while end_ts > start_ts:
            params = {'category': 'linear', 'symbol': symbol, 'interval': str(TIMEFRAME), 'limit': 1000, 'end': end_ts}
            r = requests.get(url, params=params).json()
            if r['retCode'] != 0 or not r['result']['list']: break
            klines = r['result']['list']
            all_kline.extend(klines)
            end_ts = int(klines[-1][0]) - 1
            time.sleep(0.05)
        df = pd.DataFrame(all_kline, columns=['ts', 'open', 'high', 'low', 'close', 'vol', 'to'])
        df = df.iloc[::-1].reset_index(drop=True)
        for c in ['open', 'high', 'low', 'close', 'vol']: df[c] = df[c].astype(float)
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
    
    # Key Levels (Strict 200)
    df['key_sup'] = df['low'].rolling(200).min()
    df['key_res'] = df['high'].rolling(200).max()
    
    # Structure Points (15 bar lookback)
    df['swing_high'] = df['high'].rolling(15).max()
    df['swing_low'] = df['low'].rolling(15).min()
    
    return df

def run_strict_wysetrade(df, rr=2.0):
    trades = []
    potential_long = None
    potential_short = None
    
    for i in range(200, len(df)-1):
        row = df.iloc[i]
        
        # Setup: Div + Strict Key Level (< 1% dist check)
        # Using exact Key Level proximity
        
        bull_div = (row['low'] <= df['low'].iloc[i-10:i].min() and 
                    row['rsi'] > df['rsi'].iloc[i-10:i].min() and 
                    row['rsi'] < 45)
                    
        bear_div = (row['high'] >= df['high'].iloc[i-10:i].max() and 
                    row['rsi'] < df['rsi'].iloc[i-10:i].max() and 
                    row['rsi'] > 55)
        
        near_sup = abs(row['low'] - row['key_sup']) / row['key_sup'] < 0.01
        near_res = abs(row['high'] - row['key_res']) / row['key_res'] < 0.01
        
        if bull_div and near_sup:
            potential_long = {'idx': i, 'trigger': df['swing_high'].iloc[i], 'sl': row['low']}
            potential_short = None
            
        if bear_div and near_res:
            potential_short = {'idx': i, 'trigger': df['swing_low'].iloc[i], 'sl': row['high']}
            potential_long = None
            
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
            # Check Risk
            if entry == sl or abs(entry-sl)/entry > 0.04: continue 
            
            risk_dist = abs(entry-sl)
            risk_pct = risk_dist/entry
            tp_dist = risk_dist * rr
            tp_price = entry + tp_dist if side == 'long' else entry - tp_dist
            
            outcome = 'loss'
            be_hit = False
            curr_sl = sl
            
            for j in range(i+1, min(i+1000, len(df))):
                c = df.iloc[j]
                
                if side=='long':
                    if c['low'] <= curr_sl: outcome='loss' if not be_hit else 'be'; break
                    if c['high'] >= entry+risk_dist: curr_sl=entry; be_hit=True # BE
                    if c['high'] >= tp_price: outcome='win'; break
                else:
                    if c['high'] >= curr_sl: outcome='loss' if not be_hit else 'be'; break
                    if c['low'] <= entry-risk_dist: curr_sl=entry; be_hit=True
                    if c['low'] <= tp_price: outcome='win'; break
            
            res_r = 0
            if outcome == 'win': res_r = rr - (WIN_COST / risk_pct)
            elif outcome == 'loss': res_r = -1.0 - (LOSS_COST / risk_pct)
            elif outcome == 'be': res_r = 0.0 - (WIN_COST / risk_pct)
            elif outcome == 'timeout': res_r = -0.1
            
            trades.append(res_r)
            
    return trades

def main():
    print("🚀 WYSETRADE SCALE TEST (30 Symbols)")
    print("-" * 50)
    
    total_trades = 0
    total_r = 0
    
    for sym in SYMBOLS:
        sys.stdout.write(f"Testing {sym}... ")
        sys.stdout.flush()
        
        df = fetch_data(sym)
        if df.empty: 
            print("Skipped")
            continue
            
        df = prepare_indicators(df)
        trades = run_strict_wysetrade(df)
        
        if not trades:
            print("0 trades")
            continue
            
        sym_r = sum(trades)
        total_trades += len(trades)
        total_r += sym_r
        
        print(f"{len(trades)} trades | {sym_r:+.1f}R")
        
    print("\n" + "="*50)
    avg_r = total_r / total_trades if total_trades else 0
    print(f"TOTAL: {total_trades} trades")
    print(f"NET R: {total_r:+.1f}R")
    print(f"AVG R: {avg_r:+.3f}R")

if __name__ == "__main__":
    main()
