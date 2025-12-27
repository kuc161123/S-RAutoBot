#!/usr/bin/env python3
"""
WYSETRADE 5M FINAL POLISH
=========================

Goal: Bridge the final 0.01R gap to profitability using ADX filter.
Base: Wysetrade (Standard, 2:1, Trailing) -> Currently -0.01R.
"""

import pandas as pd
import numpy as np
import requests
import time
import itertools

SYMBOLS = [
    'BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'BNBUSDT', 'XRPUSDT', 
    'ADAUSDT', 'DOGEUSDT', 'AVAXUSDT', 'LINKUSDT', 'DOTUSDT'
]

DAYS = 60
WIN_COST_PCT = 0.0006
LOSS_COST_PCT = 0.00125

def fetch_data(symbol: str) -> pd.DataFrame:
    try:
        url = "https://api.bybit.com/v5/market/kline"
        all_kline = []
        end_ts = int(time.time() * 1000)
        start_ts = int((time.time() - DAYS * 24 * 3600) * 1000)
        while end_ts > start_ts:
            params = {'category': 'linear', 'symbol': symbol, 'interval': '5', 'limit': 1000, 'end': end_ts}
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

def prepare_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    close = df['close']
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 1e-9)
    df['rsi'] = 100 - (100 / (1 + rs))
    df['key_sup'] = df['low'].rolling(200).min()
    df['key_res'] = df['high'].rolling(200).max()
    
    # ADX
    h, l = df['high'], df['low']
    c_prev = close.shift(1)
    tr = pd.concat([h-l, (h-c_prev).abs(), (l-c_prev).abs()], axis=1).max(axis=1)
    atr = tr.rolling(14).mean()
    plus_dm = h.diff()
    minus_dm = l.diff()
    plus_dm = np.where((plus_dm > minus_dm) & (plus_dm > 0), plus_dm, 0.0)
    minus_dm = np.where((minus_dm > plus_dm) & (minus_dm > 0), minus_dm, 0.0)
    tr_s = tr.rolling(14).sum()
    plus_di = 100 * (pd.Series(plus_dm).rolling(14).sum() / tr_s)
    minus_di = 100 * (pd.Series(minus_dm).rolling(14).sum() / tr_s)
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di).abs()) * 100
    df['adx'] = dx.rolling(14).mean()
    
    return df

def test_strategy(datasets, adx_min):
    total_r = 0
    trades_count = 0
    wins = 0
    rr = 2.0
    
    for sym, df in datasets.items():
        potential_long = None
        potential_short = None
        cooldown = 0
        
        for i in range(200, len(df)-1):
            if cooldown > 0:
                cooldown -= 1
                continue
            row = df.iloc[i]
            
            # ADX Filter
            if row['adx'] < adx_min: continue
            
            # Setup
            is_bull = (row['low'] <= df['low'].iloc[i-10:i].min() and row['rsi'] > df['rsi'].iloc[i-10:i].min() and row['rsi'] < 40 and abs(row['low'] - row['key_sup'])/row['key_sup'] < 0.01)
            is_bear = (row['high'] >= df['high'].iloc[i-10:i].max() and row['rsi'] < df['rsi'].iloc[i-10:i].max() and row['rsi'] > 60 and abs(row['high'] - row['key_res'])/row['key_res'] < 0.01)

            if is_bull: potential_long = {'idx': i, 'sl': row['low'], 'trigger': df['high'].iloc[i-15:i].max()}
            if is_bear: potential_short = {'idx': i, 'sl': row['high'], 'trigger': df['low'].iloc[i-15:i].min()}

            # Trigger
            entry = 0; sl = 0; tp = 0; side = None
            if potential_long:
                if i - potential_long['idx'] > 20: potential_long = None
                elif row['close'] > potential_long['trigger']:
                    side = 'long'; entry = df.iloc[i+1]['open']; sl = potential_long['sl']; potential_long = None
            elif potential_short:
                if i - potential_short['idx'] > 20: potential_short = None
                elif row['close'] < potential_short['trigger']:
                    side = 'short'; entry = df.iloc[i+1]['open']; sl = potential_short['sl']; potential_short = None
            
            if side:
                if entry == sl or abs(entry-sl)/entry > 0.02: continue
                risk_dist = abs(entry-sl)
                risk_pct = risk_dist/entry
                tp_dist = risk_dist * rr
                tp_price = entry + tp_dist if side == 'long' else entry - tp_dist
                
                # Sim with Trailing
                outcome = 'loss'
                curr_sl = sl
                be_hit = False
                
                for j in range(i+1, min(i+500, len(df))):
                    c = df.iloc[j]
                    if side == 'long':
                        if c['low'] <= curr_sl:
                            outcome = 'loss' if not be_hit else 'be'
                            break
                        if c['high'] >= entry + risk_dist: curr_sl = entry; be_hit = True # BE at 1R
                        if c['high'] >= tp_price: outcome = 'win'; break
                    else:
                        if c['high'] >= curr_sl:
                            outcome = 'loss' if not be_hit else 'be'
                            break
                        if c['low'] <= entry - risk_dist: curr_sl = entry; be_hit = True # BE at 1R
                        if c['low'] <= tp_price: outcome = 'win'; break
                
                res_r = 0
                if outcome == 'win': res_r = rr - (WIN_COST_PCT / risk_pct); wins += 1
                elif outcome == 'loss': res_r = -1.0 - (LOSS_COST_PCT / risk_pct)
                elif outcome == 'be': res_r = 0.0 - (WIN_COST_PCT / risk_pct)
                
                total_r += res_r
                trades_count += 1
                cooldown = 5
                
    return {'net': total_r, 'trades': trades_count, 'avg': total_r/trades_count if trades_count else 0}

def main():
    print("Loading...")
    datasets = {}
    for sym in SYMBOLS:
        df = fetch_data(sym)
        if not df.empty:
            # Convert TS to datetime for hour filtering
            df['datetime'] = pd.to_datetime(df['ts'].astype(float), unit='ms')
            datasets[sym] = prepare_indicators(df)
    
    print(f"Loaded {len(datasets)} symbols. Testing TIME filters (ADX > 20)...")
    print(f"{'Session':<20} | {'Trades':<6} {'Net R':<8} {'Avg R':<8}")
    print("-" * 50)
    
    sessions = [
        ('Full Day (0-24)', 0, 24),
        ('London (7-15)', 7, 15),
        ('NY Session (13-21)', 13, 21),
        ('Asia (0-8)', 0, 8),
        ('London/NY (8-20)', 8, 20)
    ]
    
    for name, start, end in sessions:
        # Custom logic to filter df inside test_strategy? No, easier to pass hours
        # We need to modify test_strategy to accept hour range
        total_r = 0
        trades_count = 0
        
        for sym, df in datasets.items():
            # Filter by hour in the loop
            res = test_session(df, start, end)
            total_r += res['net']
            trades_count += res['trades']
            
        avg = total_r/trades_count if trades_count else 0
        print(f"{name:<20} | {trades_count:<6} {total_r:<8.1f} {avg:<8.3f}")

def test_session(df, start_h, end_h):
    # Quick inline copy of test_strategy logic adapted for hours
    total_r = 0
    trades = 0
    df['hour'] = df['datetime'].dt.hour
    
    potential_long = None; potential_short = None; cooldown = 0
    rr = 2.0
    
    for i in range(200, len(df)-1):
        if cooldown > 0: cooldown-=1; continue
        row = df.iloc[i]
        
        # TIME FILTER
        if not (start_h <= row['hour'] < end_h):
            continue
        
        # ADX > 20
        if row['adx'] < 20: continue
        
        # Setup
        is_bull = (row['low'] <= df['low'].iloc[i-10:i].min() and row['rsi'] > df['rsi'].iloc[i-10:i].min() and row['rsi'] < 40 and abs(row['low'] - row['key_sup'])/row['key_sup'] < 0.01)
        is_bear = (row['high'] >= df['high'].iloc[i-10:i].max() and row['rsi'] < df['rsi'].iloc[i-10:i].max() and row['rsi'] > 60 and abs(row['high'] - row['key_res'])/row['key_res'] < 0.01)

        if is_bull: potential_long = {'idx': i, 'sl': row['low'], 'trigger': df['high'].iloc[i-15:i].max()}
        if is_bear: potential_short = {'idx': i, 'sl': row['high'], 'trigger': df['low'].iloc[i-15:i].min()}

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
            if entry == sl or abs(entry-sl)/entry > 0.02: continue
            risk_dist = abs(entry-sl); risk_pct = risk_dist/entry
            tp_dist = risk_dist * rr; tp_price = entry + tp_dist if side == 'long' else entry - tp_dist
            
            outcome = 'loss'; curr_sl = sl; be_hit = False
            for j in range(i+1, min(i+500, len(df))):
                c = df.iloc[j]
                if side == 'long':
                    if c['low'] <= curr_sl: outcome = 'loss' if not be_hit else 'be'; break
                    if c['high'] >= entry + risk_dist: curr_sl = entry; be_hit = True
                    if c['high'] >= tp_price: outcome = 'win'; break
                else:
                    if c['high'] >= curr_sl: outcome = 'loss' if not be_hit else 'be'; break
                    if c['low'] <= entry - risk_dist: curr_sl = entry; be_hit = True
                    if c['low'] <= tp_price: outcome = 'win'; break
            
            res_r = 0
            if outcome == 'win': res_r = rr - (WIN_COST_PCT / risk_pct)
            elif outcome == 'loss': res_r = -1.0 - (LOSS_COST_PCT / risk_pct)
            elif outcome == 'be': res_r = 0.0 - (WIN_COST_PCT / risk_pct)
            
            total_r += res_r; trades += 1; cooldown = 5
            
    return {'net': total_r, 'trades': trades}


if __name__ == "__main__":
    main()
