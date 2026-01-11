#!/usr/bin/env python3
"""
DEBUG PRECISION TRADE (CONCISE)
===============================
Trace 1000PEPEUSDT trades to check if SL logic is failing.
"""

import requests
import pandas as pd
import numpy as np
import time

# CONFIG
SYMBOL = '1000PEPEUSDT'
DAYS = 30
RSI_PERIOD = 14
EMA_PERIOD = 200
SIGNAL_TF = '60'
EXECUTION_TF = '5'
RR = 10.0
ATR_MULT = 2.0
DIV_TYPE = 'REG_BEAR'

BASE_URL = "https://api.bybit.com"

def fetch_klines(symbol, interval, days):
    end_ts = int(time.time() * 1000)
    start_ts = end_ts - (days * 24 * 60 * 60 * 1000)
    all_candles = []
    current_end = end_ts
    while current_end > start_ts:
        params = {'category': 'linear', 'symbol': symbol, 'interval': interval, 'limit': 1000, 'end': current_end}
        try:
            resp = requests.get(f"{BASE_URL}/v5/market/kline", params=params, timeout=10)
            data = resp.json().get('result', {}).get('list', [])
            if not data: break
            all_candles.extend(data)
            current_end = int(data[-1][0]) - 1
            if len(data) < 1000: break
            time.sleep(0.05)
        except: break
    if not all_candles: return pd.DataFrame()
    df = pd.DataFrame(all_candles, columns=['start', 'open', 'high', 'low', 'close', 'volume', 'turnover'])
    df['start'] = pd.to_datetime(df['start'].astype(int), unit='ms')
    for c in ['open', 'high', 'low', 'close', 'volume']: df[c] = df[c].astype(float)
    df = df.sort_values('start').reset_index(drop=True)
    return df

def find_pivots(data, left=3, right=3):
    pivot_highs = np.full(len(data), np.nan)
    pivot_lows = np.full(len(data), np.nan)
    for i in range(left, len(data) - right):
        window = data[i-left : i+right+1]
        center = data[i]
        if len(window) != (left + right + 1): continue
        if center == max(window) and list(window).count(center) == 1: pivot_highs[i] = center
        if center == min(window) and list(window).count(center) == 1: pivot_lows[i] = center
    return pivot_highs, pivot_lows

def main():
    print(f"DEBUGGING {SYMBOL} (SL CHECK)")
    
    df_1h = fetch_klines(SYMBOL, SIGNAL_TF, DAYS)
    df_5m = fetch_klines(SYMBOL, EXECUTION_TF, DAYS)
    
    # Indicators
    delta = df_1h['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = -delta.where(delta < 0, 0).rolling(14).mean()
    rs = gain / (loss + 1e-10)
    df_1h['rsi'] = 100 - (100 / (1 + rs))
    df_1h['ema'] = df_1h['close'].ewm(span=EMA_PERIOD, adjust=False).mean()
    
    high = df_1h['high']
    low = df_1h['low']
    close = df_1h['close'].shift()
    tr = pd.concat([high-low, (high-close).abs(), (low-close).abs()], axis=1).max(axis=1)
    df_1h['atr'] = tr.rolling(14).mean()
    
    price_ph, price_pl = find_pivots(df_1h['close'].values)
    
    count = 0 
    
    for i in range(205, len(df_1h)-20):
        # Scan for REG_BEAR
        curr_price = df_1h['close'].iloc[i]
        curr_ema = df_1h['ema'].iloc[i]
        
        if curr_price < curr_ema:
            p_highs = [j for j in range(i-3, i-50, -1) if not np.isnan(price_ph[j])]
            if len(p_highs) >= 2:
                curr_idx = p_highs[0]
                prev_idx = p_highs[1]
                if (i - curr_idx) <= 10:
                    if price_ph[curr_idx] > price_ph[prev_idx] and df_1h['rsi'].iloc[curr_idx] < df_1h['rsi'].iloc[prev_idx]:
                        
                        # EXECUTE
                        swing_low = df_1h['low'].iloc[curr_idx:i+1].min()
                        for k in range(1, 7):
                            idx = i + k
                            c = df_1h.iloc[idx]
                            
                            if c['close'] < swing_low:
                                entry_candle = df_1h.iloc[idx+1]
                                entry_time = entry_candle['start']
                                entry_price = entry_candle['open'] * (1 - 0.0002)
                                sl_dist = c['atr'] * ATR_MULT
                                sl = entry_price + sl_dist
                                tp = entry_price - (sl_dist * RR)
                                
                                print(f"\n[{count}] SIGNAL: {entry_time}")
                                print(f"    Entry: {entry_price:.8f}")
                                print(f"    SL:    {sl:.8f} (+{sl_dist:.8f})")
                                print(f"    TP:    {tp:.8f}")
                                
                                # 5M Trace
                                sub_5m = df_5m[df_5m['start'] >= entry_time]
                                hit = False
                                
                                for ri, row in sub_5m.iterrows():
                                    # SL CHECK
                                    if row['high'] >= sl:
                                        print(f"❌ SL HIT @ {row['start']} | High {row['high']:.8f} >= {sl:.8f}")
                                        hit = True
                                        break
                                    
                                    # TP CHECK
                                    if row['low'] <= tp:
                                        print(f"✅ TP HIT @ {row['start']} | Low {row['low']:.8f} <= {tp:.8f}")
                                        hit = True
                                        break
                                        
                                    if (row['start'] - entry_time).total_seconds() > 3600 * 48:
                                        print("⏱ TIMEOUT (>48h)")
                                        hit = True
                                        break
                                
                                count += 1
                                if count > 5: return
                                break

if __name__ == "__main__":
    main()
