#!/usr/bin/env python3
"""
ADVANCED VOLUME & DELTA DISCOVERY ENGINE (5M)
=============================================
Testing if non-indicator based confirmations like Volume Delta and Absorption Wicks
can solve the 5M divergence problem.

Models:
1. DELTA_DIV: Price lower low, Cumulative Delta higher low (Aggressive Absorption).
2. WICK_REJECTION: Divergence + Candle wick > 60% of total range (Pin-bar on support).
3. VOLUME_CLIMAX: Divergence + Buy/Sell volume 4x higher than prior candle (Exhaustion).
"""

import pandas as pd
import numpy as np
import requests
import time
import concurrent.futures

# Configuration
DAYS = 200
TIMEFRAME = '5'
SYMBOL_COUNT = 30
FEE = 0.0006
SLIPPAGE = 0.0003

def get_symbols():
    try:
        url = "https://api.bybit.com/v5/market/tickers?category=linear"
        resp = requests.get(url, timeout=10).json()
        tickers = resp.get('result', {}).get('list', [])
        usdt = [t for t in tickers if t['symbol'].endswith('USDT')]
        usdt.sort(key=lambda x: float(x.get('turnover24h', 0)), reverse=True)
        return [t['symbol'] for t in usdt[:SYMBOL_COUNT]]
    except: return []

def fetch_data(symbol):
    try:
        url = "https://api.bybit.com/v5/market/kline"
        all_kline = []
        end_ts = int(time.time() * 1000)
        start_ts = int((time.time() - DAYS * 24 * 3600) * 1000)
        while end_ts > start_ts:
            params = {'category': 'linear', 'symbol': symbol, 'interval': TIMEFRAME, 'limit': 1000, 'end': end_ts}
            r = requests.get(url, params=params).json()
            if r['retCode'] != 0 or not r['result']['list']: break
            klines = r['result']['list']
            all_kline.extend(klines)
            end_ts = int(klines[-1][0]) - 1
            time.sleep(0.01)
        if not all_kline: return pd.DataFrame()
        df = pd.DataFrame(all_kline, columns=['ts', 'open', 'high', 'low', 'close', 'vol', 'to'])
        df = df.iloc[::-1].reset_index(drop=True)
        for c in ['open', 'high', 'low', 'close', 'vol']: df[c] = df[c].astype(float)
        return df
    except: return pd.DataFrame()

def calc_indicators(df):
    if len(df) < 100: return pd.DataFrame()
    df = df.copy()
    close, high, low, vol = df['close'], df['high'], df['low'], df['vol']
    
    # Delta Approximation (VbP - Volume by Price)
    # We'll use (Close-Open)/(High-Low) as a proxy for Candle Delta
    df['delta'] = ((close - df['open']) / (high - low + 1e-9)) * vol
    df['cum_delta'] = df['delta'].cumsum()
    
    # ATR for Wick calculation
    df['atr'] = (high - low).rolling(20).mean()
    df['body_size'] = abs(close - df['open'])
    df['wick_size_total'] = (high - low) - df['body_size']
    df['lower_wick'] = (np.minimum(close, df['open']) - low)
    df['upper_wick'] = (high - np.maximum(close, df['open']))
    
    # RSI for Divergence context
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df['rsi'] = 100 - (100 / (1 + gain/(loss+1e-9)))
    
    # Swings
    lb = 14
    df['swing_l'] = low.rolling(lb).min().shift(1)
    df['swing_h'] = high.rolling(lb).max().shift(1)
    df['rsi_l'] = df['rsi'].rolling(lb).min().shift(1)
    df['rsi_h'] = df['rsi'].rolling(lb).max().shift(1)
    df['delta_l'] = df['cum_delta'].rolling(lb).min().shift(1)
    df['delta_h'] = df['cum_delta'].rolling(lb).max().shift(1)
    
    return df

def run_strategy(df, name):
    trades = []
    for i in range(50, len(df)-2):
        row = df.iloc[i]
        candle_range = row['high'] - row['low']
        if candle_range == 0: continue
        
        side = None
        if name == "DELTA_DIV":
            # Price lower low, Delta higher low (Absorption)
            if row['low'] <= row['swing_l'] and row['cum_delta'] > row['delta_l'] and row['rsi'] < 35:
                side = 'long'
            if row['high'] >= row['swing_h'] and row['cum_delta'] < row['delta_h'] and row['rsi'] > 65:
                side = 'short'
        
        elif name == "WICK_REJECTION":
            # RSI Divergence confirmed by a long lower/upper wick (60% of candle)
            div_bull = row['low'] <= row['swing_l'] and row['rsi'] > row['rsi_l']
            if div_bull and (row['lower_wick'] / candle_range) > 0.6:
                side = 'long'
            
            div_bear = row['high'] >= row['swing_h'] and row['rsi'] < row['rsi_h']
            if div_bear and (row['upper_wick'] / candle_range) > 0.6:
                side = 'short'
                
        elif name == "VOLUME_CLIMAX":
            # Divergence confirmed by Volume > 4x average (Exhaustion climax)
            vol_avg = df.iloc[i-10:i]['vol'].mean()
            if (row['low'] <= row['swing_l'] and row['rsi'] > row['rsi_l']) and row['vol'] > vol_avg * 4:
                side = 'long'
            if (row['high'] >= row['swing_h'] and row['rsi'] < row['rsi_h']) and row['vol'] > vol_avg * 4:
                side = 'short'

        if not side: continue
        
        # Realistic Entry
        entry = df.iloc[i+1]['open'] * (1 + (SLIPPAGE if side == 'long' else -SLIPPAGE))
        sl_dist = row['atr'] * 1.5
        tp_dist = sl_dist * 3.0
        sl, tp = (entry - sl_dist, entry + tp_dist) if side == 'long' else (entry + sl_dist, entry - tp_dist)
        
        outcome = None
        for j in range(i+1, min(i+1+150, len(df))):
            c = df.iloc[j]
            if side == 'long':
                if c['low'] <= sl: outcome = 'loss'; break
                if c['high'] >= tp: outcome = 'win'; break
            else:
                if c['high'] >= sl: outcome = 'loss'; break
                if c['low'] <= tp: outcome = 'win'; break
        
        if outcome:
            risk_pct = sl_dist / entry
            fee_cost = (FEE + SLIPPAGE) / risk_pct
            res_r = 3.0 - fee_cost if outcome == 'win' else -1.0 - fee_cost
            trades.append({'r': res_r, 'win': outcome == 'win'})
    return trades

def process_symbol(sym):
    df = fetch_data(sym)
    if df.empty: return {}
    df = calc_indicators(df)
    results = {}
    for name in ["DELTA_DIV", "WICK_REJECTION", "VOLUME_CLIMAX"]:
        results[name] = run_strategy(df, name)
    return results

def main():
    symbols = get_symbols()
    print(f"ADVANCED VOLUME DISCOVERY: Testing {len(symbols)} symbols on 5M...")
    
    total_trades = {"DELTA_DIV": [], "WICK_REJECTION": [], "VOLUME_CLIMAX": []}
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(process_symbol, sym): sym for sym in symbols}
        count = 0
        for future in concurrent.futures.as_completed(futures):
            sym = futures[future]
            try:
                res = future.result()
                for name in total_trades:
                    total_trades[name].extend(res.get(name, []))
                count += 1
                if count % 5 == 0: print(f"  Processed {count}/{len(symbols)} symbols...")
            except Exception as e:
                print(f"  Error processing {sym}: {e}")

    print("\n" + "="*80)
    print(f"{'Strategy':<20} | {'WR':<10} | {'Avg R':<10} | {'Total R':<10} | {'N':<6}")
    print("-"*80)
    for name, trades in total_trades.items():
        if trades:
            wr = sum(1 for t in trades if t['win']) / len(trades) * 100
            total_r = sum(t['r'] for t in trades)
            avg_r = total_r / len(trades)
            print(f"{name:<20} | {wr:8.1f}% | {avg_r:+9.3f}R | {total_r:+9.1f}R | {len(trades):<6}")
        else:
            print(f"{name:<20} | NO TRADES")
    print("="*80)

if __name__ == "__main__":
    main()
