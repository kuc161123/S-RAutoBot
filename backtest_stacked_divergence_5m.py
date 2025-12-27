#!/usr/bin/env python3
"""
STACKED DIVERGENCE DISCOVERY ENGINE (5M)
========================================
Testing if requiring multiple types of confirmation divergences increases probability.

Models:
1. RSI + OBV Stacked: Price and Volume both show divergence.
2. RSI + MACD Stacked: Price and Momentum both show divergence.
3. RSI + MFI Stacked: Price and Money Flow both show divergence.
4. QUAD_STACK: All 4 indicators (RSI, OBV, MACD, MFI) must diverge.
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
        BAD = ['XAUTUSDT', 'PAXGUSDT', 'USTCUSDT', 'USDCUSDT', 'BUSDUSDT', 'DAIUSDT']
        return [t['symbol'] for t in usdt if t['symbol'] not in BAD][:SYMBOL_COUNT]
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
    
    # ATR
    df['atr'] = (high - low).rolling(20).mean()
    
    # RSI
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df['rsi'] = 100 - (100 / (1 + gain/(loss+1e-9)))
    
    # OBV
    obv = [0]
    for i in range(1, len(df)):
        if close.iloc[i] > close.iloc[i-1]: obv.append(obv[-1] + vol.iloc[i])
        elif close.iloc[i] < close.iloc[i-1]: obv.append(obv[-1] - vol.iloc[i])
        else: obv.append(obv[-1])
    df['obv'] = obv
    
    # MACD Histogram
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    df['macd_hist'] = macd - signal
    
    # MFI (Money Flow Index)
    tp = (high + low + close) / 3
    rmf = tp * vol
    pmf = rmf.where(tp > tp.shift(1), 0).rolling(14).sum()
    nmf = rmf.where(tp < tp.shift(1), 0).rolling(14).sum()
    mfr = pmf / (nmf + 1e-9)
    df['mfi'] = 100 - (100 / (1 + mfr))
    
    # Swings
    lb = 14
    df['price_low'] = low.rolling(lb).min().shift(1)
    df['price_high'] = high.rolling(lb).max().shift(1)
    df['rsi_low'] = df['rsi'].rolling(lb).min().shift(1)
    df['rsi_high'] = df['rsi'].rolling(lb).max().shift(1)
    df['obv_low'] = df['obv'].rolling(lb).min().shift(1)
    df['obv_high'] = df['obv'].rolling(lb).max().shift(1)
    df['macd_low'] = df['macd_hist'].rolling(lb).min().shift(1)
    df['macd_high'] = df['macd_hist'].rolling(lb).max().shift(1)
    df['mfi_low'] = df['mfi'].rolling(lb).min().shift(1)
    df['mfi_high'] = df['mfi'].rolling(lb).max().shift(1)
    
    return df

def run_strategy(df, name):
    trades = []
    for i in range(50, len(df)-2):
        row = df.iloc[i]
        
        # Bullish Divergences
        rsi_bull = row['low'] <= row['price_low'] and row['rsi'] > row['rsi_low'] and row['rsi'] < 35
        obv_bull = row['low'] <= row['price_low'] and row['obv'] > row['obv_low']
        macd_bull = row['low'] <= row['price_low'] and row['macd_hist'] > row['macd_low']
        mfi_bull = row['low'] <= row['price_low'] and row['mfi'] > row['mfi_low'] and row['mfi'] < 35
        
        # Bearish Divergences
        rsi_bear = row['high'] >= row['price_high'] and row['rsi'] < row['rsi_high'] and row['rsi'] > 65
        obv_bear = row['high'] >= row['price_high'] and row['obv'] < row['obv_high']
        macd_bear = row['high'] >= row['price_high'] and row['macd_hist'] < row['macd_high']
        mfi_bear = row['high'] >= row['price_high'] and row['mfi'] < row['mfi_high'] and row['mfi'] > 65
        
        side = None
        if name == "RSI+OBV":
            if rsi_bull and obv_bull: side = 'long'
            if rsi_bear and obv_bear: side = 'short'
        elif name == "RSI+MACD":
            if rsi_bull and macd_bull: side = 'long'
            if rsi_bear and macd_bear: side = 'short'
        elif name == "RSI+MFI":
            if rsi_bull and mfi_bull: side = 'long'
            if rsi_bear and mfi_bear: side = 'short'
        elif name == "QUAD_STACK":
            if rsi_bull and obv_bull and macd_bull and mfi_bull: side = 'long'
            if rsi_bear and obv_bear and macd_bear and mfi_bear: side = 'short'
            
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
    for name in ["RSI+OBV", "RSI+MACD", "RSI+MFI", "QUAD_STACK"]:
        results[name] = run_strategy(df, name)
    return results

def main():
    symbols = get_symbols()
    print(f"STACKED DIVERGENCE DISCOVERY: Testing {len(symbols)} symbols on 5M...")
    
    total_trades = {"RSI+OBV": [], "RSI+MACD": [], "RSI+MFI": [], "QUAD_STACK": []}
    
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
