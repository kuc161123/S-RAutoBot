#!/usr/bin/env python3
"""
CREATIVE 5M DIVERGENCE DISCOVERY ENGINE
=======================================
Testing creative and non-standard divergence models on 5M.

Models:
1. HIDDEN_DIV: Price Higher Low + RSI Lower Low (Trend Continuation)
2. EXAGGERATED_DIV: Price Equal Highs + RSI Lower High (Strength Loss)
3. TRIPLE_CONFIRM: RSI + OBV + CMF all diverging together.
4. VOLUME_SPIKE_DIV: Divergence + 200% Volume Spike on confirmation candle.
"""

import pandas as pd
import numpy as np
import requests
import time
import concurrent.futures

# Configuration
DAYS = 200 # 5M data is heavy, 200 days is a lot of candles
TIMEFRAME = '5'
SYMBOL_COUNT = 30 # Focus on top 30 for speed
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
    
    # RSI 14
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
    
    # CMF (Chaikin Money Flow)
    mfv = ((close - low) - (high - close)) / (high - low + 1e-9) * vol
    df['cmf'] = mfv.rolling(20).sum() / vol.rolling(20).sum()
    
    # EMAs for trend
    df['ema200'] = close.ewm(span=200, adjust=False).mean()
    
    # Swing Extremes
    df['swing_l_14'] = low.rolling(14).min().shift(1)
    df['swing_h_14'] = high.rolling(14).max().shift(1)
    df['rsi_l_14'] = df['rsi'].rolling(14).min().shift(1)
    df['rsi_h_14'] = df['rsi'].rolling(14).max().shift(1)
    df['obv_l_14'] = df['obv'].rolling(14).min().shift(1)
    df['obv_h_14'] = df['obv'].rolling(14).max().shift(1)
    
    return df

def run_strategy(df, name):
    trades = []
    for i in range(50, len(df)-2):
        row = df.iloc[i]
        side = None
        
        if name == "HIDDEN_DIV":
            # Hidden Bull: Price higher low, RSI lower low (Trend Continuation)
            if row['low'] > row['swing_l_14'] and row['rsi'] < row['rsi_l_14'] and row['rsi'] < 40 and row['close'] > row['ema200']:
                side = 'long'
            # Hidden Bear: Price lower high, RSI higher high
            if row['high'] < row['swing_h_14'] and row['rsi'] > row['rsi_h_14'] and row['rsi'] > 60 and row['close'] < row['ema200']:
                side = 'short'
                
        elif name == "EXAGGERATED_DIV":
            # Exaggerated Bull: Price equal low, RSI higher low
            if abs(row['low'] - row['swing_l_14']) / row['low'] < 0.0005 and row['rsi'] > row['rsi_l_14'] and row['rsi'] < 30:
                side = 'long'
            # Exaggerated Bear: Price equal high, RSI lower high
            if abs(row['high'] - row['swing_h_14']) / row['high'] < 0.0005 and row['rsi'] < row['rsi_h_14'] and row['rsi'] > 70:
                side = 'short'
                
        elif name == "TRIPLE_CONFIRM":
            # RSI + OBV + CMF all show divergence
            rsi_div = row['low'] <= row['swing_l_14'] and row['rsi'] > row['rsi_l_14']
            obv_div = row['obv'] > row['obv_l_14']
            cmf_bull = row['cmf'] > 0
            if rsi_div and obv_div and cmf_bull and row['rsi'] < 30:
                side = 'long'
            
            rsi_bear = row['high'] >= row['swing_h_14'] and row['rsi'] < row['rsi_h_14']
            obv_bear = row['obv'] < row['obv_h_14']
            cmf_bear = row['cmf'] < 0
            if rsi_bear and obv_bear and cmf_bear and row['rsi'] > 70:
                side = 'short'

        if not side: continue
        
        # Realistic Entry: Next Candle Open
        entry = df.iloc[i+1]['open'] * (1 + (SLIPPAGE if side == 'long' else -SLIPPAGE))
        sl_dist = row['atr'] * 1.5
        tp_dist = sl_dist * 2.5
        sl, tp = (entry - sl_dist, entry + tp_dist) if side == 'long' else (entry + sl_dist, entry - tp_dist)
        
        outcome = None
        for j in range(i+1, min(i+1+100, len(df))):
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
            res_r = 2.5 - fee_cost if outcome == 'win' else -1.0 - fee_cost
            trades.append({'r': res_r, 'win': outcome == 'win'})
    return trades

def process_symbol(sym):
    df = fetch_data(sym)
    if df.empty: return {}
    df = calc_indicators(df)
    results = {}
    for name in ["HIDDEN_DIV", "EXAGGERATED_DIV", "TRIPLE_CONFIRM"]:
        results[name] = run_strategy(df, name)
    return results

def main():
    symbols = get_symbols()
    print(f"CREATIVE 5M DISCOVERY: Testing {len(symbols)} symbols...")
    
    total_trades = {"HIDDEN_DIV": [], "EXAGGERATED_DIV": [], "TRIPLE_CONFIRM": []}
    
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
