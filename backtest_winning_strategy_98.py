#!/usr/bin/env python3
"""
SCALED WINNING STRATEGY TEST (98 SYMBOLS)
=========================================
Testing the 4H EMA Trend Stack strategy on all 98 symbols for 1 year.
Golden Combo: RSI 35, 1.5x SL, 4.5:1 RR
"""

import pandas as pd
import numpy as np
import requests
import time
import concurrent.futures

# Settings
DAYS = 365
INTERVAL = '240' # 4H
RSI_TRIGGER = 35
SL_MULT = 1.5
RR = 4.5
FEE = 0.0006
SLIPPAGE = 0.0003

def get_symbols(n=98):
    try:
        url = "https://api.bybit.com/v5/market/tickers?category=linear"
        resp = requests.get(url, timeout=10).json()
        tickers = resp.get('result', {}).get('list', [])
        usdt = [t for t in tickers if t['symbol'].endswith('USDT')]
        usdt.sort(key=lambda x: float(x.get('turnover24h', 0)), reverse=True)
        BAD = ['XAUTUSDT', 'PAXGUSDT', 'USTCUSDT', 'USDCUSDT', 'BUSDUSDT', 'DAIUSDT']
        return [t['symbol'] for t in usdt[:n] if t['symbol'] not in BAD][:n]
    except Exception as e:
        print(f"Error fetching symbols: {e}")
        return []

def fetch_data(symbol):
    try:
        url = "https://api.bybit.com/v5/market/kline"
        all_kline = []
        end_ts = int(time.time() * 1000)
        start_ts = int((time.time() - DAYS * 24 * 3600) * 1000)
        while end_ts > start_ts:
            params = {'category': 'linear', 'symbol': symbol, 'interval': INTERVAL, 'limit': 1000, 'end': end_ts}
            r = requests.get(url, params=params).json()
            if r['retCode'] != 0 or not r['result']['list']: break
            klines = r['result']['list']
            all_kline.extend(klines)
            end_ts = int(klines[-1][0]) - 1
            time.sleep(0.02)
        if not all_kline: return pd.DataFrame()
        df = pd.DataFrame(all_kline, columns=['ts', 'open', 'high', 'low', 'close', 'vol', 'to'])
        df = df.iloc[::-1].reset_index(drop=True)
        for c in ['open', 'high', 'low', 'close', 'vol']: df[c] = df[c].astype(float)
        return df
    except: return pd.DataFrame()

def calc_indicators(df):
    if len(df) < 200: return pd.DataFrame()
    df = df.copy()
    close = df['close']
    
    # RSI 14
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df['rsi'] = 100 - (100 / (1 + gain/(loss+1e-9)))
    
    # EMAs
    df['ema20'] = close.ewm(span=20, adjust=False).mean()
    df['ema50'] = close.ewm(span=50, adjust=False).mean()
    df['ema200'] = close.ewm(span=200, adjust=False).mean()
    df['atr'] = (df['high'] - df['low']).rolling(20).mean()
    
    return df

def run_strategy(df, symbol):
    trades = []
    if len(df) < 201: return []
    
    for i in range(200, len(df)-2):
        row = df.iloc[i]
        prev = df.iloc[i-1]
        
        side = None
        # Long: 20 > 50 > 200 + RSI cross UP trigger
        if row['ema20'] > row['ema50'] > row['ema200'] and prev['rsi'] < RSI_TRIGGER and row['rsi'] >= RSI_TRIGGER:
            side = 'long'
        # Short: 20 < 50 < 200 + RSI cross DOWN trigger
        if row['ema20'] < row['ema50'] < row['ema200'] and prev['rsi'] > (100-RSI_TRIGGER) and row['rsi'] <= (100-RSI_TRIGGER):
            side = 'short'
            
        if not side: continue
        
        # Realistic Entry: Next Candle Open
        entry_price = df.iloc[i+1]['open']
        entry = entry_price * (1 + (SLIPPAGE if side == 'long' else -SLIPPAGE))
        
        atr = row['atr']
        if pd.isna(atr) or atr <= 0: continue
        
        sl_dist = atr * SL_MULT
        tp_dist = sl_dist * RR
        sl, tp = (entry - sl_dist, entry + tp_dist) if side == 'long' else (entry + sl_dist, entry - tp_dist)
        
        outcome = None
        for j in range(i+1, min(i+1+500, len(df))): # Look ahead up to 500 candles (~3 months on 4H)
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
            res_r = RR - fee_cost if outcome == 'win' else -1.0 - fee_cost
            trades.append({'symbol': symbol, 'r': res_r, 'win': outcome == 'win'})
    return trades

def process_symbol(sym):
    df = fetch_data(sym)
    if df.empty: return []
    df = calc_indicators(df)
    return run_strategy(df, sym)

def main():
    symbols = get_symbols(98)
    print(f"SCALED TEST: Testing {len(symbols)} symbols for 365 days...")
    print(f"Strategy: 4H EMA Trend Stack (RSI {RSI_TRIGGER}, {SL_MULT}x SL, {RR}x RR)")
    
    all_trades = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(process_symbol, sym): sym for sym in symbols}
        count = 0
        for future in concurrent.futures.as_completed(futures):
            sym = futures[future]
            try:
                trades = future.result()
                all_trades.extend(trades)
                count += 1
                if count % 10 == 0:
                    print(f"  Processed {count}/{len(symbols)} symbols...")
            except Exception as e:
                print(f"  Error processing {sym}: {e}")

    if all_trades:
        total_r = sum(t['r'] for t in all_trades)
        wr = sum(1 for t in all_trades if t['win']) / len(all_trades) * 100
        avg_r = total_r / len(all_trades)
        
        # Per symbol breakdown
        sym_perf = {}
        for t in all_trades:
            s = t['symbol']
            if s not in sym_perf: sym_perf[s] = {'r': 0, 'n': 0}
            sym_perf[s]['r'] += t['r']
            sym_perf[s]['n'] += 1
            
        print("\n" + "="*60)
        print("FINAL SCALED RESULTS")
        print("="*60)
        print(f"Total Trades:   {len(all_trades)}")
        print(f"Win Rate:       {wr:.1f}%")
        print(f"Avg R / Trade:  {avg_r:+.3f}R")
        print(f"Total R:        {total_r:+.1f}R")
        print("-" * 60)
        
        # Top 10 symbols
        sorted_syms = sorted(sym_perf.items(), key=lambda x: x[1]['r'], reverse=True)
        print("TOP 10 SYMBOLS:")
        for sym, stats in sorted_syms[:10]:
            print(f"  {sym:<12} | {stats['r']:>7.1f}R | N={stats['n']}")
            
        # Worst 10 symbols
        print("\nWORST 10 SYMBOLS:")
        for sym, stats in sorted_syms[-10:]:
            print(f"  {sym:<12} | {stats['r']:>7.1f}R | N={stats['n']}")
            
        print("="*60)
    else:
        print("No trades found.")

if __name__ == "__main__":
    main()
