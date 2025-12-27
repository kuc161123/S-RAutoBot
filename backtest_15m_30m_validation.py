#!/usr/bin/env python3
"""
15M/30M TIMEFRAME VALIDATION
============================
Testing exact bot configuration on higher timeframes where noise is reduced.

Strategy: RSI Divergence + Structure Break
Testing on: 15M and 30M timeframes
- Lookback: 10
- Max Wait: 10 candles
- SL: 0.8x ATR
- RR: 3.0:1
- Entry: Next candle open after structure break
"""

import pandas as pd
import numpy as np
import requests
import time
import concurrent.futures

# Configuration
DAYS = 365  # More data for HTF
SYMBOL_COUNT = 40
RSI_PERIOD = 14
LOOKBACK = 10
MAX_WAIT = 10
SL_MULT = 0.8
RR = 3.0
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

def fetch_data(symbol, interval):
    try:
        url = "https://api.bybit.com/v5/market/kline"
        all_kline = []
        end_ts = int(time.time() * 1000)
        start_ts = int((time.time() - DAYS * 24 * 3600) * 1000)
        while end_ts > start_ts:
            params = {'category': 'linear', 'symbol': symbol, 'interval': interval, 'limit': 1000, 'end': end_ts}
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
    close, high, low = df['close'], df['high'], df['low']
    
    # RSI
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(RSI_PERIOD).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(RSI_PERIOD).mean()
    df['rsi'] = 100 - (100 / (1 + gain/(loss+1e-9)))
    
    # ATR
    df['atr'] = (high - low).rolling(20).mean()
    
    # Swings
    df['swing_low'] = low.rolling(LOOKBACK).min()
    df['swing_high'] = high.rolling(LOOKBACK).max()
    df['rsi_low'] = df['rsi'].rolling(LOOKBACK).min()
    df['rsi_high'] = df['rsi'].rolling(LOOKBACK).max()
    
    return df

def run_strategy(df):
    trades = []
    
    for i in range(50, len(df) - 50):
        row = df.iloc[i]
        
        # RSI Divergence
        bullish_div = (row['low'] <= row['swing_low']) and (row['rsi'] > row['rsi_low']) and (row['rsi'] < 35)
        bearish_div = (row['high'] >= row['swing_high']) and (row['rsi'] < row['rsi_high']) and (row['rsi'] > 65)
        
        if not (bullish_div or bearish_div):
            continue
        
        side = 'long' if bullish_div else 'short'
        
        # Wait for Structure Break
        bos_found, bos_idx = False, None
        for j in range(i+1, min(i+1+MAX_WAIT, len(df)-10)):
            c = df.iloc[j]
            if side == 'long' and c['close'] > row['swing_high']:
                bos_found, bos_idx = True, j
                break
            elif side == 'short' and c['close'] < row['swing_low']:
                bos_found, bos_idx = True, j
                break
        
        if not bos_found:
            continue
        
        # Entry at NEXT candle open
        if bos_idx + 1 >= len(df):
            continue
            
        entry = df.iloc[bos_idx + 1]['open'] * (1 + (SLIPPAGE if side == 'long' else -SLIPPAGE))
        
        atr = row['atr']
        if pd.isna(atr) or atr <= 0:
            continue
        
        sl_dist = atr * SL_MULT
        tp_dist = sl_dist * RR
        sl, tp = (entry - sl_dist, entry + tp_dist) if side == 'long' else (entry + sl_dist, entry - tp_dist)
        
        # Execute
        outcome = None
        for k in range(bos_idx+1, min(bos_idx+1+200, len(df))):
            c = df.iloc[k]
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
            trades.append({'r': res_r, 'win': outcome == 'win'})
    
    return trades

def process_symbol(sym, interval):
    df = fetch_data(sym, interval)
    if df.empty: return []
    df = calc_indicators(df)
    return run_strategy(df)

def main():
    symbols = get_symbols()
    print("="*80)
    print("15M/30M TIMEFRAME VALIDATION")
    print("="*80)
    print(f"Testing {len(symbols)} symbols over 365 days")
    print(f"Strategy: RSI Divergence + Structure Break")
    print(f"Config: Lookback={LOOKBACK}, MaxWait={MAX_WAIT}, SL={SL_MULT}x, RR={RR}:1")
    print()
    
    for timeframe, tf_name in [('15', '15M'), ('30', '30M')]:
        print(f"\n{'='*80}")
        print(f"TESTING {tf_name} TIMEFRAME")
        print(f"{'='*80}")
        
        all_trades = []
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = {executor.submit(process_symbol, sym, timeframe): sym for sym in symbols}
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
            wr = sum(1 for t in all_trades if t['win']) / len(all_trades) * 100
            total_r = sum(t['r'] for t in all_trades)
            avg_r = total_r / len(all_trades)
            
            print(f"\n{tf_name} RESULTS:")
            print(f"  Total Trades:    {len(all_trades)}")
            print(f"  Win Rate:        {wr:.1f}%")
            print(f"  Avg R/Trade:     {avg_r:+.3f}R")
            print(f"  Total R:         {total_r:+.1f}R")
            print("-" * 80)
            
            if avg_r > 0:
                print(f"  ✅ {tf_name} IS PROFITABLE!")
            else:
                print(f"  ❌ {tf_name} shows negative expectancy")
        else:
            print(f"\n{tf_name}: NO TRADES FOUND")
    
    print("\n" + "="*80)
    print("VALIDATION COMPLETE")
    print("="*80)

if __name__ == "__main__":
    main()
