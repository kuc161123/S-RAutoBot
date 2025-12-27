#!/usr/bin/env python3
"""
BOS PULLBACK CONFIRMATION BACKTEST (5M)
=======================================
Testing the current RSI divergence strategy with "Pullback-to-BOS" confirmation.

Logic:
1. Detect RSI divergence (price lower low, RSI higher low)
2. Wait for structure break (price breaks above previous swing high)
3. NEW: Wait for pullback to the BOS level
4. NEW: Enter on confirmation candle close if it bounces in trade direction
"""

import pandas as pd
import numpy as np
import requests
import time
import concurrent.futures

# Configuration
DAYS = 200
TIMEFRAME = '5'
SYMBOL_COUNT = 20
RSI_PERIOD = 14
LOOKBACK = 10
FEE = 0.0006
SLIPPAGE = 0.0003
SL_MULT = 0.8
RR = 3.0

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
    
    for i in range(50, len(df) - 100):  # Need lookahead for pullback confirmation
        row = df.iloc[i]
        
        # Step 1: Detect Divergence
        bullish_div = (row['low'] <= row['swing_low']) and (row['rsi'] > row['rsi_low']) and (row['rsi'] < 35)
        bearish_div = (row['high'] >= row['swing_high']) and (row['rsi'] < row['rsi_high']) and (row['rsi'] > 65)
        
        if not (bullish_div or bearish_div):
            continue
        
        side = 'long' if bullish_div else 'short'
        
        # Step 2: Wait for BOS (Structure Break)
        bos_found = False
        bos_level = None
        bos_idx = None
        
        for j in range(i+1, min(i+1+10, len(df)-50)):  # Look ahead max 10 candles for BOS
            c = df.iloc[j]
            if side == 'long' and c['high'] > row['swing_high']:
                bos_found = True
                bos_level = row['swing_high']
                bos_idx = j
                break
            elif side == 'short' and c['low'] < row['swing_low']:
                bos_found = True
                bos_level = row['swing_low']
                bos_idx = j
                break
        
        if not bos_found:
            continue
        
        # Step 3: Wait for Pullback to BOS level
        pullback_found = False
        pullback_idx = None
        
        for k in range(bos_idx+1, min(bos_idx+1+20, len(df)-10)):  # Look ahead max 20 candles for pullback
            c = df.iloc[k]
            # Pullback criteria: Price touches or goes through BOS level
            if side == 'long' and c['low'] <= bos_level * 1.002:  # Within 0.2% tolerance
                pullback_found = True
                pullback_idx = k
                break
            elif side == 'short' and c['high'] >= bos_level * 0.998:
                pullback_found = True
                pullback_idx = k
                break
        
        if not pullback_found:
            continue
        
        # Step 4: Wait for Confirmation Bounce
        # Check if next candle closes in the trade direction
        if pullback_idx + 1 >= len(df):
            continue
            
        confirmation_candle = df.iloc[pullback_idx + 1]
        
        # Bounce confirmation: Next candle close is above/below pullback candle
        bounce_confirmed = False
        if side == 'long' and confirmation_candle['close'] > df.iloc[pullback_idx]['close']:
            bounce_confirmed = True
        elif side == 'short' and confirmation_candle['close'] < df.iloc[pullback_idx]['close']:
            bounce_confirmed = True
        
        if not bounce_confirmed:
            continue
        
        # ENTRY: On confirmation candle CLOSE (realistic)
        entry = confirmation_candle['close'] * (1 + (SLIPPAGE if side == 'long' else -SLIPPAGE))
        
        atr = row['atr']
        if pd.isna(atr) or atr <= 0:
            continue
        
        sl_dist = atr * SL_MULT
        tp_dist = sl_dist * RR
        sl, tp = (entry - sl_dist, entry + tp_dist) if side == 'long' else (entry + sl_dist, entry - tp_dist)
        
        # Execute trade from confirmation candle onwards
        outcome = None
        for m in range(pullback_idx+1, min(pullback_idx+1+150, len(df))):
            c = df.iloc[m]
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

def process_symbol(sym):
    df = fetch_data(sym)
    if df.empty: return []
    df = calc_indicators(df)
    return run_strategy(df)

def main():
    symbols = get_symbols()
    print(f"BOS PULLBACK CONFIRMATION TEST: Testing {len(symbols)} symbols on 5M...")
    print(f"Strategy: RSI Divergence + BOS + Pullback + Bounce Confirmation")
    print(f"Config: SL={SL_MULT}x ATR, RR={RR}:1\n")
    
    all_trades = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(process_symbol, sym): sym for sym in symbols}
        count = 0
        for future in concurrent.futures.as_completed(futures):
            sym = futures[future]
            try:
                trades = future.result()
                all_trades.extend(trades)
                count += 1
                print(f"  Processed {count}/{len(symbols)} symbols... ({len(trades)} trades)")
            except Exception as e:
                print(f"  Error processing {sym}: {e}")
    
    print("\n" + "="*70)
    print("RESULTS: BOS PULLBACK CONFIRMATION STRATEGY")
    print("="*70)
    
    if all_trades:
        wr = sum(1 for t in all_trades if t['win']) / len(all_trades) * 100
        total_r = sum(t['r'] for t in all_trades)
        avg_r = total_r / len(all_trades)
        
        print(f"Total Trades:    {len(all_trades)}")
        print(f"Win Rate:        {wr:.1f}%")
        print(f"Avg R/Trade:     {avg_r:+.3f}R")
        print(f"Total R:         {total_r:+.1f}R")
        print("-" * 70)
        
        if avg_r > 0:
            print("✅ STRATEGY IS PROFITABLE!")
        else:
            print("❌ Strategy shows negative expectancy")
    else:
        print("NO TRADES FOUND")
    
    print("="*70)

if __name__ == "__main__":
    main()
