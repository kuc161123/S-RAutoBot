#!/usr/bin/env python3
"""
300 SYMBOL 15M 1-YEAR BACKTEST (ASYNC FAST)
===========================================
Comprehensive scan of ALL Bybit USDT perpetuals (300+) over 1 year on 15M timeframe.
Uses asyncio + aiohttp for high-performance data fetching.
"""

import asyncio
import aiohttp
import pandas as pd
import numpy as np
from datetime import datetime
import os
import warnings
import time

warnings.filterwarnings('ignore')

# === CONFIGURATION ===
TIMEFRAME = '15'  # 15M
DATA_DAYS = 365   # 1 Year
MIN_TURNOVER = 500_000
MAX_SYMBOLS = 300

# COSTS
SLIPPAGE_PCT = 0.0002
FEE_PCT = 0.0006
SL_MULT = 1.0

# API
BASE_URL = "https://api.bybit.com"
MAX_CONCURRENT_REQUESTS = 10  # Be careful with rate limits

# STRATEGY
RR_RATIOS = [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
MAX_WAIT_CANDLES = 12
RSI_PERIOD = 14
EMA_PERIOD = 200

async def get_all_symbols():
    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(f"{BASE_URL}/v5/market/tickers?category=linear") as resp:
                data = await resp.json()
                tickers = data.get('result', {}).get('list', [])
                
                usdt = []
                for t in tickers:
                    if t['symbol'].endswith('USDT'):
                        turnover = float(t.get('turnover24h', 0))
                        if turnover >= MIN_TURNOVER:
                            usdt.append({'symbol': t['symbol'], 'turnover': turnover})
                
                usdt.sort(key=lambda x: x['turnover'], reverse=True)
                return [t['symbol'] for t in usdt[:MAX_SYMBOLS]]
        except Exception as e:
            print(f"Error fetching symbols: {e}")
            return []

async def fetch_symbol_data(session, symbol):
    end_ts = int(datetime.now().timestamp() * 1000)
    start_ts = end_ts - (DATA_DAYS * 24 * 60 * 60 * 1000)
    
    all_candles = []
    current_end = end_ts
    
    # We need ~35 calls. We'll do them sequentially per symbol but symbols parallel
    # to avoid complexity with stitching, but effectively parallel across symbols
    
    attempts = 0
    while current_end > start_ts and attempts < 45:
        attempts += 1
        params = {
            'category': 'linear',
            'symbol': symbol,
            'interval': TIMEFRAME,
            'limit': 1000,
            'end': current_end
        }
        
        try:
            async with session.get(f"{BASE_URL}/v5/market/kline", params=params) as resp:
                data = await resp.json()
                candles = data.get('result', {}).get('list', [])
                
                if not candles:
                    break
                    
                all_candles.extend(candles)
                oldest = int(candles[-1][0])
                current_end = oldest - 1
                
                if len(candles) < 1000:
                    break
                    
        except Exception:
            await asyncio.sleep(1)
            continue
            
    if not all_candles:
        return None
        
    df = pd.DataFrame(all_candles, columns=['start', 'open', 'high', 'low', 'close', 'volume', 'turnover'])
    df['start'] = pd.to_datetime(df['start'].astype(int), unit='ms')
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = df[col].astype(float)
        
    df.set_index('start', inplace=True)
    df.sort_index(inplace=True)
    df = df[~df.index.duplicated(keep='first')]
    return df

def process_data_and_backtest(df, symbol):
    # Indicators
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(RSI_PERIOD).mean()
    avg_loss = loss.rolling(RSI_PERIOD).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    df['rsi'] = 100 - (100 / (1 + rs))
    
    hl = df['high'] - df['low']
    hc = abs(df['high'] - df['close'].shift())
    lc = abs(df['low'] - df['close'].shift())
    df['atr'] = pd.concat([hl, hc, lc], axis=1).max(axis=1).rolling(14).mean()
    df['ema'] = df['close'].ewm(span=EMA_PERIOD, adjust=False).mean()
    df.dropna(inplace=True)
    
    if len(df) < 5000: return None
    
    # Divergences
    close = df['close'].values
    high = df['high'].values
    low = df['low'].values
    rsi = df['rsi'].values
    n = len(df)
    
    def find_pivots(src):
        p_highs = np.full(n, np.nan)
        p_lows = np.full(n, np.nan)
        for i in range(3, n-3):
            window = src[i-3:i+4]
            if src[i] == max(window): p_highs[i] = src[i]
            if src[i] == min(window): p_lows[i] = src[i]
        return p_highs, p_lows

    price_ph, price_pl = find_pivots(close)
    signals = []
    
    for i in range(50, n):
        # Bullish
        p_lows = []
        for j in range(i-3, max(0, i-50), -1):
            if not np.isnan(price_pl[j]):
                p_lows.append((j, price_pl[j]))
                if len(p_lows) >= 2: break
        if len(p_lows) == 2:
            idx, val = p_lows[0]
            prev_idx, prev_val = p_lows[1]
            if (i - idx) <= 10 and val < prev_val and rsi[idx] > rsi[prev_idx]:
                 # Filter nearby
                if not any(s['side'] == 'long' and abs(s['idx'] - idx) < 5 for s in signals):
                    signals.append({'idx': idx, 'conf_idx': i, 'side': 'long', 'swing': max(high[idx:i+1]), 'price': val})

        # Bearish
        p_highs = []
        for j in range(i-3, max(0, i-50), -1):
            if not np.isnan(price_ph[j]):
                p_highs.append((j, price_ph[j]))
                if len(p_highs) >= 2: break
        if len(p_highs) == 2:
            idx, val = p_highs[0]
            prev_idx, prev_val = p_highs[1]
            if (i - idx) <= 10 and val > prev_val and rsi[idx] < rsi[prev_idx]:
                if not any(s['side'] == 'short' and abs(s['idx'] - idx) < 5 for s in signals):
                    signals.append({'idx': idx, 'conf_idx': i, 'side': 'short', 'swing': min(low[idx:i+1]), 'price': val})
    
    if len(signals) < 10: return None
    
    # Backtest R:R
    best_res = {'total_r': -999}
    rows = list(df.itertuples())
    
    for rr in RR_RATIOS:
        trades = []
        for sig in signals:
            idx = sig['conf_idx']
            if idx >= len(rows)-1: continue
            
            # Trend Filter
            if sig['side'] == 'long' and rows[idx].close < rows[idx].ema: continue
            if sig['side'] == 'short' and rows[idx].close > rows[idx].ema: continue
            
            # BOS Entry
            entry_idx = None
            for j in range(idx+1, min(idx+1+MAX_WAIT_CANDLES, len(rows))):
                if sig['side'] == 'long' and rows[j].close > sig['swing']:
                    entry_idx = j+1; break
                if sig['side'] == 'short' and rows[j].close < sig['swing']:
                    entry_idx = j+1; break
            
            if not entry_idx or entry_idx >= len(rows): continue
            
            entry = rows[entry_idx]
            ep = entry.open
            sl_dist = entry.atr * SL_MULT
            
            if sig['side'] == 'long':
                entry_p = ep * (1+SLIPPAGE_PCT)
                tp = entry_p + (sl_dist*rr)
                sl = entry_p - sl_dist
            else:
                entry_p = ep * (1-SLIPPAGE_PCT)
                tp = entry_p - (sl_dist*rr)
                sl = entry_p + sl_dist
                
            res = -0.1
            for k in range(entry_idx, min(entry_idx+300, len(rows))):
                row = rows[k]
                if sig['side'] == 'long':
                    if row.low <= sl: res = -1.0; break
                    if row.high >= tp: res = rr; break
                else:
                    if row.high >= sl: res = -1.0; break
                    if row.low <= tp: res = rr; break
            
            risk_pct = abs(entry_p - sl)/entry_p if entry_p else 0.01
            fee = (FEE_PCT*2)/risk_pct
            trades.append(res - fee)
            
        if not trades: continue
        
        tr = sum(trades)
        if tr > best_res['total_r']:
            wr = len([t for t in trades if t > 0])/len(trades)
            best_res = {
                'symbol': symbol,
                'best_rr': rr,
                'total_r': round(tr, 2),
                'win_rate': round(wr, 3),
                'trades': len(trades)
            }
            
    return best_res if best_res['total_r'] != -999 else None

async def worker(queue, session, results):
    while True:
        symbol = await queue.get()
        try:
            print(f"Fetching {symbol}...", end=" ", flush=True)
            df = await fetch_symbol_data(session, symbol)
            
            if df is not None:
                # Run CPU-bound task in separate thread to not block async loop
                res = await asyncio.to_thread(process_data_and_backtest, df, symbol)
                if res:
                    results.append(res)
                    icon = "✅" if res['total_r'] > 0 else "❌"
                    print(f"{icon} {res['total_r']}R")
                else:
                    print(f"⏭️ No trades")
            else:
                print("Total Fail")
                
        except Exception as e:
            print(f"Error {symbol}: {e}")
        finally:
            queue.task_done()

async def main():
    print("="*60)
    print("ASYNC 15M BACKTEST (300 SYMBOLS)")
    print("="*60)
    
    symbols = await get_all_symbols()
    print(f"Processing {len(symbols)} symbols...")
    
    queue = asyncio.Queue()
    for s in symbols: queue.put_nowait(s)
    
    results = []
    
    async with aiohttp.ClientSession() as session:
        tasks = []
        for _ in range(MAX_CONCURRENT_REQUESTS):
            task = asyncio.create_task(worker(queue, session, results))
            tasks.append(task)
            
        await queue.join()
        for task in tasks: task.cancel()
        
    # Save Results
    df_res = pd.DataFrame(results)
    if not df_res.empty:
        df_res.sort_values('total_r', ascending=False, inplace=True)
        df_res.to_csv('300_symbol_15m_1year_async_results.csv', index=False)
        
        print("\n" + "="*60)
        print("FINAL RESULTS (15M 1-YEAR)")
        print("="*60)
        print(f"Total: {len(df_res)}")
        print(f"Profitable: {len(df_res[df_res['total_r']>0])}")
        print(df_res.head(20))
    else:
        print("No results found.")

if __name__ == "__main__":
    asyncio.run(main())
