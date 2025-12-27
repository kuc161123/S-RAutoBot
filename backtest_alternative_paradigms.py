#!/usr/bin/env python3
"""
ALTERNATIVE PARADIGM DISCOVERY ENGINE
=====================================
Testing fundamentally different trading approaches (NO divergence).

Strategies:
1. MEAN_REVERSION_BB: Buy BB lower band, sell BB upper band (2 SD)
2. MOMENTUM_BREAKOUT: Buy ATR breakout above 20-bar high
3. VWAP_REVERSION: Mean reversion to VWAP with volume confirmation
4. EMA_CROSSOVER: Classic 9/21 EMA crossover (trend following)
5. RANGE_FADE: Fade range extremes (buy low, sell high in consolidation)
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
    
    # Bollinger Bands
    df['bb_mid'] = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    df['bb_upper'] = df['bb_mid'] + (bb_std * 2)
    df['bb_lower'] = df['bb_mid'] - (bb_std * 2)
    
    # ATR
    df['atr'] = (high - low).rolling(20).mean()
    
    # VWAP (session approximation using 100-period)
    df['vwap'] = (close * vol).rolling(100).sum() / vol.rolling(100).sum()
    
    # EMAs
    df['ema9'] = close.ewm(span=9, adjust=False).mean()
    df['ema21'] = close.ewm(span=21, adjust=False).mean()
    
    # Range detection
    df['range_high'] = high.rolling(50).max()
    df['range_low'] = low.rolling(50).min()
    df['range_size'] = df['range_high'] - df['range_low']
    
    # Breakout levels
    df['breakout_high'] = high.rolling(20).max().shift(1)
    df['breakout_low'] = low.rolling(20).min().shift(1)
    
    return df

def run_strategy(df, name):
    trades = []
    
    for i in range(100, len(df) - 10):
        row = df.iloc[i]
        prev = df.iloc[i-1]
        
        side = None
        sl_mult = 1.5
        rr = 2.5
        
        if name == "MEAN_REVERSION_BB":
            # Buy at lower BB, sell at upper BB
            if row['close'] <= row['bb_lower']:
                side = 'long'
            elif row['close'] >= row['bb_upper']:
                side = 'short'
                
        elif name == "MOMENTUM_BREAKOUT":
            # Buy breakout above 20-bar high with volume
            vol_avg = df.iloc[i-10:i]['vol'].mean()
            if row['high'] > row['breakout_high'] and row['vol'] > vol_avg * 1.5:
                side = 'long'
            elif row['low'] < row['breakout_low'] and row['vol'] > vol_avg * 1.5:
                side = 'short'
                
        elif name == "VWAP_REVERSION":
            # Mean reversion to VWAP
            dist_from_vwap = abs(row['close'] - row['vwap']) / row['vwap']
            if dist_from_vwap > 0.005:  # 0.5% away from VWAP
                if row['close'] < row['vwap']:
                    side = 'long'
                else:
                    side = 'short'
                    
        elif name == "EMA_CROSSOVER":
            # Classic EMA crossover
            if prev['ema9'] <= prev['ema21'] and row['ema9'] > row['ema21']:
                side = 'long'
            elif prev['ema9'] >= prev['ema21'] and row['ema9'] < row['ema21']:
                side = 'short'
                
        elif name == "RANGE_FADE":
            # Fade range extremes
            if pd.isna(row['range_size']) or row['range_size'] == 0:
                continue
            price_in_range = (row['close'] - row['range_low']) / row['range_size']
            if price_in_range < 0.1:  # Near bottom of range
                side = 'long'
            elif price_in_range > 0.9:  # Near top of range
                side = 'short'
        
        if not side:
            continue
        
        # Entry at next candle open
        entry = df.iloc[i+1]['open'] * (1 + (SLIPPAGE if side == 'long' else -SLIPPAGE))
        
        atr = row['atr']
        if pd.isna(atr) or atr <= 0:
            continue
        
        sl_dist = atr * sl_mult
        tp_dist = sl_dist * rr
        sl, tp = (entry - sl_dist, entry + tp_dist) if side == 'long' else (entry + sl_dist, entry - tp_dist)
        
        # Execute
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
            res_r = rr - fee_cost if outcome == 'win' else -1.0 - fee_cost
            trades.append({'r': res_r, 'win': outcome == 'win'})
    
    return trades

def process_symbol(sym):
    df = fetch_data(sym)
    if df.empty: return {}
    df = calc_indicators(df)
    results = {}
    for name in ["MEAN_REVERSION_BB", "MOMENTUM_BREAKOUT", "VWAP_REVERSION", "EMA_CROSSOVER", "RANGE_FADE"]:
        results[name] = run_strategy(df, name)
    return results

def main():
    symbols = get_symbols()
    print("="*80)
    print("ALTERNATIVE PARADIGM DISCOVERY ENGINE")
    print("="*80)
    print(f"Testing {len(symbols)} symbols on 5M...")
    print("Strategies: Mean Reversion, Momentum, VWAP, Crossover, Range Fade")
    print()
    
    total_trades = {
        "MEAN_REVERSION_BB": [],
        "MOMENTUM_BREAKOUT": [],
        "VWAP_REVERSION": [],
        "EMA_CROSSOVER": [],
        "RANGE_FADE": []
    }
    
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
                if count % 5 == 0:
                    print(f"  Processed {count}/{len(symbols)} symbols...")
            except Exception as e:
                print(f"  Error processing {sym}: {e}")
    
    print("\n" + "="*80)
    print(f"{'Strategy':<25} | {'WR':<10} | {'Avg R':<10} | {'Total R':<10} | {'N':<6}")
    print("-"*80)
    
    winners = []
    for name, trades in total_trades.items():
        if trades:
            wr = sum(1 for t in trades if t['win']) / len(trades) * 100
            total_r = sum(t['r'] for t in trades)
            avg_r = total_r / len(trades)
            print(f"{name:<25} | {wr:8.1f}% | {avg_r:+9.3f}R | {total_r:+9.1f}R | {len(trades):<6}")
            
            if avg_r > 0:
                winners.append((name, avg_r, wr, total_r, len(trades)))
        else:
            print(f"{name:<25} | NO TRADES")
    
    print("="*80)
    
    if winners:
        print("\n🏆 PROFITABLE STRATEGIES FOUND:")
        for name, avg_r, wr, total_r, n in winners:
            print(f"  {name}: {avg_r:+.3f}R avg | {wr:.1f}% WR | {total_r:+.1f}R total | N={n}")
    else:
        print("\n❌ No profitable strategies found in this paradigm set")

if __name__ == "__main__":
    main()
