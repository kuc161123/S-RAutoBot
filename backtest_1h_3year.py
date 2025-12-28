#!/usr/bin/env python3
"""
1H MEGA-OPTIMIZATION (3 YEARS DATA)
===================================
Fair comparison with 4H backtest using same parameters:
- 3 years (1095 days) of historical data
- 100 liquid symbols
- EMA 200 trend filter
- Per-symbol R:R optimization
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime
import time
import os
import yaml
import warnings
warnings.filterwarnings('ignore')

# === CONFIGURATION ===
TIMEFRAME = '60'  # 1 Hour
DATA_DAYS = 1095  # 3 Years
MIN_TURNOVER = 5_000_000  # $5M daily volume
MAX_SYMBOLS = 100
MAX_WAIT_CANDLES = 6
SL_MULT = 1.0

# RR Ratios to test
RR_RATIOS = [2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 7.0, 8.0]

# Costs 
SLIPPAGE_PCT = 0.0002
FEE_PCT = 0.0006

BASE_URL = "https://api.bybit.com"

# Indicator settings
RSI_PERIOD = 14
EMA_PERIOD = 200

def get_top_symbols(n=100):
    """Fetch top N liquid USDT perpetuals"""
    try:
        print(f"Fetching top {n} liquid symbols...")
        resp = requests.get(f"{BASE_URL}/v5/market/tickers?category=linear", timeout=10)
        tickers = resp.json().get('result', {}).get('list', [])
        
        usdt = []
        for t in tickers:
            if t['symbol'].endswith('USDT'):
                turnover = float(t.get('turnover24h', 0))
                if turnover >= MIN_TURNOVER:
                    usdt.append({'symbol': t['symbol'], 'turnover': turnover})
        
        usdt.sort(key=lambda x: x['turnover'], reverse=True)
        symbols = [t['symbol'] for t in usdt[:n]]
        print(f"Selected {len(symbols)} symbols")
        return symbols
    except Exception as e:
        print(f"Error: {e}")
        return []

def fetch_klines(symbol, interval, days):
    """Fetch klines with pagination for 3 years of 1H data"""
    end_ts = int(datetime.now().timestamp() * 1000)
    start_ts = end_ts - (days * 24 * 60 * 60 * 1000)
    
    all_candles = []
    current_end = end_ts
    max_iterations = 50  # 3 years of 1H = ~26,280 candles = ~27 API calls
    
    while current_end > start_ts and max_iterations > 0:
        max_iterations -= 1
        params = {
            'category': 'linear', 
            'symbol': symbol, 
            'interval': interval, 
            'limit': 1000, 
            'end': current_end
        }
        
        try:
            resp = requests.get(f"{BASE_URL}/v5/market/kline", params=params, timeout=15)
            data = resp.json().get('result', {}).get('list', [])
            
            if not data:
                break
                
            all_candles.extend(data)
            oldest = int(data[-1][0])
            current_end = oldest - 1
            
            if len(data) < 1000:
                break
                
            time.sleep(0.15)
            
        except Exception as e:
            time.sleep(1)
            continue
    
    if not all_candles:
        return pd.DataFrame()
    
    df = pd.DataFrame(all_candles, columns=['start', 'open', 'high', 'low', 'close', 'volume', 'turnover'])
    df['start'] = pd.to_datetime(df['start'].astype(int), unit='ms')
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = df[col].astype(float)
        
    df.set_index('start', inplace=True)
    df.sort_index(inplace=True)
    df = df[~df.index.duplicated(keep='first')]
    
    return df

def prepare_data(df):
    """Calculate indicators"""
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
    
    return df.dropna()

def find_pivots(data, left=3, right=3):
    n = len(data)
    pivot_highs = np.full(n, np.nan)
    pivot_lows = np.full(n, np.nan)
    
    for i in range(left, n - right):
        window = data[i-left : i+right+1]
        center = data[i]
        
        if len(window) != (left + right + 1): continue
        
        if center == max(window) and list(window).count(center) == 1:
            pivot_highs[i] = center
            
        if center == min(window) and list(window).count(center) == 1:
            pivot_lows[i] = center
            
    return pivot_highs, pivot_lows

def detect_divergences(df):
    close = df['close'].values
    high = df['high'].values
    low = df['low'].values
    rsi = df['rsi'].values
    n = len(df)
    
    price_ph, price_pl = find_pivots(close, 3, 3)
    signals = []
    
    for i in range(50, n):
        # BULLISH
        p_lows = []
        for j in range(i-3, max(0, i-50), -1):
            if not np.isnan(price_pl[j]):
                p_lows.append((j, price_pl[j]))
                if len(p_lows) >= 2: break
        
        if len(p_lows) == 2:
            curr_idx, curr_val = p_lows[0]
            prev_idx, prev_val = p_lows[1]
            
            if (i - curr_idx) <= 10:
                if curr_val < prev_val and rsi[curr_idx] > rsi[prev_idx]:
                    is_new = all(s['idx'] != curr_idx or s['side'] != 'long' for s in signals)
                    if is_new:
                        swing_high = max(high[curr_idx:i+1])
                        signals.append({
                            'idx': curr_idx,
                            'conf_idx': i,
                            'side': 'long',
                            'swing': swing_high,
                            'price': curr_val
                        })

        # BEARISH
        p_highs = [] 
        for j in range(i-3, max(0, i-50), -1):
            if not np.isnan(price_ph[j]):
                p_highs.append((j, price_ph[j]))
                if len(p_highs) >= 2: break
        
        if len(p_highs) == 2:
            curr_idx, curr_val = p_highs[0]
            prev_idx, prev_val = p_highs[1]
            
            if (i - curr_idx) <= 10:
                if curr_val > prev_val and rsi[curr_idx] < rsi[prev_idx]:
                    is_new = all(s['idx'] != curr_idx or s['side'] != 'short' for s in signals)
                    if is_new:
                        swing_low = min(low[curr_idx:i+1])
                        signals.append({
                            'idx': curr_idx,
                            'conf_idx': i,
                            'side': 'short',
                            'swing': swing_low,
                            'price': curr_val
                        })
                            
    return signals

def backtest_symbol(df, signals, rr):
    rows = list(df.itertuples())
    trades = []
    
    for sig in signals:
        start_idx = sig['conf_idx']
        side = sig['side']
        
        curr_price = rows[start_idx].close
        ema = rows[start_idx].ema
        
        if side == 'long' and curr_price < ema: continue
        if side == 'short' and curr_price > ema: continue
        
        entry_idx = None
        
        for j in range(start_idx + 1, min(start_idx + 1 + MAX_WAIT_CANDLES, len(rows))):
            row = rows[j]
            if side == 'long':
                if row.close > sig['swing']:
                    entry_idx = j + 1
                    break
            else:
                if row.close < sig['swing']:
                    entry_idx = j + 1
                    break
        
        if not entry_idx or entry_idx >= len(rows): continue
        
        entry_row = rows[entry_idx]
        entry_price = entry_row.open
        atr = entry_row.atr
        sl_dist = atr * SL_MULT
        
        if side == 'long':
            entry_price *= (1 + SLIPPAGE_PCT)
            tp_price = entry_price + (sl_dist * rr)
            sl_price = entry_price - sl_dist
        else:
            entry_price *= (1 - SLIPPAGE_PCT)
            tp_price = entry_price - (sl_dist * rr)
            sl_price = entry_price + sl_dist
            
        result = None
        
        for k in range(entry_idx, min(entry_idx + 300, len(rows))):
            row = rows[k]
            if side == 'long':
                if row.low <= sl_price:
                    result = -1.0
                    break
                if row.high >= tp_price:
                    result = rr
                    break
            else:
                if row.high >= sl_price:
                    result = -1.0
                    break
                if row.low <= tp_price:
                    result = rr
                    break
                    
        if result is None:
            result = -0.1
            
        risk_pct = abs(entry_price - sl_price) / entry_price
        if risk_pct == 0: risk_pct = 0.01
        total_fee_cost = (FEE_PCT * 2) / risk_pct
        final_r = result - total_fee_cost
        trades.append(final_r)
        
    return trades

def main():
    print("="*80)
    print("1H MEGA-OPTIMIZATION (3 YEARS DATA)")
    print("Fair comparison with 4H backtest")
    print("="*80)
    
    symbols = get_top_symbols(MAX_SYMBOLS)
    results = []
    output_file = '1h_3year_optimization_results.csv'
    
    # Resume support
    processed = set()
    if os.path.exists(output_file):
        try:
            existing = pd.read_csv(output_file)
            processed = set(existing['symbol'].unique())
            print(f"Resuming... {len(processed)} symbols already done.")
        except:
            pass
    
    for i, sym in enumerate(symbols):
        if sym in processed:
            continue
            
        print(f"\n[{i+1}/{len(symbols)}] {sym}...", end=" ", flush=True)
        
        try:
            df = fetch_klines(sym, TIMEFRAME, DATA_DAYS)
            print(f"({len(df)} candles)", end=" ", flush=True)
            
            if len(df) < 5000:
                print("⏭️ Insufficient data")
                continue
                
            df = prepare_data(df)
            signals = detect_divergences(df)
            
            if len(signals) < 10:
                print(f"⏭️ Only {len(signals)} signals")
                continue
                
            best_r = -999
            best_rr = 0
            best_wr = 0
            best_n = 0
            
            for rr in RR_RATIOS:
                trades = backtest_symbol(df, signals, rr)
                if not trades: continue
                
                n = len(trades)
                total_r = sum(trades)
                wr = len([t for t in trades if t > 0]) / n
                
                if total_r > best_r:
                    best_r = total_r
                    best_rr = rr
                    best_wr = wr
                    best_n = n
            
            status = "✅" if best_r > 0 else "❌"
            print(f"{status} {best_r:+.1f}R @ {best_rr}:1 (WR:{best_wr*100:.0f}%, N:{best_n})")
            
            res = {
                'symbol': sym,
                'best_rr': best_rr,
                'total_r': round(best_r, 2),
                'win_rate': round(best_wr, 3),
                'trades': best_n,
                'avg_r': round(best_r / best_n, 3) if best_n > 0 else 0
            }
            
            # Save immediately
            df_res = pd.DataFrame([res])
            hdr = not os.path.exists(output_file)
            df_res.to_csv(output_file, mode='a', header=hdr, index=False)
            results.append(res)
                
        except Exception as e:
            print(f"❌ Error: {e}")
            
    # Final summary
    print("\n" + "="*80)
    print("FINAL RESULTS - 1H (3 YEARS)")
    print("="*80)
    
    try:
        final_df = pd.read_csv(output_file)
        
        winners = final_df[final_df['total_r'] > 0]
        losers = final_df[final_df['total_r'] <= 0]
        
        total_r = final_df['total_r'].sum()
        avg_r = final_df['total_r'].mean()
        
        print(f"\n📊 SUMMARY:")
        print(f"   Total Symbols Tested: {len(final_df)}")
        print(f"   Profitable: {len(winners)} ({len(winners)/len(final_df)*100:.0f}%)")
        print(f"   Losing: {len(losers)} ({len(losers)/len(final_df)*100:.0f}%)")
        print(f"   Combined R: {total_r:+.1f}")
        print(f"   Avg R/Symbol: {avg_r:+.2f}")
        
        print(f"\n🏆 TOP 10 PERFORMERS:")
        top10 = final_df.nlargest(10, 'total_r')
        for _, row in top10.iterrows():
            print(f"   {row['symbol']}: {row['total_r']:+.1f}R @ {row['best_rr']}:1 (WR:{row['win_rate']*100:.0f}%, N:{row['trades']})")
        
        print(f"\n📉 BOTTOM 5:")
        bottom5 = final_df.nsmallest(5, 'total_r')
        for _, row in bottom5.iterrows():
            print(f"   {row['symbol']}: {row['total_r']:+.1f}R")
        
        # Compare to 4H
        print(f"\n📊 VS 4H COMPARISON:")
        print(f"   1H (3yr): {total_r:+.1f}R total | {len(winners)} winners | {avg_r:+.2f}R avg")
        print(f"   4H (3yr): +445R total | 24 winners | +18R avg (per winner)")
        
    except Exception as e:
        print(f"Error reading results: {e}")

if __name__ == "__main__":
    main()
