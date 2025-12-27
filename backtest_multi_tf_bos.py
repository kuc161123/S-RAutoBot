#!/usr/bin/env python3
"""
MULTI-TIMEFRAME BOS DIVERGENCE TEST
====================================
Testing 15M, 30M, 1H with best filters:
- Volume > 1.5x MA
- Trend Filter (EMA200)
- Candle direction confirmation
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Config
TIMEFRAMES = ['15', '30', '60']  # 15M, 30M, 1H
DATA_DAYS = 120
NUM_SYMBOLS = 50
MAX_WAIT_CANDLES = 10
SL_ATR_MULT = 0.8

# Test R:R
RR_RATIOS = [1.0, 1.2, 1.5, 2.0]

# Costs
SLIPPAGE_PCT = 0.0003
FEE_PCT = 0.0006

RSI_PERIOD = 14
LOOKBACK_BARS = 10
MIN_PIVOT_DISTANCE = 5

BASE_URL = "https://api.bybit.com"

def get_symbols(limit):
    resp = requests.get(f"{BASE_URL}/v5/market/tickers?category=linear", timeout=10)
    tickers = resp.json().get('result', {}).get('list', [])
    usdt = [t for t in tickers if t['symbol'].endswith('USDT')]
    usdt.sort(key=lambda x: float(x.get('turnover24h', 0)), reverse=True)
    return [t['symbol'] for t in usdt[:limit]]

def fetch_klines(symbol, interval, days):
    end_ts = int(datetime.now().timestamp() * 1000)
    candles_needed = days * 24 * 60 // int(interval)
    all_candles = []
    current_end = end_ts
    
    while len(all_candles) < candles_needed:
        params = {'category': 'linear', 'symbol': symbol, 'interval': interval, 
                  'limit': 1000, 'end': current_end}
        try:
            resp = requests.get(f"{BASE_URL}/v5/market/kline", params=params, timeout=10)
            data = resp.json().get('result', {}).get('list', [])
            if not data: break
            all_candles.extend(data)
            current_end = int(data[-1][0]) - 1
            if len(data) < 1000: break
        except: break
    
    if not all_candles: return pd.DataFrame()
    
    df = pd.DataFrame(all_candles, columns=['start', 'open', 'high', 'low', 'close', 'volume', 'turnover'])
    df['start'] = pd.to_datetime(df['start'].astype(int), unit='ms')
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = df[col].astype(float)
    df.set_index('start', inplace=True)
    df.sort_index(inplace=True)
    return df

def add_indicators(df):
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
    
    df['ema200'] = df['close'].ewm(span=200, adjust=False).mean()
    df['vol_ma'] = df['volume'].rolling(20).mean()
    df['bullish'] = df['close'] > df['open']
    df['bearish'] = df['close'] < df['open']
    
    return df.dropna()

def find_pivots(data, left=3, right=3):
    n = len(data)
    pivot_highs = np.full(n, np.nan)
    pivot_lows = np.full(n, np.nan)
    for i in range(left, n - right):
        is_high = all(data[j] < data[i] for j in range(i - left, i + right + 1) if j != i)
        is_low = all(data[j] > data[i] for j in range(i - left, i + right + 1) if j != i)
        if is_high: pivot_highs[i] = data[i]
        if is_low: pivot_lows[i] = data[i]
    return pivot_highs, pivot_lows

def detect_divergences(df):
    if len(df) < 100: return []
    
    close = df['close'].values
    high = df['high'].values
    low = df['low'].values
    rsi = df['rsi'].values
    n = len(df)
    
    price_ph, price_pl = find_pivots(close, 3, 3)
    signals = []
    
    for i in range(30, n - 15):
        curr_pl = curr_pli = prev_pl = prev_pli = None
        for j in range(i, max(i - LOOKBACK_BARS, 0), -1):
            if not np.isnan(price_pl[j]):
                if curr_pl is None: curr_pl, curr_pli = price_pl[j], j
                elif prev_pl is None and j < curr_pli - MIN_PIVOT_DISTANCE:
                    prev_pl, prev_pli = price_pl[j], j
                    break
        
        curr_ph = curr_phi = prev_ph = prev_phi = None
        for j in range(i, max(i - LOOKBACK_BARS, 0), -1):
            if not np.isnan(price_ph[j]):
                if curr_ph is None: curr_ph, curr_phi = price_ph[j], j
                elif prev_ph is None and j < curr_phi - MIN_PIVOT_DISTANCE:
                    prev_ph, prev_phi = price_ph[j], j
                    break
        
        # Bullish divergence
        if curr_pl and prev_pl:
            if curr_pl < prev_pl and rsi[curr_pli] > rsi[prev_pli]:
                if rsi[i] < 40:
                    swing_high = max(high[max(0, i-LOOKBACK_BARS):i+1])
                    signals.append({'idx': i, 'side': 'long', 'swing': swing_high})
                    continue
        
        # Bearish divergence
        if curr_ph and prev_ph:
            if curr_ph > prev_ph and rsi[curr_phi] < rsi[prev_phi]:
                if rsi[i] > 60:
                    swing_low = min(low[max(0, i-LOOKBACK_BARS):i+1])
                    signals.append({'idx': i, 'side': 'short', 'swing': swing_low})
    
    return signals

def run_backtest(df, signals, rr, use_vol=True, use_trend=True, use_candle=True, vol_mult=1.5):
    rows = list(df.itertuples())
    atr = df['atr'].values
    bullish = df['bullish'].values
    bearish = df['bearish'].values
    volume = df['volume'].values
    vol_ma = df['vol_ma'].values
    close_arr = df['close'].values
    ema200 = df['ema200'].values
    
    trades = []
    
    for sig in signals:
        div_idx = sig['idx']
        side = sig['side']
        
        # Trend filter
        if use_trend:
            if side == 'long' and close_arr[div_idx] < ema200[div_idx]:
                continue
            if side == 'short' and close_arr[div_idx] > ema200[div_idx]:
                continue
        
        bos_idx = None
        for j in range(div_idx + 1, min(div_idx + 1 + MAX_WAIT_CANDLES, len(rows) - 10)):
            # Volume filter
            if use_vol and volume[j] < vol_ma[j] * vol_mult:
                continue
            
            if side == 'long':
                if rows[j].close > sig['swing']:
                    if use_candle and not bullish[j]:
                        continue
                    bos_idx = j
                    break
            else:
                if rows[j].close < sig['swing']:
                    if use_candle and not bearish[j]:
                        continue
                    bos_idx = j
                    break
        
        if bos_idx is None:
            continue
        
        entry_idx = bos_idx + 1
        if entry_idx >= len(rows) - 50:
            continue
        
        entry_price = rows[entry_idx].open
        if side == 'long':
            entry_price *= (1 + SLIPPAGE_PCT)
        else:
            entry_price *= (1 - SLIPPAGE_PCT)
        
        entry_atr = atr[entry_idx - 1]
        sl_dist = entry_atr * SL_ATR_MULT
        
        if side == 'long':
            sl = entry_price - sl_dist
            tp = entry_price + (sl_dist * rr)
        else:
            sl = entry_price + sl_dist
            tp = entry_price - (sl_dist * rr)
        
        result = None
        for k in range(entry_idx, min(entry_idx + 200, len(rows))):
            row = rows[k]
            if side == 'long':
                if row.low <= sl: result = 'loss'; break
                if row.high >= tp: result = 'win'; break
            else:
                if row.high >= sl: result = 'loss'; break
                if row.low <= tp: result = 'win'; break
        
        if result:
            risk_pct = sl_dist / entry_price
            fee_cost = (FEE_PCT * 2 + SLIPPAGE_PCT) / risk_pct
            r = (rr - fee_cost) if result == 'win' else (-1.0 - fee_cost)
            trades.append({'result': result, 'r': r})
    
    return trades

def main():
    print("="*80)
    print("MULTI-TIMEFRAME BOS DIVERGENCE TEST")
    print("="*80)
    print(f"Timeframes: 15M, 30M, 1H")
    print(f"Data: {DATA_DAYS} days, {NUM_SYMBOLS} symbols")
    print(f"Filters: Vol 1.5x + Trend EMA200 + Candle Color")
    print("="*80)
    
    symbols = get_symbols(NUM_SYMBOLS)
    
    best_configs = []
    
    for tf in TIMEFRAMES:
        tf_name = {'15': '15M', '30': '30M', '60': '1H'}[tf]
        print(f"\n{'='*80}")
        print(f"TIMEFRAME: {tf_name}")
        print(f"{'='*80}")
        print(f"Fetching data...")
        
        # Fetch data for this timeframe
        symbol_data = {}
        for idx, sym in enumerate(symbols):
            df = fetch_klines(sym, tf, DATA_DAYS)
            if df.empty or len(df) < 300:
                continue
            df = add_indicators(df)
            if len(df) < 100:
                continue
            signals = detect_divergences(df)
            if signals:
                symbol_data[sym] = (df, signals)
            if (idx + 1) % 10 == 0:
                print(f"  {idx+1}/{len(symbols)} symbols...")
        
        print(f"Symbols with signals: {len(symbol_data)}")
        
        configs = [
            {'name': 'No Filter', 'vol': False, 'trend': False, 'candle': False},
            {'name': 'Vol Only', 'vol': True, 'trend': False, 'candle': False},
            {'name': 'Trend Only', 'vol': False, 'trend': True, 'candle': False},
            {'name': 'Vol + Trend', 'vol': True, 'trend': True, 'candle': False},
            {'name': 'Vol + Trend + Candle', 'vol': True, 'trend': True, 'candle': True},
        ]
        
        print(f"\n{'Config':<25} | {'RR':<4} | {'N':<5} | {'WR':<7} | {'Avg R':<10} | {'Status'}")
        print("-"*75)
        
        for rr in RR_RATIOS:
            for cfg in configs:
                all_trades = []
                for sym, (df, signals) in symbol_data.items():
                    trades = run_backtest(
                        df, signals, rr,
                        use_vol=cfg['vol'],
                        use_trend=cfg['trend'],
                        use_candle=cfg['candle']
                    )
                    all_trades.extend(trades)
                
                if all_trades:
                    wins = sum(1 for t in all_trades if t['result'] == 'win')
                    n = len(all_trades)
                    wr = wins / n * 100
                    total_r = sum(t['r'] for t in all_trades)
                    avg_r = total_r / n
                    
                    status = "✅ PROFIT" if avg_r > 0 else "❌"
                    print(f"{cfg['name']:<25} | {rr:<4.1f} | {n:<5} | {wr:>5.1f}% | {avg_r:>+8.4f}R | {status}")
                    
                    if avg_r > 0:
                        best_configs.append({
                            'tf': tf_name,
                            'cfg': cfg['name'],
                            'rr': rr,
                            'n': n,
                            'wr': wr,
                            'avg_r': avg_r,
                            'total_r': total_r
                        })
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    if best_configs:
        print(f"\n🏆 FOUND {len(best_configs)} PROFITABLE CONFIGURATIONS!")
        best_configs.sort(key=lambda x: x['avg_r'], reverse=True)
        print("\nTop configs by Avg R/Trade:")
        for cfg in best_configs[:5]:
            print(f"  ✅ {cfg['tf']} | {cfg['cfg']} | RR={cfg['rr']} | {cfg['wr']:.1f}% WR | {cfg['avg_r']:+.4f}R avg | {cfg['total_r']:+.1f}R total ({cfg['n']} trades)")
    else:
        print("\n❌ NO PROFITABLE CONFIGURATIONS FOUND ACROSS ALL TIMEFRAMES")

if __name__ == "__main__":
    main()
