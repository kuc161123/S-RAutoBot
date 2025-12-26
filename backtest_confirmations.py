#!/usr/bin/env python3
"""
CONFIRMATION FILTERS BACKTEST (5M)
===================================

Goal: Add CONFIRMATION signals to reduce false divergence entries.
Base: RSI Divergence (current bot)
New: Require additional confirmation before entry.

Confirmations to Test:
1. ENGULFING: Bullish/Bearish engulfing candle
2. HAMMER: Long lower wick (>2x body) for longs, upper wick for shorts
3. STRUCTURE_BREAK: Close > Recent Swing High (like Wysetrade)
4. VOLUME_SPIKE: Volume > 1.5x MA(20)
5. TREND_ALIGN: Price > EMA50 for longs, < EMA50 for shorts
6. COMBO: Multiple confirmations required

Author: AutoBot Architect
"""

import pandas as pd
import numpy as np
import requests
import time
import itertools
import sys

SYMBOLS = [
    'BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'XRPUSDT', 'DOGEUSDT',
    'BNBUSDT', 'ADAUSDT', 'AVAXUSDT', 'LINKUSDT', 'DOTUSDT',
    'LTCUSDT', 'NEARUSDT', 'APTUSDT', 'SUIUSDT', 'ARBUSDT',
    'OPUSDT', 'ATOMUSDT', 'UNIUSDT', 'INJUSDT', 'TONUSDT'
]

DAYS = 60
TIMEFRAME = 5
WIN_COST = 0.0006
LOSS_COST = 0.00125

def fetch_data(symbol):
    try:
        url = "https://api.bybit.com/v5/market/kline"
        all_kline = []
        end_ts = int(time.time() * 1000)
        start_ts = int((time.time() - DAYS * 24 * 3600) * 1000)
        while end_ts > start_ts:
            params = {'category': 'linear', 'symbol': symbol, 'interval': str(TIMEFRAME), 'limit': 1000, 'end': end_ts}
            r = requests.get(url, params=params).json()
            if r['retCode'] != 0 or not r['result']['list']: break
            klines = r['result']['list']
            all_kline.extend(klines)
            end_ts = int(klines[-1][0]) - 1
            time.sleep(0.04)
        if not all_kline: return pd.DataFrame()
        df = pd.DataFrame(all_kline, columns=['ts', 'open', 'high', 'low', 'close', 'vol', 'to'])
        df = df.iloc[::-1].reset_index(drop=True)
        for c in ['open', 'high', 'low', 'close', 'vol']: df[c] = df[c].astype(float)
        df['datetime'] = pd.to_datetime(df['ts'].astype(float), unit='ms')
        df['hour'] = df['datetime'].dt.hour
        return df
    except: return pd.DataFrame()

def calc_indicators(df):
    df = df.copy()
    close = df['close']
    
    # RSI
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 1e-9)
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # ATR
    h, l, c_prev = df['high'], df['low'], close.shift(1)
    tr = pd.concat([h-l, (h-c_prev).abs(), (l-c_prev).abs()], axis=1).max(axis=1)
    df['atr'] = tr.rolling(14).mean()
    
    # EMAs
    df['ema50'] = close.ewm(span=50).mean()
    df['ema200'] = close.ewm(span=200).mean()
    
    # Volume
    df['vol_ma'] = df['vol'].rolling(20).mean()
    
    # Divergence Windows
    df['price_low_14'] = df['low'].rolling(14).min()
    df['price_high_14'] = df['high'].rolling(14).max()
    df['rsi_low_14'] = df['rsi'].rolling(14).min()
    df['rsi_high_14'] = df['rsi'].rolling(14).max()
    
    # Swing Points (for structure break)
    df['swing_high_10'] = df['high'].rolling(10).max()
    df['swing_low_10'] = df['low'].rolling(10).min()
    
    # Candle Patterns
    df['body'] = abs(df['close'] - df['open'])
    df['upper_wick'] = df['high'] - df[['close', 'open']].max(axis=1)
    df['lower_wick'] = df[['close', 'open']].min(axis=1) - df['low']
    df['total_range'] = df['high'] - df['low']
    
    # Engulfing
    df['bullish_engulf'] = (
        (df['close'] > df['open']) &  # Current green
        (df['close'].shift(1) < df['open'].shift(1)) &  # Prev red
        (df['open'] < df['close'].shift(1)) &  # Opens below prev close
        (df['close'] > df['open'].shift(1))  # Closes above prev open
    )
    
    df['bearish_engulf'] = (
        (df['close'] < df['open']) &
        (df['close'].shift(1) > df['open'].shift(1)) &
        (df['open'] > df['close'].shift(1)) &
        (df['close'] < df['open'].shift(1))
    )
    
    # Hammer / Shooting Star
    df['hammer'] = (
        (df['lower_wick'] > 2 * df['body']) &
        (df['upper_wick'] < df['body']) &
        (df['close'] > df['open'])  # Bullish
    )
    
    df['shooting_star'] = (
        (df['upper_wick'] > 2 * df['body']) &
        (df['lower_wick'] < df['body']) &
        (df['close'] < df['open'])  # Bearish
    )
    
    return df

def detect_divergences(df):
    df = df.copy()
    
    # Regular Bullish
    df['reg_bull'] = (
        (df['low'] <= df['price_low_14']) &
        (df['rsi'] > df['rsi_low_14'].shift(14)) &
        (df['rsi'] < 40)
    )
    
    # Regular Bearish
    df['reg_bear'] = (
        (df['high'] >= df['price_high_14']) &
        (df['rsi'] < df['rsi_high_14'].shift(14)) &
        (df['rsi'] > 60)
    )
    
    return df

def backtest_confirmation(datasets, config):
    total_r = 0
    total_trades = 0
    
    rr = config['rr']
    sl_mult = config['sl_mult']
    confirmation = config['confirmation']
    time_filter = config['time_filter']
    
    for sym, df in datasets.items():
        df = detect_divergences(df)
        
        cooldown = 0
        for i in range(50, len(df)-1):
            if cooldown > 0: cooldown -= 1; continue
            
            row = df.iloc[i]
            
            # Time Filter
            if time_filter == 'ny_session' and not (13 <= row['hour'] < 21): continue
            
            # Divergence Signal
            side = None
            if row['reg_bull']: side = 'long'
            elif row['reg_bear']: side = 'short'
            
            if not side: continue
            
            # CONFIRMATION CHECK
            confirmed = False
            
            if confirmation == 'engulfing':
                if side == 'long' and row['bullish_engulf']: confirmed = True
                if side == 'short' and row['bearish_engulf']: confirmed = True
                
            elif confirmation == 'hammer':
                if side == 'long' and row['hammer']: confirmed = True
                if side == 'short' and row['shooting_star']: confirmed = True
                
            elif confirmation == 'structure_break':
                # Check NEXT candle for structure break
                if i+1 >= len(df): continue
                next_row = df.iloc[i+1]
                if side == 'long' and next_row['close'] > row['swing_high_10']: confirmed = True
                if side == 'short' and next_row['close'] < row['swing_low_10']: confirmed = True
                
            elif confirmation == 'volume_spike':
                if row['vol'] > row['vol_ma'] * 1.5: confirmed = True
                
            elif confirmation == 'trend_align':
                if side == 'long' and row['close'] > row['ema50']: confirmed = True
                if side == 'short' and row['close'] < row['ema50']: confirmed = True
                
            elif confirmation == 'combo_2':
                # Volume + Trend
                vol_ok = row['vol'] > row['vol_ma'] * 1.3
                trend_ok = (side == 'long' and row['close'] > row['ema50']) or (side == 'short' and row['close'] < row['ema50'])
                if vol_ok and trend_ok: confirmed = True
                
            elif confirmation == 'none':
                confirmed = True
            
            if not confirmed: continue
            
            # Entry
            entry = df.iloc[i+1]['open']
            atr = row['atr']
            if pd.isna(atr) or atr == 0: continue
            
            sl_dist = atr * sl_mult
            if sl_dist/entry > 0.05: continue
            tp_dist = sl_dist * rr
            
            if side == 'long':
                sl = entry - sl_dist
                tp = entry + tp_dist
            else:
                sl = entry + sl_dist
                tp = entry - tp_dist
            
            # Simulate
            outcome = 'timeout'
            for j in range(i+1, min(i+300, len(df))):
                c = df.iloc[j]
                if side == 'long':
                    if c['low'] <= sl: outcome = 'loss'; break
                    if c['high'] >= tp: outcome = 'win'; break
                else:
                    if c['high'] >= sl: outcome = 'loss'; break
                    if c['low'] <= tp: outcome = 'win'; break
            
            risk_pct = sl_dist / entry
            res_r = 0
            if outcome == 'win': res_r = rr - (WIN_COST / risk_pct)
            elif outcome == 'loss': res_r = -1.0 - (LOSS_COST / risk_pct)
            elif outcome == 'timeout': res_r = -0.1
            
            total_r += res_r
            total_trades += 1
            cooldown = 5
    
    return {'net_r': total_r, 'trades': total_trades, 'avg_r': total_r/total_trades if total_trades else -999}

def main():
    print("🚀 CONFIRMATION FILTERS BACKTEST")
    print("Loading data...")
    
    datasets = {}
    for sym in SYMBOLS:
        df = fetch_data(sym)
        if not df.empty: datasets[sym] = calc_indicators(df)
    
    print(f"Loaded {len(datasets)} symbols.\n")
    
    # Test Grid
    GRID = {
        'rr': [2.0, 2.5, 3.0],
        'sl_mult': [0.8, 1.0, 1.2],
        'confirmation': ['none', 'engulfing', 'hammer', 'structure_break', 'volume_spike', 'trend_align', 'combo_2'],
        'time_filter': ['all_day', 'ny_session']
    }
    
    combos = [dict(zip(GRID.keys(), v)) for v in itertools.product(*GRID.values())]
    print(f"Testing {len(combos)} configurations...\n")
    
    best_config = None
    best_r = -9999
    
    print(f"{'#':<5} {'RR':<4} {'SL':<4} {'Confirmation':<18} {'Time':<12} | {'Trades':<6} {'Net R':<8} {'Avg R':<8}")
    print("-" * 85)
    
    for i, cfg in enumerate(combos):
        res = backtest_confirmation(datasets, cfg)
        
        if res['avg_r'] > best_r and res['trades'] > 30:
            best_r = res['avg_r']
            best_config = cfg
            print(f"{i+1:<5} {cfg['rr']:<4} {cfg['sl_mult']:<4} {cfg['confirmation']:<18} {cfg['time_filter']:<12} | {res['trades']:<6} {res['net_r']:<8.1f} {res['avg_r']:<8.3f} ✅ NEW BEST")
        
        if (i+1) % 20 == 0:
            sys.stdout.write(f"\rProgress: {i+1}/{len(combos)}...")
            sys.stdout.flush()
    
    print("\n" + "="*85)
    print("🏆 CHAMPION CONFIG")
    print(best_config)
    print(f"Avg R: {best_r:+.3f}")

if __name__ == "__main__":
    main()
