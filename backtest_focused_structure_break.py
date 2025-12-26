#!/usr/bin/env python3
"""
FOCUSED BACKTEST - Structure Break (4 Candle Wait)
==================================================

Test ONLY the successful configuration:
- RR: 3.0
- SL: 0.8 ATR
- Confirmation: Structure Break (wait up to 4 candles)
- Time: All-Day
- Divergence: Regular only (bullish + bearish)

Original result (1 candle wait): +1.854R/trade (92 trades)
Testing with 4 candle wait to match bot config
"""

import pandas as pd
import numpy as np
import requests
import time

SYMBOLS = [
    'BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'XRPUSDT', 'DOGEUSDT',
    'BNBUSDT', 'ADAUSDT', 'AVAXUSDT', 'LINKUSDT', 'DOTUSDT',
    'LTCUSDT', 'NEARUSDT', 'APTUSDT', 'SUIUSDT', 'ARBUSDT',
    'OPUSDT', 'ATOMUSDT', 'UNIUSDT', 'INJUSDT', 'TONUSDT'
]

DAYS = 60
TIMEFRAME = 5
RR = 3.0
SL_MULT = 0.8
MAX_WAIT_CANDLES = 4  # Match bot config
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
    
    # Swing Points (10 bars)
    df['swing_high_10'] = df['high'].rolling(10).max()
    df['swing_low_10'] = df['low'].rolling(10).min()
    
    # Divergence detection
    df['price_low_14'] = df['low'].rolling(14).min()
    df['price_high_14'] = df['high'].rolling(14).max()
    df['rsi_low_14'] = df['rsi'].rolling(14).min()
    df['rsi_high_14'] = df['rsi'].rolling(14).max()
    
    return df

def backtest_strategy(datasets):
    all_trades = []
    
    for sym, df in datasets.items():
        # Detect divergences
        df['reg_bull'] = (
            (df['low'] <= df['price_low_14']) &
            (df['rsi'] > df['rsi_low_14'].shift(14)) &
            (df['rsi'] < 40)
        )
        
        df['reg_bear'] = (
            (df['high'] >= df['price_high_14']) &
            (df['rsi'] < df['rsi_high_14'].shift(14)) &
            (df['rsi'] > 60)
        )
        
        cooldown = 0
        for i in range(50, len(df)-1):
            if cooldown > 0: cooldown -= 1; continue
            
            row = df.iloc[i]
            
            # Divergence signal
            side = None
            if row['reg_bull']: side = 'long'
            elif row['reg_bear']: side = 'short'
            
            if not side: continue
            
            # STRUCTURE BREAK CHECK (wait up to 4 candles)
            structure_broken = False
            candles_waited = 0
            
            for ahead in range(1, MAX_WAIT_CANDLES + 1):
                if i+ahead >= len(df): break
                check_row = df.iloc[i+ahead]
                candles_waited = ahead
                
                if side == 'long' and check_row['close'] > row['swing_high_10']:
                    structure_broken = True
                    break
                if side == 'short' and check_row['close'] < row['swing_low_10']:
                    structure_broken = True
                    break
            
            if not structure_broken: continue
            
            # Entry (candle after structure break)
            entry_idx = i + candles_waited + 1
            if entry_idx >= len(df): continue
            
            entry = df.iloc[entry_idx]['open']
            atr = row['atr']
            if pd.isna(atr) or atr == 0: continue
            
            sl_dist = atr * SL_MULT
            if sl_dist/entry > 0.05: continue
            tp_dist = sl_dist * RR
            
            if side == 'long':
                sl = entry - sl_dist
                tp = entry + tp_dist
            else:
                sl = entry + sl_dist
                tp = entry - tp_dist
            
            # Simulate
            outcome = 'timeout'
            for j in range(entry_idx, min(entry_idx+300, len(df))):
                c = df.iloc[j]
                if side == 'long':
                    if c['low'] <= sl: outcome = 'loss'; break
                    if c['high'] >= tp: outcome = 'win'; break
                else:
                    if c['high'] >= sl: outcome = 'loss'; break
                    if c['low'] <= tp: outcome = 'win'; break
            
            risk_pct = sl_dist / entry
            res_r = 0
            if outcome == 'win': res_r = RR - (WIN_COST / risk_pct)
            elif outcome == 'loss': res_r = -1.0 - (LOSS_COST / risk_pct)
            elif outcome == 'timeout': res_r = -0.1
            
            all_trades.append({
                'symbol': sym,
                'side': side,
                'r': res_r,
                'win': outcome == 'win',
                'candles_waited': candles_waited
            })
            
            cooldown = 5
    
    return all_trades

def main():
    print("🎯 FOCUSED BACKTEST - Structure Break (4 Candle Wait)")
    print("=" * 70)
    print(f"Config: RR={RR}, SL={SL_MULT}x ATR, Max Wait={MAX_WAIT_CANDLES} candles")
    print("=" * 70)
    print("\nLoading data...")
    
    datasets = {}
    for sym in SYMBOLS:
        df = fetch_data(sym)
        if not df.empty: datasets[sym] = calc_indicators(df)
    
    print(f"Loaded {len(datasets)} symbols.\n")
    print("Running backtest...")
    
    trades = backtest_strategy(datasets)
    
    if trades:
        total_r = sum(t['r'] for t in trades)
        avg_r = total_r / len(trades)
        wins = sum(1 for t in trades if t['win'])
        wr = wins / len(trades) * 100
        avg_wait = sum(t['candles_waited'] for t in trades) / len(trades)
        
        print("\n" + "=" * 70)
        print("📊 RESULTS")
        print("=" * 70)
        print(f"Total Trades:        {len(trades)}")
        print(f"Net R:               {total_r:+.1f}R")
        print(f"Avg R per Trade:     {avg_r:+.3f}R")
        print(f"Win Rate:            {wr:.1f}%")
        print(f"Avg Candles Waited:  {avg_wait:.1f} ({avg_wait * 5:.0f} mins)")
        print("=" * 70)
        
        # Comparison
        print("\n📈 COMPARISON TO ORIGINAL (1 candle wait)")
        print("-" * 70)
        print(f"{'Metric':<25} {'Original':<15} {'4-Candle Wait':<15} {'Change'}")
        print("-" * 70)
        print(f"{'Trades':<25} {92:<15} {len(trades):<15} {len(trades)-92:+}")
        print(f"{'Avg R':<25} {'+1.854R':<15} {f'{avg_r:+.3f}R':<15}")
        print(f"{'Win Rate':<25} {'~60%':<15} {f'{wr:.1f}%':<15}")

if __name__ == "__main__":
    main()
