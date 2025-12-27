#!/usr/bin/env python3
"""
CREATIVE LAB BACKTEST ENGINE
============================

Goal: Find a High-Profit, High-Frequency Strategy on 30M/1H timeframes.
Scope: Top 20 Crypto Assets.
Strategies Tested:
1. WYSE_RELAXED: RSI Div + Trends (EMA200) + Structure Break (No Key Level filter).
2. TREND_BREAK: Price > EMA200 + Break of N-period High.
3. BB_SQUEEZE: Bollinger Band Squeeze + Breakout.

Author: AutoBot Architect
"""

import pandas as pd
import numpy as np
import requests
import time
import itertools
from typing import Dict, List

# ============================================
# CONFIGURATION
# ============================================

SYMBOLS = [
    'BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'BNBUSDT', 'XRPUSDT', 
    'ADAUSDT', 'DOGEUSDT', 'AVAXUSDT', 'LINKUSDT', 'DOTUSDT',
    'LTCUSDT', 'MATICUSDT', 'UNIUSDT', 'ATOMUSDT', 'IMXUSDT',
    'NEARUSDT', 'ETCUSDT', 'FILUSDT', 'HBARUSDT', 'APTUSDT'
]

TIMEFRAMES = [30, 60] # 30M, 1H
DAYS = 120 # 4 months

# Fee Model (Realistic Bybit)
WIN_COST_PCT = 0.0006
LOSS_COST_PCT = 0.00125

# ============================================
# DATA & INDICATORS
# ============================================

def fetch_data(symbol: str, interval: int) -> pd.DataFrame:
    try:
        url = "https://api.bybit.com/v5/market/kline"
        all_kline = []
        end_ts = int(time.time() * 1000)
        start_ts = int((time.time() - DAYS * 24 * 3600) * 1000)
        
        while end_ts > start_ts:
            params = {'category': 'linear', 'symbol': symbol, 'interval': str(interval), 'limit': 1000, 'end': end_ts}
            r = requests.get(url, params=params).json()
            if r['retCode'] != 0 or not r['result']['list']: break
            klines = r['result']['list']
            all_kline.extend(klines)
            end_ts = int(klines[-1][0]) - 1
            time.sleep(0.05)
            
        df = pd.DataFrame(all_kline, columns=['ts', 'open', 'high', 'low', 'close', 'vol', 'to'])
        df = df.iloc[::-1].reset_index(drop=True)
        for c in ['open', 'high', 'low', 'close', 'vol']: df[c] = df[c].astype(float)
        return df
    except: return pd.DataFrame()

def prepare_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    close = df['close']
    
    # RSI 14
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 1e-9)
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # EMAs
    df['ema50'] = close.ewm(span=50).mean()
    df['ema200'] = close.ewm(span=200).mean()
    
    # ATR
    h, l, c_prev = df['high'], df['low'], close.shift(1)
    tr = pd.concat([h-l, (h-c_prev).abs(), (l-c_prev).abs()], axis=1).max(axis=1)
    df['atr'] = tr.rolling(14).mean()
    
    # Donchian / Swing Highs (20 period)
    df['high_20'] = df['high'].rolling(20).max()
    df['low_20'] = df['low'].rolling(20).min()
    
    # Bollinger Bands
    sma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    df['bb_upper'] = sma20 + (2 * std20)
    df['bb_lower'] = sma20 - (2 * std20)
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / sma20
    
    return df

# ============================================
# STRATEGIES
# ============================================

def run_wyse_relaxed(df: pd.DataFrame, rr=2.0, trailing=True) -> List[Dict]:
    """
    RSI Divergence + Trend Alignment (EMA 200) + Structure Break.
    Less strict than original Wysetrade (No Key Level).
    """
    trades = []
    cooldown = 0
    potential_long = None
    potential_short = None
    
    for i in range(200, len(df)-1):
        if cooldown > 0: cooldown -= 1; continue
        row = df.iloc[i]
        
        # Setup: Div + Trend
        # Bull: Price > EMA200 (Uptrend Pullback) AND Bull Div
        # Bear: Price < EMA200 (Downtrend Pullback) AND Bear Div
        
        # Simplified Div Check (Price Low min 10, RSI HL)
        bull_div = (row['low'] <= df['low'].iloc[i-10:i].min() and row['rsi'] > df['rsi'].iloc[i-10:i].min() and row['rsi'] < 45)
        bear_div = (row['high'] >= df['high'].iloc[i-10:i].max() and row['rsi'] < df['rsi'].iloc[i-10:i].max() and row['rsi'] > 55)
        
        # Trend Filter
        bull_trend = row['close'] > row['ema200']
        bear_trend = row['close'] < row['ema200']
        
        if bull_div and bull_trend:
            potential_long = {'idx': i, 'trigger': df['high'].iloc[i-5:i].max(), 'sl': df['low'].iloc[i-5:i].min()} # Tighter trigger
            potential_short = None
            
        if bear_div and bear_trend:
            potential_short = {'idx': i, 'trigger': df['low'].iloc[i-5:i].min(), 'sl': df['high'].iloc[i-5:i].max()}
            potential_long = None
            
        # Trigger
        entry=0; sl=0; side=None
        if potential_long:
            if i - potential_long['idx'] > 15: potential_long = None
            elif row['close'] > potential_long['trigger']:
                side='long'; entry=df.iloc[i+1]['open']; sl=potential_long['sl']; potential_long=None
                
        elif potential_short:
             if i - potential_short['idx'] > 15: potential_short = None
             elif row['close'] < potential_short['trigger']:
                 side='short'; entry=df.iloc[i+1]['open']; sl=potential_short['sl']; potential_short=None
                 
        if side:
            trades.append(simulate_trade(df, i, side, entry, sl, rr, trailing))
            cooldown = 5
            
    return trades

def run_trend_break(df: pd.DataFrame, rr=2.0, trailing=True) -> List[Dict]:
    """
    Pure Trend Following.
    Long: Price > EMA50 > EMA200. Break of 20-period High.
    Short: Price < EMA50 < EMA200. Break of 20-period Low.
    """
    trades = []
    cooldown = 0
    
    for i in range(200, len(df)-1):
        if cooldown > 0: cooldown -= 1; continue
        row = df.iloc[i]
        
        # Trend Align
        bull_trend = row['ema50'] > row['ema200'] and row['close'] > row['ema50']
        bear_trend = row['ema50'] < row['ema200'] and row['close'] < row['ema50']
        
        side=None; entry=0; sl=0
        
        if bull_trend and row['close'] > df['high'].iloc[i-20:i].max():
            side='long'; entry=df.iloc[i+1]['open']
            sl = df['low'].iloc[i-10:i].min() # SL at recent swing low
            
        elif bear_trend and row['close'] < df['low'].iloc[i-20:i].min():
            side='short'; entry=df.iloc[i+1]['open']
            sl = df['high'].iloc[i-10:i].max()
            
        if side:
            trades.append(simulate_trade(df, i, side, entry, sl, rr, trailing))
            cooldown = 10 
            
    return trades

def run_bb_squeeze(df: pd.DataFrame, rr=1.5, trailing=False) -> List[Dict]:
    """
    Volatility Expansion.
    Setup: BB Width < 0.05 (Squeeze).
    Trigger: Close > Upper (Long) or Close < Lower (Short).
    """
    trades = []
    cooldown = 0
    
    for i in range(200, len(df)-1):
        if cooldown > 0: cooldown -= 1; continue
        row = df.iloc[i]
        
        # Check Squeeze lookback
        was_squeeze = df['bb_width'].iloc[i-5:i].min() < 0.10 # Recent squeeze
        
        side=None; entry=0; sl=0
        
        if was_squeeze and row['close'] > row['bb_upper']:
            side='long'; entry=df.iloc[i+1]['open']
            sl = row['bb_lower'] # Wide stop
            
        elif was_squeeze and row['close'] < row['bb_lower']:
            side='short'; entry=df.iloc[i+1]['open']
            sl = row['bb_upper']
            
        if side:
            trades.append(simulate_trade(df, i, side, entry, sl, rr, trailing))
            cooldown = 10
            
    return trades

def simulate_trade(df, i, side, entry, sl, rr, trailing):
    # Sanity checks
    if entry <= 0 or sl <= 0: return {'res': -0.1} # Error
    risk = abs(entry - sl)
    if risk == 0 or risk/entry > 0.05: return {'res': 0} # Skip too wide/tight
    
    tp_dist = risk * rr
    tp_price = entry + tp_dist if side == 'long' else entry - tp_dist
    
    outcome = 'loss'
    bars = 0
    curr_sl = sl
    be_hit = False
    
    for j in range(i+1, min(i+1000, len(df))):
        c = df.iloc[j]
        bars += 1
        
        if side == 'long':
            if c['low'] <= curr_sl:
                outcome = 'loss' if not be_hit else 'be'
                break
            if c['high'] >= tp_price and not trailing:
                outcome = 'win'
                break
                
            # Trail
            if trailing:
                if c['high'] >= entry + risk: curr_sl = entry; be_hit = True
                if c['high'] >= entry + (risk*2): curr_sl = entry + risk
                if c['high'] >= tp_price: outcome = 'win'; break
                
        else: # Short
            if c['high'] >= curr_sl:
                outcome = 'loss' if not be_hit else 'be'
                break
            if c['low'] <= tp_price and not trailing:
                outcome = 'win'
                break
                
            if trailing:
                if c['low'] <= entry - risk: curr_sl = entry; be_hit = True
                if c['low'] <= entry - (risk*2): curr_sl = entry - risk
                if c['low'] <= tp_price: outcome = 'win'; break
                
    risk_pct = risk/entry
    res_r = 0
    if outcome == 'win': res_r = rr - (WIN_COST_PCT / risk_pct)
    elif outcome == 'loss': res_r = -1.0 - (LOSS_COST_PCT / risk_pct)
    elif outcome == 'be': res_r = 0.0 - (WIN_COST_PCT / risk_pct)
    elif outcome == 'timeout': res_r = -0.1 # Small penalty for stuck trade
    
    return {'res': res_r, 'bars': bars}

# ============================================
# MAIN
# ============================================

def main():
    print("🚀 CREATIVE LAB: 30M/1H STRATEGY SEARCH")
    print("-" * 60)
    
    for tf in TIMEFRAMES:
        print(f"\n📊 TIMEFRAME {tf}m")
        # Load Data
        print("   Loading data...", end='\r')
        datasets = {}
        for sym in SYMBOLS:
            df = fetch_data(sym, tf)
            if not df.empty: datasets[sym] = prepare_indicators(df)
        print(f"   Loaded {len(datasets)} symbols.")
        
        # Test Strategies
        strategies = [
            ('Wyse Relaxed (2R, Trail)', lambda df: run_wyse_relaxed(df, 2.0, True)),
            ('Trend Break (2.5R, Trail)', lambda df: run_trend_break(df, 2.5, True)),
            ('BB Squeeze (1.5R, Fixed)', lambda df: run_bb_squeeze(df, 1.5, False))
        ]
        
        print(f"   {'Strategy':<25} | {'Trades':<8} {'Net R':<8} {'Avg R':<8}")
        print("   " + "-" * 55)
        
        for name, func in strategies:
            all_trades = []
            for sym, df in datasets.items():
                all_trades.extend(func(df))
            
            if not all_trades:
                print(f"   {name:<25} | 0        0.0      0.0")
                continue
                
            total_r = sum(t['res'] for t in all_trades)
            avg_r = total_r / len(all_trades)
            
            print(f"   {name:<25} | {len(all_trades):<8} {total_r:<8.1f} {avg_r:<8.3f} {'🏆' if avg_r > 0.05 and len(all_trades) > 200 else ''}")

if __name__ == "__main__":
    main()
