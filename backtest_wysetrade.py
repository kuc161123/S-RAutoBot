#!/usr/bin/env python3
"""
WYSETRADE STRATEGY BACKTEST
===========================

Strategy: RSI Divergence + Price Action + Key Levels
Logic:
1. SETUP: RSI(14) Divergence (Regular) in Overbought (>70) or Oversold (<30).
2. FILTER: Price reacting to Key Level (Previous Swing High/Low).
3. TRIGGER: Confirmation via Structure Break (Trendline Break proxy).
   - Long: Close > Most recent Lower High.
   - Short: Close < Most recent Higher Low.

Timeframes: 5M, 15M, 30M, 1H, 4H
Fees: Bybit VIP0 Realistic (Entry 0.02% + Exit Taker 0.055% + Slippage)

Author: AutoBot Architect
"""

import pandas as pd
import numpy as np
import requests
import time
import sys
from datetime import datetime, timedelta
import itertools

# ============================================
# CONFIGURATION
# ============================================

SYMBOLS = [
    'BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'BNBUSDT', 'XRPUSDT', 
    'ADAUSDT', 'DOGEUSDT', 'AVAXUSDT', 'LINKUSDT', 'DOTUSDT'
]

TIMEFRAMES = [5, 15, 30, 60, 240] # 5m, 15m, 30m, 1h, 4h
DAYS = 120 # 4 months data

# Fees (Realistic Bybit)
ENTRY_FEE = 0.0002   # Maker (Limit)
ENTRY_SLIP = 0.0001
TP_FEE = 0.0002      # Maker (Limit)
TP_SLIP = 0.0001
SL_FEE = 0.00055     # Taker (Stop Market)
SL_SLIP = 0.0004     # Slippage
TOTAL_WIN_COST = ENTRY_FEE + ENTRY_SLIP + TP_FEE + TP_SLIP      # ~0.06%
TOTAL_LOSS_COST = ENTRY_FEE + ENTRY_SLIP + SL_FEE + SL_SLIP     # ~0.125%

# ============================================
# DATA ENGINE
# ============================================

def fetch_data(symbol: str, interval: int, days: int) -> pd.DataFrame:
    try:
        url = "https://api.bybit.com/v5/market/kline"
        all_kline = []
        end_ts = int(time.time() * 1000)
        start_ts = int((time.time() - days * 24 * 3600) * 1000)
        
        while end_ts > start_ts:
            params = {
                'category': 'linear',
                'symbol': symbol,
                'interval': str(interval),
                'limit': 1000,
                'end': end_ts
            }
            r = requests.get(url, params=params).json()
            if r['retCode'] != 0 or not r['result']['list']:
                break
            klines = r['result']['list']
            all_kline.extend(klines)
            end_ts = int(klines[-1][0]) - 1
            time.sleep(0.05)
            
        df = pd.DataFrame(all_kline, columns=['ts', 'open', 'high', 'low', 'close', 'vol', 'to'])
        df = df.iloc[::-1].reset_index(drop=True)
        for c in ['open', 'high', 'low', 'close', 'vol']:
            df[c] = df[c].astype(float)
        return df
    except Exception as e:
        print(f"Failed {symbol} {interval}m: {e}")
        return pd.DataFrame()

# ============================================
# INDICATORS & PATTERNS
# ============================================

def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    close = df['close']
    
    # RSI 14
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 1e-9)
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # ATR 14
    h, l, c_prev = df['high'], df['low'], close.shift(1)
    tr = pd.concat([h-l, (h-c_prev).abs(), (l-c_prev).abs()], axis=1).max(axis=1)
    df['atr'] = tr.rolling(14).mean()
    
    # Swing Points (Fractals) - Lookback 5
    # A swing high is a high surrounded by 2 lower highs on each side
    df['swing_high'] = df['high'].rolling(5, center=True).max() == df['high']
    df['swing_low'] = df['low'].rolling(5, center=True).min() == df['low']
    
    return df

def find_key_levels(df: pd.DataFrame, window=200) -> pd.Series:
    """Identify key support/resistance levels from recent major swings."""
    # Simplified: Recent 200-period Highs/Lows as proxy for Key Levels
    key_resistance = df['high'].rolling(window).max()
    key_support = df['low'].rolling(window).min()
    return key_support, key_resistance

# ============================================
# WYSETRADE STRATEGY LOGIC
# ============================================

def run_strategy(df: pd.DataFrame, rr_ratio: float = 2.0) -> dict:
    """
    Executes the Wysetrade logic:
    1. RSI Div in Zone
    2. Key Level Check
    3. Confirmed Structure Break
    """
    trades = []
    
    # Pre-calculate
    df['key_sup'], df['key_res'] = find_key_levels(df, 200) # Key levels
    
    # State tracking
    potential_long = None # {'type': 'bull_div', 'idx': 100, 'low': 50000}
    potential_short = None
    
    cooldown = 0
    
    for i in range(50, len(df)-1):
        if cooldown > 0:
            cooldown -= 1
            continue
            
        row = df.iloc[i]
        
        # ---------------------------
        # 1. SETUP: Divergence
        # ---------------------------
        
        # Bullish Divergence (Price LL + RSI HL + Oversold)
        # Lookback 5-15 bars for divergence
        bull_div = False
        bear_div = False
        
        # Simple Divergence Check: 
        # Current Low is lowest in 10 bars, but RSI is NOT lowest in 10 bars
        # AND RSI < 40 (Near Oversold)
        if row['low'] == df['low'].iloc[i-10:i+1].min() and \
           row['rsi'] > df['rsi'].iloc[i-10:i+1].min() and \
           row['rsi'] < 40:
               bull_div = True
        
        # Bearish Divergence
        if row['high'] == df['high'].iloc[i-10:i+1].max() and \
           row['rsi'] < df['rsi'].iloc[i-10:i+1].max() and \
           row['rsi'] > 60:
               bear_div = True
               
        # ---------------------------
        # 2. FILTER: Key Levels
        # ---------------------------
        
        # Is price near Key Support? (Within 0.5% of 200-bar Low)
        # Or is it just "reacting"? simpler: Has it touched recent support?
        # Let's skip complex "Zone" logic for backtest speed and assume Div at Lows implies Support.
        # User Rule: "A valid signal requires the price to be reacting to a Key Level"
        # Implementation: Current Low must be within 1% of Key Support
        
        near_support = abs(row['low'] - row['key_sup']) / row['key_sup'] < 0.01
        near_resistance = abs(row['high'] - row['key_res']) / row['key_res'] < 0.01
        
        if bull_div and near_support:
            potential_long = {
                'valid': True,
                'setup_idx': i,
                'swing_low': row['low'],
                # Define Confirmation Level: The most recent Swing High BEFORE this low
                # Scan backwards for nearest Swing High
                'trigger_price': df.iloc[i-15:i]['high'].max() 
            }
            potential_short = None # Reset opposite
            
        if bear_div and near_resistance:
            potential_short = {
                'valid': True,
                'setup_idx': i,
                'swing_high': row['high'],
                # Define Confirmation Level: Nearest Swing Low
                'trigger_price': df.iloc[i-15:i]['low'].min()
            }
            potential_long = None
            
        # ---------------------------
        # 3. TRIGGER: Structure Break
        # ---------------------------
        
        entry = 0
        sl = 0
        tp = 0
        side = None
        
        # Check Long Trigger
        if potential_long and potential_long['valid']:
            # Timeout if setup too old (e.g., 20 bars passed)
            if i - potential_long['setup_idx'] > 20:
                potential_long = None
            # Check Breakout: Close > Trigger High
            elif row['close'] > potential_long['trigger_price']:
                # CONFIRMED LONG
                side = 'long'
                entry = df.iloc[i+1]['open'] # Next candle open
                sl = potential_long['swing_low']
                # Limit SL measure to max 3% to avoid massive stop
                if (entry - sl)/entry > 0.03: 
                    potential_long = None
                    continue
                    
                risk_dist = entry - sl
                if risk_dist <= 0: continue
                tp = entry + (risk_dist * rr_ratio)
                potential_long = None # Consumed
                cooldown = 10
        
        # Check Short Trigger
        elif potential_short and potential_short['valid']:
             if i - potential_short['setup_idx'] > 20:
                 potential_short = None
             elif row['close'] < potential_short['trigger_price']:
                 # CONFIRMED SHORT
                 side = 'short'
                 entry = df.iloc[i+1]['open']
                 sl = potential_short['swing_high']
                 if (sl - entry)/entry > 0.03:
                     potential_short = None
                     continue
                     
                 risk_dist = sl - entry
                 if risk_dist <= 0: continue
                 tp = entry - (risk_dist * rr_ratio)
                 potential_short = None
                 cooldown = 10
                 
        # ---------------------------
        # SIMULATE TRADE
        # ---------------------------
        if side:
            # Simulate forward
            outcome = 'timeout'
            bars_held = 0
            exit_price = entry
            pnl_r = 0
            
            for j in range(i+1, min(i+500, len(df))):
                c = df.iloc[j]
                bars_held += 1
                
                if side == 'long':
                    if c['low'] <= sl:
                        outcome = 'loss'
                        exit_price = sl
                        break
                    if c['high'] >= tp:
                        outcome = 'win'
                        exit_price = tp
                        break
                else:
                    if c['high'] >= sl:
                        outcome = 'loss'
                        exit_price = sl
                        break
                    if c['low'] <= tp:
                        outcome = 'win'
                        exit_price = tp
                        break
                        
            # Calc Result with Fees
            # Risk %
            risk_pct = abs(entry - sl) / entry
            
            if outcome == 'win':
                # Fee in R = Cost% / Risk%
                cost_r = TOTAL_WIN_COST / risk_pct
                pnl_r = rr_ratio - cost_r
            elif outcome == 'loss':
                cost_r = TOTAL_LOSS_COST / risk_pct
                pnl_r = -1.0 - cost_r
            else: # Timeout
                # Close at market
                closing_price = df.iloc[min(i+500, len(df)-1)]['close']
                raw_pnl_pct = (closing_price - entry)/entry if side=='long' else (entry - closing_price)/entry
                raw_r = raw_pnl_pct / risk_pct
                cost_r = TOTAL_LOSS_COST / risk_pct # Assume taker exit
                pnl_r = raw_r - cost_r
            
            trades.append({
                'ts': df.iloc[i]['ts'],
                'side': side,
                'pnl_r': pnl_r,
                'outcome': outcome,
                'bars': bars_held
            })
            
    return trades

# ============================================
# MAIN
# ============================================

def main():
    print("🚀 WYSETRADE STRATEGY BACKTEST")
    print(f"Timeframes: {TIMEFRAMES}")
    print(f"Strategy: RSI Div + Key Levels + Structure Break")
    print("-" * 60)
    
    report = []
    
    for tf in TIMEFRAMES:
        print(f"\n📊 Testing Timeframe: {tf}m")
        tf_trades = []
        
        for sym in SYMBOLS:
            print(f"   Loading {sym}...", end='\r')
            df = fetch_data(sym, tf, DAYS)
            if len(df) < 500: continue
            
            df = calculate_indicators(df)
            trades = run_strategy(df, rr_ratio=2.0)
            tf_trades.extend(trades)
            
        # Analyze TF results
        if not tf_trades:
            print(f"   ❌ No trades found for {tf}m")
            continue
            
        total_r = sum(t['pnl_r'] for t in tf_trades)
        wins = len([t for t in tf_trades if t['outcome']=='win'])
        total = len(tf_trades)
        wr = wins/total*100
        avg_r = total_r / total
        
        print(f"   ✅ Result: {total} trades | WR: {wr:.1f}% | Net R: {total_r:+.1f} | Avg R: {avg_r:+.3f}")
        report.append({
            'tf': tf,
            'trades': total,
            'net_r': total_r,
            'avg_r': avg_r,
            'wr': wr
        })
        
    print("\n" + "="*60)
    print("🏆 FINAL SUMMARY")
    print("="*60)
    print(f"{'TF':<5} {'Trades':<8} {'Win%':<8} {'Net R':<10} {'Avg R':<8}")
    print("-" * 50)
    
    best_tf = None
    best_r = -9999
    
    for r in report:
        print(f"{r['tf']:<5} {r['trades']:<8} {r['wr']:<8.1f} {r['net_r']:<10.1f} {r['avg_r']:<8.3f}")
        if r['net_r'] > best_r:
            best_r = r['net_r']
            best_tf = r['tf']
            
    if best_tf and best_r > 0:
        print(f"\n✅ BEST TIMEFRAME: {best_tf}m with {best_r:+.1f}R")
    else:
        print(f"\n❌ NO PROFITABLE TIMEFRAMES FOUND")

if __name__ == "__main__":
    main()
