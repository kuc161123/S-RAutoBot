#!/usr/bin/env python3
"""
WYSETRADE 5M OPTIMIZER
======================

Goal: Flip the 5M timeframe to profitable by optimizing entry aggression and R:R.
Base Strategy: Wysetrade (RSI Div + Key Levels + Breakout)
Current 5M Performance: 33.9% WR @ 2:1 R:R (Net -0.06R/trade) -> Breakeven before fees.

Variables to Test:
1. ENTRY_MODE: 
   - 'Standard': Break of previous Swing High (Safe, late)
   - 'Aggressive': Break of highest high of last 3 bars (Early, risky)
2. RR_RATIO: [2.0, 2.5, 3.0, 4.0]
3. TRAILING_SL: [False, True]
   - If True: Move SL to Breakeven after 1R, Trail by 2R distance.

Author: AutoBot Optimizer
"""

import pandas as pd
import numpy as np
import requests
import time
import itertools
from typing import List, Dict

# ============================================
# CONFIGURATION
# ============================================

SYMBOLS = [
    'BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'BNBUSDT', 'XRPUSDT', 
    'ADAUSDT', 'DOGEUSDT', 'AVAXUSDT', 'LINKUSDT', 'DOTUSDT'
]

DAYS = 60 # 60 days 5M data

PARAMS = {
    'entry_mode': ['Standard', 'Aggressive'],
    'rr_ratio': [2.0, 3.0, 4.0],
    'trailing_sl': [False, True]
}

# Precise Fees (Bybit VIP0)
# Win Cost: Entry Maker (0.02) + Exit Maker (0.02) + Slip (0.01x2) = 0.06%
# Loss Cost: Entry Maker (0.02) + Exit Taker (0.055) + Slip (0.01 + 0.04) = 0.125%
WIN_COST_PCT = 0.0006
LOSS_COST_PCT = 0.00125

# ============================================
# DATA & INDICATORS
# ============================================

def fetch_data(symbol: str) -> pd.DataFrame:
    try:
        url = "https://api.bybit.com/v5/market/kline"
        all_kline = []
        end_ts = int(time.time() * 1000)
        start_ts = int((time.time() - DAYS * 24 * 3600) * 1000)
        
        while end_ts > start_ts:
            params = {'category': 'linear', 'symbol': symbol, 'interval': '5', 'limit': 1000, 'end': end_ts}
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
    
    # Key Levels (200 bar lookback)
    df['key_sup'] = df['low'].rolling(200).min()
    df['key_res'] = df['high'].rolling(200).max()
    
    # Aggressive Trigger Levels (Last 3 bars)
    df['high_3'] = df['high'].rolling(3).max()
    df['low_3'] = df['low'].rolling(3).min()
    
    return df

# ============================================
# STRATEGY ENGINE
# ============================================

def test_strategy(datasets: Dict[str, pd.DataFrame], config: Dict) -> Dict:
    total_r = 0
    trades_count = 0
    wins = 0
    
    rr = config['rr_ratio']
    mode = config['entry_mode']
    trail = config['trailing_sl']
    
    for sym, df in datasets.items():
        # Iterate (Cannot vectorize easily due to state machine)
        # We optimize by pre-calculating Div signals? 
        # No, let's just run fast loop. 60 days 5M is ~17k bars. 10 symbols = 170k bars. Fast enough.
        
        potential_long = None
        potential_short = None
        cooldown = 0
        
        for i in range(200, len(df)-1):
            if cooldown > 0:
                cooldown -= 1
                continue
            
            row = df.iloc[i]
            
            # 1. SETUP: Divergence + Key Level
            # Bull Div: Current Low is min in 10, RSI HL, RSI < 40
            is_bull = False
            if row['low'] <= df['low'].iloc[i-10:i].min() and \
               row['rsi'] > df['rsi'].iloc[i-10:i].min() and \
               row['rsi'] < 40 and \
               abs(row['low'] - row['key_sup'])/row['key_sup'] < 0.01:
                   is_bull = True
            
            # Bear Div
            is_bear = False
            if row['high'] >= df['high'].iloc[i-10:i].max() and \
               row['rsi'] < df['rsi'].iloc[i-10:i].max() and \
               row['rsi'] > 60 and \
               abs(row['high'] - row['key_res'])/row['key_res'] < 0.01:
                   is_bear = True
                   
            if is_bull:
                trigger = 0
                if mode == 'Standard':
                    trigger = df['high'].iloc[i-15:i].max() # Recent swing high
                else: # Aggressive
                    trigger = df['high'].iloc[i-3:i].max() # High of last 3
                
                potential_long = {'idx': i, 'sl': row['low'], 'trigger': trigger}
                potential_short = None
                
            if is_bear:
                trigger = 0
                if mode == 'Standard':
                    trigger = df['low'].iloc[i-15:i].min()
                else:
                    trigger = df['low'].iloc[i-3:i].min()
                
                potential_short = {'idx': i, 'sl': row['high'], 'trigger': trigger}
                potential_long = None
                
            # 2. TRIGGER
            entry = 0
            sl = 0
            tp = 0
            side = None
            
            if potential_long:
                if i - potential_long['idx'] > 20: potential_long = None
                elif row['close'] > potential_long['trigger']:
                    side = 'long'
                    entry = df.iloc[i+1]['open']
                    sl = potential_long['sl']
                    potential_long = None
                    
            elif potential_short:
                if i - potential_short['idx'] > 20: potential_short = None
                elif row['close'] < potential_short['trigger']:
                    side = 'short'
                    entry = df.iloc[i+1]['open']
                    sl = potential_short['sl']
                    potential_short = None
                    
            # 3. SIMULATE
            if side:
                if side=='long' and (entry-sl)/entry > 0.02: continue # SL too wide
                if side=='short' and (sl-entry)/entry > 0.02: continue
                if entry == sl: continue

                risk_dist = abs(entry - sl)
                if risk_dist <= 0: continue
                risk_pct = risk_dist / entry
                
                tp_dist = risk_dist * rr
                tp_price = entry + tp_dist if side == 'long' else entry - tp_dist
                
                outcome = 'loss'
                # Trailing logic:
                # If Price moves 1R favorable, Move SL to Breakeven
                # Then Trail?
                # Simplified Trail: If reached 1R, SL = Entry. Then if reached 2R, SL = 1R.
                
                curr_sl = sl
                be_hit = False
                
                for j in range(i+1, min(i+500, len(df))):
                    c = df.iloc[j]
                    
                    if side == 'long':
                        if c['low'] <= curr_sl:
                            outcome = 'loss' if not be_hit else 'be'
                            break
                        if c['high'] >= tp_price and not trail:
                            outcome = 'win'
                            break
                            
                        # Trailing Logic
                        if trail:
                            if c['high'] >= entry + risk_dist: # 1R reached
                                curr_sl = entry # BE
                                be_hit = True
                            if c['high'] >= entry + (risk_dist * 2) and rr > 2:
                                curr_sl = entry + risk_dist # Lock 1R
                            if c['high'] >= tp_price:
                                outcome = 'win'
                                break
                    else:
                        if c['high'] >= curr_sl:
                            outcome = 'loss' if not be_hit else 'be'
                            break
                        if c['low'] <= tp_price and not trail:
                            outcome = 'win'
                            break
                            
                        if trail:
                            if c['low'] <= entry - risk_dist:
                                curr_sl = entry
                                be_hit = True
                            if c['low'] <= entry - (risk_dist * 2) and rr > 2:
                                curr_sl = entry - risk_dist
                            if c['low'] <= tp_price:
                                outcome = 'win'
                                break

                # Calc R
                res_r = 0
                if outcome == 'win':
                    res_r = rr - (WIN_COST_PCT / risk_pct)
                    wins += 1
                elif outcome == 'loss':
                    res_r = -1.0 - (LOSS_COST_PCT / risk_pct)
                elif outcome == 'be':
                    res_r = 0.0 - (WIN_COST_PCT / risk_pct) # Paid fees on BE
                    
                total_r += res_r
                trades_count += 1
                cooldown = 5
                
    return {
        'net_r': total_r,
        'trades': trades_count,
        'wr': wins/trades_count*100 if trades_count else 0,
        'avg_r': total_r/trades_count if trades_count else 0
    }

def main():
    print("🚀 WYSETRADE 5M OPTIMIZER STARTING")
    
    # Load Data
    print("Loading data...")
    datasets = {}
    for sym in SYMBOLS:
        df = fetch_data(sym)
        if not df.empty:
            datasets[sym] = prepare_indicators(df)
            
    print(f"Data Loaded: {len(datasets)} symbols")
    
    # Generate Combinations
    keys = PARAMS.keys()
    values = PARAMS.values()
    combos = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    print(f"{'Mode':<10} {'R:R':<5} {'Trail':<6} | {'Trades':<6} {'Win%':<6} {'Net R':<8} {'Avg R':<8}")
    print("-" * 65)
    
    best_config = None
    best_r = -9999
    
    for cfg in combos:
        res = test_strategy(datasets, config=cfg)
        print(f"{cfg['entry_mode']:<10} {cfg['rr_ratio']:<5} {str(cfg['trailing_sl']):<6} | {res['trades']:<6} {res['wr']:<6.1f} {res['net_r']:<8.1f} {res['avg_r']:<8.3f}")
        
        if res['net_r'] > best_r and res['trades'] > 50:
            best_r = res['net_r']
            best_config = cfg
            
    print("\n🏆 CHAMPION CONFIG")
    print(best_config)
    print(f"Net R: {best_r:+.1f}")

if __name__ == "__main__":
    main()
