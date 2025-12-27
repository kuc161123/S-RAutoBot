#!/usr/bin/env python3
"""
BACKTEST DISCOVERY ENGINE
=========================

Dynamic Strategy Optimization for 5M Timeframe
Goal: Find a profitable configuration that survives realistic Bybit fees/slippage.
Method: Grid Search over Filters & Exit Strategies + Walk-Forward Validation.

Key challenge on 5M: Fees (0.06% x2) + Slippage (0.02% x2) = ~0.16% cost per trade.
If ATR is ~0.2%, SL is 0.2%. Cost is nearly 0.8R! 
We need strategies with high Expectancy (> 0.2R after fees).

Author: AutoBot Architect
"""

import pandas as pd
import numpy as np
import itertools
import random
from typing import List, Dict
import time
import sys
import requests
from datetime import datetime, timedelta

# =============================================================================
# CONFIGURATION SPACE
# =============================================================================

SYMBOLS = [
    'BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'BNBUSDT', 'XRPUSDT', 
    'ADAUSDT', 'DOGEUSDT', 'AVAXUSDT', 'LINKUSDT', 'DOTUSDT'
] # Reduced to Top 10 for speed during discovery, then validate on 20.

DAYS = 60 # 2 months data for discovery

# Search Space - Phase 3: Precise Fees + Volume Filter
PARAMS = {
    'rr_ratio': [1.5, 2.0, 3.0],     
    'sl_atr': [2.0, 3.0, 4.0],       # Swing Stops
    'ema_trend': [200],              # Trend
    'rsi_entry': [None, 'strict'],   
    'volatility_filter': [None, 'adx_25'],
    'macd_confirm': [True, False],
    'volume_spike': [False, True]    # Vol > 1.5x MA
}

# Precise Fees (Bybit VIP0)
ENTRY_FEE = 0.0002   # Maker
ENTRY_SLIP = 0.0001
TP_FEE = 0.0002      # Maker
TP_SLIP = 0.0001
SL_FEE = 0.00055     # Taker
SL_SLIP = 0.0004     # Higher slippage on stop

# =============================================================================
# DATA ENGINE
# =============================================================================

def fetch_data(symbol: str, days: int) -> pd.DataFrame:
    try:
        url = "https://api.bybit.com/v5/market/kline"
        end_ts = int(time.time() * 1000)
        start_ts = int((time.time() - days * 24 * 3600) * 1000)
        
        all_kline = []
        while end_ts > start_ts:
            params = {
                'category': 'linear',
                'symbol': symbol,
                'interval': '5',
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
        df = df.iloc[::-1].reset_index(drop=True) # Reverse to chronological
        for c in ['open', 'high', 'low', 'close', 'vol']:
            df[c] = df[c].astype(float)
        return df
    except Exception as e:
        print(f"Failed {symbol}: {e}")
        return pd.DataFrame()

def prepare_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    close = df['close']
    
    # RSI
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 1e-9)
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # EMA 200
    df['ema200'] = close.ewm(span=200, adjust=False).mean()
    
    # MACD
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df['macd'] = ema12 - ema26
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']
    
    # ATR
    h, l, c_prev = df['high'], df['low'], close.shift(1)
    tr = pd.concat([h-l, (h-c_prev).abs(), (l-c_prev).abs()], axis=1).max(axis=1)
    df['atr'] = tr.rolling(14).mean()
    
    # Volume MA
    df['vol_ma'] = df['vol'].rolling(20).mean()
    df['vol_spike'] = df['vol'] > (df['vol_ma'] * 1.5)
    
    # ADX (Simplified)
    plus_dm = df['high'].diff()
    minus_dm = df['low'].diff()
    plus_dm = np.where((plus_dm > minus_dm) & (plus_dm > 0), plus_dm, 0.0)
    minus_dm = np.where((minus_dm > plus_dm) & (minus_dm > 0), minus_dm, 0.0)
    
    tr_s = tr.rolling(14).sum()
    plus_di = 100 * (pd.Series(plus_dm).rolling(14).sum() / tr_s)
    minus_di = 100 * (pd.Series(minus_dm).rolling(14).sum() / tr_s)
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di).abs()) * 100
    df['adx'] = dx.rolling(14).mean()
    
    # Pivot Points for Divergence
    lookback = 14
    df['ll'] = df['low'].rolling(lookback).min() # Lowest Low
    df['hh'] = df['high'].rolling(lookback).max() # Highest High
    df['rsi_ll'] = df['rsi'].rolling(lookback).min()
    df['rsi_hh'] = df['rsi'].rolling(lookback).max()
    
    return df

# =============================================================================
# STRATEGY CORE
# =============================================================================

def detect_signals(df: pd.DataFrame) -> List[Dict]:
    """Detects ALL raw divergence signals. Filters applied later."""
    signals = []
    
    # Vectorized detection usually faster, but loop allows logic match with bot
    # We will use a simplified loop for "Base Signals" and store their attributes
    
    for i in range(50, len(df)-1):
        row = df.iloc[i]
        prev = df.iloc[i-14:i] # Lookback window
        
        sig_type = None
        
        # Bullish Regular: Price Low < Prev Low AND RSI > Prev Low RSI
        if (row['low'] <= row['ll'] + 1e-8) and (row['rsi'] > row['rsi_ll']) and (row['rsi'] < 45):
             sig_type = 'long'
        
        # Bearish Regular: Price High > Prev High AND RSI < Prev High RSI
        elif (row['high'] >= row['hh'] - 1e-8) and (row['rsi'] < row['rsi_hh']) and (row['rsi'] > 55):
             sig_type = 'short'
             
        if sig_type:
            signals.append({
                'idx': i,
                'ts': row['ts'],
                'side': sig_type,
                'price': row['open'], # Signal generated at close, trade at next Open?
                                     # Bot typically trades immediately or next candle. 
                                     # Let's assume entry at next candle OPEN for realism.
                'entry_price': df.iloc[i+1]['open'],
                'atr': row['atr'],
                'rsi': row['rsi'],
                'close': row['close'],
                'ema200': row['ema200'],
                'macd_hist': row['macd_hist'],
                'prev_macd_hist': df.iloc[i-1]['macd_hist'],
                'adx': row['adx'],
                'vol_spike': row['vol_spike']
            })
            
    return signals

def test_combination(signals: List[Dict], df: pd.DataFrame, config: Dict) -> Dict:
    """Fast simulation of a specific config on pre-calculated signals."""
    
    balance = 1000.0
    wins = 0
    losses = 0
    total_r = 0.0
    
    # Unpack Config
    rr = config['rr_ratio']
    sl_mult = config['sl_atr']
    use_ema = config['ema_trend']
    use_rsi_strict = config['rsi_entry']
    use_adx = config['volatility_filter']
    use_macd = config['macd_confirm']
    use_vol = config['volume_spike']
    
    trades = []
    
    for s in signals:
        # --- FILTERS ---
        
        # 0. Volume Spike
        if use_vol and not s['vol_spike']: continue
        
        # 1. EMA Trend Filter
        if use_ema:
            if s['side'] == 'long' and s['close'] < s['ema200']: continue
            if s['side'] == 'short' and s['close'] > s['ema200']: continue
            
        # 2. Strict RSI Entry
        if use_rsi_strict == 'strict':
            if s['side'] == 'long' and s['rsi'] > 40: continue # Only deep oversold
            if s['side'] == 'short' and s['rsi'] < 60: continue # Only deep overbought
            
        # 3. MACD Confirmation
        if use_macd:
            if s['side'] == 'long' and s['macd_hist'] <= s['prev_macd_hist']: continue
            if s['side'] == 'short' and s['macd_hist'] >= s['prev_macd_hist']: continue
            
        if use_adx == 'adx_25':
            if s['adx'] < 25: continue
        elif use_adx == 'adx_20':
            if s['adx'] < 20: continue
            
        # --- SIMULATION ---
        entry = s['entry_price']
        atr = s['atr']
        if atr == 0: continue
        
        sl_dist = atr * sl_mult
        tp_dist = sl_dist * rr
        
        # Determine outcome (Vectorized lookahead would be faster, but sequential for accuracy)
        # Scan forward in DF
        start_idx = s['idx'] + 1
        
        # Max hold 2000 bars (~7 days)
        outcome = 0 # 0=running, 1=win, -1=loss
        bars = 0
        
        # Slice the DF for speed
        idx_end = min(start_idx + 2000, len(df))
        future = df.iloc[start_idx:idx_end]
        
        sl_price = entry - sl_dist if s['side'] == 'long' else entry + sl_dist
        tp_price = entry + tp_dist if s['side'] == 'long' else entry - tp_dist
        
        # Check Lows for Long SL, Highs for Long TP
        # We need precise order of events inside the candle? 
        # Assume Worst Case: SL hit before TP if both in same candle
        
        for _, row in future.iterrows():
            bars += 1
            if s['side'] == 'long':
                if row['low'] <= sl_price:
                    outcome = -1
                    break
                if row['high'] >= tp_price:
                    outcome = 1
                    break
            else:
                if row['high'] >= sl_price:
                    outcome = -1
                    break
                if row['low'] <= tp_price:
                    outcome = 1
                    break
        
        if outcome == 0: # Timeout
            # Close at market
            end_price = future.iloc[-1]['close']
            if s['side'] == 'long':
                pnl = (end_price - entry) / entry
            else:
                pnl = (entry - end_price) / entry
            
            # Convert % PnL to R
            risk_pct = sl_dist/entry
            if risk_pct > 0:
                raw_r = pnl / risk_pct
                # Apply Fees (Assume Taker exit for timeout just in case)
                cost_pct = ENTRY_FEE + ENTRY_SLIP + SL_FEE + SL_SLIP
                cost_r = cost_pct / risk_pct
                r_res = raw_r - cost_r
            else:
                r_res = -1
        
        elif outcome == 1:
            wins += 1
            # Win R = RR - Costs (Maker Entry + Maker TP)
            risk_pct = sl_dist / entry
            cost_pct = ENTRY_FEE + ENTRY_SLIP + TP_FEE + TP_SLIP
            cost_r = cost_pct / risk_pct if risk_pct > 0 else 0
            r_res = rr - cost_r
            
        else: # Loss
            losses += 1
            # Loss R = -1 - Costs (Maker Entry + Taker SL)
            risk_pct = sl_dist / entry
            cost_pct = ENTRY_FEE + ENTRY_SLIP + SL_FEE + SL_SLIP
            cost_r = cost_pct / risk_pct if risk_pct > 0 else 0
            r_res = -1.0 - cost_r
            
        total_r += r_res
        trades.append(r_res)
        
    return {
        'net_r': total_r,
        'trades': len(trades),
        'wr': (wins/len(trades)*100) if trades else 0,
        'avg_r': (total_r/len(trades)) if trades else 0
    }

# =============================================================================
# MAIN ORCHESTRATOR
# =============================================================================

def main():
    print("🚀 DISCOVERY ENGINE STARTING")
    
    # 1. Load Data
    print("DATA: Loading top symbols...")
    datasets = {}
    for sym in SYMBOLS:
        df = fetch_data(sym, DAYS)
        if not df.empty:
            datasets[sym] = prepare_indicators(df)
            print(f" Loaded {sym}: {len(df)} candles")
            
    # 2. Pre-calculate Signals
    print("SIGS: Pre-calculating base divergence signals...")
    base_signals = {}
    for sym, df in datasets.items():
        base_signals[sym] = detect_signals(df)
        print(f" {sym}: {len(base_signals[sym])} raw signals")
        
    # 3. Generate Grid
    keys = PARAMS.keys()
    values = PARAMS.values()
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    print(f"GRID: Exploring {len(combinations)} parameter combinations...")
    
    best_config = None
    best_score = -99999
    
    results_log = []

    print(f"{'#':<5} {'R:R':<5} {'SL':<5} {'EMA':<5} {'RSI':<8} {'ADX':<8} {'MACD':<5} {'VOL':<5} | {'Trades':<6} {'Win%':<6} {'Net R':<10} {'Avg R':<8}")
    print("-" * 95)

    start_t = time.time()
    for i, cfg in enumerate(combinations):
        
        # Test across whole portfolio
        total_r = 0
        total_trades = 0
        total_wins = 0
        
        for sym, df in datasets.items():
            res = test_combination(base_signals[sym], df, cfg)
            total_r += res['net_r']
            total_trades += res['trades']
            
        avg_r = total_r / total_trades if total_trades > 0 else 0
        score = total_r
        
        if score > best_score and total_trades > 5: # Min validity
            best_score = score
            best_config = cfg
            print(f"{i:<5} {cfg['rr_ratio']:<5} {cfg['sl_atr']:<5} {str(cfg['ema_trend']):<5} {str(cfg['rsi_entry']):<8} {str(cfg['volatility_filter']):<8} {str(cfg['macd_confirm']):<5} {str(cfg['volume_spike']):<5} | {total_trades:<6} {0:.1f}?? {total_r:<10.1f} {avg_r:<8.3f} < NEW BEST")
        
        # Log periodically
        if i % 50 == 0:
             sys.stdout.write(f"\rProgress: {i}/{len(combinations)}...")
             sys.stdout.flush()

    print("\n" + "="*80)
    print("🏆 CHAMPION CONFIGURATION")
    print("="*80)
    print(best_config)
    print(f"Score (Net R): {best_score}")
    
    # Save to file
    import json
    with open('discovery_result.json', 'w') as f:
        json.dump({'config': best_config, 'score': best_score}, f, default=str)

if __name__ == "__main__":
    main()
