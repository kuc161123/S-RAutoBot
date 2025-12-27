#!/usr/bin/env python3
"""
QUANTEDGE RESEARCH PIPELINE V2
==============================
Iteration 2: Testing refined strategies with tighter parameters and additional filters.

Based on Round 1 failures:
- Mean Reversion: 0% fold profitability - abandon
- Squeeze: 20% fold profitability - abandon
- Trend Following: Showed some promise - refine with ADX filter

New candidates for Round 2:
1. Trend Following + ADX Filter (only trade when ADX > 25)
2. EMA Crossover with Trend Confirmation
3. ATR Breakout with Volume Spike
"""

import pandas as pd
import numpy as np
import sys
sys.path.insert(0, '.')
from quantedge.backtester_core import QuantEdgeBacktester

# Load Data
def load_research_data():
    df = pd.read_parquet("quantedge/BTCUSDT_15m_3y.parquet")
    df = df.copy()
    
    # Technical Indicators
    df['atr'] = (df['high'] - df['low']).rolling(14).mean()
    df['ema200'] = df['close'].ewm(span=200, adjust=False).mean()
    df['ema50'] = df['close'].ewm(span=50, adjust=False).mean()
    df['ema20'] = df['close'].ewm(span=20, adjust=False).mean()
    
    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df['rsi'] = 100 - (100 / (1 + gain/(loss + 1e-9)))
    
    # ADX
    plus_dm = df['high'].diff()
    minus_dm = df['low'].diff().abs() * -1
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm < 0), 0).abs()
    tr = pd.concat([df['high'] - df['low'], 
                    (df['high'] - df['close'].shift()).abs(),
                    (df['low'] - df['close'].shift()).abs()], axis=1).max(axis=1)
    atr14 = tr.rolling(14).mean()
    plus_di = 100 * (plus_dm.rolling(14).mean() / atr14)
    minus_di = 100 * (minus_dm.rolling(14).mean() / atr14)
    dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di + 1e-9))
    df['adx'] = dx.rolling(14).mean()
    
    # Volume SMA
    df['vol_sma'] = df['volume'].rolling(20).mean()
    
    # Bollinger Bands
    df['ma20'] = df['close'].rolling(20).mean()
    df['std20'] = df['close'].rolling(20).std()
    
    return df.dropna().reset_index(drop=True)

# CANDIDATE 1: Trend Following + ADX Filter
def strategy_trend_adx(df, lookback=24, adx_thresh=25):
    df = df.copy()
    high_channel = df['high'].rolling(lookback).max().shift(1)
    low_channel = df['low'].rolling(lookback).min().shift(1)
    
    # Only trade when ADX > threshold (trending market)
    trending = df['adx'] > adx_thresh
    
    buy_sig = (df['close'] > high_channel) & (df['close'] > df['ema200']) & trending
    sell_sig = (df['close'] < low_channel) & (df['close'] < df['ema200']) & trending
    
    signals = pd.Series(0, index=df.index)
    signals.loc[buy_sig] = 1
    signals.loc[sell_sig] = -1
    return signals

# CANDIDATE 2: EMA Crossover + Trend
def strategy_ema_cross(df, fast=20, slow=50):
    df = df.copy()
    fast_ema = df['close'].ewm(span=fast, adjust=False).mean()
    slow_ema = df['close'].ewm(span=slow, adjust=False).mean()
    
    # Crossover signals
    cross_up = (fast_ema > slow_ema) & (fast_ema.shift(1) <= slow_ema.shift(1))
    cross_down = (fast_ema < slow_ema) & (fast_ema.shift(1) >= slow_ema.shift(1))
    
    # Only trade with EMA200 trend
    buy_sig = cross_up & (df['close'] > df['ema200'])
    sell_sig = cross_down & (df['close'] < df['ema200'])
    
    signals = pd.Series(0, index=df.index)
    signals.loc[buy_sig] = 1
    signals.loc[sell_sig] = -1
    return signals

# CANDIDATE 3: ATR Breakout + Volume
def strategy_atr_breakout(df, atr_mult=2.0, vol_mult=1.5):
    df = df.copy()
    upper = df['ma20'] + (df['atr'] * atr_mult)
    lower = df['ma20'] - (df['atr'] * atr_mult)
    
    # Volume spike confirmation
    vol_spike = df['volume'] > (df['vol_sma'] * vol_mult)
    
    buy_sig = (df['close'] > upper.shift(1)) & vol_spike & (df['close'] > df['ema200'])
    sell_sig = (df['close'] < lower.shift(1)) & vol_spike & (df['close'] < df['ema200'])
    
    signals = pd.Series(0, index=df.index)
    signals.loc[buy_sig] = 1
    signals.loc[sell_sig] = -1
    return signals

def print_wfo_summary(name, wfo_results, bt_engine):
    if not wfo_results:
        print(f"\n--- WFO Summary for {name}: NO RESULTS ---")
        return None
        
    print(f"\n--- WFO Summary for {name} ---")
    all_trades = [f['trades'] for f in wfo_results if not f['trades'].empty]
    if not all_trades:
        print("  No trades generated")
        return None
        
    all_oos_trades = pd.concat(all_trades)
    metrics = bt_engine.calculate_metrics(all_oos_trades)
    
    print(f"Aggregate OOS Stats:")
    print(f"  Total PnL: {metrics['total_pnl']:.4f}")
    print(f"  Win Rate:  {metrics['win_rate']:.2%}")
    print(f"  Sharpe:    {metrics['sharpe']:.2f}")
    print(f"  Max DD:    {metrics['max_dd']:.2%}")
    print(f"  N Trades:  {metrics['trades']}")
    
    print(f"Per Fold Detail:")
    folds_prof = 0
    for f in wfo_results:
        m = bt_engine.calculate_metrics(f['trades'])
        is_prof = m['total_pnl'] > 0
        folds_prof += 1 if is_prof else 0
        mark = "✅" if is_prof else "❌"
        print(f"  Fold {f['fold']} {mark} | PnL: {m['total_pnl']:+7.4f} | Sharpe: {m['sharpe']:5.2f} | trades: {m['trades']}")
    
    prof_rate = folds_prof / len(wfo_results) if wfo_results else 0
    status = 'PASS' if prof_rate >= 0.6 else 'FAIL'
    print(f"Fold Profitable Rate: {prof_rate:.0%} ({status})")
    
    return {
        'name': name,
        'metrics': metrics,
        'fold_prof_rate': prof_rate,
        'passed': prof_rate >= 0.6
    }

def run_research():
    print("="*60)
    print("QUANTEDGE RESEARCH PIPELINE V2")
    print("="*60)
    
    df = load_research_data()
    print(f"Loaded {len(df)} bars of BTCUSDT 15m data")
    
    bt = QuantEdgeBacktester(df)
    results = []
    
    # Test Candidate 1: Trend + ADX
    print("\n" + "="*60)
    print("Testing Candidate 1: Trend Following + ADX Filter")
    tf_grid = [
        {'lookback': 24, 'adx_thresh': 25, 'rr': 2.0, 'sl_atr_mult': 1.5},
        {'lookback': 48, 'adx_thresh': 20, 'rr': 2.5, 'sl_atr_mult': 2.0},
    ]
    wfo_tf = bt.walk_forward(strategy_trend_adx, tf_grid)
    r = print_wfo_summary("Trend + ADX", wfo_tf, bt)
    if r: results.append(r)
    
    # Test Candidate 2: EMA Crossover
    print("\n" + "="*60)
    print("Testing Candidate 2: EMA Crossover + Trend")
    ema_grid = [
        {'fast': 12, 'slow': 26, 'rr': 2.0, 'sl_atr_mult': 1.5},
        {'fast': 20, 'slow': 50, 'rr': 2.5, 'sl_atr_mult': 2.0},
    ]
    wfo_ema = bt.walk_forward(strategy_ema_cross, ema_grid)
    r = print_wfo_summary("EMA Cross", wfo_ema, bt)
    if r: results.append(r)
    
    # Test Candidate 3: ATR Breakout + Volume
    print("\n" + "="*60)
    print("Testing Candidate 3: ATR Breakout + Volume Spike")
    atr_grid = [
        {'atr_mult': 1.5, 'vol_mult': 1.3, 'rr': 2.0, 'sl_atr_mult': 1.5},
        {'atr_mult': 2.0, 'vol_mult': 1.5, 'rr': 2.5, 'sl_atr_mult': 2.0},
    ]
    wfo_atr = bt.walk_forward(strategy_atr_breakout, atr_grid)
    r = print_wfo_summary("ATR Breakout", wfo_atr, bt)
    if r: results.append(r)
    
    # Summary
    print("\n" + "="*60)
    print("ROUND 2 SUMMARY")
    print("="*60)
    
    passed = [r for r in results if r['passed']]
    if passed:
        print(f"\n🎉 {len(passed)} STRATEGIES PASSED!")
        for p in passed:
            print(f"  ✅ {p['name']}: {p['fold_prof_rate']:.0%} folds profitable, Sharpe={p['metrics']['sharpe']:.2f}")
    else:
        print("\n❌ No strategies passed the 60% fold profitability criterion")
        print("Proceeding to Round 3 with different approaches...")

if __name__ == "__main__":
    run_research()
