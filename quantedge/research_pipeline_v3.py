#!/usr/bin/env python3
"""
QUANTEDGE RESEARCH PIPELINE V3
==============================
Iteration 3: Pivoting strategy approach after 2 rounds of failures.

Key insight from Round 1-2:
- All breakout/trend strategies on 15M show ~30% WR and negative expectancy
- This suggests crypto 15M is mean-reverting, not trending

Round 3 Hypothesis: 
- Try PURE mean reversion with strict RSI extremes (not BB)
- Try range-bound strategies
- Try multi-timeframe confirmation (align with 1H trend)
"""

import pandas as pd
import numpy as np
import sys
sys.path.insert(0, '.')
from quantedge.backtester_core import QuantEdgeBacktester

def load_research_data():
    df = pd.read_parquet("quantedge/BTCUSDT_15m_3y.parquet")
    df = df.copy()
    
    # Core indicators
    df['atr'] = (df['high'] - df['low']).rolling(14).mean()
    df['ema200'] = df['close'].ewm(span=200, adjust=False).mean()
    df['ema50'] = df['close'].ewm(span=50, adjust=False).mean()
    
    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df['rsi'] = 100 - (100 / (1 + gain/(loss + 1e-9)))
    
    # Stochastic RSI
    rsi_min = df['rsi'].rolling(14).min()
    rsi_max = df['rsi'].rolling(14).max()
    df['stoch_rsi'] = (df['rsi'] - rsi_min) / (rsi_max - rsi_min + 1e-9)
    
    # VWAP (simplified - rolling)
    df['vwap'] = (df['close'] * df['volume']).rolling(48).sum() / (df['volume'].rolling(48).sum() + 1e-9)
    
    # 1H trend proxy (4 bars = 1 hour)
    df['ema_1h'] = df['close'].ewm(span=16, adjust=False).mean()  # ~1H EMA
    df['trend_1h'] = np.where(df['close'] > df['ema_1h'], 1, -1)
    
    # Volatility regime
    df['atr_pct'] = df['atr'] / df['close']
    df['vol_regime'] = np.where(df['atr_pct'] > df['atr_pct'].rolling(100).quantile(0.7), 'high', 'low')
    
    return df.dropna().reset_index(drop=True)

# CANDIDATE 1: RSI Extreme Reversion
def strategy_rsi_extreme(df, rsi_low=20, rsi_high=80):
    df = df.copy()
    
    # Buy when RSI < 20 and rising
    rsi_rising = df['rsi'] > df['rsi'].shift(1)
    rsi_falling = df['rsi'] < df['rsi'].shift(1)
    
    buy_sig = (df['rsi'] < rsi_low) & rsi_rising
    sell_sig = (df['rsi'] > rsi_high) & rsi_falling
    
    signals = pd.Series(0, index=df.index)
    signals.loc[buy_sig] = 1
    signals.loc[sell_sig] = -1
    return signals

# CANDIDATE 2: VWAP Reversion with Trend Filter
def strategy_vwap_reversion(df, dev_mult=1.5):
    df = df.copy()
    
    vwap_dev = df['atr'] * dev_mult
    oversold = df['close'] < (df['vwap'] - vwap_dev)
    overbought = df['close'] > (df['vwap'] + vwap_dev)
    
    # Only trade with 1H trend
    buy_sig = oversold & (df['trend_1h'] == 1)
    sell_sig = overbought & (df['trend_1h'] == -1)
    
    signals = pd.Series(0, index=df.index)
    signals.loc[buy_sig] = 1
    signals.loc[sell_sig] = -1
    return signals

# CANDIDATE 3: Stoch RSI + Low Volatility
def strategy_stoch_low_vol(df, stoch_low=0.2, stoch_high=0.8):
    df = df.copy()
    
    low_vol = df['vol_regime'] == 'low'
    
    buy_sig = (df['stoch_rsi'] < stoch_low) & low_vol & (df['close'] > df['ema200'])
    sell_sig = (df['stoch_rsi'] > stoch_high) & low_vol & (df['close'] < df['ema200'])
    
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
    print("QUANTEDGE RESEARCH PIPELINE V3")
    print("Mean Reversion & Multi-Timeframe Focus")
    print("="*60)
    
    df = load_research_data()
    print(f"Loaded {len(df)} bars of BTCUSDT 15m data")
    
    bt = QuantEdgeBacktester(df)
    results = []
    
    # Test Candidate 1: RSI Extreme
    print("\n" + "="*60)
    print("Testing Candidate 1: RSI Extreme Reversion")
    rsi_grid = [
        {'rsi_low': 15, 'rsi_high': 85, 'rr': 1.5, 'sl_atr_mult': 1.0},
        {'rsi_low': 20, 'rsi_high': 80, 'rr': 2.0, 'sl_atr_mult': 1.5},
    ]
    wfo = bt.walk_forward(strategy_rsi_extreme, rsi_grid)
    r = print_wfo_summary("RSI Extreme", wfo, bt)
    if r: results.append(r)
    
    # Test Candidate 2: VWAP Reversion
    print("\n" + "="*60)
    print("Testing Candidate 2: VWAP Reversion + 1H Trend")
    vwap_grid = [
        {'dev_mult': 1.2, 'rr': 1.5, 'sl_atr_mult': 1.0},
        {'dev_mult': 1.5, 'rr': 2.0, 'sl_atr_mult': 1.5},
    ]
    wfo = bt.walk_forward(strategy_vwap_reversion, vwap_grid)
    r = print_wfo_summary("VWAP Reversion", wfo, bt)
    if r: results.append(r)
    
    # Test Candidate 3: Stoch RSI Low Vol
    print("\n" + "="*60)
    print("Testing Candidate 3: Stoch RSI + Low Volatility")
    stoch_grid = [
        {'stoch_low': 0.15, 'stoch_high': 0.85, 'rr': 1.5, 'sl_atr_mult': 1.0},
        {'stoch_low': 0.20, 'stoch_high': 0.80, 'rr': 2.0, 'sl_atr_mult': 1.5},
    ]
    wfo = bt.walk_forward(strategy_stoch_low_vol, stoch_grid)
    r = print_wfo_summary("Stoch Low Vol", wfo, bt)
    if r: results.append(r)
    
    # Summary
    print("\n" + "="*60)
    print("ROUND 3 SUMMARY")
    print("="*60)
    
    passed = [r for r in results if r['passed']]
    if passed:
        print(f"\n🎉 {len(passed)} STRATEGIES PASSED!")
        for p in passed:
            print(f"  ✅ {p['name']}: {p['fold_prof_rate']:.0%} folds profitable, Sharpe={p['metrics']['sharpe']:.2f}")
    else:
        print("\n❌ No strategies passed the 60% fold profitability criterion")
        print("Need to try fundamentally different approaches...")

if __name__ == "__main__":
    run_research()
