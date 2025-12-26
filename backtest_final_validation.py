#!/usr/bin/env python3
"""
ROBUST VALIDATION - 1-Candle Structure Break Strategy
=====================================================

Comprehensive testing of the EXACT profitable configuration:
- RR: 3.0
- SL: 0.8 ATR
- Structure Break: 1 candle wait (STRICT)
- Time: All-day
- Divergence: Regular only

Tests Include:
1. Walk-Forward (6 periods of 10 days each)
2. Per-Symbol Performance
3. Month-by-Month Breakdown
4. Monte Carlo Simulation (1000 runs)
5. Recent 30-Day Performance (live readiness)

Original Result: +1.854R/trade (92 trades, 60 days)
"""

import pandas as pd
import numpy as np
import requests
import time
import random
from datetime import datetime, timedelta

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
WIN_COST = 0.0006
LOSS_COST = 0.00125

# ============================================================================
# DATA FETCHING
# ============================================================================

def fetch_data(symbol, days=DAYS):
    try:
        url = "https://api.bybit.com/v5/market/kline"
        all_kline = []
        end_ts = int(time.time() * 1000)
        start_ts = int((time.time() - days * 24 * 3600) * 1000)
        
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
    
    # Divergence detection (REGULAR ONLY)
    df['price_low_14'] = df['low'].rolling(14).min()
    df['price_high_14'] = df['high'].rolling(14).max()
    df['rsi_low_14'] = df['rsi'].rolling(14).min()
    df['rsi_high_14'] = df['rsi'].rolling(14).max()
    
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
    
    return df

# ============================================================================
# STRATEGY
# ============================================================================

def run_strategy(df, symbol=''):
    trades = []
    
    cooldown = 0
    for i in range(50, len(df)-1):
        if cooldown > 0: cooldown -= 1; continue
        
        row = df.iloc[i]
        
        # Divergence signal
        side = None
        if row['reg_bull']: side = 'long'
        elif row['reg_bear']: side = 'short'
        
        if not side: continue
        
        # STRUCTURE BREAK CHECK (1 candle only - STRICT)
        if i+1 >= len(df): continue
        next_row = df.iloc[i+1]
        
        structure_broken = False
        if side == 'long' and next_row['close'] > row['swing_high_10']:
            structure_broken = True
        if side == 'short' and next_row['close'] < row['swing_low_10']:
            structure_broken = True
        
        if not structure_broken: continue
        
        # Entry (FIXED: enter at i+1, not i+2!)
        entry = df.iloc[i+1]['open']  # Enter on the NEXT candle's open
        
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
        for j in range(i+1, min(i+301, len(df))):
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
        
        trades.append({
            'symbol': symbol,
            'datetime': df.iloc[i+1]['datetime'],
            'side': side,
            'r': res_r,
            'win': outcome == 'win'
        })
        
        cooldown = 5
    
    return trades

# ============================================================================
# VALIDATION TESTS
# ============================================================================

def walk_forward_test(datasets):
    """Split into 6 periods of 10 days each"""
    print("\n📊 WALK-FORWARD VALIDATION (6 periods × 10 days)")
    print("-" * 70)
    
    period_results = []
    
    for period in range(6):
        period_trades = []
        
        for sym, df in datasets.items():
            period_size = len(df) // 6
            start_idx = period * period_size
            end_idx = start_idx + period_size if period < 5 else len(df)
            
            df_period = df.iloc[start_idx:end_idx].copy()
            trades = run_strategy(df_period, sym)
            period_trades.extend(trades)
        
        if period_trades:
            total_r = sum(t['r'] for t in period_trades)
            avg_r = total_r / len(period_trades)
            wins = sum(1 for t in period_trades if t['win'])
            wr = wins / len(period_trades) * 100
            
            period_results.append({
                'period': period + 1,
                'trades': len(period_trades),
                'net_r': total_r,
                'avg_r': avg_r,
                'wr': wr
            })
            
            status = "✅" if avg_r > 0 else "❌"
            print(f"Period {period+1}: {len(period_trades):3} trades | {total_r:+7.1f}R | {avg_r:+.3f}R | WR: {wr:.1f}% {status}")
    
    if period_results:
        profitable = sum(1 for p in period_results if p['avg_r'] > 0)
        avg_r = np.mean([p['avg_r'] for p in period_results])
        print(f"\n✅ Profitable Periods: {profitable}/6")
        print(f"📊 Average Expectancy: {avg_r:+.3f}R")
    
    return period_results

def per_symbol_breakdown(all_trades):
    """Show performance per symbol"""
    print("\n📈 PER-SYMBOL BREAKDOWN")
    print("-" * 70)
    print(f"{'Symbol':<12} | {'N':<4} {'Net R':<8} {'Avg R':<8} {'WR':<6}")
    print("-" * 70)
    
    symbol_stats = {}
    for t in all_trades:
        sym = t['symbol']
        if sym not in symbol_stats:
            symbol_stats[sym] = []
        symbol_stats[sym].append(t)
    
    for sym in sorted(symbol_stats.keys()):
        trades = symbol_stats[sym]
        net_r = sum(t['r'] for t in trades)
        avg_r = net_r / len(trades)
        wins = sum(1 for t in trades if t['win'])
        wr = wins / len(trades) * 100
        status = "✅" if avg_r > 0 else "❌"
        print(f"{sym:<12} | {len(trades):<4} {net_r:<8.1f} {avg_r:<8.3f} {wr:<6.1f}% {status}")

def monte_carlo_simulation(all_trades, runs=1000):
    """Test robustness via random sampling"""
    print("\n🎲 MONTE CARLO SIMULATION (1000 runs)")
    print("-" * 70)
    
    if not all_trades: return
    
    trade_r = [t['r'] for t in all_trades]
    mc_results = []
    
    for _ in range(runs):
        sample = random.choices(trade_r, k=len(trade_r))
        mc_results.append(sum(sample))
    
    mc_results.sort()
    
    worst_5 = mc_results[int(runs * 0.05)]
    median = mc_results[runs // 2]
    best_5 = mc_results[int(runs * 0.95)]
    profitable = sum(1 for r in mc_results if r > 0) / runs * 100
    
    print(f"Worst 5%:    {worst_5:+.1f}R")
    print(f"Median:      {median:+.1f}R")
    print(f"Best 5%:     {best_5:+.1f}R")
    print(f"Profitable:  {profitable:.1f}% of runs")
    
    return profitable

def recent_30_day_test(symbols):
    """Test on most recent 30 days"""
    print("\n🔥 RECENT 30-DAY PERFORMANCE (Live Readiness)")
    print("-" * 70)
    
    datasets_30 = {}
    for sym in symbols:
        df = fetch_data(sym, days=30)
        if not df.empty: datasets_30[sym] = calc_indicators(df)
    
    all_trades = []
    for sym, df in datasets_30.items():
        trades = run_strategy(df, sym)
        all_trades.extend(trades)
    
    if all_trades:
        total_r = sum(t['r'] for t in all_trades)
        avg_r = total_r / len(all_trades)
        wins = sum(1 for t in all_trades if t['win'])
        wr = wins / len(all_trades) * 100
        
        print(f"Trades:      {len(all_trades)}")
        print(f"Net R:       {total_r:+.1f}R")
        print(f"Avg R:       {avg_r:+.3f}R")
        print(f"Win Rate:    {wr:.1f}%")
        
        if avg_r > 0:
            print("✅ READY FOR LIVE TRADING")
        else:
            print("⚠️ Recent performance degraded")

# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 70)
    print("🎯 ROBUST VALIDATION - 1-Candle Structure Break")
    print("=" * 70)
    print(f"Config: RR={RR}, SL={SL_MULT}×ATR, 1-Candle Wait (STRICT)")
    print("=" * 70)
    
    print("\n📥 Loading 60-day data...")
    datasets = {}
    for sym in SYMBOLS:
        df = fetch_data(sym)
        if not df.empty: datasets[sym] = calc_indicators(df)
    print(f"✅ Loaded {len(datasets)} symbols")
    
    # Run full backtest
    print("\n🔄 Running full 60-day backtest...")
    all_trades = []
    for sym, df in datasets.items():
        trades = run_strategy(df, sym)
        all_trades.extend(trades)
    
    # Main results
    if all_trades:
        total_r = sum(t['r'] for t in all_trades)
        avg_r = total_r / len(all_trades)
        wins = sum(1 for t in all_trades if t['win'])
        wr = wins / len(all_trades) * 100
        
        print("\n" + "=" * 70)
        print("📊 60-DAY FULL RESULTS")
        print("=" * 70)
        print(f"Total Trades:    {len(all_trades)}")
        print(f"Net R:           {total_r:+.1f}R")
        print(f"Avg R per Trade: {avg_r:+.3f}R")
        print(f"Win Rate:        {wr:.1f}%")
        print("=" * 70)
        
        # Validation tests
        wf_results = walk_forward_test(datasets)
        per_symbol_breakdown(all_trades)
        mc_prob = monte_carlo_simulation(all_trades)
        recent_30_day_test(SYMBOLS)
        
        # Final verdict
        print("\n" + "=" * 70)
        print("🏆 VALIDATION VERDICT")
        print("=" * 70)
        
        profitable_periods = sum(1 for p in wf_results if p['avg_r'] > 0) if wf_results else 0
        
        if avg_r > 0.5 and profitable_periods >= 4 and mc_prob >= 90:
            print("✅ STRATEGY VALIDATED - READY FOR LIVE")
            print(f"   ✓ Positive expectancy: {avg_r:+.3f}R")
            print(f"   ✓ Walk-forward: {profitable_periods}/6 periods profitable")
            print(f"   ✓ Monte Carlo: {mc_prob:.1f}% probability")
            print(f"   ✓ Expected: ~{len(all_trades)//2} trades/month at {avg_r:+.3f}R each")
        else:
            print("⚠️ STRATEGY NEEDS REVIEW")
            print(f"   - Expectancy: {avg_r:+.3f}R")
            print(f"   - Walk-forward: {profitable_periods}/6 periods")
            print(f"   - Monte Carlo: {mc_prob:.1f}%")

if __name__ == "__main__":
    main()
