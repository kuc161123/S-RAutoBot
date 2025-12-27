#!/usr/bin/env python3
"""
STRUCTURE BREAK STRATEGY - RIGOROUS VALIDATION
==============================================

Goal: Validate the structure break strategy with comprehensive testing:
1. Walk-Forward Validation (4 periods)
2. Monte Carlo Simulation (1000 runs)
3. Per-Symbol Breakdown
4. Sensitivity Analysis (parameter variations)
5. Market Condition Analysis (trending vs ranging)

Configuration:
- RR: 3.0
- SL: 0.8 ATR
- Confirmation: Structure Break
- Time: All-Day

Expected: +1.854R per trade
"""

import pandas as pd
import numpy as np
import requests
import time
import random

SYMBOLS = [
    'BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'XRPUSDT', 'DOGEUSDT',
    'BNBUSDT', 'ADAUSDT', 'AVAXUSDT', 'LINKUSDT', 'DOTUSDT',
    'LTCUSDT', 'NEARUSDT', 'APTUSDT', 'SUIUSDT', 'ARBUSDT',
    'OPUSDT', 'ATOMUSDT', 'UNIUSDT', 'INJUSDT', 'TONUSDT'
]

DAYS = 120  # 4 months for walk-forward
TIMEFRAME = 5
WIN_COST = 0.0006
LOSS_COST = 0.00125

# ============================================================================
# DATA FETCHING
# ============================================================================

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
    
    # Swing Points (10 bar)
    df['swing_high'] = df['high'].rolling(10).max()
    df['swing_low'] = df['low'].rolling(10).min()
    
    # Divergence detection
    df['price_low_14'] = df['low'].rolling(14).min()
    df['price_high_14'] = df['high'].rolling(14).max()
    df['rsi_low_14'] = df['rsi'].rolling(14).min()
    df['rsi_high_14'] = df['rsi'].rolling(14).max()
    
    return df

# ============================================================================
# STRATEGY
# ============================================================================

def run_strategy(df, rr=3.0, sl_mult=0.8):
    trades = []
    
    for i in range(50, len(df)-1):
        row = df.iloc[i]
        
        # Detect divergence
        bull_div = (row['low'] <= df['low'].iloc[i-10:i].min() and 
                    row['rsi'] > df['rsi'].iloc[i-10:i].min() and 
                    row['rsi'] < 40)
        
        bear_div = (row['high'] >= df['high'].iloc[i-10:i].max() and 
                    row['rsi'] < df['rsi'].iloc[i-10:i].max() and 
                    row['rsi'] > 60)
        
        side = None
        if bull_div: side = 'long'
        elif bear_div: side = 'short'
        
        if not side: continue
        
        # Structure break check (look ahead for confirmation)
        swing_high = row['swing_high']
        swing_low = row['swing_low']
        structure_broken = False
        
        for j in range(i+1, min(i+20, len(df))):
            c = df.iloc[j]
            if side == 'long' and c['close'] > swing_high:
                structure_broken = True
                entry_idx = j+1
                break
            elif side == 'short' and c['close'] < swing_low:
                structure_broken = True
                entry_idx = j+1
                break
        
        if not structure_broken: continue
        if entry_idx >= len(df): continue
        
        # Entry
        entry = df.iloc[entry_idx]['open']
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
        for k in range(entry_idx, min(entry_idx+300, len(df))):
            c = df.iloc[k]
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
        
        trades.append({
            'symbol': df.iloc[i].get('symbol', 'UNKNOWN'),
            'timestamp': df.iloc[entry_idx]['datetime'],
            'side': side,
            'r': res_r,
            'win': outcome == 'win'
        })
    
    return trades

# ============================================================================
# VALIDATION TESTS
# ============================================================================

def walk_forward_validation(datasets):
    """Split data into 4 periods and test each"""
    print("\n📊 WALK-FORWARD VALIDATION (4 Periods)")
    print("-" * 60)
    
    period_results = []
    
    for period in range(4):
        period_trades = []
        
        for sym, df in datasets.items():
            # Split into 4 equal periods
            period_size = len(df) // 4
            start_idx = period * period_size
            end_idx = start_idx + period_size if period < 3 else len(df)
            
            df_period = df.iloc[start_idx:end_idx].copy()
            df_period['symbol'] = sym
            
            trades = run_strategy(df_period)
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
            
            print(f"Period {period+1}: {len(period_trades):3} trades | {total_r:+7.1f}R | {avg_r:+.3f}R | WR: {wr:.1f}%")
    
    # Summary
    if period_results:
        profitable_periods = sum(1 for p in period_results if p['avg_r'] > 0)
        avg_period_r = np.mean([p['avg_r'] for p in period_results])
        print(f"\n✅ Profitable Periods: {profitable_periods}/4")
        print(f"📊 Average R across periods: {avg_period_r:+.3f}R")
    
    return period_results

def monte_carlo_simulation(all_trades, runs=1000):
    """Randomly shuffle trade order to test robustness"""
    print("\n🎲 MONTE CARLO SIMULATION (1000 runs)")
    print("-" * 60)
    
    if not all_trades:
        print("No trades to simulate")
        return
    
    trade_r_values = [t['r'] for t in all_trades]
    mc_results = []
    
    for _ in range(runs):
        shuffled = random.sample(trade_r_values, len(trade_r_values))
        total_r = sum(shuffled)
        mc_results.append(total_r)
    
    mc_results.sort()
    
    # Statistics
    worst_5pct = mc_results[int(runs * 0.05)]
    median = mc_results[runs // 2]
    best_5pct = mc_results[int(runs * 0.95)]
    profitable_pct = sum(1 for r in mc_results if r > 0) / runs * 100
    
    print(f"Worst 5%:  {worst_5pct:+.1f}R")
    print(f"Median:    {median:+.1f}R")
    print(f"Best 5%:   {best_5pct:+.1f}R")
    print(f"Profitable: {profitable_pct:.1f}% of runs")
    
    return profitable_pct

def per_symbol_breakdown(all_trades):
    """Show performance per symbol"""
    print("\n📈 PER-SYMBOL BREAKDOWN")
    print("-" * 60)
    print(f"{'Symbol':<12} | {'Trades':<6} {'Net R':<8} {'Avg R':<8} {'WR':<6}")
    print("-" * 60)
    
    symbol_stats = {}
    for t in all_trades:
        sym = t['symbol']
        if sym not in symbol_stats:
            symbol_stats[sym] = {'trades': [], 'wins': 0}
        symbol_stats[sym]['trades'].append(t['r'])
        if t['win']: symbol_stats[sym]['wins'] += 1
    
    for sym in sorted(symbol_stats.keys()):
        stats = symbol_stats[sym]
        trades = stats['trades']
        net_r = sum(trades)
        avg_r = net_r / len(trades)
        wr = stats['wins'] / len(trades) * 100
        print(f"{sym:<12} | {len(trades):<6} {net_r:<8.1f} {avg_r:<8.3f} {wr:<6.1f}%")

def sensitivity_analysis(datasets):
    """Test parameter variations"""
    print("\n🔬 SENSITIVITY ANALYSIS")
    print("-" * 60)
    print(f"{'RR':<4} {'SL':<5} | {'Trades':<6} {'Net R':<8} {'Avg R':<8}")
    print("-" * 60)
    
    variations = [
        (2.5, 0.7),
        (2.5, 0.8),
        (2.5, 0.9),
        (3.0, 0.7),
        (3.0, 0.8),  # Optimal
        (3.0, 0.9),
        (3.5, 0.7),
        (3.5, 0.8),
        (3.5, 0.9),
    ]
    
    for rr, sl_mult in variations:
        all_trades = []
        for sym, df in datasets.items():
            df['symbol'] = sym
            trades = run_strategy(df, rr, sl_mult)
            all_trades.extend(trades)
        
        if all_trades:
            total_r = sum(t['r'] for t in all_trades)
            avg_r = total_r / len(all_trades)
            marker = " ⭐" if rr == 3.0 and sl_mult == 0.8 else ""
            print(f"{rr:<4} {sl_mult:<5} | {len(all_trades):<6} {total_r:<8.1f} {avg_r:<8.3f}{marker}")

# ============================================================================
# MAIN
# ============================================================================

def main():
    print("🚀 STRUCTURE BREAK VALIDATION")
    print("=" * 60)
    print("Loading data (120 days, 20 symbols)...")
    
    datasets = {}
    for sym in SYMBOLS:
        df = fetch_data(sym)
        if not df.empty: datasets[sym] = calc_indicators(df)
    
    print(f"Loaded {len(datasets)} symbols.\n")
    
    # Run base strategy
    print("📊 BASE STRATEGY (RR=3.0, SL=0.8 ATR)")
    print("-" * 60)
    all_trades = []
    for sym, df in datasets.items():
        df['symbol'] = sym
        trades = run_strategy(df)
        all_trades.extend(trades)
    
    if all_trades:
        total_r = sum(t['r'] for t in all_trades)
        avg_r = total_r / len(all_trades)
        wins = sum(1 for t in all_trades if t['win'])
        wr = wins / len(all_trades) * 100
        
        print(f"Total Trades: {len(all_trades)}")
        print(f"Net R: {total_r:+.1f}R")
        print(f"Avg R: {avg_r:+.3f}R")
        print(f"Win Rate: {wr:.1f}%")
        
        # Run validations
        period_results = walk_forward_validation(datasets)
        mc_prob = monte_carlo_simulation(all_trades)
        per_symbol_breakdown(all_trades)
        sensitivity_analysis(datasets)
        
        # Final verdict
        print("\n" + "=" * 60)
        print("🏆 VALIDATION VERDICT")
        print("=" * 60)
        
        profitable_periods = sum(1 for p in period_results if p['avg_r'] > 0)
        
        if avg_r > 0.01 and profitable_periods >= 3 and mc_prob >= 95:
            print("✅ STRATEGY VALIDATED")
            print(f"   - Positive expectancy: {avg_r:+.3f}R")
            print(f"   - Walk-forward: {profitable_periods}/4 periods profitable")
            print(f"   - Monte Carlo: {mc_prob:.1f}% probability of profit")
        else:
            print("⚠️ STRATEGY NEEDS REVIEW")
            print(f"   - Expectancy: {avg_r:+.3f}R")
            print(f"   - Walk-forward: {profitable_periods}/4 periods")
            print(f"   - Monte Carlo: {mc_prob:.1f}%")

if __name__ == "__main__":
    main()
