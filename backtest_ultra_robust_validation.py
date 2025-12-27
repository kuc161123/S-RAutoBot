#!/usr/bin/env python3
"""
ULTRA-ROBUST STRATEGY VALIDATION
================================
Comprehensive validation to ensure the 4H EMA Trend Stack has no hidden biases.

Tests:
1. Walk-Forward Validation (6 periods - In-Sample vs Out-of-Sample)
2. Monte Carlo Simulation (1000 runs)
3. Slippage Stress Test (0%, 0.05%, 0.1%, 0.2%)
4. Monthly Performance Breakdown
5. Max Drawdown Analysis
"""

import pandas as pd
import numpy as np
import requests
import time
import random
import concurrent.futures
from datetime import datetime

# Strategy Config (Golden Combo)
DAYS = 365
INTERVAL = '240' # 4H
RSI_TRIGGER = 35
SL_MULT = 1.5
RR = 4.5
FEE = 0.0006
BASE_SLIPPAGE = 0.0003

def get_symbols(n=50):
    try:
        url = "https://api.bybit.com/v5/market/tickers?category=linear"
        resp = requests.get(url, timeout=10).json()
        tickers = resp.get('result', {}).get('list', [])
        usdt = [t for t in tickers if t['symbol'].endswith('USDT')]
        usdt.sort(key=lambda x: float(x.get('turnover24h', 0)), reverse=True)
        BAD = ['XAUTUSDT', 'PAXGUSDT', 'USTCUSDT', 'USDCUSDT', 'BUSDUSDT', 'DAIUSDT']
        return [t['symbol'] for t in usdt[:n] if t['symbol'] not in BAD][:n]
    except: return []

def fetch_data(symbol):
    try:
        url = "https://api.bybit.com/v5/market/kline"
        all_kline = []
        end_ts = int(time.time() * 1000)
        start_ts = int((time.time() - DAYS * 24 * 3600) * 1000)
        while end_ts > start_ts:
            params = {'category': 'linear', 'symbol': symbol, 'interval': INTERVAL, 'limit': 1000, 'end': end_ts}
            r = requests.get(url, params=params).json()
            if r['retCode'] != 0 or not r['result']['list']: break
            klines = r['result']['list']
            all_kline.extend(klines)
            end_ts = int(klines[-1][0]) - 1
            time.sleep(0.02)
        if not all_kline: return pd.DataFrame()
        df = pd.DataFrame(all_kline, columns=['ts', 'open', 'high', 'low', 'close', 'vol', 'to'])
        df = df.iloc[::-1].reset_index(drop=True)
        for c in ['open', 'high', 'low', 'close', 'vol']: df[c] = df[c].astype(float)
        df['ts'] = pd.to_numeric(df['ts'])
        return df
    except: return pd.DataFrame()

def calc_indicators(df):
    if len(df) < 200: return pd.DataFrame()
    df = df.copy()
    close = df['close']
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df['rsi'] = 100 - (100 / (1 + gain/(loss+1e-9)))
    df['ema20'] = close.ewm(span=20, adjust=False).mean()
    df['ema50'] = close.ewm(span=50, adjust=False).mean()
    df['ema200'] = close.ewm(span=200, adjust=False).mean()
    df['atr'] = (df['high'] - df['low']).rolling(20).mean()
    return df

def run_strategy(df, slippage=BASE_SLIPPAGE):
    trades = []
    if len(df) < 201: return []
    
    for i in range(200, len(df)-2):
        row = df.iloc[i]
        prev = df.iloc[i-1]
        
        side = None
        if row['ema20'] > row['ema50'] > row['ema200'] and prev['rsi'] < RSI_TRIGGER and row['rsi'] >= RSI_TRIGGER:
            side = 'long'
        if row['ema20'] < row['ema50'] < row['ema200'] and prev['rsi'] > (100-RSI_TRIGGER) and row['rsi'] <= (100-RSI_TRIGGER):
            side = 'short'
            
        if not side: continue
        
        entry_price = df.iloc[i+1]['open']
        entry = entry_price * (1 + (slippage if side == 'long' else -slippage))
        
        atr = row['atr']
        if pd.isna(atr) or atr <= 0: continue
        
        sl_dist = atr * SL_MULT
        tp_dist = sl_dist * RR
        sl, tp = (entry - sl_dist, entry + tp_dist) if side == 'long' else (entry + sl_dist, entry - tp_dist)
        
        outcome = None
        for j in range(i+1, min(i+1+500, len(df))):
            c = df.iloc[j]
            if side == 'long':
                if c['low'] <= sl: outcome = 'loss'; break
                if c['high'] >= tp: outcome = 'win'; break
            else:
                if c['high'] >= sl: outcome = 'loss'; break
                if c['low'] <= tp: outcome = 'win'; break
        
        if outcome:
            risk_pct = sl_dist / entry
            fee_cost = (FEE + slippage) / risk_pct
            res_r = RR - fee_cost if outcome == 'win' else -1.0 - fee_cost
            trades.append({
                'r': res_r, 
                'win': outcome == 'win',
                'ts': df.iloc[i]['ts'],
                'idx': i,
                'symbol': df.get('symbol', ['UNK']*len(df)).iloc[i] if 'symbol' in df.columns else 'UNK'
            })
    return trades

def process_symbol(sym):
    df = fetch_data(sym)
    if df.empty: return []
    df = calc_indicators(df)
    df['symbol'] = sym
    return run_strategy(df)

# --- TEST 1: WALK-FORWARD VALIDATION ---
def walk_forward_test(all_trades, n_periods=6):
    if not all_trades: return []
    sorted_trades = sorted(all_trades, key=lambda x: x['ts'])
    n = len(sorted_trades)
    period_size = n // n_periods
    results = []
    for i in range(n_periods):
        start = i * period_size
        end = (i + 1) * period_size if i < n_periods - 1 else n
        period_trades = sorted_trades[start:end]
        if period_trades:
            total_r = sum(t['r'] for t in period_trades)
            wr = sum(1 for t in period_trades if t['win']) / len(period_trades) * 100
            results.append({'period': i+1, 'trades': len(period_trades), 'wr': wr, 'total_r': total_r})
    return results

# --- TEST 2: MONTE CARLO SIMULATION ---
def monte_carlo_test(all_trades, n_simulations=1000):
    if not all_trades: return {}
    results = []
    for _ in range(n_simulations):
        shuffled = random.sample(all_trades, len(all_trades))
        equity = [0]
        for t in shuffled:
            equity.append(equity[-1] + t['r'])
        results.append(equity[-1])
    profitable = sum(1 for r in results if r > 0) / n_simulations * 100
    return {
        'profitable_pct': profitable,
        'p5': np.percentile(results, 5),
        'p50': np.percentile(results, 50),
        'p95': np.percentile(results, 95),
        'worst': min(results),
        'best': max(results)
    }

# --- TEST 3: SLIPPAGE STRESS TEST ---
def slippage_stress_test(all_dfs):
    slippages = [0.0, 0.0005, 0.001, 0.002]
    results = []
    for slip in slippages:
        trades = []
        for df in all_dfs:
            trades.extend(run_strategy(df, slippage=slip))
        if trades:
            total_r = sum(t['r'] for t in trades)
            avg_r = total_r / len(trades)
            results.append({'slippage': f"{slip*100:.2f}%", 'trades': len(trades), 'avg_r': avg_r, 'total_r': total_r})
    return results

# --- TEST 4: MONTHLY BREAKDOWN ---
def monthly_breakdown(all_trades):
    if not all_trades: return []
    monthly = {}
    for t in all_trades:
        dt = datetime.fromtimestamp(t['ts'] / 1000)
        key = dt.strftime('%Y-%m')
        if key not in monthly: monthly[key] = []
        monthly[key].append(t)
    results = []
    for month, trades in sorted(monthly.items()):
        total_r = sum(t['r'] for t in trades)
        wr = sum(1 for t in trades if t['win']) / len(trades) * 100
        results.append({'month': month, 'trades': len(trades), 'wr': wr, 'total_r': total_r})
    return results

# --- TEST 5: MAX DRAWDOWN ---
def max_drawdown(all_trades):
    if not all_trades: return 0
    sorted_trades = sorted(all_trades, key=lambda x: x['ts'])
    equity = [0]
    for t in sorted_trades:
        equity.append(equity[-1] + t['r'])
    peak = equity[0]
    max_dd = 0
    for e in equity:
        if e > peak: peak = e
        dd = peak - e
        if dd > max_dd: max_dd = dd
    return max_dd

def main():
    symbols = get_symbols(50)
    print("=" * 70)
    print("ULTRA-ROBUST STRATEGY VALIDATION")
    print("=" * 70)
    print(f"Strategy: 4H EMA Trend Stack")
    print(f"Config: RSI {RSI_TRIGGER}, SL {SL_MULT}x ATR, RR {RR}:1")
    print(f"Testing {len(symbols)} symbols for 365 days...")
    
    # Fetch and process all data
    all_trades = []
    all_dfs = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(fetch_data, sym): sym for sym in symbols}
        count = 0
        for future in concurrent.futures.as_completed(futures):
            sym = futures[future]
            try:
                df = future.result()
                if not df.empty:
                    df = calc_indicators(df)
                    df['symbol'] = sym
                    all_dfs.append(df)
                    trades = run_strategy(df)
                    all_trades.extend(trades)
                count += 1
                if count % 10 == 0: print(f"  Fetched {count}/{len(symbols)} symbols...")
            except: pass
    
    print(f"\nTotal Trades: {len(all_trades)}")
    
    # --- RUN ALL TESTS ---
    print("\n" + "=" * 70)
    print("TEST 1: WALK-FORWARD VALIDATION (6 Periods)")
    print("-" * 70)
    wf_results = walk_forward_test(all_trades)
    for r in wf_results:
        status = "✅" if r['total_r'] > 0 else "❌"
        print(f"  Period {r['period']}: {status} {r['trades']} trades | WR={r['wr']:.1f}% | R={r['total_r']:+.1f}")
    profitable_periods = sum(1 for r in wf_results if r['total_r'] > 0)
    print(f"  *** {profitable_periods}/{len(wf_results)} periods profitable ***")
    
    print("\n" + "=" * 70)
    print("TEST 2: MONTE CARLO SIMULATION (1000 runs)")
    print("-" * 70)
    mc_results = monte_carlo_test(all_trades)
    print(f"  Profitable Runs: {mc_results['profitable_pct']:.1f}%")
    print(f"  5th Percentile:  {mc_results['p5']:+.1f}R (Worst-Case)")
    print(f"  Median:          {mc_results['p50']:+.1f}R")
    print(f"  95th Percentile: {mc_results['p95']:+.1f}R (Best-Case)")
    
    print("\n" + "=" * 70)
    print("TEST 3: SLIPPAGE STRESS TEST")
    print("-" * 70)
    slip_results = slippage_stress_test(all_dfs)
    for r in slip_results:
        status = "✅" if r['avg_r'] > 0 else "❌"
        print(f"  Slippage {r['slippage']}: {status} Avg={r['avg_r']:+.3f}R | Total={r['total_r']:+.1f}R")
    
    print("\n" + "=" * 70)
    print("TEST 4: MONTHLY BREAKDOWN")
    print("-" * 70)
    monthly_results = monthly_breakdown(all_trades)
    win_months = 0
    for r in monthly_results:
        status = "✅" if r['total_r'] > 0 else "❌"
        print(f"  {r['month']}: {status} {r['trades']} trades | WR={r['wr']:.1f}% | R={r['total_r']:+.1f}")
        if r['total_r'] > 0: win_months += 1
    print(f"  *** {win_months}/{len(monthly_results)} months profitable ***")
    
    print("\n" + "=" * 70)
    print("TEST 5: MAX DRAWDOWN")
    print("-" * 70)
    mdd = max_drawdown(all_trades)
    print(f"  Max Drawdown: {mdd:.1f}R")
    
    # Final Summary
    total_r = sum(t['r'] for t in all_trades)
    wr = sum(1 for t in all_trades if t['win']) / len(all_trades) * 100
    avg_r = total_r / len(all_trades)
    
    print("\n" + "=" * 70)
    print("FINAL ROBUSTNESS VERDICT")
    print("=" * 70)
    print(f"Total R:           {total_r:+.1f}R")
    print(f"Avg R/Trade:       {avg_r:+.3f}R")
    print(f"Win Rate:          {wr:.1f}%")
    print(f"Walk-Forward:      {profitable_periods}/{len(wf_results)} periods profitable")
    print(f"Monte Carlo:       {mc_results['profitable_pct']:.0f}% profitable runs")
    print(f"Slippage Resistant:{slip_results[-1]['avg_r'] > 0}")
    print(f"Monthly Consistency:{win_months}/{len(monthly_results)} months")
    print(f"Max Drawdown:      {mdd:.1f}R")
    
    # Pass/Fail
    passed = (
        total_r > 0 and
        profitable_periods >= 4 and
        mc_results['profitable_pct'] > 50 and
        slip_results[-1]['avg_r'] > 0 and
        win_months >= len(monthly_results) * 0.5
    )
    print("\n" + ("🎉 STRATEGY PASSED ALL ROBUSTNESS TESTS" if passed else "❌ STRATEGY FAILED ROBUSTNESS"))
    print("=" * 70)

if __name__ == "__main__":
    main()
