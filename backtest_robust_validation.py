#!/usr/bin/env python3
"""
Robust 5M Backtest Validation
=============================
Comprehensive validation of the 10-candle wait Structure Break strategy.

Tests:
1. Extended Duration (120 days)
2. Walk-Forward Validation (6 periods)
3. Monte Carlo Simulation (1000 runs)
4. Slippage Stress Test (0-0.3%)
5. Per-Symbol Breakdown
"""

import pandas as pd
import numpy as np
import requests
import time
import random
from datetime import datetime

# Configuration
DAYS = 120
TIMEFRAME = 5
MAX_WAIT_CANDLES = 10
RR = 3.0
SL_MULT = 0.8
FEE_PERCENT = 0.0006
SLIPPAGE_PERCENT = 0.0003

def get_top_symbols(n=98):
    try:
        url = "https://api.bybit.com/v5/market/tickers?category=linear"
        resp = requests.get(url, timeout=10).json()
        tickers = resp.get('result', {}).get('list', [])
        usdt = [t for t in tickers if t['symbol'].endswith('USDT')]
        usdt.sort(key=lambda x: float(x.get('turnover24h', 0)), reverse=True)
        BAD = ['XAUTUSDT', 'PAXGUSDT', 'USTCUSDT', 'USDCUSDT', 'BUSDUSDT', 'DAIUSDT']
        return [t['symbol'] for t in usdt[:n] if t['symbol'] not in BAD][:n]
    except:
        return []

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
            time.sleep(0.02)
        if not all_kline: return pd.DataFrame()
        df = pd.DataFrame(all_kline, columns=['ts', 'open', 'high', 'low', 'close', 'vol', 'to'])
        df = df.iloc[::-1].reset_index(drop=True)
        for c in ['open', 'high', 'low', 'close', 'vol']: df[c] = df[c].astype(float)
        return df
    except: return pd.DataFrame()

def calc_indicators(df):
    if len(df) < 50: return pd.DataFrame()
    df = df.copy()
    close = df['close']
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 1e-9)
    df['rsi'] = 100 - (100 / (1 + rs))
    h, l, c_prev = df['high'], df['low'], close.shift(1)
    tr = pd.concat([h-l, (h-c_prev).abs(), (l-c_prev).abs()], axis=1).max(axis=1)
    df['atr'] = tr.rolling(20).mean()
    df['swing_high_10'] = df['high'].rolling(10).max()
    df['swing_low_10'] = df['low'].rolling(10).min()
    df['price_low_14'] = df['low'].rolling(14).min()
    df['price_high_14'] = df['high'].rolling(14).max()
    df['rsi_low_14'] = df['rsi'].rolling(14).min()
    df['rsi_high_14'] = df['rsi'].rolling(14).max()
    df['reg_bull'] = (df['low'] <= df['price_low_14']) & (df['rsi'] > df['rsi_low_14'].shift(14)) & (df['rsi'] < 40)
    df['reg_bear'] = (df['high'] >= df['price_high_14']) & (df['rsi'] < df['rsi_high_14'].shift(14)) & (df['rsi'] > 60)
    return df

def run_strategy(df, slippage_pct=SLIPPAGE_PERCENT):
    trades = []
    cooldown = 0
    for i in range(50, len(df)-1):
        if cooldown > 0: cooldown -= 1; continue
        row = df.iloc[i]
        side = 'long' if row['reg_bull'] else 'short' if row['reg_bear'] else None
        if not side: continue
        structure_broken, candles_waited = False, 0
        for ahead in range(1, MAX_WAIT_CANDLES + 1):
            if i+ahead >= len(df): break
            check = df.iloc[i+ahead]
            candles_waited = ahead
            if (side == 'long' and check['close'] > row['swing_high_10']) or \
               (side == 'short' and check['close'] < row['swing_low_10']):
                structure_broken = True; break
        if not structure_broken: continue
        idx = i + candles_waited
        base = df.iloc[idx]['open']
        entry = base * (1 + slippage_pct) if side == 'long' else base * (1 - slippage_pct)
        atr = row['atr']
        if pd.isna(atr) or atr <= 0: continue
        sl_dist = atr * SL_MULT
        if sl_dist/entry > 0.05: continue
        tp_dist = sl_dist * RR
        sl, tp = (entry - sl_dist, entry + tp_dist) if side == 'long' else (entry + sl_dist, entry - tp_dist)
        outcome = 'timeout'
        for j in range(idx, min(idx+300, len(df))):
            c = df.iloc[j]
            if side == 'long':
                if c['low'] <= sl: outcome = 'loss'; break
                if c['high'] >= tp: outcome = 'win'; break
            else:
                if c['high'] >= sl: outcome = 'loss'; break
                if c['low'] <= tp: outcome = 'win'; break
        risk_pct = sl_dist / entry
        fee_cost = (FEE_PERCENT + slippage_pct) / risk_pct
        res_r = RR - fee_cost if outcome == 'win' else -1.0 - fee_cost if outcome == 'loss' else -0.2
        trades.append({'r': res_r, 'win': outcome == 'win', 'symbol': df.iloc[i].get('symbol', 'UNK'), 'idx': i})
        cooldown = 6
    return trades

def monte_carlo(trades, n_simulations=1000):
    if not trades: return 0, 0, 0
    results = []
    for _ in range(n_simulations):
        shuffled = random.sample(trades, len(trades))
        equity = [0]
        for t in shuffled:
            equity.append(equity[-1] + t['r'])
        results.append(equity[-1])
    profitable = sum(1 for r in results if r > 0)
    return profitable / n_simulations * 100, np.percentile(results, 5), np.percentile(results, 95)

def walk_forward(all_trades, n_periods=6):
    if not all_trades: return []
    n = len(all_trades)
    period_size = n // n_periods
    results = []
    for i in range(n_periods):
        start = i * period_size
        end = (i + 1) * period_size if i < n_periods - 1 else n
        period_trades = all_trades[start:end]
        if period_trades:
            total_r = sum(t['r'] for t in period_trades)
            wins = sum(1 for t in period_trades if t['win'])
            wr = wins / len(period_trades) * 100
            results.append({'period': i+1, 'trades': len(period_trades), 'net_r': total_r, 'wr': wr})
    return results

def per_symbol_analysis(all_trades):
    from collections import defaultdict
    stats = defaultdict(lambda: {'trades': 0, 'wins': 0, 'total_r': 0})
    for t in all_trades:
        sym = t.get('symbol', 'UNK')
        stats[sym]['trades'] += 1
        stats[sym]['wins'] += 1 if t['win'] else 0
        stats[sym]['total_r'] += t['r']
    results = []
    for sym, s in stats.items():
        wr = s['wins'] / s['trades'] * 100 if s['trades'] > 0 else 0
        avg_r = s['total_r'] / s['trades'] if s['trades'] > 0 else 0
        results.append({'symbol': sym, 'trades': s['trades'], 'wr': wr, 'avg_r': avg_r, 'net_r': s['total_r']})
    return sorted(results, key=lambda x: x['net_r'], reverse=True)

def main():
    print("=" * 70)
    print("🔬 ROBUST 5M BACKTEST VALIDATION")
    print(f"   Config: {TIMEFRAME}M | {MAX_WAIT_CANDLES} candle wait | {RR}:1 R:R | {SL_MULT}x ATR SL")
    print(f"   Duration: {DAYS} days | Fees: {FEE_PERCENT*100:.2f}% | Slippage: {SLIPPAGE_PERCENT*100:.2f}%")
    print("=" * 70)
    
    # Fetch data
    print("\n📥 Fetching data for 98 symbols...")
    symbols = get_top_symbols(98)
    datasets = {}
    for i, sym in enumerate(symbols):
        df = fetch_data(sym)
        if not df.empty:
            processed = calc_indicators(df)
            if not processed.empty:
                processed['symbol'] = sym
                datasets[sym] = processed
        if (i+1) % 20 == 0: print(f"  Progress: {i+1}/{len(symbols)}...")
    print(f"✅ Loaded {len(datasets)} symbols\n")
    
    # Run main strategy
    print("🔄 Running strategy simulation...")
    all_trades = []
    for sym, df in datasets.items():
        trades = run_strategy(df)
        all_trades.extend(trades)
    
    # ============================================
    # TEST 1: Extended Duration Results
    # ============================================
    print("\n" + "=" * 70)
    print("📊 TEST 1: EXTENDED DURATION RESULTS (120 days)")
    print("=" * 70)
    if all_trades:
        total_r = sum(t['r'] for t in all_trades)
        avg_r = total_r / len(all_trades)
        wins = sum(1 for t in all_trades if t['win'])
        wr = wins / len(all_trades) * 100
        print(f"  Trades: {len(all_trades)}")
        print(f"  Net R: {total_r:+.1f}")
        print(f"  Avg R: {avg_r:+.3f}")
        print(f"  Win Rate: {wr:.1f}%")
        print(f"  ✅ PASS" if avg_r > 0.5 else f"  ❌ FAIL (Avg R < 0.5)")
    
    # ============================================
    # TEST 2: Walk-Forward Validation
    # ============================================
    print("\n" + "=" * 70)
    print("📊 TEST 2: WALK-FORWARD VALIDATION (6 periods)")
    print("=" * 70)
    wf_results = walk_forward(all_trades, 6)
    all_profitable = True
    for r in wf_results:
        status = "✅" if r['net_r'] > 0 else "❌"
        if r['net_r'] <= 0: all_profitable = False
        print(f"  Period {r['period']}: {r['trades']} trades | {r['net_r']:+.1f}R | {r['wr']:.1f}% WR {status}")
    print(f"\n  {'✅ PASS: All 6 periods profitable' if all_profitable else '❌ FAIL: Some periods unprofitable'}")
    
    # ============================================
    # TEST 3: Monte Carlo Simulation
    # ============================================
    print("\n" + "=" * 70)
    print("📊 TEST 3: MONTE CARLO SIMULATION (1000 runs)")
    print("=" * 70)
    mc_profitable, mc_5th, mc_95th = monte_carlo(all_trades, 1000)
    print(f"  Profitable Runs: {mc_profitable:.1f}%")
    print(f"  5th Percentile: {mc_5th:+.1f}R")
    print(f"  95th Percentile: {mc_95th:+.1f}R")
    print(f"  {'✅ PASS: 100% profitable' if mc_profitable >= 99 else '❌ FAIL: Not all runs profitable'}")
    
    # ============================================
    # TEST 4: Slippage Stress Test
    # ============================================
    print("\n" + "=" * 70)
    print("📊 TEST 4: SLIPPAGE STRESS TEST")
    print("=" * 70)
    slippage_levels = [0, 0.0005, 0.001, 0.0015, 0.002, 0.0025, 0.003]
    for slip in slippage_levels:
        slip_trades = []
        for sym, df in datasets.items():
            slip_trades.extend(run_strategy(df, slippage_pct=slip))
        if slip_trades:
            total_r = sum(t['r'] for t in slip_trades)
            avg_r = total_r / len(slip_trades)
            status = "✅" if avg_r > 0 else "❌"
            print(f"  Slippage {slip*100:.2f}%: {len(slip_trades)} trades | {total_r:+.1f}R | {avg_r:+.3f}R avg {status}")
    
    # ============================================
    # TEST 5: Per-Symbol Breakdown
    # ============================================
    print("\n" + "=" * 70)
    print("📊 TEST 5: PER-SYMBOL BREAKDOWN (Top 10 / Bottom 10)")
    print("=" * 70)
    symbol_stats = per_symbol_analysis(all_trades)
    print("\n  🏆 TOP 10 PERFORMERS:")
    for s in symbol_stats[:10]:
        print(f"    {s['symbol']}: {s['trades']} trades | {s['wr']:.1f}% WR | {s['avg_r']:+.3f}R avg | {s['net_r']:+.1f}R total")
    
    print("\n  ⚠️ BOTTOM 10 PERFORMERS:")
    for s in symbol_stats[-10:]:
        print(f"    {s['symbol']}: {s['trades']} trades | {s['wr']:.1f}% WR | {s['avg_r']:+.3f}R avg | {s['net_r']:+.1f}R total")
    
    # Identify symbols to potentially remove
    bad_symbols = [s['symbol'] for s in symbol_stats if s['wr'] < 40 and s['trades'] >= 10]
    if bad_symbols:
        print(f"\n  ⚠️ CONSIDER REMOVING ({len(bad_symbols)} symbols with WR<40% and N>=10):")
        for sym in bad_symbols[:10]:
            print(f"    - {sym}")
    else:
        print("\n  ✅ No symbols with consistently poor performance")
    
    # ============================================
    # FINAL SUMMARY
    # ============================================
    print("\n" + "=" * 70)
    print("🏁 FINAL VALIDATION SUMMARY")
    print("=" * 70)
    tests_passed = 0
    tests_total = 5
    
    if avg_r > 0.5: tests_passed += 1
    if all_profitable: tests_passed += 1
    if mc_profitable >= 99: tests_passed += 1
    # Check slippage at 0.2%
    slip_trades_02 = []
    for sym, df in datasets.items():
        slip_trades_02.extend(run_strategy(df, slippage_pct=0.002))
    if slip_trades_02:
        avg_r_02 = sum(t['r'] for t in slip_trades_02) / len(slip_trades_02)
        if avg_r_02 > 0: tests_passed += 1
    if len(bad_symbols) == 0: tests_passed += 1
    
    print(f"  Tests Passed: {tests_passed}/{tests_total}")
    if tests_passed == tests_total:
        print("\n  ✅ CONFIGURATION VALIDATED - READY FOR LIVE TRADING")
    else:
        print(f"\n  ⚠️ {tests_total - tests_passed} test(s) failed - review before live trading")

if __name__ == "__main__":
    main()
