#!/usr/bin/env python3
"""
COMPREHENSIVE 100 SYMBOL VALIDATION
====================================

Rigorous testing of 100 symbols to confirm profitability:
1. Walk-Forward (6 periods)
2. Monte Carlo (1000 runs)
3. Symbol Quality Analysis
4. Recent 30-Day Check
5. Drawdown Analysis
"""

import pandas as pd
import numpy as np
import requests
import time
import random

DAYS = 60
TIMEFRAME = 5
RR = 3.0
SL_MULT = 0.8
WIN_COST = 0.0006
LOSS_COST = 0.00125

def get_top_100_symbols():
    try:
        url = "https://api.bybit.com/v5/market/tickers?category=linear"
        resp = requests.get(url, timeout=10).json()
        tickers = resp.get('result', {}).get('list', [])
        usdt_pairs = [t for t in tickers if t['symbol'].endswith('USDT')]
        usdt_pairs.sort(key=lambda x: float(x.get('turnover24h', 0)), reverse=True)
        BAD = ['XAUTUSDT', 'PAXGUSDT', 'USTCUSDT', 'USDCUSDT', 'BUSDUSDT', 'DAIUSDT']
        return [t['symbol'] for t in usdt_pairs[:100] if t['symbol'] not in BAD][:100]
    except:
        return []

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
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 1e-9)
    df['rsi'] = 100 - (100 / (1 + rs))
    h, l, c_prev = df['high'], df['low'], close.shift(1)
    tr = pd.concat([h-l, (h-c_prev).abs(), (l-c_prev).abs()], axis=1).max(axis=1)
    df['atr'] = tr.rolling(14).mean()
    df['swing_high_10'] = df['high'].rolling(10).max()
    df['swing_low_10'] = df['low'].rolling(10).min()
    df['price_low_14'] = df['low'].rolling(14).min()
    df['price_high_14'] = df['high'].rolling(14).max()
    df['rsi_low_14'] = df['rsi'].rolling(14).min()
    df['rsi_high_14'] = df['rsi'].rolling(14).max()
    df['reg_bull'] = (df['low'] <= df['price_low_14']) & (df['rsi'] > df['rsi_low_14'].shift(14)) & (df['rsi'] < 40)
    df['reg_bear'] = (df['high'] >= df['price_high_14']) & (df['rsi'] < df['rsi_high_14'].shift(14)) & (df['rsi'] > 60)
    return df

def run_strategy(df, symbol=''):
    trades = []
    cooldown = 0
    for i in range(50, len(df)-1):
        if cooldown > 0: cooldown -= 1; continue
        row = df.iloc[i]
        side = None
        if row['reg_bull']: side = 'long'
        elif row['reg_bear']: side = 'short'
        if not side: continue
        if i+1 >= len(df): continue
        next_row = df.iloc[i+1]
        structure_broken = False
        if side == 'long' and next_row['close'] > row['swing_high_10']: structure_broken = True
        if side == 'short' and next_row['close'] < row['swing_low_10']: structure_broken = True
        if not structure_broken: continue
        entry = df.iloc[i+1]['open']
        atr = row['atr']
        if pd.isna(atr) or atr == 0: continue
        sl_dist = atr * SL_MULT
        if sl_dist/entry > 0.05: continue
        tp_dist = sl_dist * RR
        if side == 'long': sl, tp = entry - sl_dist, entry + tp_dist
        else: sl, tp = entry + sl_dist, entry - tp_dist
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
        trades.append({'symbol': symbol, 'r': res_r, 'win': outcome == 'win'})
        cooldown = 5
    return trades

# ============================================================================
# VALIDATION TESTS
# ============================================================================

def walk_forward(datasets):
    print("\n📊 WALK-FORWARD VALIDATION (6 × 10-day periods)")
    print("-" * 70)
    results = []
    for period in range(6):
        trades = []
        for sym, df in datasets.items():
            size = len(df) // 6
            start, end = period * size, (period + 1) * size if period < 5 else len(df)
            df_p = df.iloc[start:end].copy()
            trades.extend(run_strategy(df_p, sym))
        if trades:
            total_r = sum(t['r'] for t in trades)
            avg_r = total_r / len(trades)
            wr = sum(1 for t in trades if t['win']) / len(trades) * 100
            results.append({'period': period+1, 'trades': len(trades), 'avg_r': avg_r, 'wr': wr})
            status = "✅" if avg_r > 0 else "❌"
            print(f"Period {period+1}: {len(trades):3} trades | {total_r:+8.1f}R | {avg_r:+.3f}R | WR: {wr:.1f}% {status}")
    if results:
        profitable = sum(1 for r in results if r['avg_r'] > 0)
        avg = np.mean([r['avg_r'] for r in results])
        print(f"\n✅ Profitable Periods: {profitable}/6")
        print(f"📊 Average Expectancy: {avg:+.3f}R")
    return results

def monte_carlo(all_trades, runs=1000):
    print("\n🎲 MONTE CARLO SIMULATION (1000 runs)")
    print("-" * 70)
    if not all_trades: return 0
    r_vals = [t['r'] for t in all_trades]
    mc = [sum(random.choices(r_vals, k=len(r_vals))) for _ in range(runs)]
    mc.sort()
    worst_5 = mc[int(runs * 0.05)]
    median = mc[runs // 2]
    best_5 = mc[int(runs * 0.95)]
    profitable = sum(1 for r in mc if r > 0) / runs * 100
    print(f"Worst 5%:    {worst_5:+.1f}R")
    print(f"Median:      {median:+.1f}R")
    print(f"Best 5%:     {best_5:+.1f}R")
    print(f"Profitable:  {profitable:.1f}% of runs")
    return profitable

def drawdown_analysis(all_trades):
    print("\n📉 DRAWDOWN ANALYSIS")
    print("-" * 70)
    if not all_trades: return
    equity = [0]
    for t in all_trades:
        equity.append(equity[-1] + t['r'])
    peak = equity[0]
    max_dd = 0
    for e in equity:
        if e > peak: peak = e
        dd = peak - e
        if dd > max_dd: max_dd = dd
    print(f"Peak Equity:    {max(equity):+.1f}R")
    print(f"Max Drawdown:   {max_dd:.1f}R")
    print(f"Final Equity:   {equity[-1]:+.1f}R")
    dd_pct = (max_dd / max(equity)) * 100 if max(equity) > 0 else 0
    print(f"DD/Peak Ratio:  {dd_pct:.1f}%")

def symbol_quality(all_trades):
    print("\n📊 SYMBOL QUALITY BREAKDOWN")
    print("-" * 70)
    stats = {}
    for t in all_trades:
        if t['symbol'] not in stats: stats[t['symbol']] = []
        stats[t['symbol']].append(t)
    results = []
    for sym, trades in stats.items():
        net = sum(t['r'] for t in trades)
        avg = net / len(trades)
        wr = sum(1 for t in trades if t['win']) / len(trades) * 100
        results.append({'symbol': sym, 'trades': len(trades), 'net_r': net, 'avg_r': avg, 'wr': wr})
    results.sort(key=lambda x: x['avg_r'], reverse=True)
    
    # Count quality tiers
    tier1 = [r for r in results if r['avg_r'] >= 2.5]  # Excellent
    tier2 = [r for r in results if 1.5 <= r['avg_r'] < 2.5]  # Good
    tier3 = [r for r in results if 0.5 <= r['avg_r'] < 1.5]  # OK
    tier4 = [r for r in results if 0 <= r['avg_r'] < 0.5]  # Marginal
    tier5 = [r for r in results if r['avg_r'] < 0]  # Losing
    
    print(f"🏆 Excellent (≥2.5R): {len(tier1)} symbols")
    print(f"✅ Good (1.5-2.5R):    {len(tier2)} symbols")
    print(f"⚡ OK (0.5-1.5R):      {len(tier3)} symbols")
    print(f"⚠️  Marginal (0-0.5R): {len(tier4)} symbols")
    print(f"❌ Losing (<0R):       {len(tier5)} symbols")
    
    if tier5:
        print(f"\nLosing symbols: {', '.join([r['symbol'] for r in tier5])}")
    
    return len(tier5)

def recent_30_day(symbols):
    print("\n🔥 RECENT 30-DAY CHECK (Live Readiness)")
    print("-" * 70)
    print("Loading recent 30-day data...")
    datasets = {}
    for i, sym in enumerate(symbols[:50]):  # Test subset for speed
        df = fetch_data(sym, days=30)
        if not df.empty: datasets[sym] = calc_indicators(df)
        if (i+1) % 10 == 0: print(f"Progress: {i+1}/50...")
    
    trades = []
    for sym, df in datasets.items():
        trades.extend(run_strategy(df, sym))
    
    if trades:
        total_r = sum(t['r'] for t in trades)
        avg_r = total_r / len(trades)
        wr = sum(1 for t in trades if t['win']) / len(trades) * 100
        print(f"\nTrades:      {len(trades)}")
        print(f"Net R:       {total_r:+.1f}R")
        print(f"Avg R:       {avg_r:+.3f}R")
        print(f"Win Rate:    {wr:.1f}%")
        if avg_r > 1.0:
            print("✅ RECENT PERFORMANCE STRONG")
        elif avg_r > 0:
            print("⚡ RECENT PERFORMANCE MODERATE")
        else:
            print("⚠️ RECENT PERFORMANCE WEAK")
        return avg_r
    return 0

# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 70)
    print("🔬 COMPREHENSIVE 100 SYMBOL VALIDATION")
    print("=" * 70)
    print("Config: RR=3.0, SL=0.8×ATR, 1-Candle Structure Break")
    print("=" * 70)
    
    print("\n📥 Fetching top 100 symbols...")
    symbols = get_top_100_symbols()
    print(f"Found {len(symbols)} symbols")
    
    print(f"\n📥 Loading 60-day data...")
    datasets = {}
    for i, sym in enumerate(symbols):
        df = fetch_data(sym)
        if not df.empty: datasets[sym] = calc_indicators(df)
        if (i+1) % 20 == 0: print(f"Progress: {i+1}/{len(symbols)}...")
    print(f"✅ Loaded {len(datasets)} symbols")
    
    print("\n🔄 Running full backtest...")
    all_trades = []
    for sym, df in datasets.items():
        all_trades.extend(run_strategy(df, sym))
    
    if all_trades:
        total_r = sum(t['r'] for t in all_trades)
        avg_r = total_r / len(all_trades)
        wr = sum(1 for t in all_trades if t['win']) / len(all_trades) * 100
        
        print("\n" + "=" * 70)
        print("📊 60-DAY FULL RESULTS")
        print("=" * 70)
        print(f"Symbols:     {len(datasets)}")
        print(f"Trades:      {len(all_trades)}")
        print(f"Net R:       {total_r:+.1f}R")
        print(f"Avg R:       {avg_r:+.3f}R")
        print(f"Win Rate:    {wr:.1f}%")
        print("=" * 70)
        
        # Validation tests
        wf_results = walk_forward(datasets)
        mc_prob = monte_carlo(all_trades)
        drawdown_analysis(all_trades)
        losing_count = symbol_quality(all_trades)
        recent_r = recent_30_day(symbols)
        
        # Final verdict
        print("\n" + "=" * 70)
        print("🏆 FINAL VALIDATION VERDICT")
        print("=" * 70)
        
        profitable_periods = sum(1 for r in wf_results if r['avg_r'] > 0) if wf_results else 0
        
        score = 0
        if avg_r > 1.5: score += 2
        elif avg_r > 0.5: score += 1
        if profitable_periods >= 5: score += 2
        elif profitable_periods >= 4: score += 1
        if mc_prob >= 95: score += 2
        elif mc_prob >= 80: score += 1
        if losing_count <= 5: score += 1
        if recent_r > 1.0: score += 2
        elif recent_r > 0: score += 1
        
        print(f"\n📊 Validation Score: {score}/10")
        
        if score >= 8:
            print("\n✅ **STRONGLY VALIDATED** - Ready for live trading!")
            print("   - High expectancy confirmed")
            print("   - Consistent across all periods")
            print("   - Minimal losing symbols")
        elif score >= 5:
            print("\n⚡ **MODERATELY VALIDATED** - Proceed with caution")
            print("   - Generally profitable but some concerns")
            print("   - Consider reducing symbol count")
        else:
            print("\n⚠️ **POORLY VALIDATED** - Do not deploy")
            print("   - Inconsistent results")
            print("   - Need further investigation")

if __name__ == "__main__":
    main()
