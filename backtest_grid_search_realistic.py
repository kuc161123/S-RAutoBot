#!/usr/bin/env python3
"""
EXHAUSTIVE REALISTIC GRID SEARCH
================================
Tests all combinations of parameters using REALISTIC execution (no look-ahead bias).
Entry at NEXT candle's OPEN after structure break confirmation.

Grid:
- Max Wait Candles: 1-10
- R:R Ratio: 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0
- SL Multiplier: 0.5, 0.8, 1.0, 1.2, 1.5, 2.0
"""

import pandas as pd
import numpy as np
import requests
import time
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import itertools

# Base Configuration
DAYS = 60  # 60 days for faster iteration
TIMEFRAME = 5
FEE_PERCENT = 0.0006
SLIPPAGE_PERCENT = 0.0003

# Grid search parameters
WAIT_CANDLES_RANGE = list(range(1, 11))  # 1-10
RR_RANGE = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]
SL_MULT_RANGE = [0.5, 0.8, 1.0, 1.2, 1.5, 2.0]

def get_top_symbols(n=30):
    """Get top symbols by volume - fewer for speed"""
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
    """Fetch OHLCV data"""
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
            time.sleep(0.015)
        if not all_kline: return pd.DataFrame()
        df = pd.DataFrame(all_kline, columns=['ts', 'open', 'high', 'low', 'close', 'vol', 'to'])
        df = df.iloc[::-1].reset_index(drop=True)
        for c in ['open', 'high', 'low', 'close', 'vol']: df[c] = df[c].astype(float)
        return df
    except: return pd.DataFrame()

def calc_indicators(df):
    """Calculate all indicators"""
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

def run_strategy_realistic(df, max_wait, rr, sl_mult, slippage_pct=SLIPPAGE_PERCENT):
    """
    REALISTIC execution: Entry at NEXT candle's OPEN after structure break.
    Returns list of trade results.
    """
    trades = []
    cooldown = 0
    
    for i in range(50, len(df)-2):
        if cooldown > 0: 
            cooldown -= 1
            continue
            
        row = df.iloc[i]
        side = 'long' if row['reg_bull'] else 'short' if row['reg_bear'] else None
        if not side: continue
        
        # Wait for structure break
        structure_broken, candles_waited = False, 0
        for ahead in range(1, max_wait + 1):
            if i+ahead >= len(df): break
            check = df.iloc[i+ahead]
            candles_waited = ahead
            if (side == 'long' and check['close'] > row['swing_high_10']) or \
               (side == 'short' and check['close'] < row['swing_low_10']):
                structure_broken = True
                break
        
        if not structure_broken: continue
        
        # REALISTIC: Enter at NEXT candle's OPEN
        confirm_idx = i + candles_waited
        entry_idx = confirm_idx + 1
        
        if entry_idx >= len(df): continue
        
        base = df.iloc[entry_idx]['open']
        entry = base * (1 + slippage_pct) if side == 'long' else base * (1 - slippage_pct)
        
        atr = row['atr']
        if pd.isna(atr) or atr <= 0: continue
        
        sl_dist = atr * sl_mult
        if sl_dist/entry > 0.05: continue  # Skip if SL > 5%
        
        tp_dist = sl_dist * rr
        sl, tp = (entry - sl_dist, entry + tp_dist) if side == 'long' else (entry + sl_dist, entry - tp_dist)
        
        outcome = 'timeout'
        for j in range(entry_idx, min(entry_idx+300, len(df))):
            c = df.iloc[j]
            if side == 'long':
                if c['low'] <= sl: outcome = 'loss'; break
                if c['high'] >= tp: outcome = 'win'; break
            else:
                if c['high'] >= sl: outcome = 'loss'; break
                if c['low'] <= tp: outcome = 'win'; break
        
        risk_pct = sl_dist / entry
        fee_cost = (FEE_PERCENT + slippage_pct) / risk_pct
        res_r = rr - fee_cost if outcome == 'win' else -1.0 - fee_cost if outcome == 'loss' else -0.2
        trades.append({'r': res_r, 'win': outcome == 'win', 'side': side})
        cooldown = 6
    
    return trades

def test_combination(args):
    """Test a single parameter combination across all symbols"""
    max_wait, rr, sl_mult, all_dfs = args
    
    all_trades = []
    for sym, df in all_dfs.items():
        if len(df) < 100: continue
        trades = run_strategy_realistic(df, max_wait, rr, sl_mult)
        all_trades.extend(trades)
    
    if not all_trades:
        return None
    
    total_r = sum(t['r'] for t in all_trades)
    wins = sum(1 for t in all_trades if t['win'])
    wr = wins / len(all_trades) * 100 if all_trades else 0
    avg_r = total_r / len(all_trades) if all_trades else 0
    
    return {
        'max_wait': max_wait,
        'rr': rr,
        'sl_mult': sl_mult,
        'trades': len(all_trades),
        'wr': wr,
        'total_r': total_r,
        'avg_r': avg_r
    }

def main():
    print("=" * 70)
    print("EXHAUSTIVE REALISTIC GRID SEARCH")
    print("=" * 70)
    print(f"Days: {DAYS} | TF: {TIMEFRAME}m | Realistic Execution (Next Candle Open)")
    print(f"Wait Range: 1-{max(WAIT_CANDLES_RANGE)} | RR Range: {min(RR_RANGE)}-{max(RR_RANGE)}")
    print(f"SL Mult Range: {min(SL_MULT_RANGE)}-{max(SL_MULT_RANGE)}")
    print()
    
    # Fetch data for all symbols first
    symbols = get_top_symbols(30)
    print(f"Fetching data for {len(symbols)} symbols...")
    
    all_dfs = {}
    for i, sym in enumerate(symbols):
        print(f"  [{i+1}/{len(symbols)}] {sym}...", end=" ", flush=True)
        df = fetch_data(sym)
        if len(df) < 100:
            print("skip")
            continue
        df = calc_indicators(df)
        if len(df) < 100:
            print("skip")
            continue
        all_dfs[sym] = df
        print(f"{len(df)} candles")
    
    print(f"\nLoaded {len(all_dfs)} symbols with data.")
    
    # Generate all combinations
    combinations = list(itertools.product(WAIT_CANDLES_RANGE, RR_RANGE, SL_MULT_RANGE))
    total = len(combinations)
    print(f"\nTesting {total} parameter combinations...")
    print()
    
    results = []
    
    for idx, (max_wait, rr, sl_mult) in enumerate(combinations):
        if idx % 20 == 0:
            print(f"  Progress: {idx}/{total} ({idx*100//total}%)")
        
        result = test_combination((max_wait, rr, sl_mult, all_dfs))
        if result:
            results.append(result)
    
    print(f"\n{'='*70}")
    print("GRID SEARCH RESULTS")
    print(f"{'='*70}")
    
    # Filter profitable configs
    profitable = [r for r in results if r['avg_r'] > 0]
    
    if not profitable:
        print("\n❌ NO PROFITABLE CONFIGURATIONS FOUND")
        print("\nTop 10 Least Losing Configurations:")
        results.sort(key=lambda x: x['avg_r'], reverse=True)
        for i, r in enumerate(results[:10]):
            print(f"  {i+1}. Wait={r['max_wait']} RR={r['rr']} SL={r['sl_mult']}x | "
                  f"WR={r['wr']:.1f}% | Avg={r['avg_r']:+.3f}R | Total={r['total_r']:+.1f}R | N={r['trades']}")
    else:
        print(f"\n✅ FOUND {len(profitable)} PROFITABLE CONFIGURATIONS!")
        profitable.sort(key=lambda x: x['avg_r'], reverse=True)
        
        print("\n🏆 TOP 20 BEST CONFIGURATIONS:")
        print("-" * 70)
        for i, r in enumerate(profitable[:20]):
            print(f"  {i+1}. Wait={r['max_wait']} RR={r['rr']} SL={r['sl_mult']}x | "
                  f"WR={r['wr']:.1f}% | Avg={r['avg_r']:+.3f}R | Total={r['total_r']:+.1f}R | N={r['trades']}")
        
        # Best overall
        best = profitable[0]
        print("\n" + "=" * 70)
        print("🥇 RECOMMENDED CONFIGURATION:")
        print("=" * 70)
        print(f"   Max Wait Candles: {best['max_wait']}")
        print(f"   Risk:Reward Ratio: {best['rr']}:1")
        print(f"   SL Multiplier: {best['sl_mult']}x ATR")
        print(f"   Expected Win Rate: {best['wr']:.1f}%")
        print(f"   Expected Avg R/Trade: {best['avg_r']:+.3f}R")
        print(f"   Backtest Total R: {best['total_r']:+.1f}R")
        print(f"   Total Trades: {best['trades']}")
        print("=" * 70)
    
    # Also check by side
    print("\n📊 SIDE ANALYSIS (Using best config if profitable, else top config):")
    best_config = profitable[0] if profitable else results[0]
    
    # Re-run best config to get side breakdown
    all_trades = []
    for sym, df in all_dfs.items():
        trades = run_strategy_realistic(df, best_config['max_wait'], best_config['rr'], best_config['sl_mult'])
        all_trades.extend(trades)
    
    longs = [t for t in all_trades if t['side'] == 'long']
    shorts = [t for t in all_trades if t['side'] == 'short']
    
    if longs:
        wr_l = sum(1 for t in longs if t['win']) / len(longs) * 100
        r_l = sum(t['r'] for t in longs)
        avg_l = r_l / len(longs)
        print(f"   📈 LONG: {len(longs)} trades | WR={wr_l:.1f}% | Avg={avg_l:+.3f}R | Total={r_l:+.1f}R")
    
    if shorts:
        wr_s = sum(1 for t in shorts if t['win']) / len(shorts) * 100
        r_s = sum(t['r'] for t in shorts)
        avg_s = r_s / len(shorts)
        print(f"   📉 SHORT: {len(shorts)} trades | WR={wr_s:.1f}% | Avg={avg_s:+.3f}R | Total={r_s:+.1f}R")
    
    # Save results to CSV
    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values('avg_r', ascending=False)
    df_results.to_csv('grid_search_realistic_results.csv', index=False)
    print(f"\n📁 Full results saved to: grid_search_realistic_results.csv")

if __name__ == "__main__":
    main()
