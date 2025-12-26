#!/usr/bin/env python3
"""
100 SYMBOLS TEST - Structure Break Strategy
===========================================

Compare performance on 100 most liquid symbols vs current 20.
Will help determine: Quality (20 symbols) vs Quantity (100 symbols)

Same config: RR=3.0, SL=0.8, 1-candle structure break
"""

import pandas as pd
import numpy as np
import requests
import time

DAYS = 60
TIMEFRAME = 5
RR = 3.0
SL_MULT = 0.8
WIN_COST = 0.0006
LOSS_COST = 0.00125

# ============================================================================
# FETCH TOP 100 SYMBOLS
# ============================================================================

def get_top_100_symbols():
    """Fetch top 100 USDT perpetuals by 24h volume"""
    try:
        url = "https://api.bybit.com/v5/market/tickers?category=linear"
        resp = requests.get(url, timeout=10).json()
        tickers = resp.get('result', {}).get('list', [])
        
        # Filter USDT pairs
        usdt_pairs = [t for t in tickers if t['symbol'].endswith('USDT')]
        
        # Sort by volume
        usdt_pairs.sort(key=lambda x: float(x.get('turnover24h', 0)), reverse=True)
        
        # Get top 100
        top_100 = [t['symbol'] for t in usdt_pairs[:100]]
        
        # Filter bad symbols
        BAD = ['XAUTUSDT', 'PAXGUSDT', 'USTCUSDT', 'USDCUSDT', 'BUSDUSDT', 'DAIUSDT']
        top_100 = [s for s in top_100 if s not in BAD]
        
        return top_100[:100]
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
    
    # Swing Points
    df['swing_high_10'] = df['high'].rolling(10).max()
    df['swing_low_10'] = df['low'].rolling(10).min()
    
    # Divergence
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

def run_strategy(df, symbol=''):
    trades = []
    cooldown = 0
    
    for i in range(50, len(df)-1):
        if cooldown > 0: cooldown -= 1; continue
        
        row = df.iloc[i]
        
        # Divergence
        side = None
        if row['reg_bull']: side = 'long'
        elif row['reg_bear']: side = 'short'
        if not side: continue
        
        # Structure break (1 candle)
        if i+1 >= len(df): continue
        next_row = df.iloc[i+1]
        
        structure_broken = False
        if side == 'long' and next_row['close'] > row['swing_high_10']:
            structure_broken = True
        if side == 'short' and next_row['close'] < row['swing_low_10']:
            structure_broken = True
        
        if not structure_broken: continue
        
        # Entry
        entry = df.iloc[i+1]['open']
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
            'r': res_r,
            'win': outcome == 'win'
        })
        
        cooldown = 5
    
    return trades

def main():
    print("=" * 70)
    print("🔬 100 SYMBOLS TEST - Structure Break Strategy")
    print("=" * 70)
    print("Fetching top 100 liquid symbols by 24h volume...")
    
    symbols = get_top_100_symbols()
    print(f"✅ Found {len(symbols)} symbols")
    print(f"Top 10: {', '.join(symbols[:10])}")
    
    print(f"\n📥 Loading {DAYS}-day data for {len(symbols)} symbols...")
    print("(This will take a few minutes...)")
    
    datasets = {}
    failed = []
    
    for i, sym in enumerate(symbols):
        if (i+1) % 10 == 0:
            print(f"Progress: {i+1}/{len(symbols)}...")
        
        df = fetch_data(sym)
        if not df.empty:
            datasets[sym] = calc_indicators(df)
        else:
            failed.append(sym)
    
    print(f"✅ Loaded {len(datasets)} symbols ({len(failed)} failed)")
    
    print("\n🔄 Running backtest...")
    all_trades = []
    
    for sym, df in datasets.items():
        trades = run_strategy(df, sym)
        all_trades.extend(trades)
    
    # Results
    if all_trades:
        total_r = sum(t['r'] for t in all_trades)
        avg_r = total_r / len(all_trades)
        wins = sum(1 for t in all_trades if t['win'])
        wr = wins / len(all_trades) * 100
        
        print("\n" + "=" * 70)
        print("📊 100 SYMBOLS - RESULTS")
        print("=" * 70)
        print(f"Symbols Tested:  {len(datasets)}")
        print(f"Total Trades:    {len(all_trades)}")
        print(f"Net R:           {total_r:+.1f}R")
        print(f"Avg R per Trade: {avg_r:+.3f}R")
        print(f"Win Rate:        {wr:.1f}%")
        print(f"Trades/Symbol:   {len(all_trades)/len(datasets):.1f}")
        print("=" * 70)
        
        # Per-symbol breakdown (top/bottom performers)
        symbol_stats = {}
        for t in all_trades:
            if t['symbol'] not in symbol_stats:
                symbol_stats[t['symbol']] = []
            symbol_stats[t['symbol']].append(t)
        
        symbol_results = []
        for sym, trades in symbol_stats.items():
            net_r = sum(t['r'] for t in trades)
            avg_r = net_r / len(trades)
            wins = sum(1 for t in trades if t['win'])
            wr = wins / len(trades) * 100
            symbol_results.append({
                'symbol': sym,
                'trades': len(trades),
                'net_r': net_r,
                'avg_r': avg_r,
                'wr': wr
            })
        
        symbol_results.sort(key=lambda x: x['avg_r'], reverse=True)
        
        print("\n🏆 TOP 10 PERFORMERS")
        print("-" * 70)
        for s in symbol_results[:10]:
            print(f"{s['symbol']:<12} | {s['trades']:>3} trades | {s['net_r']:>+7.1f}R | {s['avg_r']:>+.3f}R | {s['wr']:>5.1f}%")
        
        print("\n⚠️ BOTTOM 10 PERFORMERS")
        print("-" * 70)
        for s in symbol_results[-10:]:
            print(f"{s['symbol']:<12} | {s['trades']:>3} trades | {s['net_r']:>+7.1f}R | {s['avg_r']:>+.3f}R | {s['wr']:>5.1f}%")
        
        # Comparison
        print("\n" + "=" * 70)
        print("📊 COMPARISON: 20 vs 100 Symbols")
        print("=" * 70)
        print(f"{'Metric':<20} | {'20 Symbols':<15} | {'100 Symbols':<15}")
        print("-" * 70)
        print(f"{'Trades':<20} | {'92':<15} | {len(all_trades):<15}")
        print(f"{'Avg R':<20} | {'+2.064R':<15} | {f'{avg_r:+.3f}R':<15}")
        print(f"{'Win Rate':<20} | {'84.8%':<15} | {f'{wr:.1f}%':<15}")
        print(f"{'Net R':<20} | {'+189.9R':<15} | {f'{total_r:+.1f}R':<15}")
        print(f"{'Trades/Symbol':<20} | {'4.6':<15} | {f'{len(all_trades)/len(datasets):.1f}':<15}")
        
        # Recommendation
        print("\n" + "=" * 70)
        print("💡 RECOMMENDATION")
        print("=" * 70)
        
        if avg_r > 2.0 and wr > 80:
            print("✅ 100 symbols MAINTAINS quality!")
            print("   - More opportunities (5x trades)")
            print("   - Similar or better per-trade performance")
            print("   → RECOMMEND: Use 100 symbols for more trades")
        elif avg_r > 1.5 and total_r > 300:
            print("✅ 100 symbols = MORE PROFIT (quantity wins)")
            print("   - Slightly lower quality per trade")
            print("   - But much higher total profit")
            print("   → RECOMMEND: Use 100 symbols for volume")
        elif avg_r < 1.0:
            print("⚠️ 100 symbols DEGRADES quality")
            print("   - Lower win rate on less liquid symbols")
            print("   - Not worth the extra trades")
            print("   → RECOMMEND: Stick with 20 high-quality symbols")
        else:
            print("⚖️ MIXED results")
            print(f"   - 20 symbols: Higher quality ({'+2.064R'} avg)")
            print(f"   - 100 symbols: More volume ({len(all_trades)} trades)")
            print("   → YOUR CHOICE: Quality vs Quantity")

if __name__ == "__main__":
    main()
